"""
XSP Iron Condor Strategy — Entry, defense monitoring, and exit logic.

Sleeve 1: Sells an iron condor on XSP at 10:45 AM, monitors for defense
trigger (SPY 1.4% move from anchor), holds to 4:00 PM cash settlement.
"""
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from bot.condor_config import CondorConfig
from bot.options_client import (
    CondorLegs,
    CondorSpreadQuote,
    build_condor_legs,
    cancel_order,
    close_condor_order,
    cancel_all_orders,
    get_buying_power,
    get_condor_spread_quote,
    get_todays_expiration,
    place_iron_condor_order,
    replace_order,
    size_condor,
    get_order,
)
from bot.clock import market_now, parse_time_str
from bot.market_data import MorningTracker

logger = logging.getLogger("bot.condor")


@dataclass
class CondorState:
    """Runtime state for an active iron condor position."""
    legs: Optional[CondorLegs] = None
    qty: int = 0
    entry_credit: float = 0.0
    entry_order_id: Optional[str] = None
    entry_time: Optional[datetime] = None
    anchor_price: float = 0.0

    # Defense
    defense_triggered: bool = False
    defense_order_id: Optional[str] = None
    defense_time: Optional[datetime] = None
    defense_filled: bool = False
    defense_fill_price: Optional[float] = None  # actual debit paid to close
    defense_close_failed: bool = False  # True if close order was rejected/cancelled

    # Early exit (discretionary close before settlement)
    early_exit_order_id: Optional[str] = None
    early_exit_filled: bool = False
    early_exit_fill_price: Optional[float] = None
    early_exit_order_dead: bool = False  # terminal — stop polling

    # Defense order terminal tracking
    defense_order_dead: bool = False  # terminal — stop polling

    # Status
    # is_open means the position is live (entry filled).
    # Before fill confirmation, only entry_order_id is set.
    # entry_order_dead: set True when the entry order reaches a terminal
    # state (cancelled/expired/rejected) so we stop polling it.
    is_open: bool = False
    is_filled: bool = False
    entry_order_dead: bool = False
    settled: bool = False

    # P&L tracking
    exit_cost: float = 0.0
    realized_pnl: float = 0.0

    # Fill optimization state
    entry_natural_credit: float = 0.0    # natural credit at entry time
    entry_mid_credit: float = 0.0        # mid credit at entry time
    entry_best_credit: float = 0.0       # best credit at entry time
    entry_limit_price: float = 0.0       # current limit on the live order
    entry_adjustments: int = 0           # number of price adjustments made
    entry_last_adjust_ts: float = 0.0    # monotonic timestamp of last adjustment
    entry_spread_quote: Optional[CondorSpreadQuote] = None  # initial quote snapshot
    entry_qty_reduced: bool = False      # True if qty was reduced for wide spreads
    entry_peak_natural: float = 0.0      # rolling peak natural for deterioration check
    missed_trade_reason: Optional[str] = None  # reason if entry was skipped/aborted


class CondorStrategy:
    """
    Manages the XSP iron condor sleeve.

    Lifecycle:
    1. enter() — at 10:45 AM, compute strikes, size, place mleg order
    2. check_defense() — every minute, check if SPY breached 1.4% from anchor
    3. close_defense() — if breached, close all legs immediately
    4. on_settlement() — at 4:00 PM, record cash settlement result
    """

    def __init__(self, config: CondorConfig, dry_run: bool = False):
        self.cfg = config
        self.dry_run = dry_run
        self.state = CondorState()

    def enter(self, tracker: MorningTracker) -> bool:
        """
        Enter the iron condor at 10:45 AM.

        Steps:
        1. Record anchor price (SPY at 10:45)
        2. Compute strikes (0.90% OTM short, 1.00% wing)
        3. Fetch XSP 0DTE contracts and find best legs
        4. Get live spread quotes — check spread health
        5. Size: floor(buying_power / max_loss_per_contract), reduce on wide spreads
        6. Place mleg limit order starting at mid credit

        Returns True if order was placed successfully.
        """
        # Reset sticky state for clean entry attempt
        self.state.missed_trade_reason = None
        self.state.entry_peak_natural = 0.0
        # Step 1: Anchor price
        anchor = tracker.spy_last
        if anchor <= 0:
            logger.error("Cannot enter condor: SPY price unavailable (%.2f)", anchor)
            self.state.missed_trade_reason = "anchor_unavailable"
            logger.info("MISSED TRADE: reason=anchor_unavailable spy_price=%.2f", anchor)
            return False

        tracker.condor_anchor = anchor
        tracker.reset_post_anchor_tracking()
        self.state.anchor_price = anchor
        logger.info("CONDOR ENTRY: Anchor price = $%.2f (SPY)", anchor)

        # Step 2-3: Build legs
        expiration = get_todays_expiration()
        legs = build_condor_legs(
            anchor_price=anchor,
            option_root=self.cfg.option_root,
            expiration_date=expiration,
            short_pct=self.cfg.short_strike_pct,
            wing_pct=self.cfg.wing_width_pct,
        )

        if legs is None:
            logger.error("CONDOR ENTRY FAILED: Could not build legs")
            self.state.missed_trade_reason = "leg_build_failed"
            logger.info("MISSED TRADE: reason=leg_build_failed")
            return False

        self.state.legs = legs

        # Step 4: Get live spread quotes
        spread_quote = get_condor_spread_quote(legs)
        self.state.entry_spread_quote = spread_quote

        if not spread_quote.valid:
            logger.error("CONDOR ENTRY FAILED: Could not get quotes for all 4 legs")
            self.state.missed_trade_reason = "quote_unavailable"
            logger.info("MISSED TRADE: reason=quote_unavailable")
            return False

        self.state.entry_natural_credit = spread_quote.natural_credit
        self.state.entry_mid_credit = spread_quote.mid_credit
        self.state.entry_best_credit = spread_quote.best_credit
        self.state.entry_peak_natural = spread_quote.natural_credit

        # Spread health gate — skip if any leg has excessively wide spread
        if spread_quote.max_leg_spread_pct > self.cfg.max_leg_spread_pct:
            logger.error(
                "CONDOR ENTRY REJECTED: max leg spread %.1f%% exceeds limit %.1f%%",
                spread_quote.max_leg_spread_pct * 100,
                self.cfg.max_leg_spread_pct * 100,
            )
            self.state.missed_trade_reason = "wide_spread"
            logger.info(
                "MISSED TRADE: reason=wide_spread max_spread=%.1f%% limit=%.1f%%",
                spread_quote.max_leg_spread_pct * 100,
                self.cfg.max_leg_spread_pct * 100,
            )
            return False

        # Walk away if natural credit is below minimum
        if spread_quote.natural_credit < self.cfg.fill_min_credit:
            logger.error(
                "CONDOR ENTRY REJECTED: natural credit $%.2f below minimum $%.2f",
                spread_quote.natural_credit, self.cfg.fill_min_credit,
            )
            self.state.missed_trade_reason = "low_credit"
            logger.info(
                "MISSED TRADE: reason=low_credit natural=$%.2f min=$%.2f",
                spread_quote.natural_credit, self.cfg.fill_min_credit,
            )
            return False

        # Step 5: Sizing
        max_wing = legs.max_wing_width
        if max_wing <= 0:
            logger.error("CONDOR ENTRY FAILED: max_wing_width is zero (bad leg geometry)")
            self.state.missed_trade_reason = "bad_geometry"
            logger.info("MISSED TRADE: reason=bad_geometry max_wing=0")
            return False

        net_max_loss = max_wing - self.cfg.target_credit
        if net_max_loss <= 0:
            logger.error(
                "CONDOR ENTRY FAILED: net max loss <= 0 (wing=$%.2f credit=$%.2f)",
                max_wing, self.cfg.target_credit,
            )
            self.state.missed_trade_reason = "bad_geometry"
            logger.info("MISSED TRADE: reason=bad_geometry net_max_loss<=0")
            return False

        # Credit / risk ratio gate — protect edge integrity
        # Note: Using natural_credit / max_wing as directional filter (not true credit/max_loss)
        credit_risk_ratio = spread_quote.natural_credit / max_wing if max_wing > 0 else 0.0
        if credit_risk_ratio < self.cfg.min_credit_risk_ratio:
            logger.error(
                "CONDOR ENTRY REJECTED: credit/risk ratio %.2f below minimum %.2f "
                "(credit=$%.2f wing=$%.2f) - note: ratio uses wing not max_loss",
                credit_risk_ratio, self.cfg.min_credit_risk_ratio,
                spread_quote.natural_credit, max_wing,
            )
            self.state.missed_trade_reason = "poor_risk_reward"
            logger.info(
                "MISSED TRADE: reason=poor_risk_reward ratio=%.2f min=%.2f credit=$%.2f wing=$%.2f",
                credit_risk_ratio, self.cfg.min_credit_risk_ratio,
                spread_quote.natural_credit, max_wing,
            )
            return False

        if abs(net_max_loss - self.cfg.max_loss_per_contract) > 0.50:
            logger.warning(
                "Condor net max loss ($%.2f) differs from config estimate ($%.2f) "
                "— sizing will use actual strikes (wing=$%.2f - credit=$%.2f)",
                net_max_loss, self.cfg.max_loss_per_contract,
                max_wing, self.cfg.target_credit,
            )

        buying_power = get_buying_power()
        qty = size_condor(buying_power, net_max_loss)
        if qty <= 0:
            logger.error("CONDOR ENTRY FAILED: Insufficient buying power ($%.2f)", buying_power)
            self.state.missed_trade_reason = "insufficient_bp"
            logger.info("MISSED TRADE: reason=insufficient_bp bp=$%.2f net_max_loss=$%.2f", buying_power, net_max_loss)
            return False

        # Reduce size on moderately wide spreads
        if spread_quote.avg_leg_spread_pct > self.cfg.wide_spread_size_reduce_pct:
            original_qty = qty
            qty = max(1, math.floor(qty * self.cfg.wide_spread_size_factor))
            self.state.entry_qty_reduced = True
            logger.warning(
                "CONDOR SIZE REDUCED: avg spread %.1f%% > %.1f%% threshold — "
                "qty %d → %d (factor=%.2f)",
                spread_quote.avg_leg_spread_pct * 100,
                self.cfg.wide_spread_size_reduce_pct * 100,
                original_qty, qty, self.cfg.wide_spread_size_factor,
            )

        self.state.qty = qty

        # Step 6: Determine starting limit price
        if self.cfg.fill_start_at_mid and spread_quote.mid_credit > 0:
            # Start at mid — don't cross the spread
            starting_credit = round(spread_quote.mid_credit, 2)
            # Ensure we don't start below natural (that would never fill)
            starting_credit = max(starting_credit, round(spread_quote.natural_credit, 2))
            logger.info(
                "SMART FILL: starting at mid=$%.2f (natural=$%.2f best=$%.2f target=$%.2f)",
                starting_credit, spread_quote.natural_credit,
                spread_quote.best_credit, self.cfg.target_credit,
            )
        else:
            starting_credit = self.cfg.target_credit

        # Place order
        try:
            order = place_iron_condor_order(
                legs=legs,
                qty=qty,
                limit_price=starting_credit,
                time_in_force=self.cfg.time_in_force,
                dry_run=self.dry_run,
            )

            if order:
                self.state.entry_order_id = order.get("id")
                self.state.entry_credit = starting_credit
                self.state.entry_limit_price = starting_credit
                self.state.entry_time = datetime.now()
                self.state.entry_last_adjust_ts = time.monotonic()
                logger.info(
                    "CONDOR ORDER SUBMITTED: %d contracts @ $%.2f credit, order_id=%s",
                    qty, starting_credit, self.state.entry_order_id,
                )
                return True
        except Exception as e:
            logger.error("CONDOR ENTRY ORDER FAILED: %s", e)
            self.state.missed_trade_reason = "order_submission_failed"
            logger.info("MISSED TRADE: reason=order_submission_failed error=%s", str(e))

        return False

    def check_fill_status(self) -> bool:
        """Check if the entry order has been filled.  Sets is_open = True only on fill."""
        if self.state.is_filled:
            return True
        if not self.state.entry_order_id or self.state.entry_order_dead:
            return False

        if self.dry_run:
            self.state.is_filled = True
            self.state.is_open = True
            return True

        try:
            order = get_order(self.state.entry_order_id)
            status = order.get("status", "")

            if status == "filled":
                self.state.is_filled = True
                self.state.is_open = True
                filled_price = float(order.get("filled_avg_price", self.cfg.target_credit))
                self.state.entry_credit = filled_price
                # Fill quality metrics
                mid = self.state.entry_mid_credit
                natural = self.state.entry_natural_credit
                mid_capture = 0.0
                if mid > 0 and natural > 0 and mid != natural:
                    mid_capture = (filled_price - natural) / (mid - natural) * 100
                slippage = mid - filled_price if mid > 0 else 0.0
                logger.info(
                    "CONDOR FILLED: %d contracts @ $%.2f credit "
                    "(mid=$%.2f natural=$%.2f slippage=$%.2f mid_capture=%.0f%% "
                    "adjustments=%d)",
                    self.state.qty, filled_price,
                    mid, natural, slippage, mid_capture,
                    self.state.entry_adjustments,
                )
                return True
            elif status in ("cancelled", "expired", "rejected"):
                logger.warning("CONDOR ORDER %s: %s — marking terminal", status.upper(), self.state.entry_order_id)
                self.state.entry_order_dead = True
                return False
            else:
                logger.debug("Condor order status: %s", status)
        except Exception as e:
            logger.warning("Failed to check condor fill status: %s", e)

        return False

    def adjust_entry_if_needed(self) -> bool:
        """
        Gradually adjust the entry limit price toward the *live* natural credit.

        Called by the orchestrator on each poll cycle while the entry order
        is open and unfilled.  Respects fill_patience_secs between adjustments
        and stops after fill_max_adjustments.

        Key behaviours:
        - Re-fetches live spread quote on each adjustment (no stale anchoring).
        - Uses atomic PATCH replace_order to avoid cancel→replace gap.
        - Falls back to cancel→replace if PATCH fails (e.g. pending_replace).
        - Aborts if conditions have deteriorated (natural dropped too far).
        - Aborts if past fill_max_entry_time cutoff.

        Returns True if an adjustment was made, False otherwise.
        """
        # Guard: only adjust if we have a live, unfilled entry order
        if (self.state.is_filled or self.state.entry_order_dead
                or not self.state.entry_order_id):
            return False

        # ── Time cutoff ──────────────────────────────────────────────
        now = market_now()
        cutoff = parse_time_str(self.cfg.fill_max_entry_time)
        if now.time() >= cutoff:
            logger.warning(
                "SMART FILL: past %s cutoff — cancelling unfilled entry",
                self.cfg.fill_max_entry_time,
            )
            try:
                cancel_order(self.state.entry_order_id)
            except Exception as e:
                logger.warning("Failed to cancel condor entry at cutoff: %s", e)
            self.state.entry_order_dead = True
            self.state.missed_trade_reason = "time_cutoff"
            logger.info(
                "MISSED TRADE: reason=time_cutoff cutoff=%s adjustments=%d last_limit=$%.2f",
                self.cfg.fill_max_entry_time, self.state.entry_adjustments,
                self.state.entry_limit_price,
            )
            return False

        # Respect patience timer
        elapsed = time.monotonic() - self.state.entry_last_adjust_ts
        if elapsed < self.cfg.fill_patience_secs:
            return False

        # Respect max adjustments
        if self.state.entry_adjustments >= self.cfg.fill_max_adjustments:
            logger.info(
                "SMART FILL: max adjustments (%d) reached — leaving order working at $%.2f "
                "until fill or entry cutoff",
                self.cfg.fill_max_adjustments, self.state.entry_limit_price,
            )
            return False

        # ── Re-fetch live quotes ─────────────────────────────────────
        try:
            live_quote = get_condor_spread_quote(self.state.legs)
        except Exception as e:
            logger.warning("SMART FILL: quote fetch failed, skipping adjustment: %s", e)
            self.state.entry_last_adjust_ts = time.monotonic()
            return False

        if not live_quote.valid:
            logger.warning("SMART FILL: incomplete live quote, skipping adjustment")
            self.state.entry_last_adjust_ts = time.monotonic()
            return False

        live_natural = live_quote.natural_credit
        live_mid = live_quote.mid_credit

        # ── Update peak natural (rolling high-water mark) ────────────
        self.state.entry_peak_natural = max(
            self.state.entry_peak_natural, live_natural,
        )

        # ── Deterioration guard (vs peak, not initial) ───────────────
        peak = self.state.entry_peak_natural
        if peak > 0:
            drop_pct = (peak - live_natural) / peak
            if drop_pct > self.cfg.fill_deterioration_pct:
                logger.warning(
                    "SMART FILL ABORT: natural deteriorated %.1f%% from peak "
                    "(peak=$%.2f → live=$%.2f, threshold=%.0f%%) — cancelling",
                    drop_pct * 100, peak, live_natural,
                    self.cfg.fill_deterioration_pct * 100,
                )
                try:
                    cancel_order(self.state.entry_order_id)
                except Exception as e:
                    logger.warning("Failed to cancel condor entry on deterioration: %s", e)
                self.state.entry_order_dead = True
                self.state.missed_trade_reason = "deterioration"
                logger.info(
                    "MISSED TRADE: reason=deterioration peak=$%.2f live=$%.2f drop=%.1f%%",
                    peak, live_natural, drop_pct * 100,
                )
                return False

        # ── Dynamic step: adapts to spread width ─────────────────────
        spread_gap = live_mid - live_natural if live_mid > live_natural else 0.0
        step = max(self.cfg.fill_adjust_step_min,
                    round(self.cfg.fill_adjust_step_frac * spread_gap, 2))

        # ── Compute new limit using LIVE natural as floor ────────────
        new_limit = round(self.state.entry_limit_price - step, 2)
        natural = round(live_natural, 2)

        # Don't go below live natural — that's the floor
        new_limit = max(new_limit, natural)

        # Walk away if below minimum credit
        if new_limit < self.cfg.fill_min_credit:
            logger.warning(
                "SMART FILL: next limit $%.2f below min credit $%.2f — "
                "cancelling order and giving up",
                new_limit, self.cfg.fill_min_credit,
            )
            try:
                cancel_order(self.state.entry_order_id)
            except Exception as e:
                logger.warning("Failed to cancel condor entry order: %s", e)
            self.state.entry_order_dead = True
            self.state.missed_trade_reason = "min_credit_breach"
            logger.info(
                "MISSED TRADE: reason=min_credit_breach limit=$%.2f min=$%.2f",
                new_limit, self.cfg.fill_min_credit,
            )
            return False

        # No change needed if we're already at or below live natural
        if new_limit == self.state.entry_limit_price:
            logger.info(
                "SMART FILL: already at live natural $%.2f — no further adjustment",
                new_limit,
            )
            self.state.entry_last_adjust_ts = time.monotonic()
            return False

        # ── Atomic replace via PATCH (no cancel→replace gap) ─────────
        old_limit = self.state.entry_limit_price
        order = replace_order(self.state.entry_order_id, new_limit)

        if order:
            self.state.entry_order_id = order.get("id", self.state.entry_order_id)
            self.state.entry_limit_price = new_limit
            self.state.entry_credit = new_limit
            self.state.entry_adjustments += 1
            self.state.entry_last_adjust_ts = time.monotonic()
            logger.info(
                "SMART FILL ADJUST #%d: $%.2f → $%.2f (step=$%.2f) "
                "(live_natural=$%.2f live_mid=$%.2f peak=$%.2f) order_id=%s",
                self.state.entry_adjustments, old_limit, new_limit, step,
                live_natural, live_mid, peak, self.state.entry_order_id,
            )
            return True

        # PATCH failed (order in pending state?) — fall back to cancel→replace
        logger.warning("SMART FILL: PATCH replace failed, falling back to cancel→replace")
        try:
            cancel_order(self.state.entry_order_id)
        except Exception as e:
            logger.warning("Failed to cancel condor entry for adjustment: %s", e)
            self.state.entry_last_adjust_ts = time.monotonic()
            return False

        try:
            order = place_iron_condor_order(
                legs=self.state.legs,
                qty=self.state.qty,
                limit_price=new_limit,
                time_in_force=self.cfg.time_in_force,
                dry_run=self.dry_run,
            )
            if order:
                self.state.entry_order_id = order.get("id")
                self.state.entry_limit_price = new_limit
                self.state.entry_credit = new_limit
                self.state.entry_adjustments += 1
                self.state.entry_last_adjust_ts = time.monotonic()
                logger.info(
                    "SMART FILL ADJUST #%d (cancel→replace): $%.2f → $%.2f (step=$%.2f) "
                    "(live_natural=$%.2f peak=$%.2f) order_id=%s",
                    self.state.entry_adjustments, old_limit, new_limit, step,
                    live_natural, peak, self.state.entry_order_id,
                )
                return True
        except Exception as e:
            logger.error("SMART FILL: replacement order failed: %s", e)
            self.state.entry_order_dead = True

        return False

    def check_defense(self, tracker: MorningTracker) -> bool:
        """
        Check if SPY has breached defense_trigger_pct from anchor.

        Uses spy_max_move_from_anchor_pct() which examines post-anchor
        high/low extremes — catching intra-interval breaches even if
        the current last-trade has recovered by the time we poll.

        Returns True if defense should be triggered.
        Only runs when the entry is actually filled (is_filled).
        """
        if not self.state.is_filled or self.state.defense_triggered:
            return False

        # Use max excursion (post-anchor high/low) not just current last
        max_move = tracker.spy_max_move_from_anchor_pct()
        if max_move is None:
            return False

        if max_move >= self.cfg.defense_trigger_pct:
            current_move = tracker.spy_move_from_anchor_pct() or 0.0
            logger.warning(
                "DEFENSE TRIGGER: SPY max_move=%.2f%% (current=%.2f%%) from anchor $%.2f (threshold: %.2f%%)",
                max_move * 100, current_move * 100,
                self.state.anchor_price, self.cfg.defense_trigger_pct * 100,
            )
            return True

        logger.debug(
            "Defense check OK: SPY=%.2f, anchor=%.2f, max_move=%.3f%% < %.2f%%",
            tracker.spy_last, self.state.anchor_price,
            max_move * 100, self.cfg.defense_trigger_pct * 100,
        )
        return False

    _DEFENSE_MAX_RETRIES = 2

    def close_defense(self) -> bool:
        """
        Emergency close all condor legs at market.

        Called when defense trigger fires.  Retries up to
        _DEFENSE_MAX_RETRIES times; on final failure, escalates
        to cancel_all_orders() to prevent further damage.

        Returns True if close order was placed.
        """
        if not self.state.is_filled or not self.state.legs:
            logger.warning("Cannot close condor: no filled position")
            return False

        logger.warning(
            "CONDOR DEFENSE CLOSE: Closing %d contracts at market",
            self.state.qty,
        )

        for attempt in range(1, self._DEFENSE_MAX_RETRIES + 1):
            try:
                order = close_condor_order(
                    legs=self.state.legs,
                    qty=self.state.qty,
                    dry_run=self.dry_run,
                )

                if order:
                    self.state.defense_triggered = True
                    self.state.defense_order_id = order.get("id")
                    self.state.defense_time = datetime.now()
                    logger.info(
                        "CONDOR DEFENSE ORDER PLACED (attempt %d): order_id=%s",
                        attempt, self.state.defense_order_id,
                    )
                    return True
            except Exception as e:
                logger.error("CONDOR DEFENSE CLOSE attempt %d FAILED: %s", attempt, e)

        # All retries exhausted — escalate
        logger.critical(
            "CONDOR DEFENSE CLOSE FAILED after %d attempts — escalating to cancel_all_orders",
            self._DEFENSE_MAX_RETRIES,
        )
        self.state.defense_triggered = True  # Mark triggered so we don't re-enter
        self.state.defense_close_failed = True
        try:
            cancel_all_orders()
            logger.warning("Escalation: cancel_all_orders() called")
        except Exception as e:
            logger.critical("cancel_all_orders() ALSO FAILED: %s", e)
        return False

    def check_defense_fill(self) -> bool:
        """
        Poll the defense-close order and record actual fill price.

        If the defense order reaches a terminal failure state (cancelled/
        rejected/expired), flags defense_close_failed so the orchestrator
        can escalate.
        """
        if self.state.defense_filled or not self.state.defense_order_id:
            return self.state.defense_filled

        if self.dry_run:
            self.state.defense_filled = True
            self.state.defense_fill_price = 0.50  # conservative dry-run estimate
            return True

        try:
            order = get_order(self.state.defense_order_id)
            status = order.get("status", "")
            if status == "filled":
                self.state.defense_filled = True
                self.state.defense_fill_price = float(order.get("filled_avg_price", 0))
                self.state.is_open = False
                self.state.defense_order_dead = True
                logger.info(
                    "CONDOR DEFENSE FILLED: debit=$%.2f per contract",
                    self.state.defense_fill_price,
                )
                return True
            elif status in ("cancelled", "expired", "rejected"):
                logger.error(
                    "CONDOR DEFENSE ORDER %s — position may still be open! Flagging for escalation.",
                    status.upper(),
                )
                self.state.defense_close_failed = True
                self.state.defense_order_dead = True
            else:
                logger.debug("Defense close order status: %s", status)
        except Exception as e:
            logger.warning("Failed to check defense fill: %s", e)
        return False

    def close_early_exit(self) -> bool:
        """
        Submit a discretionary early-exit close order.

        Unlike close_defense(), this does NOT set defense_triggered.
        The orchestrator polls check_early_exit_fill() and records
        EXIT_REASON_DISCRETIONARY on confirmed fill.

        Returns True if the close order was placed.
        """
        if not self.state.is_filled or not self.state.legs:
            logger.warning("Cannot early-exit condor: no filled position")
            return False

        logger.info(
            "CONDOR EARLY EXIT: Closing %d contracts at market",
            self.state.qty,
        )

        try:
            order = close_condor_order(
                legs=self.state.legs,
                qty=self.state.qty,
                dry_run=self.dry_run,
            )
            if order:
                self.state.early_exit_order_id = order.get("id")
                logger.info(
                    "CONDOR EARLY EXIT ORDER PLACED: order_id=%s",
                    self.state.early_exit_order_id,
                )
                return True
        except Exception as e:
            logger.error("CONDOR EARLY EXIT ORDER FAILED: %s", e)
        return False

    def check_early_exit_fill(self) -> bool:
        """
        Poll the discretionary early-exit close order for a fill.

        Returns True once the fill is confirmed.  The orchestrator
        should record the PDT exit only after this returns True.
        """
        if self.state.early_exit_filled or not self.state.early_exit_order_id:
            return self.state.early_exit_filled

        if self.dry_run:
            self.state.early_exit_filled = True
            self.state.early_exit_fill_price = 0.50
            self.state.is_open = False
            return True

        try:
            order = get_order(self.state.early_exit_order_id)
            status = order.get("status", "")
            if status == "filled":
                self.state.early_exit_filled = True
                self.state.early_exit_fill_price = float(order.get("filled_avg_price", 0))
                self.state.is_open = False
                logger.info(
                    "CONDOR EARLY EXIT FILLED: debit=$%.2f per contract",
                    self.state.early_exit_fill_price,
                )
                return True
            elif status in ("cancelled", "expired", "rejected"):
                logger.error(
                    "CONDOR EARLY EXIT ORDER %s — position may still be open!",
                    status.upper(),
                )
                self.state.early_exit_order_dead = True
            else:
                logger.debug("Condor early-exit order status: %s", status)
        except Exception as e:
            logger.warning("Failed to check condor early-exit fill: %s", e)
        return False

    def on_settlement(self, tracker: MorningTracker) -> dict:
        """
        Handle 4:00 PM cash settlement.

        If defense was triggered, P&L = entry credit − actual close debit.
        If held to expiry, P&L = entry credit − estimated intrinsic value.

        NOTE: The instrument traded is XSP, which settles off a special
        opening/closing settlement value (SOQ/SET), not literally SPY
        last trade.  We use tracker.spy_last as a proxy.  The reported
        P&L is therefore an *estimate*; actual broker settlement may
        differ by a small amount.  For precise accounting, reconcile
        against the broker's end-of-day statement.
        """
        if not self.state.is_filled:
            status = "never_filled" if self.state.entry_order_id else "no_position"
            return {
                "status": status,
                "missed_trade_reason": self.state.missed_trade_reason,
            }

        multiplier = 100
        credit_received = self.state.entry_credit * self.state.qty * multiplier

        if self.state.defense_triggered:
            # Use actual fill price if available, otherwise mark as unknown
            if self.state.defense_fill_price is not None:
                exit_debit = self.state.defense_fill_price * self.state.qty * multiplier
            else:
                # Fill price never confirmed — flag it
                exit_debit = 0.0
                logger.warning("Defense close fill price unknown; P&L may be inaccurate")
            self.state.exit_cost = exit_debit
            self.state.realized_pnl = credit_received - exit_debit
            status = "defense_closed"
        else:
            # Held to settlement — compute intrinsic value of the spread
            spy_settle = tracker.spy_last
            legs = self.state.legs
            intrinsic = 0.0

            if legs and spy_settle > 0:
                # Put spread intrinsic (we are short the higher-strike put)
                short_put_itm = max(legs.short_put_strike - spy_settle, 0.0)
                long_put_itm = max(legs.long_put_strike - spy_settle, 0.0)
                put_spread_cost = short_put_itm - long_put_itm  # what we owe

                # Call spread intrinsic (we are short the lower-strike call)
                short_call_itm = max(spy_settle - legs.short_call_strike, 0.0)
                long_call_itm = max(spy_settle - legs.long_call_strike, 0.0)
                call_spread_cost = short_call_itm - long_call_itm  # what we owe

                intrinsic = max(put_spread_cost, 0.0) + max(call_spread_cost, 0.0)

            settlement_debit = intrinsic * self.state.qty * multiplier
            self.state.exit_cost = settlement_debit
            self.state.realized_pnl = credit_received - settlement_debit
            status = "settled_otm" if intrinsic == 0.0 else "settled_itm"

            logger.info(
                "Condor intrinsic at settle: SPY=$%.2f put_spread=$%.4f call_spread=$%.4f total=$%.4f",
                spy_settle,
                max(legs.short_put_strike - spy_settle, 0.0) - max(legs.long_put_strike - spy_settle, 0.0) if legs else 0,
                max(spy_settle - legs.short_call_strike, 0.0) - max(spy_settle - legs.long_call_strike, 0.0) if legs else 0,
                intrinsic,
            )

        self.state.settled = True
        self.state.is_open = False

        # Fill quality metrics (vs original entry-time quotes)
        # Note: These measure overall entry process quality, not slippage vs live market at fill time
        mid = self.state.entry_mid_credit
        natural = self.state.entry_natural_credit
        filled = self.state.entry_credit
        mid_capture = 0.0
        if mid > 0 and natural > 0 and mid != natural:
            mid_capture = (filled - natural) / (mid - natural) * 100
        slippage = mid - filled if mid > 0 else 0.0

        result = {
            "status": status,
            "contracts": self.state.qty,
            "entry_credit": round(self.state.entry_credit, 4),
            "credit_received": round(credit_received, 2),
            "exit_cost": round(self.state.exit_cost, 2),
            "anchor_price": self.state.anchor_price,
            "spy_settlement": round(tracker.spy_last, 2),
            "defense_triggered": self.state.defense_triggered,
            "realized_pnl": round(self.state.realized_pnl, 2),
            "fill_quality": {
                "natural_credit": round(natural, 4),
                "mid_credit": round(mid, 4),
                "actual_credit": round(filled, 4),
                "slippage_vs_mid": round(slippage, 4),
                "mid_capture_pct": round(mid_capture, 1),
                "adjustments": self.state.entry_adjustments,
                "qty_reduced": self.state.entry_qty_reduced,
            },
        }

        logger.info(
            "CONDOR SETTLEMENT: status=%s contracts=%d credit=$%.2f exit_cost=$%.2f pnl=$%.2f",
            status, self.state.qty, credit_received,
            self.state.exit_cost, self.state.realized_pnl,
        )
        return result

    def get_summary(self) -> dict:
        """Return current state summary for logging."""
        return {
            "is_open": self.state.is_open,
            "is_filled": self.state.is_filled,
            "qty": self.state.qty,
            "anchor_price": self.state.anchor_price,
            "entry_credit": self.state.entry_credit,
            "defense_triggered": self.state.defense_triggered,
            "missed_trade_reason": self.state.missed_trade_reason,
            "fill_quality": {
                "natural_credit": self.state.entry_natural_credit,
                "mid_credit": self.state.entry_mid_credit,
                "peak_natural": self.state.entry_peak_natural,
                "limit_price": self.state.entry_limit_price,
                "adjustments": self.state.entry_adjustments,
                "qty_reduced": self.state.entry_qty_reduced,
            },
            "legs": {
                "long_put": getattr(self.state.legs, "long_put", None),
                "short_put": getattr(self.state.legs, "short_put", None),
                "short_call": getattr(self.state.legs, "short_call", None),
                "long_call": getattr(self.state.legs, "long_call", None),
            } if self.state.legs else None,
        }
