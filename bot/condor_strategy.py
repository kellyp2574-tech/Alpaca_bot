"""
XSP Iron Condor Strategy — Entry, defense monitoring, and exit logic.

Sleeve 1: Sells an iron condor on XSP at 11:30 AM, monitors for defense
trigger (SPY 1% move from anchor), holds to 4:00 PM cash settlement.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from bot.condor_config import CondorConfig
from bot.options_client import (
    CondorLegs,
    build_condor_legs,
    close_condor_order,
    cancel_all_orders,
    get_buying_power,
    get_todays_expiration,
    place_iron_condor_order,
    size_condor,
    get_order,
)
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


class CondorStrategy:
    """
    Manages the XSP iron condor sleeve.

    Lifecycle:
    1. enter() — at 11:30 AM, compute strikes, size, place mleg order
    2. check_defense() — every minute, check if SPY breached 1% from anchor
    3. close_defense() — if breached, close all legs immediately
    4. on_settlement() — at 4:00 PM, record cash settlement result
    """

    def __init__(self, config: CondorConfig, dry_run: bool = False):
        self.cfg = config
        self.dry_run = dry_run
        self.state = CondorState()

    def enter(self, tracker: MorningTracker) -> bool:
        """
        Enter the iron condor at 11:30 AM.

        Steps:
        1. Record anchor price (SPY at 11:30)
        2. Compute strikes (0.90% OTM short, 1.00% wing)
        3. Fetch XSP 0DTE contracts and find best legs
        4. Size: floor(buying_power / max_loss_per_contract)
        5. Place mleg limit order for target credit

        Returns True if order was placed successfully.
        """
        # Step 1: Anchor price
        anchor = tracker.spy_last
        if anchor <= 0:
            logger.error("Cannot enter condor: SPY price unavailable (%.2f)", anchor)
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
            return False

        self.state.legs = legs

        # Step 4: Sizing
        buying_power = get_buying_power()
        qty = size_condor(buying_power, self.cfg.max_loss_per_contract)
        if qty <= 0:
            logger.error("CONDOR ENTRY FAILED: Insufficient buying power ($%.2f)", buying_power)
            return False

        self.state.qty = qty

        # Step 5: Place order
        try:
            order = place_iron_condor_order(
                legs=legs,
                qty=qty,
                limit_price=self.cfg.target_credit,
                time_in_force=self.cfg.time_in_force,
                dry_run=self.dry_run,
            )

            if order:
                self.state.entry_order_id = order.get("id")
                self.state.entry_credit = self.cfg.target_credit
                self.state.entry_time = datetime.now()
                # is_open stays False until fill confirmed in check_fill_status()
                logger.info(
                    "CONDOR ORDER SUBMITTED: %d contracts @ $%.2f credit, order_id=%s",
                    qty, self.cfg.target_credit, self.state.entry_order_id,
                )
                return True
        except Exception as e:
            logger.error("CONDOR ENTRY ORDER FAILED: %s", e)

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
                logger.info(
                    "CONDOR FILLED: %d contracts @ $%.2f credit",
                    self.state.qty, filled_price,
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
            else:
                logger.debug("Defense close order status: %s", status)
        except Exception as e:
            logger.warning("Failed to check defense fill: %s", e)
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
            return {"status": status}

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
            "legs": {
                "long_put": getattr(self.state.legs, "long_put", None),
                "short_put": getattr(self.state.legs, "short_put", None),
                "short_call": getattr(self.state.legs, "short_call", None),
                "long_call": getattr(self.state.legs, "long_call", None),
            } if self.state.legs else None,
        }
