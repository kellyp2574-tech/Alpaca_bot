"""
0DTE Options Trading Bot — XSP Iron Condor + XND Directional
Runs from 9:00 AM until 4:15 PM ET daily.

Two independent sleeves:
  Sleeve 1: XSP Iron Condor (every day at 11:30 AM)
  Sleeve 2: XND Directional (conditional, 10:45 AM if filters pass)
"""
import logging
import os
import time
import json
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

from bot import config as cfg
from bot.condor_config import StrategyConfig
from bot.options_client import (
    get_equity,
    get_option_positions,
    close_option_position,
    cancel_all_orders,
    get_account_info,
)
from bot.market_data import (
    MorningTracker,
    get_vix_previous_close,
    refresh_tracker,
)
from bot.condor_strategy import CondorStrategy
from bot.directional_strategy import DirectionalStrategy
from bot.pdt_guard import (
    PDTGuard,
    EXIT_REASON_SCHEDULED,
    EXIT_REASON_DISCRETIONARY,
    EXIT_REASON_DEFENSE,
    EXIT_REASON_EMERGENCY,
)

# ═══════════════════════════════════════════════════
# Timezone
# ═══════════════════════════════════════════════════
MARKET_TZ = ZoneInfo("America/New_York")


def market_now() -> datetime:
    return datetime.now(MARKET_TZ)


def parse_time(time_str: str) -> datetime:
    """Parse HH:MM into today's datetime in ET."""
    h, m = map(int, time_str.split(":"))
    now = market_now()
    return now.replace(hour=h, minute=m, second=0, microsecond=0)


# ═══════════════════════════════════════════════════
# Logging setup
# ═══════════════════════════════════════════════════
os.makedirs(cfg.LOG_DIR, exist_ok=True)
LOG_PATH = cfg.LOG_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("bot")
logger.info(f"Logging initialized: {LOG_PATH}")


class CondorBot:
    """
    0DTE Options Bot — Daily schedule:

    09:00  Bot starts, pull account info, VIX prev close
    09:30  Market opens, begin tracking SPY/QQQ bars
    10:30  Morning assessment (directional filters)
    10:45  Directional entry (if qualified)
    11:30  Condor entry (every day)
    11:30–16:00  Defense monitoring (SPY 1% trigger)
    16:00  Cash settlement
    16:15  Shutdown
    """

    def __init__(self, dry_run: bool = False):
        logger.info("CondorBot.__init__ starting...")
        self.dry_run = dry_run
        self.config = StrategyConfig()

        # Strategy modules
        self.condor = CondorStrategy(self.config.condor, dry_run=dry_run)
        self.directional = DirectionalStrategy(self.config.directional, dry_run=dry_run)

        # PDT guard — persistent day-trade ledger
        self.pdt_guard = PDTGuard()

        # Market data tracker
        self.tracker = MorningTracker()

        # State flags
        self.session_started = False
        self.bars_tracking = False
        self.assessment_done = False
        self.directional_entered = False
        self.condor_entered = False
        self.settlement_done = False

        # Exit reason tracking for reporting
        self._condor_exit_reason: str = ""
        self._directional_exit_reason: str = ""
        self._pdt_blocked_early_exit: bool = False

        # Account info
        self.start_equity = 0.0
        self.start_buying_power = 0.0

        # Timing
        self._last_defense_check = 0.0
        self._last_price_refresh = 0.0
        self._last_status_log = 0.0

        # Defense escalation cap — prevents infinite re-close loop if the
        # broker persistently rejects defense orders.
        self._defense_escalation_count = 0
        self.MAX_DEFENSE_ESCALATIONS = 3

        # Precompute schedule times (assumes single-day process)
        sched = self.config.schedule
        self._t_market_open = parse_time(sched.market_open).time()
        self._t_assessment = parse_time(sched.morning_assessment).time()
        self._t_dir_entry = parse_time(sched.directional_entry).time()
        self._t_condor_entry = parse_time(sched.condor_entry).time()
        self._t_settle = parse_time(sched.market_close).time()
        self._t_shutdown = parse_time(sched.bot_shutdown).time()

    # ─── Session lifecycle ─────────────────────────────────────────────────

    def _start_session(self):
        """9:00 AM — Pull account info, VIX close, log starting state."""
        logger.info("=" * 60)
        logger.info("SESSION START" + (" [DRY RUN]" if self.dry_run else ""))
        logger.info("=" * 60)

        try:
            acct = get_account_info()
            self.start_equity = float(acct.get("equity", 0))
            self.start_buying_power = float(acct.get("buying_power", 0))
            cash = float(acct.get("cash", 0))
            logger.info(
                "Account: equity=$%.2f buying_power=$%.2f cash=$%.2f",
                self.start_equity, self.start_buying_power, cash,
            )
        except Exception as e:
            logger.error("Failed to fetch account info: %s", e)

        # VIX previous close
        self.tracker.vix_prev_close = get_vix_previous_close()
        if self.tracker.vix_prev_close is not None:
            logger.info("VIX previous close: %.2f", self.tracker.vix_prev_close)
        else:
            logger.warning("VIX data unavailable — directional filters will fail")

        self.session_started = True

    def _start_bar_tracking(self):
        """
        9:30 AM — Begin recording SPY and QQQ prices.

        If the bot starts after 9:30 AM, refresh_tracker() will backfill
        the daily-bar open/high/low from the Alpaca snapshot so that
        morning range calculations are as accurate as possible even on
        a late start.
        """
        now = market_now()
        market_open_t = parse_time(self.config.schedule.market_open)
        late_minutes = (now - market_open_t).total_seconds() / 60
        if late_minutes > 5:
            logger.warning(
                "Late start detected: market has been open %.0f min. "
                "Open prices will be backfilled from Alpaca daily bar.",
                late_minutes,
            )

        logger.info("Market open — starting SPY/QQQ price tracking")
        self.tracker = refresh_tracker(self.tracker)
        self.bars_tracking = True
        logger.info(
            "Initial prices: SPY=$%.2f (open=$%.2f) QQQ=$%.2f (open=$%.2f)",
            self.tracker.spy_last, self.tracker.spy_open or 0,
            self.tracker.qqq_last, self.tracker.qqq_open or 0,
        )

    def _refresh_prices(self):
        """Refresh SPY/QQQ prices periodically."""
        now_mono = time.monotonic()
        if now_mono - self._last_price_refresh < 30:
            return
        self._last_price_refresh = now_mono

        try:
            self.tracker = refresh_tracker(self.tracker)
        except Exception as e:
            logger.warning("Price refresh failed: %s", e)

    # ─── Morning assessment ────────────────────────────────────────────────

    def _morning_assessment(self):
        """10:30 AM — Check directional filters."""
        logger.info("=" * 40)
        logger.info("MORNING ASSESSMENT — 10:30 AM")
        logger.info("=" * 40)

        # Final price refresh before assessment
        self.tracker = refresh_tracker(self.tracker)

        qualified = self.directional.assess_filters(self.tracker)
        self.assessment_done = True

        logger.info(
            "QQQ open=$%.2f high=$%.2f low=$%.2f last=$%.2f range=%.3f%% dir=%.3f%%",
            self.tracker.qqq_open or 0, self.tracker.qqq_high,
            self.tracker.qqq_low, self.tracker.qqq_last,
            self.tracker.qqq_morning_range_pct * 100,
            self.tracker.qqq_morning_direction_pct * 100,
        )

        if not qualified:
            logger.info("Directional trade: NO TRADE today")

    def _enter_directional(self):
        """10:45 AM — Enter directional trade if qualified."""
        if not self.directional.state.qualified:
            logger.info("Directional: Skipping — not qualified")
            self.directional_entered = True
            return

        logger.info("=" * 40)
        logger.info("DIRECTIONAL ENTRY — 10:45 AM")
        logger.info("=" * 40)

        # Refresh prices right before entry
        self.tracker = refresh_tracker(self.tracker)

        success = self.directional.enter(self.tracker)
        self.directional_entered = True

        if success:
            logger.info("Directional trade entered successfully")
        else:
            logger.warning("Directional trade entry failed")

    # ─── Condor entry ──────────────────────────────────────────────────────

    def _enter_condor(self):
        """11:30 AM — Enter iron condor."""
        logger.info("=" * 40)
        logger.info("CONDOR ENTRY — 11:30 AM")
        logger.info("=" * 40)

        # Refresh prices to get precise anchor
        self.tracker = refresh_tracker(self.tracker)

        success = self.condor.enter(self.tracker)
        self.condor_entered = True

        if success:
            summary = self.condor.get_summary()
            logger.info("Condor entered: %s", json.dumps(summary, default=str))
        else:
            logger.error("Condor entry failed — no position today")

    # ─── PDT guard integration ──────────────────────────────────────────────

    def _record_condor_entry_fill(self):
        """Record condor entry fill with PDT guard."""
        try:
            symbol = "XSP_CONDOR"  # Synthetic — mleg has multiple symbols
            self.pdt_guard.record_entry_fill(
                symbol=symbol,
                strategy="condor",
                order_id=self.condor.state.entry_order_id or "",
            )
        except Exception as e:
            logger.warning("PDT: Failed to record condor entry fill: %s", e)

    def _record_directional_entry_fill(self):
        """Record directional entry fill with PDT guard."""
        try:
            self.pdt_guard.record_entry_fill(
                symbol=self.directional.state.option_symbol or "XND_UNKNOWN",
                strategy="directional",
                order_id=self.directional.state.entry_order_id or "",
            )
        except Exception as e:
            logger.warning("PDT: Failed to record directional entry fill: %s", e)

    def _attempt_discretionary_early_exit(self, sleeve: str) -> bool:
        """
        Gate a discretionary early exit through the PDT guard.

        Call this before any non-mandatory early close (e.g., profit-taking,
        time-based early exit).  Mandatory exits (defense, emergency) must
        NOT use this method — they bypass PDT checks entirely.

        Returns True if the exit was allowed and executed, False if blocked.
        """
        try:
            equity = get_equity()
        except Exception as e:
            logger.warning("PDT: Cannot check equity for early exit: %s — blocking", e)
            self._pdt_blocked_early_exit = True
            return False

        if not self.pdt_guard.can_take_discretionary_day_trade(equity):
            self._pdt_blocked_early_exit = True
            logger.warning(
                "PDT GUARD: Discretionary early exit BLOCKED for %s sleeve", sleeve,
            )
            return False

        # Exit is allowed — execute the close
        if sleeve == "condor" and self.condor.state.is_filled and self.condor.state.is_open:
            logger.info("DISCRETIONARY EARLY EXIT: closing condor")
            success = self.condor.close_early_exit()
            if success:
                logger.info("Condor early-exit order submitted: %s",
                            self.condor.state.early_exit_order_id)
                return True
            return False

        elif sleeve == "directional" and self.directional.state.is_filled and self.directional.state.is_open:
            logger.info("DISCRETIONARY EARLY EXIT: closing directional")
            try:
                from bot.options_client import place_single_option_order
                order = place_single_option_order(
                    symbol=self.directional.state.option_symbol,
                    qty=self.directional.state.qty,
                    side="sell",
                    order_type="market",
                    dry_run=self.dry_run,
                )
                if order:
                    # Do NOT record PDT exit here — wait for confirmed fill.
                    # Store the order ID so the main loop can poll for it.
                    self.directional.state.early_exit_order_id = order.get("id")
                    logger.info("Directional early-exit order submitted: %s", order.get("id"))
                    return True
            except Exception as e:
                logger.error("Failed to submit directional early-exit order: %s", e)
            return False

        logger.debug("No open position for %s sleeve — nothing to early-exit", sleeve)
        return False

    def _record_condor_exit(self, exit_reason: str):
        """Record condor exit fill with PDT guard."""
        self._condor_exit_reason = exit_reason
        try:
            self.pdt_guard.record_exit_fill(
                symbol="XSP_CONDOR",
                strategy="condor",
                exit_reason=exit_reason,
            )
        except Exception as e:
            logger.warning("PDT: Failed to record condor exit: %s", e)

    def _record_directional_exit(self, exit_reason: str):
        """Record directional exit fill with PDT guard."""
        self._directional_exit_reason = exit_reason
        try:
            self.pdt_guard.record_exit_fill(
                symbol=self.directional.state.option_symbol or "XND_UNKNOWN",
                strategy="directional",
                exit_reason=exit_reason,
            )
        except Exception as e:
            logger.warning("PDT: Failed to record directional exit: %s", e)

    # ─── Defense monitoring ────────────────────────────────────────────────

    def _check_defense(self):
        """
        Periodic defense check: has SPY breached 1% from anchor?

        NOTE — Tick-level limitation:
        Defense monitoring relies on sampled last-trade prices refreshed
        every ~30 s.  A very fast intra-second excursion that reverses
        between refreshes could be missed.  True tick-level defense would
        require a streaming WebSocket connection, which is outside the
        current polling architecture.  The post-anchor max-excursion
        tracking (spy_post_anchor_high/low) mitigates this by remembering
        the worst sampled price, but it is not a substitute for tick data.
        """
        if not self.condor.state.is_filled or self.condor.state.defense_triggered:
            return

        now_mono = time.monotonic()
        interval = self.config.schedule.defense_check_interval
        if now_mono - self._last_defense_check < interval:
            return
        self._last_defense_check = now_mono

        # Refresh SPY price
        self._refresh_prices()

        if self.condor.check_defense(self.tracker):
            logger.warning("DEFENSE TRIGGERED — closing condor immediately")
            self.condor.close_defense()
            # NOTE: Do NOT record PDT exit here — the close order has only
            # been *submitted*, not confirmed.  The exit is recorded in the
            # defense-fill polling block once defense_filled transitions True.

    # ─── Settlement ────────────────────────────────────────────────────────

    def _handle_settlement(self):
        """4:00 PM — Cash settlement for both sleeves."""
        logger.info("=" * 40)
        logger.info("SETTLEMENT — 4:00 PM")
        logger.info("=" * 40)

        # Final price refresh for accurate settlement
        self.tracker = refresh_tracker(self.tracker)

        # Condor settlement — pass tracker for intrinsic value calculation
        condor_result = self.condor.on_settlement(self.tracker)
        logger.info("Condor result: %s", json.dumps(condor_result, default=str))

        # Record condor exit if it wasn't already recorded (defense case)
        if self.condor.state.is_filled and not self._condor_exit_reason:
            self._record_condor_exit(EXIT_REASON_SCHEDULED)

        # Directional settlement — pass tracker for NDX estimate
        directional_result = self.directional.on_settlement(self.tracker)
        logger.info("Directional result: %s", json.dumps(directional_result, default=str))

        # Record directional exit
        if self.directional.state.is_filled and not self._directional_exit_reason:
            self._record_directional_exit(EXIT_REASON_SCHEDULED)

        self.settlement_done = True

    # ─── Shutdown ──────────────────────────────────────────────────────────

    def _shutdown(self):
        """4:15 PM — Log final P&L and shut down."""
        logger.info("=" * 60)
        logger.info("SHUTDOWN — 4:15 PM")
        logger.info("=" * 60)

        try:
            acct = get_account_info()
            end_equity = float(acct.get("equity", 0))
            daily_pnl = end_equity - self.start_equity
            daily_pct = (daily_pnl / self.start_equity * 100) if self.start_equity > 0 else 0

            logger.info("Start equity:  $%.2f", self.start_equity)
            logger.info("End equity:    $%.2f", end_equity)
            logger.info("Daily P&L:     $%.2f (%.2f%%)", daily_pnl, daily_pct)
        except Exception as e:
            logger.error("Failed to fetch final account info: %s", e)

        # Log strategy summaries
        logger.info("Condor summary: %s", json.dumps(self.condor.get_summary(), default=str))
        logger.info("Directional summary: %s", json.dumps(self.directional.get_summary(), default=str))

        # Save daily report
        self._save_daily_report()

    def _save_daily_report(self):
        """Save a JSON report for today's session."""
        report_dir = os.path.join(cfg.STATE_DIR, "reports")
        os.makedirs(report_dir, exist_ok=True)

        today = market_now().strftime("%Y-%m-%d")
        report_path = os.path.join(report_dir, f"daily_report_{today}.json")

        report = {
            "date": today,
            "start_equity": self.start_equity,
            "start_buying_power": self.start_buying_power,
            "condor": self.condor.get_summary(),
            "directional": self.directional.get_summary(),
            "dry_run": self.dry_run,
            # Exit reason tracking
            "condor_exit_reason": self._condor_exit_reason or "none",
            "directional_exit_reason": self._directional_exit_reason or "none",
            "pdt_blocked_early_exit": self._pdt_blocked_early_exit,
            # Settlement proxy disclaimers — P&L figures are estimates
            "settlement_notes": {
                "condor": "XSP settles off SOQ/SET, not SPY last trade. P&L is a proxy estimate.",
                "directional": "XND settles off NDX special calc, not QQQ×40. P&L is a proxy estimate.",
                "reconcile": "Reconcile against broker end-of-day statement for precise accounting.",
            },
        }

        try:
            acct = get_account_info()
            end_equity = float(acct.get("equity", 0))
            end_bp = float(acct.get("buying_power", 0))
            report["end_equity"] = end_equity
            report["end_buying_power"] = end_bp
            report["daily_pnl"] = end_equity - self.start_equity
        except Exception as e:
            logger.warning("Could not fetch final account info for report: %s", e)
            report["end_equity"] = None
            report["end_buying_power"] = None
            report["daily_pnl"] = None

        # PDT guard status
        try:
            report["pdt_guard"] = self.pdt_guard.get_status()
        except Exception:
            pass

        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info("Daily report saved: %s", report_path)
        except Exception as e:
            logger.error("Failed to save daily report: %s", e)

    # ─── Periodic status logging ───────────────────────────────────────────

    def _log_status(self):
        """Log status every 5 minutes."""
        now_mono = time.monotonic()
        if now_mono - self._last_status_log < 300:
            return
        self._last_status_log = now_mono

        now = market_now()
        logger.info(
            "STATUS [%s]: SPY=$%.2f QQQ=$%.2f | condor_open=%s defense=%s | directional_open=%s",
            now.strftime("%H:%M"),
            self.tracker.spy_last, self.tracker.qqq_last,
            self.condor.state.is_open, self.condor.state.defense_triggered,
            self.directional.state.is_open,
        )

        if self.condor.state.is_open and self.tracker.condor_anchor:
            current_move = self.tracker.spy_move_from_anchor_pct()
            max_move = self.tracker.spy_max_move_from_anchor_pct()
            logger.info(
                "  Condor: anchor=$%.2f current_move=%.3f%% max_excursion=%.3f%% (trigger=%.2f%%)",
                self.tracker.condor_anchor,
                (current_move or 0.0) * 100,
                (max_move or 0.0) * 100,
                self.config.condor.defense_trigger_pct * 100,
            )

    # ─── Main loop ─────────────────────────────────────────────────────────

    def run(self):
        """Main bot loop: 9:00 AM → 4:15 PM."""
        logger.info("CondorBot starting — waiting for 9:00 AM ET")

        try:
            # Wait until 9:00 AM if started early
            start_time = parse_time(self.config.schedule.bot_start)
            now = market_now()
            if now < start_time:
                wait_secs = (start_time - now).total_seconds()
                logger.info("Waiting %.1f minutes until 9:00 AM", wait_secs / 60)
                time.sleep(max(wait_secs, 0))

            # 9:00 AM — Session start
            self._start_session()

            # Main event loop
            while True:
                now = market_now()
                t = now.time()

                if t >= self._t_shutdown:
                    if not self.settlement_done:
                        self._handle_settlement()
                    self._shutdown()
                    break

                # 9:30 AM — Start bar tracking
                if t >= self._t_market_open and not self.bars_tracking:
                    self._start_bar_tracking()

                # Refresh prices periodically after market open
                if self.bars_tracking:
                    self._refresh_prices()

                # 10:30 AM — Morning assessment
                if t >= self._t_assessment and not self.assessment_done and self.bars_tracking:
                    self._morning_assessment()

                # 10:45 AM — Directional entry
                if t >= self._t_dir_entry and not self.directional_entered and self.assessment_done:
                    self._enter_directional()

                # 11:30 AM — Condor entry
                if t >= self._t_condor_entry and not self.condor_entered and self.bars_tracking:
                    self._enter_condor()

                # Check fill statuses (skip terminal/dead orders)
                if (self.condor.state.entry_order_id
                        and not self.condor.state.is_filled
                        and not self.condor.state.entry_order_dead):
                    was_filled_before = self.condor.state.is_filled
                    self.condor.check_fill_status()
                    # Record entry fill with PDT guard
                    if self.condor.state.is_filled and not was_filled_before:
                        self._record_condor_entry_fill()

                if (self.directional.state.entry_order_id
                        and not self.directional.state.is_filled
                        and not self.directional.state.entry_order_dead):
                    was_filled_before = self.directional.state.is_filled
                    self.directional.check_fill_status()
                    # Record entry fill with PDT guard
                    if self.directional.state.is_filled and not was_filled_before:
                        self._record_directional_entry_fill()

                # Poll defense-close fill if defense was triggered
                if self.condor.state.defense_triggered and not self.condor.state.defense_filled:
                    was_defense_filled = self.condor.state.defense_filled
                    self.condor.check_defense_fill()

                    # Record PDT exit only on confirmed defense fill
                    if self.condor.state.defense_filled and not was_defense_filled:
                        self._record_condor_exit(EXIT_REASON_DEFENSE)

                    # Escalate only if the defense order reached a truly
                    # terminal state (cancelled/rejected/expired).  Do NOT
                    # re-close if the order is merely still pending — that
                    # would risk duplicate close attempts.
                    if (self.condor.state.defense_close_failed
                            and not self.condor.state.defense_filled):
                        self._defense_escalation_count += 1
                        if self._defense_escalation_count <= self.MAX_DEFENSE_ESCALATIONS:
                            logger.critical(
                                "Defense close order terminally failed — "
                                "emergency re-close attempt %d/%d",
                                self._defense_escalation_count,
                                self.MAX_DEFENSE_ESCALATIONS,
                            )
                            self.condor.state.defense_close_failed = False
                            self.condor.state.defense_order_id = None
                            self.condor.state.defense_triggered = False
                            self.condor.close_defense()
                        else:
                            logger.critical(
                                "Defense escalation limit (%d) reached — "
                                "giving up on automated close. MANUAL INTERVENTION REQUIRED.",
                                self.MAX_DEFENSE_ESCALATIONS,
                            )

                # Poll condor early-exit fill (discretionary, separate from defense)
                if (self.condor.state.early_exit_order_id
                        and not self.condor.state.early_exit_filled):
                    was_filled = self.condor.state.early_exit_filled
                    self.condor.check_early_exit_fill()
                    if self.condor.state.early_exit_filled and not was_filled:
                        self._record_condor_exit(EXIT_REASON_DISCRETIONARY)

                # Poll directional early-exit fill
                if (self.directional.state.early_exit_order_id
                        and not self.directional.state.early_exit_filled):
                    was_filled = self.directional.state.early_exit_filled
                    self.directional.check_early_exit_fill()
                    if self.directional.state.early_exit_filled and not was_filled:
                        self._record_directional_exit(EXIT_REASON_DISCRETIONARY)

                # Defense monitoring (only when condor is actually filled)
                # Skip if an early exit is already in progress
                if (self.condor.state.is_filled
                        and not self.condor.state.early_exit_order_id):
                    self._check_defense()

                # 4:00 PM — Settlement
                if t >= self._t_settle and not self.settlement_done:
                    self._handle_settlement()

                # Periodic status
                self._log_status()

                # Sleep to avoid tight loop
                time.sleep(10)

        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt — shutting down")
            self._emergency_shutdown()
        except Exception as e:
            logger.exception("FATAL ERROR in main loop: %s", e)
            self._emergency_shutdown()
            raise

    def _emergency_shutdown(self):
        """
        Emergency shutdown: cancel all orders, then attempt to close
        all open option positions at market.

        This is a best-effort flatten — if any close fails, it is
        logged for manual cleanup.  PDT rules are intentionally
        ignored here; keeping positions open through a crash is a
        larger risk than an extra day-trade count.
        """
        logger.warning("EMERGENCY SHUTDOWN")
        try:
            cancel_all_orders()
            logger.info("All orders cancelled")
        except Exception as e:
            logger.error("Failed to cancel orders: %s", e)

        # Attempt to flatten all open option positions
        try:
            positions = get_option_positions()
            if positions:
                logger.warning(
                    "EMERGENCY FLATTEN: %d open option position(s) — closing at market",
                    len(positions),
                )
                for p in positions:
                    sym = p.get("symbol", "")
                    qty = p.get("qty", "?")
                    try:
                        result = close_option_position(sym)
                        if result:
                            logger.info("  CLOSED %s qty=%s", sym, qty)
                        else:
                            logger.error("  FAILED to close %s qty=%s — MANUAL CLEANUP REQUIRED", sym, qty)
                    except Exception as e:
                        logger.error("  Exception closing %s: %s — MANUAL CLEANUP REQUIRED", sym, e)
            else:
                logger.info("No open option positions at emergency shutdown")
        except Exception as e:
            logger.error("Failed to check/close positions: %s", e)

        # Record emergency exits in PDT ledger
        if self.condor.state.is_filled and not self._condor_exit_reason:
            self._record_condor_exit(EXIT_REASON_EMERGENCY)
        if self.directional.state.is_filled and not self._directional_exit_reason:
            self._record_directional_exit(EXIT_REASON_EMERGENCY)

        logger.info("Condor state: %s", json.dumps(self.condor.get_summary(), default=str))
        logger.info("Directional state: %s", json.dumps(self.directional.get_summary(), default=str))

        # Save report even on crash
        try:
            self._save_daily_report()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="0DTE Options Bot — XSP Iron Condor + XND Directional")
    parser.add_argument("--dry-run", action="store_true", help="Log signals without submitting orders")
    args = parser.parse_args()

    bot = CondorBot(dry_run=args.dry_run)
    bot.run()


if __name__ == "__main__":
    main()
