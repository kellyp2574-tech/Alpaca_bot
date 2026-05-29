"""ETF router runtime — extracted from integrated_main.py.

Owns the morning ETF router workflow:
  - 09:00-09:25 startup (`run_startup_phase`, `cancel_stale_etf_orders`)
  - 09:30 tape initialization (`initialize_tape_recording`)
  - 09:30-10:00 tape updates (`update_tape`)
  - 10:00 routing decision (`make_router_decision`) and entry execution
    (`execute_etf_entry`)
  - 11:00 / 14:00 / 15:00 exit checkpoints (`check_etf_exits`,
    `execute_etf_exit`)
  - EOD summary + artifact (`build_etf_router_summary`,
    `save_etf_router_artifact`)

Every function takes the orchestrator instance (`bot`) as its first argument
and mutates its state directly. State ownership stays with
`CombinedOvernightReboundBot`; this module only houses the logic.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, time as dt_time
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from bot import config
from bot.etf_router import RouterBranch, RouterDecision
from bot.universe_builder import filter_execution_ready

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")


# ────────────────────────────────────────────────────────────────────
# Summary + artifact (used by state_io EOD report)
# ────────────────────────────────────────────────────────────────────

def build_etf_router_summary(bot) -> Dict[str, Any]:
    """Compact ETF router summary for run_health + standalone artifact.

    Captures what was decided, whether an entry actually fired, and
    the realized fill price so a post-day review can be done from a
    single JSON file. Returns ``{"enabled": False}`` when the router is
    disabled (e.g. ``ETF_ROUTER_ENABLED=False``).
    """
    if not getattr(config, "ETF_ROUTER_ENABLED", False):
        return {"enabled": False}

    decision = bot.router_decision
    summary: Dict[str, Any] = {
        "enabled": True,
        "tape_initialized": bot.tape_initialized,
        "decision_made": bot.router_decision_made,
        "branch": bot.router_branch,
        "mr_blocked_today": bot.mr_blocked_today,
        "router_traded_today": bot.router_traded_today,
        "no_trade_breakout": {
            "active": getattr(bot, "no_trade_breakout_active", False),
            "done": getattr(bot, "no_trade_breakout_done", False),
            "entered": getattr(bot, "no_trade_breakout_entered_today", False),
            "subtype": getattr(bot, "no_trade_breakout_subtype", None),
            "qqq_range": getattr(bot, "no_trade_breakout_range", None),
            "morning_high": getattr(bot, "no_trade_breakout_high", None),
            "morning_low": getattr(bot, "no_trade_breakout_low", None),
        },
    }
    if decision is not None:
        try:
            summary["decision"] = decision.to_dict()
        except Exception:
            logger.warning("ETF router decision.to_dict() failed", exc_info=True)
    if bot.etf_position:
        # The exit path nulls out etf_position on a successful exit, so
        # a non-null value here means we still have an open ETF
        # position at EOD (which would be a bug — log it loud).
        logger.warning(f"ETF router EOD: still holding {bot.etf_position.get('symbol')}")
        summary["open_position_at_eod"] = bot.etf_position
    return summary


def save_etf_router_artifact(bot) -> None:
    """Write state/logs/etf_router_YYYY-MM-DD.json for forensic review."""
    today = date.today().isoformat()
    path = os.path.join(config.LOG_DIR, f"etf_router_{today}.json")
    payload = {"date": today, **build_etf_router_summary(bot)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(f"ETF router artifact saved: {path}")


# ────────────────────────────────────────────────────────────────────
# Startup phase + stale-order cleanup
# ────────────────────────────────────────────────────────────────────

def run_startup_phase(bot) -> None:
    """9:00-9:25 AM: Startup and pre-market preparation."""
    logger.info("=" * 60)
    logger.info("STARTUP PHASE (9:00-9:25)")
    logger.info("=" * 60)

    # Load and log config
    logger.info(f"ETF Router enabled: {getattr(config, 'ETF_ROUTER_ENABLED', False)}")
    logger.info(f"MR Overnight enabled: {getattr(config, 'MR_OVERNIGHT_ENABLED', True)}")
    logger.info(f"MR Permission mode: {getattr(config, 'MR_PERMISSION_MODE', 'skip_if_router_traded')}")

    # Reconcile account state
    try:
        account = bot.position_mgr.get_account()
        if account:
            cash = float(account.get("cash", 0))
            buying_power = float(account.get("buying_power", 0))
            logger.info(f"Account state - Cash: ${cash:,.2f}, BP: ${buying_power:,.2f}")
        else:
            logger.warning("Could not fetch account state")
    except Exception as e:
        logger.error(f"Error fetching account: {e}")

    # Check broker positions
    broker_positions = bot.position_mgr.get_broker_positions()
    if broker_positions is not None:
        logger.info(f"Broker positions: {len(broker_positions)}")
        for pos in broker_positions:
            logger.info(f"  - {pos.get('symbol')}: {pos.get('qty')} shares")
    else:
        logger.warning("Could not fetch broker positions")

    # Cancel stale orders (ETF router symbols)
    cancel_stale_etf_orders(bot)

    bot.startup_done = True
    bot._save_state()
    logger.info("Startup phase complete")


def cancel_stale_etf_orders(bot) -> None:
    """Cancel only stale ETF router orders from a previous session.

    IMPORTANT: must NOT touch overnight premarket limit sells placed
    between 05:00 and 06:00 — those are on equity positions, never
    on ETF router symbols. Earlier versions used
    ``cancel_all_open_orders()`` here which wiped those limits before
    they had any chance to fill in the 09:00-09:25 pre-open window.
    """
    etf_symbols = getattr(
        config,
        "ETF_ROUTER_SYMBOLS",
        ["QQQ", "SPY", "IWM", "XLK", "VXX", "SQQQ", "UVXY", "TQQQ"],
    )
    logger.info(f"Cancelling any stale ETF-only orders for: {etf_symbols}")
    try:
        cancelled = bot.position_mgr.cancel_orders_for_symbols(etf_symbols)
        logger.info(f"Stale ETF order cleanup complete: {cancelled} canceled")
    except Exception as e:
        logger.error(f"Error cancelling stale ETF orders: {e}")


# ────────────────────────────────────────────────────────────────────
# Tape recording + updates
# ────────────────────────────────────────────────────────────────────

def initialize_tape_recording(bot) -> None:
    """9:30 AM: Record canonical 09:30 opens and start tape recording.

    Source priority (highest to lowest fidelity):
      1. First 1-min bar of today via get_intraday_bars_for_signal — this
         is the canonical 09:30 opening print.
      2. snapshot.daily_bar.o, but ONLY if daily_bar.t parses to today.
         At ~09:30:00 the daily_bar field often still holds the prior
         trading day's bar; we must reject those.
      3. snapshot.last_trade.p, but ONLY if last_trade.t is within the
         last 60s. This guards against stale pre-market prints (e.g.
         a 09:29:50 IEX trade being treated as the 09:30 open).

    If none of the sources are usable for a given symbol, that symbol is
    skipped. ``ETFTapeSnapshot.is_valid()`` will then be False for it and
    ``ETFRouter.make_decision`` will gracefully resolve to NO_TRADE.

    If NO symbols produce a usable open (the most common case when this
    runs at exactly 09:30:00.x and the 1-min bar hasn't formed yet), we
    leave ``tape_initialized = False`` so the next 1s tick retries.
    """
    logger.info("=" * 60)
    logger.info("TAPE INITIALIZATION (9:30)")
    logger.info("=" * 60)

    etf_symbols = getattr(
        config,
        "ETF_ROUTER_SYMBOLS",
        ["QQQ", "SPY", "IWM", "XLK", "VXX", "SQQQ", "UVXY", "TQQQ"],
    )
    now = datetime.now(_ET)
    today_iso = now.date().isoformat()

    try:
        # 1) Canonical source: first 1-min bar of today, 09:30 -> 09:31.
        minute_bars: Dict[str, List[dict]] = {}
        try:
            minute_bars = bot.alpaca.get_intraday_bars_for_signal(
                etf_symbols, today_iso, start="09:30", end="09:31",
            )
        except Exception:
            logger.warning("Tape init: 09:30 minute-bar fetch failed; will rely on snapshot fallbacks", exc_info=True)

        # 2) Snapshot fallbacks (also captured for "latest" logging).
        snapshots = bot.alpaca.get_snapshots(etf_symbols) or {}

        opens: Dict[str, float] = {}
        sources: Dict[str, str] = {}

        for symbol in etf_symbols:
            open_price: Optional[float] = None
            source = "none"

            # ── Source 1: today's first 1-min bar ──────────────────────
            bars = minute_bars.get(symbol) or []
            if bars:
                first_bar = bars[0]
                bar_dt = bot._bar_dt(first_bar)
                bar_open = bot._bar_float(first_bar, "o", "open")
                if (
                    bar_open is not None
                    and bar_open > 0
                    and bar_dt is not None
                    and bar_dt.date() == now.date()
                    and bar_dt.hour == 9
                    and bar_dt.minute == 30
                ):
                    open_price = bar_open
                    source = "minute_bar_0930"

            snap = snapshots.get(symbol, {}) or {}

            # ── Source 2: parsed snapshot last_price if fresh (<= 60s) ──
            # NOTE: get_snapshots returns flattened dicts; the raw
            # latestTrade/dailyBar fields are not exposed as nested
            # objects. There is no exposed daily_bar timestamp, so we
            # cannot safely use the daily_bar open as a fallback — it
            # may be the prior trading day's bar. Rely on a fresh
            # last_price instead.
            if open_price is None:
                lt_p = snap.get("last_price")
                lt_t = snap.get("timestamp")
                if lt_p and lt_t:
                    try:
                        lt_dt = datetime.fromisoformat(str(lt_t).replace("Z", "+00:00"))
                        age_s = (now - lt_dt.astimezone(_ET)).total_seconds()
                    except (TypeError, ValueError):
                        age_s = 9999.0
                    if 0 <= age_s <= 60:
                        open_price = float(lt_p)
                        source = f"last_price_fresh_{age_s:.0f}s"
                    else:
                        logger.warning(
                            f"Tape init {symbol}: rejecting last_price — age={age_s:.0f}s (limit 60s)"
                        )

            if open_price is not None and open_price > 0:
                opens[symbol] = open_price
                bot.etf_opens_930[symbol] = open_price
                sources[symbol] = source
                latest = snap.get("last_price", "N/A")
                logger.info(
                    f"ETF open {symbol}: ${open_price:.2f} (source={source}, "
                    f"latest={latest}, ts={now.strftime('%H:%M:%S')})"
                )
            else:
                logger.warning(
                    f"Tape init {symbol}: no usable 09:30 open yet — will retry on next tick"
                )

        logger.info(f"9:30 Opens summary: {opens}")
        logger.info(f"Open sources: {sources}")

        # If we couldn't seed ANY symbol yet, defer init so the next 1s
        # tick retries (the 1-min bar typically lands within ~30s of the
        # open with the IEX feed).
        if not opens:
            logger.warning("Tape init: no symbols had a usable open; retrying next tick")
            return

        # If we got SOME but not all, still proceed — the missing ones
        # will get populated by subsequent update_tape() calls (their
        # open_930 will then be the first tape print, which is acceptable
        # for the symbols that couldn't be sourced cleanly).
        bot.etf_router.start_recording(opens, datetime.now(_ET))
        bot.tape_recording_active = True
        bot.tape_initialized = True
        bot._save_state()

    except Exception as e:
        logger.error(f"Error initializing tape: {e}", exc_info=True)


def update_tape(bot, force: bool = False) -> None:
    """Update tape with latest prices during 9:30-10:00 window.

    Uses the parsed snapshot ``last_price`` field.

    Throttled (API efficiency #1) to one snapshot fetch per
    ``ETF_TAPE_UPDATE_INTERVAL_SECONDS`` (default 5s). The main loop
    ticks every 1s in this hot window, but the router only needs the
    9:30 open, the 9:45 continuation print, and the 10:00 final
    price — sub-second granularity is wasted budget.

    Pass ``force=True`` to bypass the throttle (used for the final
    tape update right before the 10:00 decision).
    """
    if not bot.tape_recording_active:
        return

    # Throttle: skip if the previous fetch was too recent.
    interval = float(getattr(config, "ETF_TAPE_UPDATE_INTERVAL_SECONDS", 5))
    now_mono = time.monotonic()
    if not force and interval > 0 and (now_mono - bot._tape_last_update_monotonic) < interval:
        return

    etf_symbols = getattr(
        config,
        "ETF_ROUTER_SYMBOLS",
        ["QQQ", "SPY", "IWM", "XLK", "VXX", "SQQQ", "UVXY", "TQQQ"],
    )

    try:
        snapshots = bot.alpaca.get_snapshots(etf_symbols)
        now = datetime.now(_ET)

        for symbol in etf_symbols:
            snap = snapshots.get(symbol, {}) or {}
            price = snap.get("last_price")
            if price:
                bot.etf_router.update_tape(symbol, float(price), now)

        bot._tape_last_update_monotonic = now_mono

    except Exception as e:
        logger.error(f"Error updating tape: {e}")


# ────────────────────────────────────────────────────────────────────
# 10:00 decision + entry execution
# ────────────────────────────────────────────────────────────────────

def make_router_decision(bot) -> None:
    """10:00 AM: Make ETF router decision and (maybe) enter.

    The daily-loss kill switch is checked BEFORE the ETF entry. The
    decision and ``mr_blocked_today`` flag are still recorded so the
    EOD report reflects what the router would have done; we just
    suppress the order submission.
    """
    logger.info("=" * 60)
    logger.info("ROUTER DECISION (10:00)")
    logger.info("=" * 60)

    bot.tape_recording_active = False
    now = datetime.now(_ET)

    # Make decision
    decision = bot.etf_router.make_decision(now)
    bot.router_decision = decision
    bot.router_decision_made = True

    # Update state
    bot.router_branch = decision.branch.value
    bot.router_traded_today = decision.mr_blocked()
    bot.mr_blocked_today = decision.mr_blocked()

    logger.info(f"Router decision: {decision.branch.value}")
    logger.info(f"MR blocked today: {bot.mr_blocked_today}")

    if decision.symbol:
        logger.info(f"Selected ETF: {decision.symbol}")
        logger.info(f"Entry: {decision.entry_time}, Exit: {decision.exit_time}")
        # Kill-switch gate (issue #5): block 10:00 ETF entry if the
        # daily-loss circuit breaker has tripped.
        if bot._check_daily_loss_kill_switch():
            logger.critical(
                f"ETF entry BLOCKED by daily-loss kill switch — {bot.kill_switch_reason}; "
                f"branch={decision.branch.value} would have bought {decision.symbol}"
            )
        else:
            execute_etf_entry(bot, decision)
    else:
        logger.info("No primary ETF router trade - MR allowed at 15:45")
        prepare_no_trade_qqq_breakout(bot, decision, now)

    bot._save_state()


def execute_etf_entry(bot, decision: RouterDecision) -> None:
    """Execute ETF entry after 10:00 decision.

    Adds (issue #4) a spread + staleness execution gate via
    ``filter_execution_ready`` and a marketable-limit order (issue #6)
    at ``ask * (1 + ETF_ENTRY_MAX_SLIPPAGE_PCT)`` so a momentary wide
    quote cannot cost more than the configured slippage cap.
    """
    if not decision.symbol:
        return

    symbol = decision.symbol
    logger.info(f"Executing ETF entry: {symbol}")

    try:
        # Calculate position size
        account = bot.position_mgr.get_account()
        if not account:
            logger.error("Cannot fetch account for ETF sizing")
            return

        equity = float(account.get("equity", 0))
        etf_capital_pct = getattr(config, "ETF_ROUTER_CAPITAL_PCT", 0.30)
        etf_budget = equity * etf_capital_pct

        # Fresh snapshot for sizing AND for the execution gate.
        snapshots = bot.alpaca.get_snapshots([symbol]) or {}
        snap = snapshots.get(symbol, {}) or {}

        # Execution gate (issue #4): tight spread, fresh quote, require quote.
        orderable, exec_rejected = filter_execution_ready(
            [symbol], snapshots,
            max_spread_pct=float(getattr(config, "ETF_ENTRY_MAX_SPREAD_PCT", 0.005)),
            require_quote=True,
            max_stale_seconds=float(getattr(config, "ETF_ENTRY_MAX_STALE_SECONDS", 10.0)),
        )
        if symbol not in orderable:
            reason = exec_rejected.get(symbol, "unknown")
            logger.warning(f"ETF entry REJECTED for {symbol}: {reason}")
            return

        ask = snap.get("ask")
        bid = snap.get("bid")
        last_price = snap.get("last_price")
        if not last_price:
            logger.error(f"ETF entry {symbol}: no last_price after gate (should not happen)")
            return

        # Size off ask (where we'd actually fill) if available, else last.
        sizing_price = float(ask) if ask else float(last_price)
        qty = int(etf_budget / sizing_price)

        if qty <= 0:
            logger.warning(
                f"ETF qty <= 0, skipping entry: budget=${etf_budget:.2f}, "
                f"sizing_price=${sizing_price:.2f}"
            )
            return

        # Marketable limit (issue #6): bound slippage above the ask.
        slippage_pct = float(getattr(config, "ETF_ENTRY_MAX_SLIPPAGE_PCT", 0.005))
        if ask:
            limit_price = float(ask) * (1.0 + slippage_pct)
            order_type = "limit"
            logger.info(
                f"ETF entry {symbol}: qty={qty}, bid={bid}, ask={ask}, "
                f"last={last_price}, marketable_limit={limit_price:.4f} "
                f"(ask + {slippage_pct:.2%})"
            )
            order, error_type = bot.position_mgr.submit_buy_order(
                symbol, qty, order_type="limit", limit_price=limit_price,
            )
        else:
            # No ask available — falls back to market order. Should be
            # rare after the execution gate, kept as a defensive path.
            logger.warning(f"ETF entry {symbol}: no ask after gate — using market order")
            order, error_type = bot.position_mgr.submit_buy_order(symbol, qty)
            limit_price = None
            order_type = "market"

        price = float(last_price)

        if order and order.get("id"):
            # Wait for fill confirmation (max 10 seconds for liquid ETFs)
            fill = bot.position_mgr.get_order_fill(order["id"], max_wait=10)
            if fill and int(fill.get("filled_qty", 0)) > 0:
                filled_qty = int(fill["filled_qty"])
                fill_price = float(fill.get("filled_avg_price", price))
                bot.etf_position = {
                    "symbol": symbol,
                    "qty": filled_qty,
                    "entry_price": fill_price,
                    "entry_time": datetime.now(_ET).isoformat(),
                    "branch": decision.branch.value,
                    "planned_exit_time": decision.exit_time.isoformat() if decision.exit_time else None,
                    "order_id": order.get("id"),
                }
                logger.info(f"ETF position opened: {symbol} {filled_qty} shares @ ${fill_price:.2f}")
            else:
                logger.warning(f"ETF entry not filled for {symbol}; canceling and leaving flat")
                bot.position_mgr._cancel_order(order["id"])
        else:
            logger.error(f"Failed to submit ETF buy order for {symbol}: {error_type}")

    except Exception as e:
        logger.error(f"Error executing ETF entry: {e}", exc_info=True)


# ────────────────────────────────────────────────────────────────────
# Exit checkpoints
# ────────────────────────────────────────────────────────────────────

def check_etf_exits(bot, current_time: dt_time) -> None:
    """Check ETF exit checkpoints at 11:00, 14:00, 15:00."""
    if not bot.etf_position:
        return

    symbol = bot.etf_position.get("symbol")
    planned_exit = bot.etf_position.get("planned_exit_time")

    if not planned_exit:
        return

    # Parse planned exit time
    try:
        if isinstance(planned_exit, str):
            # Handle HH:MM:SS format
            parts = planned_exit.split(":")
            exit_h, exit_m = int(parts[0]), int(parts[1])
        else:
            exit_h, exit_m = planned_exit.hour, planned_exit.minute
    except Exception:
        return

    exit_time = dt_time(exit_h, exit_m)

    # Check if it's time to exit
    if current_time >= exit_time:
        logger.info(f"ETF exit time reached: {symbol} at {current_time}")
        execute_etf_exit(bot)


def execute_etf_exit(bot) -> None:
    """Execute ETF exit order with fill confirmation and duplicate guard."""
    if not bot.etf_position:
        return

    symbol = bot.etf_position.get("symbol")
    qty = bot.etf_position.get("qty", 0)

    # Check if exit order already submitted (duplicate guard)
    existing_exit_id = bot.etf_position.get("exit_order_id")
    exit_submitted_at = bot.etf_position.get("exit_submitted_at")

    if existing_exit_id:
        logger.info(f"ETF exit order already exists for {symbol}; checking fill status")
        fill = bot.position_mgr.get_order_fill(existing_exit_id, max_wait=2)

        if fill and int(fill.get("filled_qty", 0)) > 0:
            filled_qty = int(fill["filled_qty"])
            if filled_qty >= qty:
                logger.info(f"ETF exit filled: {symbol} {filled_qty} shares")
                bot.etf_position = None
                bot._save_state()
            else:
                remaining = qty - filled_qty
                bot.etf_position["qty"] = remaining
                bot.etf_position["exit_order_id"] = None  # Clear to allow retry
                logger.warning(f"ETF exit partial fill: {symbol} {filled_qty}/{qty}, {remaining} remaining")
                bot._save_state()
            return

        # If exit has been pending too long, cancel and retry once
        if exit_submitted_at:
            submitted_dt = datetime.fromisoformat(exit_submitted_at)
            age_seconds = (datetime.now(_ET) - submitted_dt).total_seconds()

            if age_seconds > 60:
                logger.warning(
                    f"ETF exit order {existing_exit_id} pending {age_seconds:.0f}s; canceling and retrying"
                )
                try:
                    bot.position_mgr._cancel_order(existing_exit_id)
                except Exception:
                    logger.warning("ETF exit cancel failed", exc_info=True)

                bot.etf_position["exit_order_id"] = None
                bot.etf_position["exit_submitted_at"] = None
                bot._save_state()
                return

        logger.info(f"ETF exit order {existing_exit_id} still pending; waiting for fill")
        return

    logger.info(f"Executing ETF exit: {symbol} {qty} shares")

    try:
        # Submit sell order
        order = bot.position_mgr._submit_sell_order(
            symbol,
            qty,
            order_type="market",
            time_in_force="day",
            extended_hours=False,
        )

        if order and order.get("id"):
            # Record exit order id for duplicate guard
            bot.etf_position["exit_order_id"] = order.get("id")
            bot.etf_position["exit_submitted_at"] = datetime.now(_ET).isoformat()
            bot._save_state()

            # Wait for fill confirmation
            fill = bot.position_mgr.get_order_fill(order["id"], max_wait=10)
            if fill and int(fill.get("filled_qty", 0)) > 0:
                filled_qty = int(fill["filled_qty"])
                if filled_qty >= qty:
                    logger.info(f"ETF exit filled: {symbol} {filled_qty} shares")
                    bot.etf_position = None
                    bot._save_state()
                else:
                    # Partial fill - update remaining qty
                    remaining = qty - filled_qty
                    bot.etf_position["qty"] = remaining
                    bot.etf_position["exit_order_id"] = None
                    logger.warning(f"ETF exit partial fill: {symbol} {filled_qty}/{qty}, {remaining} remaining")
                    bot._save_state()
            else:
                logger.warning(f"ETF exit submitted but not filled yet for {symbol}; will retry on next tick")
        else:
            logger.error(f"Failed to submit ETF sell order for {symbol}")

    except Exception as e:
        logger.error(f"Error executing ETF exit: {e}", exc_info=True)


# ────────────────────────────────────────────────────────────────────
# No-trade QQQ range-breakout fallback
# ────────────────────────────────────────────────────────────────────

def _parse_hhmm(value: str) -> dt_time:
    parts = str(value).split(":")
    return dt_time(int(parts[0]), int(parts[1]))


def classify_no_trade_live_subtype(bot) -> str:
    """Classify a primary-router NO_TRADE day into a live subtype.

    The research files used precomputed live subtypes. The live bot did not
    have that classifier wired in, so this runtime classifier intentionally
    uses only the already-recorded 09:30-10:00 ETF tape. Keep this simple and
    auditable; if you later recover the exact research classifier, swap it in
    here without changing the execution plumbing.
    """
    returns = bot.etf_router.tape.get_returns_summary()
    qqq = returns.get("QQQ")
    spy = returns.get("SPY")
    iwm = returns.get("IWM")
    xlk = returns.get("XLK")
    vxx = returns.get("VXX")
    uvxy = returns.get("UVXY")

    qqq = 0.0 if qqq is None else float(qqq)
    spy = 0.0 if spy is None else float(spy)
    iwm = 0.0 if iwm is None else float(iwm)
    xlk = 0.0 if xlk is None else float(xlk)
    vxx = 0.0 if vxx is None else float(vxx)
    uvxy = 0.0 if uvxy is None else float(uvxy)

    if vxx > 25 or uvxy > 75:
        return "LIVE_VOL_WARNING"

    if qqq > 5 and spy > -5:
        if xlk > 0:
            return "LIVE_BULLISH_MESSY"
        return "LIVE_WEAK_GREEN"

    if qqq < -10 or spy < -10:
        return "LIVE_MILD_RISK_OFF"

    if iwm < -15 and qqq > -10 and spy > -10:
        return "LIVE_SMALL_CAP_WEAKNESS"

    return "LIVE_FLAT_CHOP"


def _get_qqq_morning_range_from_bars(bot, today_iso: str) -> tuple:
    """Return (range_pct, high, low) for QQQ 09:30 <= t < 10:00.

    Prefer the historical 1-minute bars so the fallback uses the same range
    definition as the research. Fall back to the live tape if the minute bars
    are unavailable at exactly 10:00.
    """
    try:
        bars_by_symbol = bot.alpaca.get_intraday_bars_for_signal(
            ["QQQ"], today_iso, start="09:30", end="10:00",
        ) or {}
        bars = bars_by_symbol.get("QQQ") or []
        highs = []
        lows = []
        first_open = None
        for bar in bars:
            bar_dt = bot._bar_dt(bar)
            if bar_dt is None or not (bar_dt.hour == 9 and 30 <= bar_dt.minute < 60):
                continue
            high = bot._bar_float(bar, "h", "high")
            low = bot._bar_float(bar, "l", "low")
            opn = bot._bar_float(bar, "o", "open")
            if high is None or low is None:
                continue
            highs.append(float(high))
            lows.append(float(low))
            if first_open is None and opn is not None and opn > 0:
                first_open = float(opn)
        if highs and lows and first_open and first_open > 0:
            hi = max(highs)
            lo = min(lows)
            return (hi - lo) / first_open, hi, lo
    except Exception:
        logger.warning("No-trade breakout: QQQ 09:30-10:00 bar range fetch failed; using tape fallback", exc_info=True)

    q = bot.etf_router.tape.qqq
    if q.open_930 and q.high and q.low and q.open_930 > 0:
        return (q.high - q.low) / q.open_930, q.high, q.low
    return None, None, None


def prepare_no_trade_qqq_breakout(bot, decision: RouterDecision, timestamp: datetime) -> None:
    """Arm the no-trade QQQ breakout fallback after a primary NO_TRADE decision."""
    bot.no_trade_breakout_active = False
    bot.no_trade_breakout_done = True
    bot.no_trade_breakout_entered_today = False
    bot.no_trade_breakout_subtype = None
    bot.no_trade_breakout_range = None
    bot.no_trade_breakout_high = None
    bot.no_trade_breakout_low = None

    if not getattr(config, "NO_TRADE_QQQ_BREAKOUT_ENABLED", False):
        return

    if decision.branch != RouterBranch.NO_TRADE or decision.symbol:
        return

    subtype = classify_no_trade_live_subtype(bot)
    allowed = set(getattr(config, "NO_TRADE_QQQ_BREAKOUT_ALLOWED_SUBTYPES", []))
    bot.no_trade_breakout_subtype = subtype

    if allowed and subtype not in allowed:
        logger.info("No-trade QQQ breakout not armed: subtype=%s not in allowed=%s", subtype, sorted(allowed))
        return

    today_iso = timestamp.date().isoformat()
    q_range, q_high, q_low = _get_qqq_morning_range_from_bars(bot, today_iso)
    threshold = float(getattr(config, "NO_TRADE_QQQ_BREAKOUT_RANGE_THRESHOLD", 0.0045))
    bot.no_trade_breakout_range = q_range
    bot.no_trade_breakout_high = q_high
    bot.no_trade_breakout_low = q_low

    if q_range is None or q_high is None or q_low is None:
        logger.warning("No-trade QQQ breakout not armed: missing QQQ range data")
        return

    if q_range < threshold:
        logger.info(
            "No-trade QQQ breakout not armed: subtype=%s qqq_range=%.3f%% < %.3f%%",
            subtype, q_range * 100.0, threshold * 100.0,
        )
        return

    bot.no_trade_breakout_active = True
    bot.no_trade_breakout_done = False
    logger.warning(
        "NO_TRADE_QQQ_BREAKOUT ARMED: subtype=%s qqq_range=%.3f%% high=%.4f low=%.4f "
        "threshold=%.3f%% exit=%s MR_ALLOWED=True",
        subtype,
        q_range * 100.0,
        q_high,
        q_low,
        threshold * 100.0,
        getattr(config, "NO_TRADE_QQQ_BREAKOUT_EXIT_TIME", "11:30"),
    )


def monitor_no_trade_qqq_breakout(bot, current_time: dt_time) -> None:
    """Poll QQQ after 10:00 and enter TQQQ/SQQQ on range break.

    Uses latest QQQ price, not minute high/low, to approximate live polling.
    Once a trade is attempted, the sleeve is marked done for the day; there is
    no re-entry.
    """
    if bot.no_trade_breakout_done or not bot.no_trade_breakout_active:
        return

    exit_t = _parse_hhmm(getattr(config, "NO_TRADE_QQQ_BREAKOUT_EXIT_TIME", "11:30"))
    if current_time >= exit_t:
        logger.info("No-trade QQQ breakout expired flat at %s", current_time)
        bot.no_trade_breakout_active = False
        bot.no_trade_breakout_done = True
        bot._save_state()
        return

    if bot.etf_position:
        return

    interval = float(getattr(config, "NO_TRADE_QQQ_BREAKOUT_POLL_SECONDS", 5.0))
    now_mono = time.monotonic()
    if interval > 0 and (now_mono - bot._no_trade_breakout_last_check_monotonic) < interval:
        return
    bot._no_trade_breakout_last_check_monotonic = now_mono

    q_high = bot.no_trade_breakout_high
    q_low = bot.no_trade_breakout_low
    if q_high is None or q_low is None:
        bot.no_trade_breakout_active = False
        bot.no_trade_breakout_done = True
        bot._save_state()
        return

    try:
        snaps = bot.alpaca.get_snapshots(["QQQ"]) or {}
        q_snap = snaps.get("QQQ", {}) or {}
        q_price = q_snap.get("last_price") or q_snap.get("close")
        if not q_price:
            logger.warning("No-trade QQQ breakout: no QQQ last price")
            return
        q_price = float(q_price)
    except Exception:
        logger.warning("No-trade QQQ breakout: QQQ snapshot failed", exc_info=True)
        return

    if q_price > float(q_high):
        logger.warning(
            "NO_TRADE_QQQ_BREAKOUT trigger UP: QQQ %.4f > morning_high %.4f -> TQQQ",
            q_price, q_high,
        )
        execute_no_trade_breakout_entry(bot, symbol="TQQQ", trigger_price=q_price, trigger_direction="up")
    elif q_price < float(q_low):
        logger.warning(
            "NO_TRADE_QQQ_BREAKOUT trigger DOWN: QQQ %.4f < morning_low %.4f -> SQQQ",
            q_price, q_low,
        )
        execute_no_trade_breakout_entry(bot, symbol="SQQQ", trigger_price=q_price, trigger_direction="down")


def execute_no_trade_breakout_entry(bot, symbol: str, trigger_price: float, trigger_direction: str) -> None:
    """Buy TQQQ/SQQQ for the no-trade breakout fallback and place trailing stop."""
    bot.no_trade_breakout_active = False
    bot.no_trade_breakout_done = True

    if bot._check_daily_loss_kill_switch():
        logger.critical(
            "No-trade QQQ breakout BLOCKED by daily-loss kill switch — %s", bot.kill_switch_reason
        )
        bot._save_state()
        return

    try:
        account = bot.position_mgr.get_account()
        if not account:
            logger.error("No-trade QQQ breakout: cannot fetch account for sizing")
            bot._save_state()
            return
        equity = float(account.get("equity", 0) or 0)
        if equity <= 0:
            logger.error("No-trade QQQ breakout: equity <= 0")
            bot._save_state()
            return

        budget = equity * float(getattr(config, "NO_TRADE_QQQ_BREAKOUT_CAPITAL_PCT", 0.50))
        snapshots = bot.alpaca.get_snapshots([symbol]) or {}
        snap = snapshots.get(symbol, {}) or {}

        orderable, rejected = filter_execution_ready(
            [symbol], snapshots,
            max_spread_pct=float(getattr(config, "NO_TRADE_QQQ_BREAKOUT_ENTRY_MAX_SPREAD_PCT", 0.005)),
            require_quote=True,
            max_stale_seconds=float(getattr(config, "NO_TRADE_QQQ_BREAKOUT_ENTRY_MAX_STALE_SECONDS", 10.0)),
        )
        if symbol not in orderable:
            logger.warning("No-trade QQQ breakout entry rejected for %s: %s", symbol, rejected.get(symbol, "unknown"))
            bot._save_state()
            return

        ask = snap.get("ask")
        bid = snap.get("bid")
        last = snap.get("last_price") or snap.get("close")
        if not last:
            logger.error("No-trade QQQ breakout: no last price for %s", symbol)
            bot._save_state()
            return

        sizing_price = float(ask) if ask else float(last)
        qty = int(budget / sizing_price)
        if qty <= 0:
            logger.warning("No-trade QQQ breakout: qty <= 0 for %s budget=$%.2f price=%.4f", symbol, budget, sizing_price)
            bot._save_state()
            return

        slippage_pct = float(getattr(config, "NO_TRADE_QQQ_BREAKOUT_ENTRY_MAX_SLIPPAGE_PCT", 0.005))
        if ask:
            limit_price = float(ask) * (1.0 + slippage_pct)
            order_type = "limit"
            logger.warning(
                "No-trade QQQ breakout BUY %s qty=%d bid=%s ask=%s last=%s limit=%.4f trigger=%s %.4f",
                symbol, qty, bid, ask, last, limit_price, trigger_direction, trigger_price,
            )
            order, error_type = bot.position_mgr.submit_buy_order(
                symbol, qty, order_type="limit", limit_price=limit_price, timeout=5,
            )
        else:
            order_type = "market"
            limit_price = None
            order, error_type = bot.position_mgr.submit_buy_order(symbol, qty, timeout=5)

        if not order or not order.get("id"):
            logger.error("No-trade QQQ breakout buy failed for %s: %s", symbol, error_type)
            bot._save_state()
            return

        fill = bot.position_mgr.get_order_fill(order["id"], max_wait=10)
        if not fill or int(fill.get("filled_qty", 0)) <= 0:
            logger.warning("No-trade QQQ breakout buy not filled for %s; canceling", symbol)
            bot.position_mgr._cancel_order(order["id"])
            bot._save_state()
            return

        filled_qty = int(fill["filled_qty"])
        fill_price = float(fill.get("filled_avg_price", last))
        trail_pct = float(getattr(config, "NO_TRADE_QQQ_BREAKOUT_TRAIL_PCT", 1.50))
        trailing_order = bot.position_mgr.submit_trailing_stop_sell_order(symbol, filled_qty, trail_pct)
        trailing_id = trailing_order.get("id") if trailing_order else None
        if not trailing_id:
            logger.error("No-trade QQQ breakout: trailing stop submit failed for %s; 11:30 failsafe exit will still run", symbol)

        bot.etf_position = {
            "symbol": symbol,
            "qty": filled_qty,
            "entry_price": fill_price,
            "entry_time": datetime.now(_ET).isoformat(),
            "branch": "NO_TRADE_QQQ_BREAKOUT",
            "planned_exit_time": _parse_hhmm(getattr(config, "NO_TRADE_QQQ_BREAKOUT_EXIT_TIME", "11:30")).isoformat(),
            "order_id": order.get("id"),
            "entry_order_type": order_type,
            "entry_limit_price": limit_price,
            "trailing_stop_order_id": trailing_id,
            "trail_percent": trail_pct,
            "mr_blocking": False,
            "trigger_direction": trigger_direction,
            "trigger_price": trigger_price,
            "live_subtype": bot.no_trade_breakout_subtype,
            "qqq_range": bot.no_trade_breakout_range,
            "qqq_morning_high": bot.no_trade_breakout_high,
            "qqq_morning_low": bot.no_trade_breakout_low,
        }
        bot.no_trade_breakout_entered_today = True
        logger.warning(
            "NO_TRADE_QQQ_BREAKOUT POSITION OPENED: %s qty=%d fill=%.4f trail=%.2f%% exit=%s MR_ALLOWED=True",
            symbol, filled_qty, fill_price, trail_pct, bot.etf_position["planned_exit_time"],
        )
        bot._save_state()

    except Exception:
        logger.error("No-trade QQQ breakout entry failed", exc_info=True)
        bot._save_state()
