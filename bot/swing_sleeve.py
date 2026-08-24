"""Swing ETF Sleeve — Portfolio 7: TQQQ -3% + UPRO -3% + NUGT 3+ down streak.

Multi-day hold (2 trading days) leveraged ETF swing strategy with SPY 200-day
MA regime filter.  Entered at 15:45 alongside the overnight TQQQ sleeve; exits
at 15:45 two trading days later (NOT at the 09:30 morning liquidation).

Signal logic:
  - TQQQ: buy when daily return <= -3%
  - UPRO: buy when daily return <= -3%
  - NUGT: buy when 3+ consecutive down days (and today is also down)
  - All filtered by SPY close > 200-day MA
  - 2-day hold, equal allocation per triggered symbol

This sleeve operates INDEPENDENTLY of the overnight TQQQ and intraday router.
Swing positions are deliberately held through the 09:30 morning liquidation
(morning_exits.py skips symbols tracked in bot.swing_positions).
"""
from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from bot import config
from bot.universe_builder import filter_execution_ready

logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")


# ──────────────────────────────────────────────────────────────────
# Public API — called by integrated_main.py
# ──────────────────────────────────────────────────────────────────

def evaluate_swing_sleeve(bot) -> None:
    """15:45 entry point: exit maturing swing positions, then evaluate new signals."""
    if not getattr(config, "SWING_SLEEVE_ENABLED", True):
        logger.info("Swing sleeve disabled")
        return

    logger.info("=" * 60)
    logger.info("SWING SLEEVE EVALUATION (15:45)")
    logger.info("=" * 60)

    # Phase 1: Exit swing positions that have reached hold period
    _check_swing_exits(bot)

    # Phase 2: Evaluate new swing entries
    _evaluate_swing_entries(bot)

    bot.swing_decision_made = True
    bot._save_state()


def get_swing_symbols(bot) -> set:
    """Return the set of symbols currently held as swing positions."""
    swing_positions = getattr(bot, "swing_positions", {})
    return {pos["symbol"] for pos in swing_positions.values() if pos.get("symbol")}


# ──────────────────────────────────────────────────────────────────
# Exit logic
# ──────────────────────────────────────────────────────────────────

def _check_swing_exits(bot) -> None:
    """Sell swing positions that have been held for >= SWING_HOLD_DAYS trading days."""
    swing_positions = getattr(bot, "swing_positions", {})
    if not swing_positions:
        logger.info("Swing exits: no open swing positions")
        return

    hold_days = int(getattr(config, "SWING_HOLD_DAYS", 2))

    # Fetch SPY daily bars to count trading days
    spy_bars = bot.alpaca.get_daily_bars(["SPY"], days=hold_days + 10)
    spy_dates = _extract_trading_dates(spy_bars.get("SPY", []))

    if not spy_dates:
        logger.warning("Swing exits: cannot fetch SPY daily bars — skipping exit check")
        return

    today_str = datetime.now(_ET).strftime("%Y-%m-%d")
    # Today may or may not be in spy_dates yet (market hasn't closed).
    # Use the last available date as "today" for counting purposes.
    # If today is in the list, use it; otherwise use the last date before today.
    trading_dates_up_to_today = [d for d in spy_dates if d <= today_str]
    if not trading_dates_up_to_today:
        logger.warning("Swing exits: no SPY trading dates up to today — skipping")
        return

    to_exit: List[Tuple[str, Dict[str, Any]]] = []

    for key, pos in list(swing_positions.items()):
        entry_date = pos.get("entry_date", "")
        symbol = pos.get("symbol", "")

        # Count trading days from entry_date (exclusive) to today (inclusive)
        elapsed = sum(1 for d in trading_dates_up_to_today if d > entry_date)

        logger.info(
            f"Swing exit check: {symbol} entered {entry_date}, "
            f"elapsed {elapsed}/{hold_days} trading days"
        )

        if elapsed >= hold_days:
            to_exit.append((key, pos))

    if not to_exit:
        logger.info(f"Swing exits: 0 positions ready to exit ({len(swing_positions)} still holding)")
        return

    logger.info(f"Swing exits: {len(to_exit)} positions ready for exit")

    for key, pos in to_exit:
        _execute_swing_exit(bot, key, pos)

    bot._save_state()


def _execute_swing_exit(bot, key: str, pos: Dict[str, Any]) -> None:
    """Submit market sell for a swing position and remove from state."""
    symbol = pos.get("symbol", "")
    qty = int(pos.get("qty", 0) or 0)

    if qty <= 0:
        logger.warning(f"Swing exit {symbol}: invalid qty={qty}, removing from state")
        bot.swing_positions.pop(key, None)
        return

    logger.info(f"Swing exit: SELL {symbol} x{qty}")

    resp = bot.position_mgr._submit_sell_order(
        symbol=symbol,
        qty=qty,
        order_type="market",
        time_in_force="day",
        extended_hours=False,
        verify_broker_qty=True,
    )

    if resp and resp.get("id"):
        order_id = resp["id"]
        logger.info(f"Swing exit {symbol}: sell submitted, order_id={order_id}")

        # Wait for fill
        fill = bot.position_mgr.get_order_fill(order_id, max_wait=15)
        if fill and int(fill.get("filled_qty", 0)) > 0:
            fill_price = float(fill.get("filled_avg_price", 0) or 0)
            entry_price = float(pos.get("entry_price", 0) or 0)
            pnl_pct = ((fill_price / entry_price) - 1) * 100 if entry_price > 0 else 0.0
            logger.info(
                f"Swing exit {symbol}: FILLED {fill.get('filled_qty')} @ ${fill_price:.2f} "
                f"(entry=${entry_price:.2f}, PnL={pnl_pct:+.2f}%)"
            )
        else:
            logger.warning(f"Swing exit {symbol}: no fill confirmation, order may still be pending")
    else:
        logger.error(f"Swing exit {symbol}: sell order FAILED — position remains in broker")

    # Remove from state regardless (if sell failed, failsafe will catch it)
    bot.swing_positions.pop(key, None)


# ──────────────────────────────────────────────────────────────────
# Entry logic
# ──────────────────────────────────────────────────────────────────

def _evaluate_swing_entries(bot) -> None:
    """Evaluate Portfolio 7 signals and submit buys for triggered ETFs."""
    swing_positions = getattr(bot, "swing_positions", {})

    # Don't add to a symbol we already hold in swing
    held_symbols = {pos["symbol"] for pos in swing_positions.values() if pos.get("symbol")}

    # Kill switch check
    if bot._check_daily_loss_kill_switch():
        logger.critical(f"Swing entries BLOCKED by kill switch — {bot.kill_switch_reason}")
        return

    # Step 1: SPY regime filter
    spy_bars = bot.alpaca.get_daily_bars(["SPY"], days=350)
    spy_daily = spy_bars.get("SPY", [])

    if not spy_daily or len(spy_daily) < 200:
        logger.warning(f"Swing entries: insufficient SPY daily bars ({len(spy_daily)}), skipping")
        return

    ma_period = int(getattr(config, "SWING_REGIME_MA_PERIOD", 200))
    closes = [float(b.get("c", 0)) for b in spy_daily if b.get("c")]
    if len(closes) < ma_period:
        logger.warning(f"Swing entries: only {len(closes)} SPY closes, need {ma_period}, skipping")
        return

    spy_ma200 = sum(closes[-ma_period:]) / ma_period
    spy_last_close = closes[-1]
    above_ma = spy_last_close > spy_ma200

    logger.info(
        f"Swing regime: SPY close=${spy_last_close:.2f}, MA{ma_period}=${spy_ma200:.2f}, "
        f"above_ma={above_ma}"
    )

    if not above_ma:
        logger.info("Swing entries: SPY below 200-day MA — regime filter blocks all entries")
        return

    # Step 2: Fetch daily bars for signal symbols to compute returns and streaks
    signal_symbols = list(getattr(config, "SWING_SYMBOLS", ["TQQQ", "UPRO", "NUGT"]))
    # Only fetch for symbols we don't already hold
    fetch_symbols = [s for s in signal_symbols if s not in held_symbols]

    if not fetch_symbols:
        logger.info("Swing entries: all signal symbols already held — no new entries")
        return

    daily_bars = bot.alpaca.get_daily_bars(fetch_symbols, days=30)
    snapshots = bot.alpaca.get_snapshots(fetch_symbols) or {}

    # Step 3: Evaluate each symbol's signal
    triggered: List[str] = []
    thresh = float(getattr(config, "SWING_THRESHOLD_PCT", -3.0))
    nugt_streak_min = int(getattr(config, "SWING_NUGT_STREAK_MIN", 3))

    for sym in fetch_symbols:
        bars = daily_bars.get(sym, [])
        snap = snapshots.get(sym, {})

        if not bars or len(bars) < 5:
            logger.warning(f"Swing {sym}: insufficient daily bars ({len(bars)})")
            continue

        # Today's return using snapshot (last_price vs prev_close)
        last_price = snap.get("last_price")
        prev_close = snap.get("prev_daily_close") or snap.get("prev_close")

        if not last_price or not prev_close or float(prev_close) <= 0:
            # Fallback: use last two daily bar closes
            if len(bars) >= 2:
                last_price = float(bars[-1].get("c", 0))
                prev_close = float(bars[-2].get("c", 0))

        if not last_price or not prev_close or float(prev_close) <= 0:
            logger.warning(f"Swing {sym}: cannot determine today's return")
            continue

        today_ret_pct = (float(last_price) / float(prev_close) - 1) * 100

        signal_fired = False
        signal_reason = ""

        if sym == "NUGT":
            # NUGT uses consecutive down streak signal
            streak = _compute_down_streak(bars, today_ret_pct)
            signal_fired = streak >= nugt_streak_min and today_ret_pct < 0
            signal_reason = f"streak={streak} (min={nugt_streak_min}), ret={today_ret_pct:+.2f}%"
        else:
            # TQQQ and UPRO use threshold signal
            signal_fired = today_ret_pct <= thresh
            signal_reason = f"ret={today_ret_pct:+.2f}% (thresh={thresh}%)"

        logger.info(f"Swing {sym}: {signal_reason} → {'TRIGGERED' if signal_fired else 'no signal'}")

        if signal_fired:
            triggered.append(sym)

    if not triggered:
        logger.info("Swing entries: no signals triggered today")
        return

    logger.info(f"Swing entries: {len(triggered)} signals triggered: {', '.join(triggered)}")

    # Step 4: Execute buys
    _execute_swing_entries(bot, triggered, snapshots)


def _execute_swing_entries(bot, triggered: List[str], snapshots: Dict[str, dict]) -> None:
    """Submit buy orders for triggered swing signals with equal allocation."""
    account = bot.position_mgr.get_account()
    if not account:
        logger.error("Swing entries: cannot fetch account for sizing")
        return

    equity = float(account.get("equity", 0))
    if equity <= 0:
        logger.error(f"Swing entries: invalid equity=${equity}")
        return

    alloc_per_sym = float(getattr(config, "SWING_ALLOCATION_PCT", 0.25))
    total_alloc = alloc_per_sym * len(triggered)

    # Check buying power
    bp = float(account.get("buying_power", 0) or account.get("cash", 0) or equity)
    budget_per_sym = equity * alloc_per_sym
    total_budget = budget_per_sym * len(triggered)

    # Reduce if buying power is insufficient
    if total_budget > bp * 0.98:
        scale = (bp * 0.98) / total_budget
        budget_per_sym *= scale
        total_budget = budget_per_sym * len(triggered)
        logger.warning(
            f"Swing entries: buying power ${bp:,.2f} < total budget ${equity * total_alloc:,.2f}, "
            f"scaled to ${total_budget:,.2f}"
        )

    logger.info(
        f"Swing entries: {len(triggered)} symbols, "
        f"${budget_per_sym:,.2f} each, total=${total_budget:,.2f} "
        f"({alloc_per_sym:.0%} x {len(triggered)} = {total_alloc:.0%} of equity)"
    )

    # Execution gate: check spread/staleness
    max_spread = float(getattr(config, "ETF_ENTRY_MAX_SPREAD_PCT", 0.005))
    max_stale = float(getattr(config, "ETF_ENTRY_MAX_STALE_SECONDS", 120.0))
    orderable, rejected = filter_execution_ready(
        triggered, snapshots,
        max_spread_pct=max_spread,
        require_quote=True,
        max_stale_seconds=max_stale,
    )

    for sym in triggered:
        if sym not in orderable:
            logger.warning(f"Swing entry {sym}: REJECTED by execution gate — {rejected.get(sym)}")
            continue

        snap = snapshots.get(sym, {})
        ask = snap.get("ask")
        last_price = snap.get("last_price")
        price = float(ask) if ask and float(ask) > 0 else float(last_price) if last_price else 0.0

        if price <= 0:
            logger.error(f"Swing entry {sym}: no valid price")
            continue

        bp_buffer = float(getattr(config, "ETF_ENTRY_BP_BUFFER_PCT", 0.98))
        shares = int(budget_per_sym * bp_buffer / price)

        if shares <= 0:
            logger.warning(f"Swing entry {sym}: shares={shares}, budget=${budget_per_sym:.2f}, price=${price:.2f}")
            continue

        # Marketable limit to bound slippage
        slippage_pct = float(getattr(config, "ETF_ENTRY_MAX_SLIPPAGE_PCT", 0.005))
        limit_price = price * (1.0 + slippage_pct)

        submitted, error_type = bot.position_mgr.submit_buy_order(
            symbol=sym,
            qty=shares,
            order_type="limit",
            limit_price=limit_price,
        )

        if not submitted or not submitted.get("id"):
            logger.error(f"Swing entry {sym}: buy FAILED — {error_type}")
            continue

        # Wait for fill
        fill = bot.position_mgr.get_order_fill(submitted["id"], max_wait=10)
        if not fill or int(fill.get("filled_qty", 0)) <= 0:
            logger.warning(f"Swing entry {sym}: not filled — canceling")
            bot.position_mgr._cancel_order(submitted["id"])
            continue

        filled_qty = int(fill["filled_qty"])
        fill_price = float(fill.get("filled_avg_price", price))

        logger.info(
            f"Swing entry {sym}: FILLED {filled_qty} @ ${fill_price:.2f} "
            f"(budget=${budget_per_sym:,.2f})"
        )

        # Record in swing_positions state
        today_str = datetime.now(_ET).strftime("%Y-%m-%d")
        key = f"{sym}_{today_str}"
        bot.swing_positions[key] = {
            "symbol": sym,
            "qty": filled_qty,
            "entry_price": fill_price,
            "entry_date": today_str,
            "entry_time": datetime.now(_ET).isoformat(),
            "strategy": "SWING_PORTFOLIO_7",
            "order_id": submitted["id"],
        }


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _extract_trading_dates(bars: List[dict]) -> List[str]:
    """Extract sorted list of date strings from daily bars."""
    dates = []
    for b in bars:
        t = b.get("t")
        if t is None:
            continue
        if isinstance(t, str):
            dates.append(t[:10])
        elif isinstance(t, (datetime, date)):
            dates.append(t.strftime("%Y-%m-%d"))
    return sorted(set(dates))


def _compute_down_streak(bars: List[dict], today_ret_pct: float) -> int:
    """Count consecutive down days ending today (inclusive).

    Uses daily bar closes for historical days and today_ret_pct for today.
    """
    if not bars:
        return 0

    streak = 0

    # Today counts as down if today_ret_pct < 0
    if today_ret_pct < 0:
        streak = 1
    else:
        return 0

    # Walk backwards through historical bars (excluding today)
    # bars[-1] is the most recent bar (today or last trading day)
    # We need to check if today's bar is in the list; if the last bar IS today,
    # start from -2; otherwise start from -1.
    today_str = datetime.now(_ET).strftime("%Y-%m-%d")
    last_bar_date = ""
    t = bars[-1].get("t")
    if isinstance(t, str):
        last_bar_date = t[:10]
    elif isinstance(t, (datetime, date)):
        last_bar_date = t.strftime("%Y-%m-%d")

    start_idx = len(bars) - 1
    if last_bar_date == today_str:
        start_idx = len(bars) - 2  # Skip today's bar, already counted

    for i in range(start_idx, -1, -1):
        bar = bars[i]
        if len(bars) < 2:
            break
        # Compute return: close / prev_close - 1
        close = float(bar.get("c", 0))
        if i > 0:
            prev_close = float(bars[i - 1].get("c", 0))
        else:
            break  # No previous bar to compare

        if prev_close <= 0:
            break

        ret = (close / prev_close - 1) * 100
        if ret < 0:
            streak += 1
        else:
            break

    return streak
