"""
Intraday MR Runtime — Morning Momentum execution layer.

Two-stage pre-market build:
  Stage 1 (~09:00):  Build universe from Massive free grouped daily.
                     Fetch T-1 and T-2 daily bars for all symbols.
                     Cache prior_ret and liquidity per symbol.
                     VIX is NOT fetched here — today's bar doesn't exist yet.

  Stage 2 (09:30:10+):
                     Fetch actual ^VIX open (today's bar, date-validated).
                     Fetch official regular-session opens for the universe.
                     Compute pm_ret = open / prev_close - 1.
                     Compute SPY/QQQ gaps.
                     Run classifier -> bot.intraday_mr_candidates.

Entry execution (09:32–09:47):
  All candidates scheduled for the same minute are submitted CONCURRENTLY.
  Each order tracked via pending_order dict until fill confirmed.
  Only mark a symbol as entered after fill is confirmed.

Exit states per position:
  OPEN         — live, monitoring TP/SL
  EXIT_PENDING — sell order submitted, awaiting fill confirmation
  CLOSED       — fill confirmed, done
  EXIT_FAILED  — sell rejected/cancelled; retry at next loop tick

15:40 hard flatten reconciles against actual broker positions —
  anything still held is force-flattened regardless of local state.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date as dt_date, timedelta
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

from bot import config
from bot.intraday_mr_classifier import (
    IntradayMRCandidate,
    build_intraday_mr_candidates,
    apply_router_exit_rule,
)

logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")

# Exit state constants
_STATE_OPEN          = "OPEN"
_STATE_EXIT_PENDING  = "EXIT_PENDING"
_STATE_CLOSED        = "CLOSED"
_STATE_EXIT_FAILED   = "EXIT_FAILED"

# Symbols always fetched for regime classification
_REGIME_SYMBOLS = ["SPY", "QQQ"]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 (~09:00): build universe + cache T-1/T-2 daily data
# ─────────────────────────────────────────────────────────────────────────────

def build_intraday_mr_universe(bot) -> None:
    """
    Stage 1 — called at ~09:00.
    Builds universe, fetches T-1/T-2 daily bars, and fetches actual VIX.
    Aborts entirely (does NOT mark universe_built) if VIX is unavailable —
    missing VIX is not evidence the day is low-volatility.
    """
    if not getattr(config, "INTRADAY_MR_ENABLED", False):
        return
    if getattr(bot, "intraday_mr_universe_built", False):
        return

    logger.info("Intraday MR Stage 1: building universe and daily bar cache")
    try:
        # ── Build universe (VIX fetch deferred to Stage 2 after market open) ──────
        universe = _build_universe_from_massive(bot)
        if not universe:
            logger.warning("Intraday MR Stage 1: empty universe from Massive — will retry")
            if _stage2_past_deadline():
                _save_intraday_mr_decision_artifact(
                    bot, decision="stage1_universe_empty", reason="Massive watchlist returned zero symbols"
                )
            return

        for sym in _REGIME_SYMBOLS:
            if sym not in universe:
                universe.append(sym)

        # ── Fetch T-1/T-2 daily bars, select by market date ──────────────────
        logger.info(f"Intraday MR: fetching daily bars for {len(universe)} symbols")
        daily_bars = bot.alpaca.get_daily_bars(universe, days=7)
        today_str = datetime.now(_ET).strftime("%Y-%m-%d")
        symbol_cache: Dict[str, dict] = {}
        for symbol, bars in daily_bars.items():
            t1, t2 = _select_t1_t2_bars(bars, today_str)
            if t1 is None or t2 is None:
                continue
            symbol_cache[symbol] = {
                "prev_close":  float(t1.get("c") or t1.get("close") or 0),
                "prev2_close": float(t2.get("c") or t2.get("close") or 0),
                "prev_volume": float(t1.get("v") or t1.get("volume") or 0),
            }

        if not symbol_cache:
            logger.warning("Intraday MR Stage 1: no symbols with valid T-1/T-2 bars — will retry")
            if _stage2_past_deadline():
                _save_intraday_mr_decision_artifact(
                    bot, decision="stage1_daily_data_incomplete", reason="No valid T-1/T-2 bars"
                )
            return

        bot.intraday_mr_symbol_cache = symbol_cache
        bot.intraday_mr_universe_list = universe
        bot.intraday_mr_universe_built = True
        logger.info(
            f"Intraday MR Stage 1 done: {len(universe)} symbols, "
            f"{len(symbol_cache)} with valid T-1/T-2 bars"
        )
        bot._save_state()
    except Exception as e:
        logger.error(f"Intraday MR Stage 1 error: {e}", exc_info=True)
        _save_intraday_mr_decision_artifact(
            bot, decision="stage1_error", reason="classifier_error", extra_meta={"error": str(e)}
        )


def _select_t1_t2_bars(bars: List[dict], today_str: str):
    """
    Return (t1_bar, t2_bar) where t1 = most recent completed session before today,
    t2 = session before t1. Selects by date stamp, not list position, to avoid
    partial today-bar issues.
    """
    completed = [
        b for b in bars
        if (b.get("t") or b.get("date") or "")[:10] < today_str
    ]
    completed.sort(key=lambda b: (b.get("t") or b.get("date") or ""))
    if len(completed) < 2:
        return None, None
    return completed[-1], completed[-2]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 (09:30:05–09:31): fetch opens, finalize regime + candidates
# ─────────────────────────────────────────────────────────────────────────────

_STAGE2_MIN_OPEN_RATIO = 0.50  # at least 50% of universe must have valid opens
_STAGE2_EARLIEST_SECONDS = 10  # do not run before 09:30:10
_STAGE2_DEADLINE_SECONDS_AFTER_OPEN = 90  # final attempt at 09:31:30


def _stage2_past_deadline() -> bool:
    now = datetime.now(_ET)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    deadline = market_open + timedelta(seconds=_STAGE2_DEADLINE_SECONDS_AFTER_OPEN)
    return now >= deadline


def _stage2_failure(bot, reason: str, extra_meta: Optional[Dict] = None) -> None:
    """Log the failure reason and write a final decision artifact if past deadline."""
    bot.intraday_mr_stage2_failure_reason = reason
    if _stage2_past_deadline() and not getattr(bot, "intraday_mr_watchlist_built", False):
        _save_intraday_mr_decision_artifact(
            bot, decision="stage2_incomplete", reason=reason, extra_meta=extra_meta
        )


def build_intraday_mr_finalize(bot) -> None:
    """
    Stage 2 — called after 09:30:10 (guard ensures official opens are populated).

    Completeness requirements before setting watchlist_built=True:
      - now >= 09:30:10
      - VIX available (fetched here, after 09:30:10)
      - SPY and QQQ official opens available
      - >= 50% of candidate universe has official opens
    If any requirement fails, returns without marking complete so the loop retries.
    """
    if not getattr(config, "INTRADAY_MR_ENABLED", False):
        _save_intraday_mr_decision_artifact(bot, decision="disabled", reason="INTRADAY_MR_ENABLED=False")
        return
    if getattr(bot, "intraday_mr_watchlist_built", False):
        return
    if not getattr(bot, "intraday_mr_universe_built", False):
        logger.warning("Intraday MR Stage 2: Stage 1 not complete — skipping")
        _stage2_failure(bot, "stage1_not_complete")
        return

    # ── Early-time guard: ensure official opens are actually populated ────────
    now = datetime.now(_ET)
    open_cutoff = now.replace(hour=9, minute=30, second=_STAGE2_EARLIEST_SECONDS, microsecond=0)
    if now < open_cutoff:
        logger.debug(f"Intraday MR Stage 2: too early ({now.strftime('%H:%M:%S')} < 09:30:{_STAGE2_EARLIEST_SECONDS:02d}) — retry")
        return

    # ── Fetch today's ^VIX open (date-validated; retries until today's bar available) ──
    vix_result = bot.alpaca.get_vix_open()  # returns (float, bar_date) or None
    if vix_result is None:
        # get_vix_open already logged the reason; stale bar = retry, error = abort
        # We cannot distinguish stale from error here without changing the API further,
        # so we simply retry (watchlist_built stays False) and let the 09:30-09:31
        # retry window sort it out. If still None at 09:31+, caller loop stops trying.
        _stage2_failure(bot, "vix_unavailable")
        return
    vix_open, vix_bar_date = vix_result
    bot.intraday_mr_vix_open = vix_open
    logger.info(f"Intraday MR Stage 2: VIX open = {vix_open:.2f} (bar_date={vix_bar_date})")

    logger.info("Intraday MR Stage 2: fetching official opens")
    try:
        feed = getattr(config, "DATA_FEED", "iex")
        universe = getattr(bot, "intraday_mr_universe_list", [])
        symbol_cache = getattr(bot, "intraday_mr_symbol_cache", {})

        snapshots = bot.alpaca.get_snapshots(universe, feed=feed)
        if not snapshots:
            logger.warning("Intraday MR Stage 2: empty snapshots — will retry")
            return

        # ── Validate SPY/QQQ official opens (required for regime gaps) ────────
        spy_snap = snapshots.get("SPY", {})
        qqq_snap = snapshots.get("QQQ", {})
        spy_open = spy_snap.get("open")
        qqq_open = qqq_snap.get("open")
        if not spy_open or not qqq_open:
            logger.warning(
                f"Intraday MR Stage 2: SPY open={spy_open}, QQQ open={qqq_open} — "
                f"official opens not yet available, retrying"
            )
            _stage2_failure(bot, "spy_qqq_open_unavailable")
            return

        # ── Build enriched dict; only use official dailyBar.open (no last_price fallback) ──
        enriched: Dict[str, dict] = {}
        n_with_open = 0
        n_missing_open = 0
        missing_open_syms: List[str] = []

        for symbol, snap in snapshots.items():
            cached = symbol_cache.get(symbol)
            if cached is None and symbol not in _REGIME_SYMBOLS:
                continue

            open_px = snap.get("open")  # official regular-session open only — NO fallback
            if not open_px:
                n_missing_open += 1
                if symbol not in _REGIME_SYMBOLS:
                    missing_open_syms.append(symbol)
                continue

            prev_close  = (cached or {}).get("prev_close")  or snap.get("prev_close")
            prev2_close = (cached or {}).get("prev2_close")
            prev_volume = (cached or {}).get("prev_volume")  or snap.get("prev_volume") or 0

            enriched[symbol] = {
                "open":        float(open_px),
                "prev_close":  prev_close,
                "prev2_close": prev2_close,
                "prev_volume": prev_volume,
            }
            if symbol not in _REGIME_SYMBOLS:
                n_with_open += 1

        # ── Completeness check ────────────────────────────────────────────────
        n_universe = max(len(symbol_cache), 1)
        open_ratio = n_with_open / n_universe
        if open_ratio < _STAGE2_MIN_OPEN_RATIO:
            logger.warning(
                f"Intraday MR Stage 2: only {n_with_open}/{n_universe} ({open_ratio:.0%}) "
                f"symbols have official opens — below {_STAGE2_MIN_OPEN_RATIO:.0%} threshold, "
                f"retrying (missing e.g.: {missing_open_syms[:5]})"
            )
            _stage2_failure(
                bot, "insufficient_open_coverage",
                extra_meta={
                    "open_ratio": round(open_ratio, 3),
                    "symbols_with_open": n_with_open,
                    "symbols_missing_open": n_missing_open,
                }
            )
            return

        enriched["_vix_open"] = float(vix_open)

        candidates = build_intraday_mr_candidates(enriched)

        bot.intraday_mr_candidates = candidates
        bot.intraday_mr_watchlist_built = True
        bot._save_state()

        decision = (
            "no_candidates" if not candidates
            else f"{len(candidates)} candidates"
        )
        _save_intraday_mr_artifact(
            bot, candidates,
            meta={
                "vix_open": vix_open,
                "vix_bar_date": vix_bar_date,
                "universe_size": len(universe),
                "symbols_with_t1t2": len(symbol_cache),
                "symbols_with_open": n_with_open,
                "symbols_missing_open": n_missing_open,
                "open_ratio": round(open_ratio, 3),
                "spy_open": spy_open,
                "qqq_open": qqq_open,
                "stage2_time": now.strftime("%H:%M:%S"),
                "decision": decision,
            }
        )
        logger.info(
            f"Intraday MR Stage 2 done: {len(candidates)} candidates, "
            f"VIX={vix_open:.2f}, opens={n_with_open}/{n_universe} "
            f"(decision={decision})"
        )
    except Exception as e:
        logger.error(f"Intraday MR Stage 2 error: {e}", exc_info=True)
        _save_intraday_mr_decision_artifact(
            bot, decision="stage2_incomplete", reason="classifier_error",
            extra_meta={"error": str(e)}
        )


def _build_universe_from_massive(bot) -> List[str]:
    """Build the intraday MR universe from Massive free grouped daily. Price $2-$100, ADV $1M+."""
    try:
        from bot.mr_free_data_pipeline import build_massive_prevday_watchlist, previous_weekday
        watchlist = build_massive_prevday_watchlist(
            massive_api_key=config.MASSIVE_API_KEY,
            trade_date=previous_weekday(),
            min_prev_close=float(getattr(config, "INTRADAY_MR_UNIVERSE_MIN_PRICE", 2.0)),
            max_prev_close=float(getattr(config, "INTRADAY_MR_UNIVERSE_MAX_PRICE", 100.0)),
            min_prev_dollar_volume=float(getattr(config, "INTRADAY_MR_MIN_ADV_DOLLARS", 1_000_000)),
            apply_prior_ret_filter=False,  # Intraday MR does NOT use prior-day return filter
        )
        symbols = [w.symbol for w in watchlist]
        logger.info(f"Intraday MR: Massive universe: {len(symbols)} symbols")
        return symbols
    except Exception as e:
        logger.warning(f"Intraday MR: Massive universe build failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 09:32–09:47: Concurrent entry execution
# ─────────────────────────────────────────────────────────────────────────────

def execute_intraday_mr_entries(bot, current_time) -> None:
    """
    Submit all candidates whose entry_time <= current_time in a single batch.
    Uses non-blocking submission: orders are recorded in bot.intraday_mr_pending_orders
    and reconciled in reconcile_intraday_mr_pending_fills() each loop tick.
    A symbol is only marked entered after a confirmed fill.
    """
    if not getattr(config, "INTRADAY_MR_ENABLED", False):
        return

    candidates: List[IntradayMRCandidate] = getattr(bot, "intraday_mr_candidates", [])
    if not candidates:
        if not getattr(bot, "_intraday_mr_no_candidates_logged", False):
            logger.info("Intraday MR entry: NO CANDIDATES — nothing to submit")
            bot._intraday_mr_no_candidates_logged = True
        return

    if not hasattr(bot, "intraday_mr_positions"):
        bot.intraday_mr_positions = {}
    if not hasattr(bot, "intraday_mr_entered_symbols"):
        bot.intraday_mr_entered_symbols = set()
    if not hasattr(bot, "intraday_mr_pending_orders"):
        bot.intraday_mr_pending_orders = {}  # symbol -> {order_id, qty, cand}

    try:
        account = bot.position_mgr.get_account()
        equity = float(account.get("equity") or account.get("buying_power") or 0)
    except Exception:
        logger.warning("Intraday MR: could not fetch account for sizing")
        return

    budget_pct = float(getattr(config, "INTRADAY_MR_BUDGET_PCT", 0.50))
    total_budget = equity * budget_pct
    n_cands = len(candidates)
    per_pos = total_budget / n_cands if n_cands > 0 else 0

    # Determine which candidates are due this tick, respecting max entry delay
    _MAX_ENTRY_DELAY_SECONDS = int(getattr(config, "INTRADAY_MR_MAX_ENTRY_DELAY_S", 60))
    due = []
    from datetime import time as dtime
    for cand in candidates:
        if cand.symbol in bot.intraday_mr_entered_symbols:
            continue
        if cand.symbol in bot.intraday_mr_pending_orders:
            continue
        h, m = cand.entry_time.split(":")
        sched = dtime(int(h), int(m))
        if current_time < sched:
            continue
        # Compute seconds past scheduled entry time
        now_dt = datetime.now(_ET)
        sched_dt = now_dt.replace(hour=int(h), minute=int(m), second=0, microsecond=0)
        elapsed = (now_dt - sched_dt).total_seconds()
        if elapsed > _MAX_ENTRY_DELAY_SECONDS:
            logger.warning(
                f"Intraday MR: skipping {cand.symbol} [{cand.sleeve_name}] — "
                f"entry time {cand.entry_time} passed {elapsed:.0f}s ago "
                f"(max delay={_MAX_ENTRY_DELAY_SECONDS}s)"
            )
            bot.intraday_mr_entered_symbols.add(cand.symbol)  # don't retry
            continue

        # Check minimum remaining hold time before exit
        exit_h, exit_m = cand.exit_time.split(":")
        exit_dt = now_dt.replace(hour=int(exit_h), minute=int(exit_m), second=0, microsecond=0)
        remaining_hold_s = (exit_dt - now_dt).total_seconds()
        min_remaining = int(getattr(config, "INTRADAY_MR_MIN_REMAINING_HOLD_S", 120))
        if remaining_hold_s < min_remaining:
            logger.warning(
                f"Intraday MR: skipping {cand.symbol} [{cand.sleeve_name}] — "
                f"only {remaining_hold_s:.0f}s remain before exit {cand.exit_time} "
                f"(min required={min_remaining}s)"
            )
            bot.intraday_mr_entered_symbols.add(cand.symbol)  # don't retry
            continue

        due.append(cand)

    if not due:
        if not getattr(bot, "_intraday_mr_no_due_logged", False):
            logger.info(
                f"Intraday MR entry: no candidates due at {current_time.strftime('%H:%M:%S')} "
                f"({len(candidates)} total candidates, none ready for this minute)"
            )
            bot._intraday_mr_no_due_logged = True
        return

    # Submit all due candidates CONCURRENTLY via a thread pool.
    # Each thread fires one market-buy REST call; results are collected below.
    # This mirrors the backtest assumption that all same-minute entries happen
    # at the open of that minute rather than sequentially over several seconds.
    def _submit_one(cand: IntradayMRCandidate):
        """Returns (cand, order, error_type, qty) or raises."""
        if per_pos <= 0 or cand.signal_price <= 0:
            return (cand, None, "invalid_budget_or_price", 0)
        qty = int(per_pos / cand.signal_price)
        if qty <= 0:
            return (cand, None, "qty_zero", 0)
        order, error_type = bot.position_mgr.submit_buy_order(
            symbol=cand.symbol, qty=qty, order_type="market",
        )
        return (cand, order, error_type, qty)

    with ThreadPoolExecutor(max_workers=len(due)) as pool:
        futures = {pool.submit(_submit_one, cand): cand for cand in due}
        for future in as_completed(futures):
            cand = futures[future]
            try:
                cand_out, order, error_type, qty = future.result()
            except Exception as e:
                logger.error(f"Intraday MR: submission error for {cand.symbol}: {e}", exc_info=True)
                bot.intraday_mr_entered_symbols.add(cand.symbol)
                continue

            if error_type or not order:
                logger.error(f"Intraday MR: buy rejected for {cand.symbol}: {error_type}")
                bot.intraday_mr_entered_symbols.add(cand.symbol)
                continue

            order_id = order.get("id")
            bot.intraday_mr_pending_orders[cand.symbol] = {
                "order_id": order_id, "qty": qty, "cand": cand,
            }
            logger.info(
                f"Intraday MR order submitted: {cand.symbol} [{cand.sleeve_name}] "
                f"x{qty} @ mkt  (order_id={order_id})"
            )

    bot._save_state()


def reconcile_intraday_mr_pending_fills(bot) -> None:
    """
    Poll pending orders for fill confirmation.
    Called each main loop tick after entries have been submitted.
    Moves confirmed fills into intraday_mr_positions (state=OPEN).
    Only marks symbol as entered after confirmed fill.
    """
    if not getattr(config, "INTRADAY_MR_ENABLED", False):
        return

    pending: dict = getattr(bot, "intraday_mr_pending_orders", {})
    if not pending:
        return

    confirmed = []
    still_pending = []

    # Retrieve the router action latch set at 10:00 (if any).
    # Use the same canonical mapping as _get_router_action so the two code
    # paths cannot disagree on what counts as a short signal.
    router_action_latch = getattr(bot, "intraday_mr_router_action", None)
    _ROUTER_SHORT_ACTIONS = ("sqqq_goldilocks",)
    router_is_short = (
        router_action_latch is not None
        and router_action_latch.lower() in _ROUTER_SHORT_ACTIONS
    )

    for symbol, pend in list(pending.items()):
        order_id = pend["order_id"]
        qty      = pend["qty"]
        cand     = pend["cand"]
        try:
            url  = f"{bot.position_mgr.base_url}/v2/orders/{order_id}"
            resp = bot.position_mgr.session.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            status     = data.get("status")
            filled_avg = data.get("filled_avg_price")
            filled_qty = int(float(data.get("filled_qty") or 0))

            # Handle any filled quantity first (full fill, partial-then-terminal, etc.)
            already_accounted = int(pend.get("accounted_filled_qty", 0))
            new_filled = filled_qty - already_accounted

            if new_filled > 0 and filled_avg:
                fill_price = float(filled_avg)
                is_addon = bool(pend.get("is_addon", False))

                if is_addon:
                    # ── Add-on fill: weighted-average from ORIGINAL position baseline ──
                    # Always recompute from original_position + the cumulative fill qty/avg
                    # reported by the broker.  This is correct for any pattern:
                    #   complete fill, multiple partials, or partial-then-cancel.
                    # We do NOT use new_filled here because filled_avg is a CUMULATIVE
                    # average, not the average of just the incremental shares.
                    orig = pend.get("original_position") or {}
                    base_qty   = int(orig.get("qty",          0))
                    base_entry = float(orig.get("entry_price", fill_price))
                    total_qty  = base_qty + filled_qty
                    new_entry  = (
                        (base_qty * base_entry + filled_qty * fill_price) / total_qty
                    ) if total_qty > 0 else fill_price

                    pos = bot.intraday_mr_positions.get(symbol)

                    if pos is None:
                        # The position disappeared (race with exit logic or restart).
                        # An actual broker fill cannot be ignored — create an emergency
                        # tracked position and flatten it immediately.
                        logger.critical(
                            f"Intraday MR add-on fill: {symbol} has no local position "
                            f"— creating emergency entry x{filled_qty} @ ${fill_price:.2f} "
                            f"and flattening"
                        )
                        bot.intraday_mr_positions[symbol] = {
                            "symbol":               symbol,
                            "theme":                orig.get("theme"),
                            "sleeve_name":          orig.get("sleeve_name", ""),
                            "entry_price":          fill_price,
                            "original_entry_price": fill_price,
                            "qty":                  filled_qty,
                            "entry_time":           orig.get("entry_time", ""),
                            "exit_time":            orig.get("exit_time", "15:50"),
                            "tp_pct":               orig.get("tp_pct"),
                            "sl_pct":               orig.get("sl_pct"),
                            "order_id":             order_id,
                            "exit_state":           _STATE_OPEN,
                            "exit_order_id":        None,
                            "exit_reason":          None,
                        }
                        _submit_intraday_mr_exit(
                            bot, symbol, reason="addon_emergency_flatten"
                        )
                    else:
                        exit_state = pos.get("exit_state")
                        pos["qty"]          = total_qty
                        pos["entry_price"]  = new_entry
                        # Preserve the original cost basis for audit / exit-rule comparison
                        if "original_entry_price" not in pos:
                            pos["original_entry_price"] = base_entry
                        logger.info(
                            f"Intraday MR add-on fill: {symbol} addon={filled_qty} @ ${fill_price:.2f}  "
                            f"total={total_qty} wavg_entry=${new_entry:.2f} exit_state={exit_state}"
                        )
                        if exit_state in (_STATE_EXIT_PENDING, _STATE_CLOSED):
                            # Add-on filled after exit process started.
                            # The already-submitted sell may cover fewer shares than
                            # the broker now holds.  Query broker and top-up the sell.
                            logger.warning(
                                f"Intraday MR add-on fill: {symbol} filled into "
                                f"{exit_state} state — querying broker for residual shares"
                            )
                            try:
                                broker_pos = bot.position_mgr.get_broker_positions() or []
                                broker_qty = next(
                                    (int(float(p.get("qty") or 0))
                                     for p in broker_pos if p.get("symbol") == symbol),
                                    0,
                                )
                                if broker_qty > 0:
                                    logger.warning(
                                        f"Intraday MR add-on residual: {symbol} broker={broker_qty} "
                                        f"— submitting exit for residual shares"
                                    )
                                    # Mark OPEN momentarily so _submit_intraday_mr_exit accepts it
                                    pos["exit_state"] = _STATE_OPEN
                                    pos["qty"]        = broker_qty
                                    _submit_intraday_mr_exit(
                                        bot, symbol, reason="addon_residual_after_exit"
                                    )
                            except Exception as ex:
                                logger.error(
                                    f"Intraday MR add-on residual: broker query failed for {symbol}: {ex}"
                                )
                    # Update running cumulative tally for partially-filled add-on polls
                    pend["accounted_filled_qty"] = filled_qty
                else:
                    theme = getattr(cand, "theme", None)

                    # Register / update position so the 15:40 hard flatten can always see it
                    bot.intraday_mr_positions[symbol] = {
                        "symbol":      symbol,
                        "theme":       theme,
                        "sleeve_name": getattr(cand, "sleeve_name", ""),
                        "entry_price": fill_price,
                        "qty":         filled_qty,
                        "entry_time":  getattr(cand, "entry_time", ""),
                        "exit_time":   getattr(cand, "exit_time", "15:50"),
                        "tp_pct":      getattr(cand, "tp_pct", None),
                        "sl_pct":      getattr(cand, "sl_pct", None),
                        "order_id":    order_id,
                        "exit_state":  _STATE_OPEN,
                        "exit_order_id": None,
                        "exit_reason": None,
                    }
                    logger.info(
                        f"Intraday MR fill: {symbol} [{getattr(cand,'sleeve_name','')}/ "
                        f"Theme-{theme}] x{filled_qty} @ ${fill_price:.2f}  "
                        f"TP={getattr(cand,'tp_pct',None)}  SL={getattr(cand,'sl_pct',None)}  "
                        f"EXIT={getattr(cand,'exit_time','')}  status={status}"
                    )
                    # If router already latched SHORT and this is a non-A fill, exit immediately
                    if router_is_short and theme != "A":
                        logger.warning(
                            f"Intraday MR late fill after router-short: "
                            f"{symbol} [Theme-{theme}] — submitting immediate exit"
                        )
                        _submit_intraday_mr_exit(bot, symbol, reason="late_fill_after_router_short")

            # Now decide whether to remove from pending
            if status == "filled":
                bot.intraday_mr_entered_symbols.add(symbol)
                confirmed.append(symbol)
            elif status in ("canceled", "expired", "rejected"):
                if filled_qty > 0:
                    logger.warning(
                        f"Intraday MR order {status} with partial fill: {symbol} "
                        f"x{filled_qty} shares registered as position"
                    )
                else:
                    logger.warning(
                        f"Intraday MR order {status}: {symbol} (order_id={order_id}) — no fill"
                    )
                bot.intraday_mr_entered_symbols.add(symbol)
                confirmed.append(symbol)
            elif status == "partially_filled":
                # Still active; track how many shares we've already accounted for
                pend["accounted_filled_qty"] = filled_qty
                still_pending.append(symbol)
            else:
                still_pending.append(symbol)

        except Exception as e:
            logger.warning(f"Intraday MR fill poll error for {symbol}: {e}")
            still_pending.append(symbol)

    for sym in confirmed:
        pending.pop(sym, None)

    if confirmed:
        bot._save_state()


# ─────────────────────────────────────────────────────────────────────────────
# 10:00: Router exit rule
# ─────────────────────────────────────────────────────────────────────────────

def apply_router_exit_at_1000(bot) -> None:
    """
    At 10:00: if router is SHORT, exit non-Theme-A.
    Also cancels pending buy orders for non-A symbols so they don't fill
    after the check and stay open. Sets bot.intraday_mr_router_action so
    that any late fills can be immediately exited by reconcile_intraday_mr_pending_fills.
    """
    if not getattr(config, "INTRADAY_MR_ENABLED", False):
        return
    if getattr(bot, "intraday_mr_router_exit_checked", False):
        return

    bot.intraday_mr_router_exit_checked = True
    router_action = _get_router_action(bot)
    bot.intraday_mr_router_action = router_action  # persist for late-fill reconciliation

    # Reconcile pending fills first so confirmed positions are included
    reconcile_intraday_mr_pending_fills(bot)

    # Cancel pending non-A buy orders
    pending: dict = getattr(bot, "intraday_mr_pending_orders", {})
    _ROUTER_SHORT_ACTIONS = ("sqqq_goldilocks",)
    is_short = router_action.lower() in _ROUTER_SHORT_ACTIONS

    if is_short:
        for symbol, pend in list(pending.items()):
            cand = pend.get("cand")
            theme = getattr(cand, "theme", None)
            if theme == "A":
                continue
            if pend.get("cancel_requested"):
                continue  # already requested; keep polling in reconcile
            order_id = pend.get("order_id")
            logger.info(
                f"Intraday MR router cancel pending buy: {symbol} "
                f"[Theme-{theme}] order_id={order_id}"
            )
            try:
                bot.position_mgr._cancel_order(order_id)
            except Exception as e:
                logger.warning(f"Intraday MR: cancel order failed for {symbol}: {e}")
            # Mark cancel requested but KEEP in pending so fill reconciliation
            # can detect a race-condition fill and exit it immediately.
            pend["cancel_requested"] = True

    # Exit confirmed non-A positions
    positions = list(getattr(bot, "intraday_mr_positions", {}).values())
    to_exit = apply_router_exit_rule(positions, router_action)
    for symbol in to_exit:
        logger.info(f"Intraday MR router exit (10:00): {symbol} (router={router_action})")
        _submit_intraday_mr_exit(bot, symbol, reason="router_short_10am")

    bot._save_state()


def _get_router_action(bot) -> str:
    """Extract router action string from bot state.

    Maps live RouterBranch values to the action labels used by
    apply_router_exit_rule / ROUTER_SHORT_ACTIONS:
      SHORT signals  → 'sqqq_goldilocks'                   (trigger MR exit)
      LONG  signals  → branch value as-is                  (hold MR positions)
      NO_TRADE       → 'none'

    Live RouterBranch enum:
      MOMENTUM_SLEEVE_ANTI  — SQQQ short/hedge → SHORT
      VXX_SPIKE_RECOVERY    — long TQQQ        → LONG  (was incorrectly SHORT)
      VXX_COLLAPSE          — long TQQQ        → LONG
      MOMENTUM_SLEEVE       — long TQQQ        → LONG
      ROUTER_LONG           — long TQQQ        → LONG
      SVIX_LONG             — long SVIX        → LONG
      NO_TRADE              — no entry         → none
    """
    decision = getattr(bot, "router_decision", None)
    if decision is None:
        return "none"
    branch = getattr(decision, "branch", None)
    if branch is None:
        return "none"
    branch_str = branch.value if hasattr(branch, "value") else str(branch)

    # Only MOMENTUM_SLEEVE_ANTI is a short/hedge signal in the live bot.
    # All VXX-based strategies (VXX_SPIKE_RECOVERY, VXX_COLLAPSE) are long TQQQ.
    if branch_str in ("MOMENTUM_SLEEVE_ANTI",):
        return "sqqq_goldilocks"

    if branch_str == "NO_TRADE":
        return "none"

    # All remaining branches (MOMENTUM_SLEEVE, VXX_SPIKE_RECOVERY, VXX_COLLAPSE,
    # ROUTER_LONG, SVIX_LONG) are long signals — do not exit MR positions.
    return branch_str.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 10:10: Router-budget reallocation to open MR positions (momentum-weighted)
# ─────────────────────────────────────────────────────────────────────────────

def reallocate_router_budget_to_mr(bot) -> None:
    """
    Called each tick in the 10:10–10:11 window when the router did NOT fire.

    Redeploys the unused router bucket (INTRADAY_REALLOC_MAX_PCT of equity,
    capped by available buying power) into open MR positions as add-on buys.

    Weighting: equal weight among positions whose return at 10:10 >= 0.
    If no position is in profit, falls back to equal weight across all open.
    This is the conservative default; change INTRADAY_REALLOC_MODE in config
    once the allocation backtest confirms a different formula.

    Add-on fills reconcile via the normal pending path but use is_addon=True
    so the reconciler increments qty + recomputes weighted-average entry
    instead of overwriting the position dictionary.

    Latch (intraday_mr_realloc_done) is set only after:
      - Intentional skip (no eligible positions), OR
      - At least one add-on order was successfully submitted, OR
      - The cutoff time (INTRADAY_REALLOC_CUTOFF) has passed.
    Temporary failures (API errors) leave the latch False so the next tick retries.
    """
    if not getattr(config, "INTRADAY_MR_ENABLED", False):
        return
    if not getattr(config, "INTRADAY_REALLOC_ENABLED", True):
        return
    if getattr(bot, "intraday_mr_realloc_done", False):
        return

    from datetime import time as dtime
    now = datetime.now(_ET)
    cutoff_str = getattr(config, "INTRADAY_REALLOC_CUTOFF", "10:11")
    ch, cm = cutoff_str.split(":")
    cutoff_time = dtime(int(ch), int(cm))
    if now.time() > cutoff_time:
        logger.warning(
            f"Intraday MR realloc: past cutoff {cutoff_str} — abandoning reallocation attempt"
        )
        bot.intraday_mr_realloc_done = True
        bot._save_state()
        return

    positions   = getattr(bot, "intraday_mr_positions", {})
    pending_now = getattr(bot, "intraday_mr_pending_orders", {})

    # Eligible: OPEN only (not EXIT_FAILED — those must follow the exit retry path,
    # not receive additional capital) AND no pending buy already in flight.
    open_pos = {
        sym: pos for sym, pos in positions.items()
        if pos.get("exit_state") == _STATE_OPEN
        and sym not in pending_now
    }

    if not open_pos:
        logger.info("Intraday MR realloc: no eligible open positions — router budget stays idle")
        bot.intraday_mr_realloc_done = True
        bot._save_state()
        return

    # ── Account / buying-power fetch ────────────────────────────────────────
    # Temporary failure → leave latch False so next tick retries.
    try:
        account      = bot.position_mgr.get_account()
        equity       = float(account.get("equity")       or 0)
        buying_power = float(account.get("buying_power") or 0)
    except Exception as e:
        logger.warning(f"Intraday MR realloc: account fetch failed, will retry: {e}")
        return

    bp_buffer       = float(getattr(config, "INTRADAY_REALLOC_BP_BUFFER",  0.98))
    max_pct         = float(getattr(config, "INTRADAY_REALLOC_MAX_PCT",    0.50))
    nominal_budget  = equity * max_pct
    available_bp    = buying_power * bp_buffer
    router_budget   = min(nominal_budget, available_bp)

    if router_budget <= 0:
        logger.warning("Intraday MR realloc: zero available buying power — skipping")
        bot.intraday_mr_realloc_done = True
        bot._save_state()
        return

    logger.info(
        f"Intraday MR realloc: nominal=${nominal_budget:.0f} "
        f"avail_bp=${available_bp:.0f} budget=${router_budget:.0f} "
        f"positions={len(open_pos)}"
    )

    # ── Live price fetch ───────────────────────────────────────────────
    # Temporary failure → leave latch False so next tick retries.
    feed = getattr(config, "DATA_FEED", "iex")
    try:
        snaps = bot.alpaca.get_snapshots(list(open_pos.keys()), feed=feed) or {}
    except Exception as e:
        logger.warning(f"Intraday MR realloc: snapshot fetch failed, will retry: {e}")
        return

    live = []
    for sym, pos in open_pos.items():
        snap = snaps.get(sym, {})
        px = snap.get("last_price") or snap.get("last") or snap.get("open")
        if not px:
            logger.warning(f"Intraday MR realloc: no price for {sym} — skipping")
            continue
        px    = float(px)
        entry = float(pos.get("entry_price", px))
        ret   = (px / entry) - 1.0 if entry > 0 else 0.0
        live.append((sym, px, ret))

    if not live:
        logger.warning("Intraday MR realloc: no live prices — skipping")
        bot.intraday_mr_realloc_done = True
        bot._save_state()
        return

    # ── Weighting formula gated on INTRADAY_REALLOC_MODE ──────────────────────
    mode = getattr(config, "INTRADAY_REALLOC_MODE", "equal_positive")
    if mode != "equal_positive":
        logger.error(
            f"Intraday MR realloc: unrecognised INTRADAY_REALLOC_MODE='{mode}'. "
            f"Only 'equal_positive' is implemented. Aborting reallocation — "
            f"update the backtest confirmation before adding new modes."
        )
        bot.intraday_mr_realloc_done = True
        bot._save_state()
        return

    winners  = [(s, p, r) for s, p, r in live if r >= 0]
    eligible = winners if winners else live
    n = len(eligible)
    weights = {sym: 1.0 / n for sym, _, _ in eligible}

    logger.info(
        f"Intraday MR realloc: {n} eligible ({len(winners)} winners / {len(live)} total)"
    )
    for sym, px, ret in live:
        w = weights.get(sym, 0.0)
        logger.info(f"  {sym}: ret={ret:.2%} px=${px:.2f} weight={w:.3f}")

    # ── Concurrent add-on submissions ───────────────────────────────────
    px_map = {sym: px for sym, px, _ in live}

    def _submit_addon(sym):
        px  = px_map[sym]
        w   = weights[sym]
        qty = int(router_budget * w / px)
        if qty <= 0:
            return (sym, None, "qty_zero", 0)
        order, error_type = bot.position_mgr.submit_buy_order(
            symbol=sym, qty=qty, order_type="market",
        )
        return (sym, order, error_type, qty)

    submitted = []   # symbols where an add-on order landed
    syms_to_submit = list(weights.keys())
    with ThreadPoolExecutor(max_workers=len(syms_to_submit)) as pool:
        futures = {pool.submit(_submit_addon, sym): sym for sym in syms_to_submit}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                sym_out, order, error_type, qty = future.result()
            except Exception as e:
                logger.error(f"Intraday MR realloc: submission error for {sym}: {e}", exc_info=True)
                continue
            if error_type or not order:
                if error_type != "qty_zero":
                    logger.error(f"Intraday MR realloc: buy rejected for {sym}: {error_type}")
                continue
            order_id = order.get("id")
            # Register as add-on pending — reconciler will handle fill
            bot.intraday_mr_pending_orders[sym] = {
                "order_id":             order_id,
                "qty":                  qty,
                "cand":                 None,          # not used for add-ons
                "is_addon":             True,
                "original_position":    dict(positions[sym]),
                "accounted_filled_qty": 0,
                "cancel_requested":     False,
            }
            submitted.append(sym)
            logger.info(
                f"Intraday MR realloc add-on: {sym} x{qty} @ mkt "
                f"weight={weights[sym]:.3f} budget=${router_budget * weights[sym]:.0f} "
                f"order_id={order_id}"
            )

    # Latch only if at least one order landed (or no eligible symbols)
    if submitted:
        bot.intraday_mr_realloc_done = True
        logger.info(f"Intraday MR realloc: {len(submitted)} add-on orders submitted — latch set")
    else:
        logger.warning(
            "Intraday MR realloc: no orders submitted — leaving latch False for next tick retry"
        )

    bot._save_state()


# ─────────────────────────────────────────────────────────────────────────────
# Throughout day: TP/SL + timed exits, pending exit reconciliation
# ─────────────────────────────────────────────────────────────────────────────

def check_intraday_mr_exits(bot, current_time) -> None:
    """
    Called each main loop tick.
    1. Reconcile any EXIT_PENDING orders (check if they filled or failed).
    2. Check open positions for timed exit or TP/SL breach.
    """
    if not getattr(config, "INTRADAY_MR_ENABLED", False):
        return

    positions = getattr(bot, "intraday_mr_positions", {})
    if not positions:
        return

    _reconcile_exit_orders(bot)

    feed = getattr(config, "DATA_FEED", "iex")
    open_symbols = [
        sym for sym, p in positions.items()
        if p.get("exit_state") in (_STATE_OPEN, _STATE_EXIT_FAILED)
    ]
    if not open_symbols:
        return

    try:
        snaps = bot.alpaca.get_snapshots(open_symbols, feed=feed)
    except Exception as e:
        logger.warning(f"Intraday MR exit check: snapshot error: {e}")
        return

    from datetime import time as dtime
    for symbol in open_symbols:
        pos = positions.get(symbol)
        if not pos:
            continue
        if pos.get("exit_state") not in (_STATE_OPEN, _STATE_EXIT_FAILED):
            continue

        # Timed exit
        h, m = pos["exit_time"].split(":")
        if current_time >= dtime(int(h), int(m)):
            logger.info(f"Intraday MR timed exit: {symbol} (exit_time={pos['exit_time']})")
            _submit_intraday_mr_exit(bot, symbol, reason="timed_exit")
            continue

        # TP / SL
        snap = snaps.get(symbol, {})
        last_price = snap.get("last_price") or snap.get("last") or snap.get("open")
        if not last_price:
            continue

        ret = float(last_price) / float(pos["entry_price"]) - 1.0
        tp_pct = pos.get("tp_pct")
        sl_pct = pos.get("sl_pct")

        if tp_pct and ret >= tp_pct:
            logger.info(f"Intraday MR TP: {symbol} ret={ret:.2%} >= tp={tp_pct:.2%}")
            _submit_intraday_mr_exit(bot, symbol, reason="tp_hit")
        elif sl_pct and ret <= -sl_pct:
            logger.info(f"Intraday MR SL: {symbol} ret={ret:.2%} <= -sl={sl_pct:.2%}")
            _submit_intraday_mr_exit(bot, symbol, reason="sl_hit")

    bot._save_state()


def _submit_intraday_mr_exit(bot, symbol: str, reason: str = "exit") -> None:
    """
    Submit a market sell for an intraday MR position.
    Transitions state: OPEN/EXIT_FAILED -> EXIT_PENDING.
    Does not change state if already EXIT_PENDING or CLOSED.
    """
    positions = getattr(bot, "intraday_mr_positions", {})
    pos = positions.get(symbol)
    if not pos:
        return

    if pos.get("exit_state") in (_STATE_EXIT_PENDING, _STATE_CLOSED):
        logger.debug(f"Intraday MR exit: {symbol} already {pos['exit_state']} — skipping")
        return

    qty = int(pos.get("qty", 0))
    if qty <= 0:
        pos["exit_state"] = _STATE_CLOSED
        return

    try:
        ok = bot.position_mgr._submit_sell_order(symbol, qty, order_type="market")
        if ok:
            # _submit_sell_order returns Optional[dict]; dict has "id" key.
            exit_order_id = ok.get("id") if isinstance(ok, dict) else None
            pos["exit_state"]    = _STATE_EXIT_PENDING
            pos["exit_order_id"] = exit_order_id
            pos["exit_reason"]   = reason
            logger.info(
                f"Intraday MR exit submitted: {symbol} x{qty} reason={reason} "
                f"order_id={exit_order_id}"
            )
        else:
            pos["exit_state"] = _STATE_EXIT_FAILED
            logger.error(f"Intraday MR exit FAILED (rejected): {symbol} x{qty}")
    except Exception as e:
        pos["exit_state"] = _STATE_EXIT_FAILED
        logger.error(f"Intraday MR exit error for {symbol}: {e}", exc_info=True)


def _reconcile_exit_orders(bot) -> None:
    """
    Poll EXIT_PENDING orders; move to CLOSED on fill or EXIT_FAILED on rejection.
    Called at the start of check_intraday_mr_exits each tick.
    """
    positions = getattr(bot, "intraday_mr_positions", {})
    pending_exits = [
        (sym, pos) for sym, pos in positions.items()
        if pos.get("exit_state") == _STATE_EXIT_PENDING
    ]
    if not pending_exits:
        return

    for symbol, pos in pending_exits:
        order_id = pos.get("exit_order_id")

        # ── ID-less EXIT_PENDING: reconcile via broker position count ──────────────
        if not order_id:
            broker_pos = bot.position_mgr.get_broker_position(symbol)
            if broker_pos is bot.position_mgr.BROKER_NOT_FOUND:
                # Broker confirms no position — treat as filled/closed
                pos["exit_state"] = _STATE_CLOSED
                logger.info(
                    f"Intraday MR exit (ID-less): {symbol} no longer at broker — marking CLOSED"
                )
            elif broker_pos is None:
                logger.warning(
                    f"Intraday MR exit (ID-less): {symbol} broker API error — will retry"
                )
            else:
                # Position still held; re-submit exit
                logger.warning(
                    f"Intraday MR exit (ID-less): {symbol} still at broker — re-submitting exit"
                )
                _submit_intraday_mr_exit(bot, symbol, reason=pos.get("exit_reason", "retry"))
            continue

        # ── Normal EXIT_PENDING with order ID ────────────────────────────────────
        try:
            url  = f"{bot.position_mgr.base_url}/v2/orders/{order_id}"
            resp = bot.position_mgr.session.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")

            if status == "filled":
                pos["exit_state"] = _STATE_CLOSED
                fill_px = data.get("filled_avg_price")
                logger.info(f"Intraday MR exit filled: {symbol} @ ${fill_px}")
            elif status in ("canceled", "expired", "rejected"):
                pos["exit_state"] = _STATE_EXIT_FAILED
                logger.warning(
                    f"Intraday MR exit order {status}: {symbol} — "
                    f"will retry next tick"
                )
        except Exception as e:
            logger.warning(f"Intraday MR exit poll error for {symbol}: {e}")


def flatten_all_intraday_mr_positions(bot) -> None:
    """
    15:40 failsafe: reconcile against actual broker positions.
    Any intraday MR symbol still held at the broker is force-flattened,
    regardless of local exit_state. This catches EXIT_FAILED, EXIT_PENDING
    orders that were silently dropped, and any state tracking bugs.
    
    NOTE: Reconciles pending exits first, skips symbols already EXIT_PENDING,
    and only resubmits for OPEN or EXIT_FAILED positions.
    """
    if not getattr(config, "INTRADAY_MR_ENABLED", False):
        return

    positions = getattr(bot, "intraday_mr_positions", {})
    if not positions:
        return

    # Step 1: Reconcile any pending exits to avoid duplicate sells
    _reconcile_exit_orders(bot)

    # Step 2: Fetch actual broker positions for ground truth
    try:
        broker_positions = bot.position_mgr.get_broker_positions()  # returns list of dicts or None
        broker_held = {p.get("symbol") for p in (broker_positions or []) if p.get("symbol")}
    except Exception as e:
        logger.error(f"Intraday MR failsafe: could not fetch broker positions: {e}")
        broker_held = set()

    tracked_symbols = set(positions.keys())
    to_flatten = set()

    # Step 3: Find positions that need flattening
    for symbol, pos in positions.items():
        state = pos.get("exit_state")
        # Include: OPEN, EXIT_FAILED, and broker-held positions
        # Exclude: EXIT_PENDING (already have live order), CLOSED
        if state == _STATE_EXIT_PENDING:
            # Already have a live exit order - don't duplicate
            continue
        if state != _STATE_CLOSED:
            to_flatten.add(symbol)

    # Any broker-held position we track (catches missed EXIT_PENDING fills)
    to_flatten |= (tracked_symbols & broker_held)

    # Step 4: Submit exits only for positions not already EXIT_PENDING
    for symbol in to_flatten:
        pos = positions.get(symbol)
        if not pos:
            continue
        
        state = pos.get("exit_state")
        if state == _STATE_EXIT_PENDING:
            # Double-check: skip if somehow now EXIT_PENDING after reconcile
            continue
        
        logger.warning(f"Intraday MR hard flatten: {symbol} (state={state})")
        
        # Reset EXIT_FAILED to OPEN so _submit_intraday_mr_exit will run
        # Leave EXIT_PENDING alone (shouldn't reach here due to check above)
        if state == _STATE_EXIT_FAILED:
            pos["exit_state"] = _STATE_OPEN
        
        _submit_intraday_mr_exit(bot, symbol, reason="hard_flatten_failsafe")

    bot._save_state()


# ─────────────────────────────────────────────────────────────────────────────
# Artifact logger
# ─────────────────────────────────────────────────────────────────────────────

def _save_intraday_mr_artifact(bot, candidates: list, meta: Optional[Dict] = None) -> None:
    """Save daily intraday MR candidate log + data quality metadata for forensic review."""
    try:
        log_dir = getattr(config, "LOG_DIR", "logs")
        today   = datetime.now(_ET).strftime("%Y-%m-%d")
        path    = os.path.join(log_dir, f"intraday_mr_{today}.json")
        data = {
            "date": today,
            "meta": meta or {},
            "candidate_count": len(candidates),
            "candidates": [
                {
                    "symbol":        c.symbol,
                    "theme":         c.theme,
                    "sleeve":        c.sleeve_name,
                    "regime":        c.regime,
                    "prior_ret":     round(c.prior_ret, 4) if c.prior_ret is not None else None,
                    "pm_ret":        round(c.pm_ret, 4),
                    "severity_score":round(c.severity_score, 4),
                    "signal_price":  round(c.signal_price, 4),
                    "entry_time":    c.entry_time,
                    "exit_time":     c.exit_time,
                    "tp_pct":        c.tp_pct,
                    "sl_pct":        c.sl_pct,
                }
                for c in candidates
            ],
        }
        os.makedirs(log_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Intraday MR artifact saved: {path}")
        bot.intraday_mr_decision_artifact_written = True
        bot._save_state()
    except Exception as e:
        logger.warning(f"Intraday MR artifact save failed: {e}")


def _save_intraday_mr_decision_artifact(bot, decision: str, reason: str,
                                        extra_meta: Optional[Dict] = None) -> None:
    """Write a final decision artifact for non-success paths (disabled, incomplete, etc.).

    Guarded by intraday_mr_decision_artifact_written so it is only written once.
    Also marks the morning build as terminal so the orchestrator stops retrying.
    """
    if getattr(bot, "intraday_mr_decision_artifact_written", False):
        return
    try:
        log_dir = getattr(config, "LOG_DIR", "logs")
        today = datetime.now(_ET).strftime("%Y-%m-%d")
        path = os.path.join(log_dir, f"intraday_mr_{today}.json")
        meta = {
            "decision": decision,
            "reason": reason,
            "stage1_complete": getattr(bot, "intraday_mr_universe_built", False),
            "stage2_complete": getattr(bot, "intraday_mr_watchlist_built", False),
            "decision_time": datetime.now(_ET).strftime("%H:%M:%S"),
        }
        if extra_meta:
            meta.update(extra_meta)
        data = {
            "date": today,
            "meta": meta,
            "candidate_count": 0,
            "candidates": [],
        }
        os.makedirs(log_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Intraday MR decision artifact saved: {path} ({decision}: {reason})")
        bot.intraday_mr_decision_artifact_written = True
        bot.intraday_mr_build_terminal = True
        bot._save_state()
    except Exception as e:
        logger.warning(f"Intraday MR decision artifact save failed: {e}")
