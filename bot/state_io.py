"""State persistence + EOD reporting for the Combined Overnight Rebound Bot.

This module owns the read/write side of bot state, plus the end-of-day
diagnostic artifact writers. The orchestrator delegates here so
``integrated_main`` stays focused on the event loop.

Every function takes the ``bot`` (a ``CombinedOvernightReboundBot``
instance) as its first argument and reads/writes ``bot.X`` attributes
in-place. State ownership remains in the orchestrator; this module is a
stateless dispatcher.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from bot import config
from bot.etf_router import parse_router_decision_from_dict
from bot.rate_limiter import get_api_call_count
from bot.universe_builder import (
    save_candidates_audit,
    save_execution_audit,
    save_run_health,
    save_universe_audit,
)


def _serialise_pending_orders(pending: dict) -> dict:
    """JSON-safe representation of intraday_mr_pending_orders (drop IntradayMRCandidate obj)."""
    out = {}
    for sym, p in pending.items():
        cand = p.get("cand")
        out[sym] = {
            "order_id":              p.get("order_id"),
            "qty":                   p.get("qty"),
            "cancel_requested":      bool(p.get("cancel_requested", False)),
            "accounted_filled_qty":  int(p.get("accounted_filled_qty", 0)),
            "is_addon":              bool(p.get("is_addon", False)),
            "original_position":     p.get("original_position"),  # dict or None
            "cand_entry_time":       getattr(cand, "entry_time",    None) if cand else None,
            "cand_exit_time":        getattr(cand, "exit_time",     None) if cand else None,
            "cand_tp_pct":           getattr(cand, "tp_pct",        None) if cand else None,
            "cand_sl_pct":           getattr(cand, "sl_pct",        None) if cand else None,
            "cand_theme":            getattr(cand, "theme",         None) if cand else None,
            "cand_sleeve_name":      getattr(cand, "sleeve_name",   None) if cand else None,
            "cand_signal_price":     getattr(cand, "signal_price",  None) if cand else None,
        }
    return out


def _deserialise_pending_orders(raw: dict) -> dict:
    """Reconstruct pending_orders from JSON.  cand is a lightweight namespace."""
    from types import SimpleNamespace
    out = {}
    for sym, p in raw.items():
        cand = SimpleNamespace(
            entry_time   = p.get("cand_entry_time"),
            exit_time    = p.get("cand_exit_time"),
            tp_pct       = p.get("cand_tp_pct"),
            sl_pct       = p.get("cand_sl_pct"),
            theme        = p.get("cand_theme"),
            sleeve_name  = p.get("cand_sleeve_name"),
            signal_price = p.get("cand_signal_price"),
        )
        out[sym] = {
            "order_id":             p.get("order_id"),
            "qty":                  p.get("qty"),
            "cancel_requested":     bool(p.get("cancel_requested",    False)),
            "accounted_filled_qty": int( p.get("accounted_filled_qty", 0)),
            "is_addon":             bool(p.get("is_addon",             False)),
            "original_position":    p.get("original_position"),
            "cand":                 cand if not p.get("is_addon") else None,
        }
    return out


def _serialise_candidates(candidates: list) -> list:
    """Convert list of IntradayMRCandidate dataclass instances to JSON-safe dicts."""
    out = []
    for c in candidates:
        try:
            out.append({
                "symbol":        c.symbol,
                "theme":         c.theme,
                "sleeve_name":   c.sleeve_name,
                "regime":        c.regime,
                "prior_ret":     c.prior_ret,
                "pm_ret":        c.pm_ret,
                "severity_score":c.severity_score,
                "signal_price":  c.signal_price,
                "entry_time":    c.entry_time,
                "exit_time":     c.exit_time,
                "tp_pct":        c.tp_pct,
                "sl_pct":        c.sl_pct,
            })
        except Exception:
            pass
    return out


def _deserialise_candidates(raw: list) -> list:
    """Reconstruct candidates list from JSON dicts as lightweight namespaces."""
    from types import SimpleNamespace
    out = []
    for d in raw:
        try:
            out.append(SimpleNamespace(
                symbol        = d.get("symbol", ""),
                theme         = d.get("theme", ""),
                sleeve_name   = d.get("sleeve_name", ""),
                regime        = d.get("regime", ""),
                prior_ret     = float(d.get("prior_ret", 0)),
                pm_ret        = float(d.get("pm_ret", 0)),
                severity_score= float(d.get("severity_score", 0)),
                signal_price  = float(d.get("signal_price", 0)),
                entry_time    = d.get("entry_time", ""),
                exit_time     = d.get("exit_time", "15:50"),
                tp_pct        = d.get("tp_pct"),
                sl_pct        = d.get("sl_pct"),
            ))
        except Exception:
            pass
    return out

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")


# ──────────────────────────────────────────────────────────────
# ETF position schema validator
# ──────────────────────────────────────────────────────────────

def validate_loaded_etf_position(raw: Any) -> Optional[Dict[str, Any]]:
    """Sanity-check an ``etf_position`` payload restored from state.

    Returns the dict unchanged when it has the minimum required fields
    (symbol, qty > 0, entry_price > 0). Returns ``None`` for anything
    malformed so the rest of the orchestrator treats the session as
    having no ETF position rather than crashing in
    ``_check_etf_exits`` / ``_execute_etf_exit`` on a missing key.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        logger.warning(f"State: etf_position not a dict ({type(raw).__name__}); discarding")
        return None
    symbol = raw.get("symbol")
    if not isinstance(symbol, str) or not symbol:
        logger.warning(f"State: etf_position missing valid symbol ({raw!r}); discarding")
        return None
    try:
        qty = int(raw.get("qty", 0))
        entry_price = float(raw.get("entry_price", 0))
    except (TypeError, ValueError):
        logger.warning(f"State: etf_position {symbol} qty/entry_price unparseable; discarding")
        return None
    if qty <= 0 or entry_price <= 0:
        logger.warning(
            f"State: etf_position {symbol} qty={qty} entry_price={entry_price} invalid; discarding"
        )
        return None
    return raw


# ──────────────────────────────────────────────────────────────
# Save / load state
# ──────────────────────────────────────────────────────────────

def save_state(bot) -> None:
    """Persist bot flags + positions to disk via ``bot.state_mgr``."""
    try:
        bot.state_mgr.save_positions(bot.position_mgr.positions)

        bot_state = {
            "date": datetime.now(_ET).strftime("%Y-%m-%d"),
            "morning_liquidation_confirmed": getattr(bot, "morning_liquidation_confirmed", False),
            "morning_exits_done": bot.morning_exits_done,
            "open_exit_plan": bot.open_exit_plan,
            "open_exit_submitted": getattr(bot, "open_exit_submitted", False),
            "morning_open_orders_cancelled": bot.morning_open_orders_cancelled,
            "open_market_rescue_done": bot.open_market_rescue_done,
            "end_of_day_reports_done": bot.end_of_day_reports_done,
            "post_exit_failsafe_done": bot.post_exit_failsafe_done,
            "data_collected": bot.data_collected,
            "scoring_done": bot.scoring_done,
            "entries_done": bot.entries_done,
            "sold_today": list(bot.sold_today),
            # Daily-loss kill switch
            "kill_switch_tripped": bot.kill_switch_tripped,
            "kill_switch_reason": bot.kill_switch_reason,
            # ETF Router state
            "router_decision": bot.router_decision.to_dict() if bot.router_decision else None,
            "router_decisions": [d.to_dict() for d in getattr(bot, "router_decisions", []) if d],
            "router_traded_today": bot.router_traded_today,
            "router_branch": bot.router_branch,
            "mr_blocked_today": bot.mr_blocked_today,
            "router_realized_exits": getattr(bot, "router_realized_exits", []),
            "etf_positions": getattr(bot, "etf_positions", {}),  # dict keyed by branch value
            "startup_done": bot.startup_done,
            "tape_initialized": bot.tape_initialized,
            "router_decision_made": bot.router_decision_made,
            # Intraday ETF sleeve state
            "intraday_etf_sleeve_filled": getattr(bot, "intraday_etf_sleeve_filled", False),
            "router_decision_1010_made": getattr(bot, "router_decision_1010_made", False),
            "router_signal_fired_today": getattr(bot, "router_signal_fired_today", False),
            "router_signals_fired": getattr(bot, "router_signals_fired", []),
            # Overnight ETF sleeve state
            "overnight_etf_fired": getattr(bot, "overnight_etf_fired", False),
            "overnight_etf_position": getattr(bot, "overnight_etf_position", None),
            "overnight_etf_decision_made": getattr(bot, "overnight_etf_decision_made", False),
            "overnight_etf_blocked_today": getattr(bot, "overnight_etf_blocked_today", False),
            # Intraday MR sleeve state
            "intraday_mr_universe_built": getattr(bot, "intraday_mr_universe_built", False),
            "intraday_mr_watchlist_built": getattr(bot, "intraday_mr_watchlist_built", False),
            "intraday_mr_build_terminal": getattr(bot, "intraday_mr_build_terminal", False),
            "intraday_mr_decision_artifact_written": getattr(bot, "intraday_mr_decision_artifact_written", False),
            "intraday_mr_router_exit_checked": getattr(bot, "intraday_mr_router_exit_checked", False),
            "intraday_mr_router_action": getattr(bot, "intraday_mr_router_action", None),
            "intraday_mr_positions": getattr(bot, "intraday_mr_positions", {}),
            "intraday_mr_entered_symbols": list(getattr(bot, "intraday_mr_entered_symbols", set())),
            "intraday_mr_pending_orders": _serialise_pending_orders(getattr(bot, "intraday_mr_pending_orders", {})),
            "intraday_mr_universe_list": getattr(bot, "intraday_mr_universe_list", []),
            "intraday_mr_symbol_cache": getattr(bot, "intraday_mr_symbol_cache", {}),
            "intraday_mr_vix_open": getattr(bot, "intraday_mr_vix_open", None),
            "intraday_mr_candidates": _serialise_candidates(getattr(bot, "intraday_mr_candidates", [])),
            "intraday_mr_realloc_done": getattr(bot, "intraday_mr_realloc_done", False),
        }
        bot.state_mgr.save_bot_state(bot_state)
    except Exception as e:
        logger.error(f"Error saving state: {e}")


def load_state(bot) -> None:
    """Restore bot flags + positions from disk (same-day recovery only)."""
    today = datetime.now(_ET).strftime("%Y-%m-%d")
    bot_state = bot.state_mgr.load_bot_state()

    if not bot_state or bot_state.get("date") != today:
        logger.info("No same-day state to restore — fresh start")
        # Reset ETF router state for new day
        bot.etf_router.reset()
        bot.router_decision = None
        bot.router_decisions = []
        bot.router_traded_today = False
        bot.router_branch = None
        bot.mr_blocked_today = False
        bot.router_realized_exits = []
        bot.etf_positions = {}
        bot.startup_done = False
        bot.tape_initialized = False
        bot.router_decision_made = False
        # Intraday ETF sleeve state
        bot.intraday_etf_sleeve_filled = False
        bot.router_decision_1010_made = False
        bot.router_signal_fired_today = False
        bot.router_signals_fired = []
        # Overnight ETF sleeve state
        bot.overnight_etf_fired = False
        bot.overnight_etf_position = None
        bot.overnight_etf_decision_made = False
        bot.overnight_etf_blocked_today = False
        # Intraday MR sleeve state
        bot.morning_liquidation_confirmed   = False
        bot.intraday_mr_universe_built      = False
        bot.intraday_mr_watchlist_built     = False
        bot.intraday_mr_build_terminal      = False
        bot.intraday_mr_decision_artifact_written = False
        bot.intraday_mr_router_exit_checked = False
        bot.intraday_mr_router_action       = None
        bot.intraday_mr_positions           = {}
        bot.intraday_mr_candidates          = []
        bot.intraday_mr_entered_symbols     = set()
        bot.intraday_mr_pending_orders      = {}
        bot.intraday_mr_universe_list       = []
        bot.intraday_mr_symbol_cache        = {}
        bot.intraday_mr_vix_open            = None
        bot.intraday_mr_realloc_done        = False
        logger.info("ETF router state reset for new trading day")
        saved = bot.state_mgr.load_positions()
        if saved:
            bot.position_mgr.load_positions(saved)
            logger.info(f"Loaded {len(saved)} saved positions")
        return

    logger.info("Restoring same-day bot state")
    bot.morning_exits_done = bot_state.get("morning_exits_done", False)
    bot.post_exit_failsafe_done = bot_state.get("post_exit_failsafe_done", False)
    bot.data_collected = bot_state.get("data_collected", False)
    bot.scoring_done = bot_state.get("scoring_done", False)
    bot.entries_done = bot_state.get("entries_done", False)
    bot.sold_today = set(bot_state.get("sold_today", []))
    bot.kill_switch_tripped = bot_state.get("kill_switch_tripped", False)
    bot.kill_switch_reason = bot_state.get("kill_switch_reason", None)
    if bot.kill_switch_tripped:
        logger.critical(
            f"Daily-loss kill switch is ACTIVE from earlier this session — "
            f"{bot.kill_switch_reason}. All entries remain blocked."
        )
    bot.open_exit_plan = bot_state.get("open_exit_plan", [])
    bot.open_exit_submitted = bot_state.get("open_exit_submitted", False)
    bot.morning_open_orders_cancelled = bot_state.get("morning_open_orders_cancelled", False)
    bot.open_market_rescue_done = bot_state.get("open_market_rescue_done", False)
    bot.end_of_day_reports_done = bot_state.get("end_of_day_reports_done", False)

    # ETF Router state
    bot.router_decision = parse_router_decision_from_dict(bot_state.get("router_decision"))
    raw_decisions = bot_state.get("router_decisions") or []
    bot.router_decisions = [parse_router_decision_from_dict(d) for d in raw_decisions if d]
    bot.router_traded_today = bot_state.get("router_traded_today", False)
    bot.router_branch = bot_state.get("router_branch", None)
    bot.mr_blocked_today = bot_state.get("mr_blocked_today", False)
    bot.router_realized_exits = bot_state.get("router_realized_exits", [])
    raw_etf_positions = bot_state.get("etf_positions") or {}
    if isinstance(raw_etf_positions, dict):
        bot.etf_positions = {
            k: validate_loaded_etf_position(v)
            for k, v in raw_etf_positions.items()
            if validate_loaded_etf_position(v) is not None
        }
    else:
        bot.etf_positions = {}
    bot.startup_done = bot_state.get("startup_done", False)
    bot.tape_initialized = bot_state.get("tape_initialized", False)
    bot.router_decision_made = bot_state.get("router_decision_made", False)
    # Intraday ETF sleeve state
    bot.intraday_etf_sleeve_filled = bot_state.get("intraday_etf_sleeve_filled", False)
    bot.router_decision_1010_made = bot_state.get("router_decision_1010_made", False)
    bot.router_signal_fired_today = bot_state.get("router_signal_fired_today", False)
    bot.router_signals_fired = bot_state.get("router_signals_fired", [])
    # Overnight ETF sleeve state
    bot.overnight_etf_fired = bot_state.get("overnight_etf_fired", False)
    bot.overnight_etf_position = bot_state.get("overnight_etf_position", None)
    bot.overnight_etf_decision_made = bot_state.get("overnight_etf_decision_made", False)
    bot.overnight_etf_blocked_today = bot_state.get("overnight_etf_blocked_today", False)
    bot.morning_liquidation_confirmed     = bot_state.get("morning_liquidation_confirmed", False)
    # Intraday MR sleeve state
    bot.intraday_mr_universe_built        = bot_state.get("intraday_mr_universe_built", False)
    bot.intraday_mr_watchlist_built       = bot_state.get("intraday_mr_watchlist_built", False)
    bot.intraday_mr_build_terminal        = bot_state.get("intraday_mr_build_terminal", False)
    bot.intraday_mr_decision_artifact_written = bot_state.get("intraday_mr_decision_artifact_written", False)
    bot.intraday_mr_router_exit_checked   = bot_state.get("intraday_mr_router_exit_checked", False)
    bot.intraday_mr_router_action         = bot_state.get("intraday_mr_router_action", None)
    bot.intraday_mr_positions             = bot_state.get("intraday_mr_positions", {})
    bot.intraday_mr_entered_symbols       = set(bot_state.get("intraday_mr_entered_symbols", []))
    bot.intraday_mr_pending_orders        = _deserialise_pending_orders(bot_state.get("intraday_mr_pending_orders", {}))
    bot.intraday_mr_universe_list         = bot_state.get("intraday_mr_universe_list", [])
    bot.intraday_mr_symbol_cache          = bot_state.get("intraday_mr_symbol_cache", {})
    bot.intraday_mr_vix_open              = bot_state.get("intraday_mr_vix_open", None)
    bot.intraday_mr_candidates            = _deserialise_candidates(bot_state.get("intraday_mr_candidates", []))
    bot.intraday_mr_realloc_done          = bot_state.get("intraday_mr_realloc_done", False)
    # Safety: if watchlist flag is set but candidates list is empty, force rebuild
    if bot.intraday_mr_watchlist_built and not bot.intraday_mr_candidates:
        logger.warning(
            "Intraday MR: watchlist_built=True but candidates empty after restore — "
            "resetting to force Stage 2 rebuild"
        )
        bot.intraday_mr_watchlist_built = False

    saved = bot.state_mgr.load_positions()
    if saved:
        bot.position_mgr.load_positions(saved)
        logger.info(f"Loaded {len(saved)} saved positions")


# ──────────────────────────────────────────────────────────────
# End-of-day reports
# ──────────────────────────────────────────────────────────────

def _mr_candidate_dict(c) -> Dict[str, Any]:
    return {
        "symbol": c.symbol,
        "sleeve": "MR",
        "selection_score": round(c.selection_score, 4),
        "signal_price": round(c.signal_price, 4),
        "day_return": round(c.day_return, 4),
        "volume_ratio": round(c.volume_ratio, 2) if c.volume_ratio_available else None,
        "volume_ratio_available": c.volume_ratio_available,
        "close_position": round(c.close_position, 3),
        "late_drop_1530_1550": round(c.late_drop_1530_1550, 4) if c.late_drop_available else None,
        "late_drop_available": c.late_drop_available,
        "adv_dollars": round(c.adv_dollars, 0),
        "adv_multiplier": float(getattr(c, "adv_multiplier", 1.0)),
        "adv_source": getattr(c, "adv_source", "unknown"),
    }


def save_end_of_day_reports(bot) -> None:
    """Write all daily diagnostic artifacts. Called on EVERY completed market day."""
    try:
        stats = bot._exec_stats
        total_candidates = len(bot.mr_candidates)

        extras: Dict[str, Any] = {
            "api_calls_total": get_api_call_count(),
            "submit_latency_ms": stats.get("submit_latency_ms"),
            "kill_switch": {
                "tripped": bot.kill_switch_tripped,
                "reason": bot.kill_switch_reason,
            },
            "etf_router": bot._build_etf_router_summary(),
        }
        save_run_health(
            diag=bot._universe_diag,
            scored_count=total_candidates,
            selected_count=stats.get("selected", 0),
            orderable_count=stats.get("orderable", 0),
            filled_count=stats.get("entries_filled", 0),
            total_deployed=stats.get("total_deployed", 0.0),
            equity=stats.get("equity", 0.0),
            exec_rejected=stats.get("exec_rejected_reasons"),
            extra=extras,
        )
    except Exception as e:
        logger.error(f"Failed to save health report: {e}")

    # Separate ETF router artifact for easier post-day analysis.
    try:
        bot._save_etf_router_artifact()
    except Exception as e:
        logger.error(f"Failed to save ETF router artifact: {e}")

    try:
        if bot._universe_diag:
            save_universe_audit(bot._universe_diag, bot.universe)
    except Exception as e:
        logger.error(f"Failed to save universe audit: {e}")

    try:
        audit_dicts = {
            "mr_selected": [_mr_candidate_dict(c) for c in bot.mr_candidates[:config.MR_MAX_PRIMARY_POSITIONS]],
            "mr_all_passed": [_mr_candidate_dict(c) for c in bot.mr_candidates],
        }
        if audit_dicts["mr_selected"]:
            save_candidates_audit(audit_dicts)
    except Exception as e:
        logger.error(f"Failed to save candidates audit: {e}")

    try:
        if bot._exec_diag:
            save_execution_audit(bot._exec_diag)
    except Exception as e:
        logger.error(f"Failed to save execution audit: {e}")


def finalize_day(bot, clear_state: bool = True) -> None:
    """End-of-day: write reports, optionally clear bot flags.

    When ``clear_state=True`` (no-entry day) we clear flags so tomorrow
    starts fresh, and only persist positions (which should be empty).
    We deliberately do NOT call ``save_state`` after clearing — that
    would re-write the flags we just cleared.
    """
    logger.info("Finalizing trading day")
    save_end_of_day_reports(bot)
    if clear_state:
        bot.state_mgr.clear_bot_state()
        bot.state_mgr.save_positions(bot.position_mgr.positions)
    else:
        save_state(bot)
