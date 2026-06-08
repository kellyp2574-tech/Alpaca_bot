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
            "router_traded_today": bot.router_traded_today,
            "router_branch": bot.router_branch,
            "mr_blocked_today": bot.mr_blocked_today,
            "etf_position": bot.etf_position,
            "startup_done": bot.startup_done,
            "tape_initialized": bot.tape_initialized,
            "router_decision_made": bot.router_decision_made,
            # Intraday ETF sleeve state
            "intraday_etf_sleeve_filled": getattr(bot, "intraday_etf_sleeve_filled", False),
            "router_decision_1010_made": getattr(bot, "router_decision_1010_made", False),
            # Overnight ETF sleeve state
            "overnight_etf_fired": getattr(bot, "overnight_etf_fired", False),
            "overnight_etf_position": getattr(bot, "overnight_etf_position", None),
            "overnight_etf_decision_made": getattr(bot, "overnight_etf_decision_made", False),
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
        bot.router_traded_today = False
        bot.router_branch = None
        bot.mr_blocked_today = False
        bot.etf_position = None
        bot.startup_done = False
        bot.tape_initialized = False
        bot.router_decision_made = False
        # Intraday ETF sleeve state
        bot.intraday_etf_sleeve_filled = False
        bot.router_decision_1010_made = False
        # Overnight ETF sleeve state
        bot.overnight_etf_fired = False
        bot.overnight_etf_position = None
        bot.overnight_etf_decision_made = False
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
    bot.router_traded_today = bot_state.get("router_traded_today", False)
    bot.router_branch = bot_state.get("router_branch", None)
    bot.mr_blocked_today = bot_state.get("mr_blocked_today", False)
    raw_etf = bot_state.get("etf_position")
    bot.etf_position = validate_loaded_etf_position(raw_etf)
    bot.startup_done = bot_state.get("startup_done", False)
    bot.tape_initialized = bot_state.get("tape_initialized", False)
    bot.router_decision_made = bot_state.get("router_decision_made", False)
    # Intraday ETF sleeve state
    bot.intraday_etf_sleeve_filled = bot_state.get("intraday_etf_sleeve_filled", False)
    bot.router_decision_1010_made = bot_state.get("router_decision_1010_made", False)
    # Overnight ETF sleeve state
    bot.overnight_etf_fired = bot_state.get("overnight_etf_fired", False)
    bot.overnight_etf_position = bot_state.get("overnight_etf_position", None)
    bot.overnight_etf_decision_made = bot_state.get("overnight_etf_decision_made", False)

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
        "volume_ratio": round(c.volume_ratio, 2),
        "close_position": round(c.close_position, 3),
        "late_drop_1530_1550": round(c.late_drop_1530_1550, 4),
        "adv_dollars": round(c.adv_dollars, 0),
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
