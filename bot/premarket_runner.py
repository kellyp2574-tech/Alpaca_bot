"""Premarket dynamic-limit runner — extracted from integrated_main.py.

Owns the 05:00 → 06:00 rolling premarket dynamic-limit classification loop
and the per-checkpoint artifact appender. Pure side-effects module: takes
the orchestrator instance (`bot`) and mutates its state directly, so all
state ownership remains in `CombinedOvernightReboundBot`.

Public functions:
  - place_premarket_dynamic_limit_sells(bot, decision_time_str=None)
  - append_premarket_limits_artifact(bot, ...)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, date, time as dt_time
from typing import Any, Dict
from zoneinfo import ZoneInfo

from bot import config

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")


def _parse_config_time(time_str: str) -> dt_time:
    """Parse an 'HH:MM' config string into a datetime.time object."""
    parts = time_str.split(":")
    return dt_time(int(parts[0]), int(parts[1]))


def place_premarket_dynamic_limit_sells(bot, decision_time_str: str = None) -> None:
    """Rolling premarket dynamic limit classification (05:00 → 06:00).

    At each 15-minute checkpoint (05:00, 05:15, 05:30, 05:45), classify only
    "decisive" symbols - those with clear enough signals to act now. Leave unclear
    symbols unresolved for the next checkpoint. At 06:00 (final checkpoint),
    classify all remaining unresolved symbols with the normal dynamic rule.

    This replaces the old 20:00 blanket limit workflow. Orders are submitted
    in a non-blocking batch style, then the normal 09:25 cleanup cancels any
    remaining open limits before the 09:30 exit logic.
    """
    if bot.premarket_dynamic_limits_done:
        logger.info("PREMARKET LIMITS: already completed; skipping duplicate call")
        return

    bot.position_mgr.reconcile_local_positions_from_broker()
    symbols = list(bot.position_mgr.positions.keys())
    if not symbols:
        logger.info("PREMARKET LIMITS: no overnight positions to classify")
        return

    broker_positions = bot.position_mgr.get_broker_positions()
    if broker_positions is None:
        logger.error("PREMARKET LIMITS: broker position read failed; skipping limit placement")
        return
    broker_by_symbol = {
        str(p.get("symbol", "")).upper(): p
        for p in broker_positions
        if p.get("symbol")
    }

    # Determine decision time
    start_time = _parse_config_time(getattr(config, "PREMARKET_DYNAMIC_START_TIME", "05:00"))
    final_time = _parse_config_time(getattr(config, "PREMARKET_DYNAMIC_FINAL_TIME", "06:00"))
    interval_min = getattr(config, "PREMARKET_DYNAMIC_CHECK_INTERVAL_MINUTES", 15)

    current_time = datetime.now(_ET).time()
    decision_time_str = decision_time_str or current_time.strftime("%H:%M")
    decision_time_dt = _parse_config_time(decision_time_str)

    decision_dt = datetime.combine(datetime.now(_ET).date(), decision_time_dt, tzinfo=_ET)
    if datetime.now(_ET) < decision_dt:
        decision_dt = datetime.now(_ET)

    final_time_str = final_time.strftime("%H:%M")

    # Batch fetch delayed SIP bars for all symbols before looping
    # This uses the historical bars endpoint with feed=sip and end=decision_dt - 16 minutes
    all_bars = bot._fetch_delayed_sip_premarket_bars(symbols, decision_dt)
    logger.info(f"PREMARKET LIMITS: batch delayed SIP bars fetched {len(all_bars)} symbols")

    placed = 0
    no_cap = 0
    skipped = 0
    waited = 0

    for symbol in symbols:
        # Skip symbols already decided in previous checkpoints
        if symbol in bot.premarket_decided_symbols:
            continue

        pos = bot.position_mgr.positions.get(symbol)
        broker_pos = broker_by_symbol.get(symbol.upper())
        if not pos or not broker_pos:
            skipped += 1
            logger.warning(f"PREMARKET LIMIT {symbol}: missing local/broker position; skipping")
            continue

        try:
            qty = min(int(pos.quantity), abs(int(float(broker_pos.get("qty", 0)))))
        except (TypeError, ValueError):
            qty = int(pos.quantity or 0)
        if qty <= 0:
            skipped += 1
            logger.warning(f"PREMARKET LIMIT {symbol}: qty <= 0; skipping")
            continue

        entry_price = float(getattr(pos, "entry_price", 0.0) or broker_pos.get("avg_entry_price", 0.0) or 0.0)
        if entry_price <= 0:
            skipped += 1
            logger.warning(f"PREMARKET LIMIT {symbol}: missing entry price; skipping")
            continue

        sleeve = str(getattr(pos, "sleeve", "UNKNOWN") or "UNKNOWN").upper()

        metrics = bot._compute_delayed_sip_premarket_metrics(
            symbol, entry_price, decision_dt, pre_fetched_bars=all_bars,
        )

        # Check if signal is decisive (act now) or should wait
        current_return = metrics.get("current_return")

        # If data is unavailable, skip this checkpoint
        if current_return is None:
            if decision_time_str >= final_time_str:
                skipped += 1
                logger.error(
                    f"PREMARKET LIMIT {symbol}: DATA UNAVAILABLE at FINAL {decision_time_str}; "
                    f"no premarket limit placed; regular 09:30/MOO exit will handle it"
                )
            else:
                waited += 1
                logger.warning(
                    f"PREMARKET LIMIT {symbol}: DATA UNAVAILABLE at {decision_time_str}; "
                    f"will check again in 15 minutes"
                )
            continue

        is_decisive, decisive_reason = bot._is_decisive_premarket_signal(
            decision_time=decision_time_str,
            final_time=final_time_str,
            current_return=current_return,
            distance_from_high=metrics.get("distance_from_high", 0.0),
            trend_from_first_bar=metrics.get("trend_from_first_bar", 0.0),
            minutes_traded=int(metrics.get("premarket_minutes", 0) or 0),
            last_bar_age_minutes=float(metrics.get("last_bar_age_minutes", 999) or 999),
            sleeve=sleeve,
            data_source=metrics.get("reason", ""),
        )

        if not is_decisive:
            waited += 1
            logger.info(
                f"PREMARKET LIMIT {symbol}: NOT DECISIVE at {decision_time_str} | "
                f"reason={decisive_reason}, ret={metrics.get('current_return', 0.0):+.2%}, "
                f"will check again in 15 minutes"
            )
            continue

        # Symbol is decisive - classify and act
        decision = bot._classify_premarket_limit(pos, metrics)

        log_metrics = (
            f"source={metrics.get('reason')}, "
            f"bars={metrics.get('premarket_minutes', 0)}, "
            f"ret={metrics.get('current_return', 0.0):+.2%}, "
            f"dist_high={metrics.get('distance_from_high', 0.0):+.2%}, "
            f"trend={metrics.get('trend_from_first_bar', 0.0):+.2%}, "
            f"stale={metrics.get('last_bar_age_minutes', 999):.0f}m"
        )

        # Mark as decided so we don't reclassify in future checkpoints
        bot.premarket_decided_symbols.add(symbol)

        if decision["action"] == "NO_ACTION":
            skipped += 1
            logger.error(
                f"PREMARKET LIMIT {symbol} [{decision_time_str}]: NO ACTION (DATA UNAVAILABLE) | "
                f"entry={entry_price:.4f}, {log_metrics}, decisive={decisive_reason}, action={decision['reason']}"
            )
            continue

        if decision["action"] == "NO_CAP":
            no_cap += 1
            logger.warning(
                f"PREMARKET LIMIT {symbol} [{decision_time_str}]: NO CAP | entry={entry_price:.4f}, "
                f"{log_metrics}, decisive={decisive_reason}, action={decision['reason']}"
            )
            continue

        limit_pct = decision.get("limit_pct")
        if limit_pct is None:
            no_cap += 1
            logger.warning(
                f"PREMARKET LIMIT {symbol} [{decision_time_str}]: NO LIMIT | entry={entry_price:.4f}, "
                f"{log_metrics}, decisive={decisive_reason}, action={decision['reason']}"
            )
            continue

        limit_price = bot.position_mgr.round_limit_price(entry_price * (1.0 + float(limit_pct)))
        resp = bot.position_mgr._submit_sell_order(
            symbol=symbol,
            qty=qty,
            order_type="limit",
            limit_price=limit_price,
            time_in_force=getattr(config, "PREMARKET_DYNAMIC_LIMIT_TIME_IN_FORCE", "day"),
            extended_hours=getattr(config, "PREMARKET_DYNAMIC_LIMIT_EXTENDED_HOURS", True),
        )
        if resp and resp.get("id"):
            bot.premarket_limit_order_ids[symbol] = resp["id"]
            placed += 1
            logger.info(
                f"PREMARKET LIMIT {symbol} [{decision_time_str}]: placed qty={qty}, entry={entry_price:.4f}, "
                f"limit_pct={limit_pct:.2%}, limit={limit_price:.4f}, {log_metrics}, "
                f"decisive={decisive_reason}, action={decision['reason']}, order_id={resp['id']}"
            )
        else:
            skipped += 1
            logger.error(
                f"PREMARKET LIMIT {symbol} [{decision_time_str}]: submit failed | "
                f"{log_metrics}, decisive={decisive_reason}, action={decision['reason']}"
            )

    # Check if we're at the final checkpoint
    is_final = decision_time_str >= final_time_str

    # If final checkpoint and data was unavailable for any symbols, log severe warning
    if is_final:
        data_unavailable_count = sum(
            1 for s in symbols
            if s not in bot.premarket_decided_symbols
            and s not in bot.premarket_limit_order_ids
        )
        if data_unavailable_count > 0:
            logger.error(
                f"PREMARKET LIMITS FINAL CHECKPOINT: {data_unavailable_count} symbols had no data available "
                f"and were not classified. These will be handled by regular morning exit logic."
            )

    logger.warning(
        f"PREMARKET LIMITS [{decision_time_str}] COMPLETE: placed={placed}, no_cap={no_cap}, "
        f"skipped={skipped}, waited={waited}, final_checkpoint={is_final}"
    )

    # Append this checkpoint's outcome to the daily premarket-limits
    # artifact (observability). One file per session with one entry
    # per 05:00 / 05:15 / 05:30 / 05:45 / 06:00 checkpoint, so the
    # entire premarket decision sequence is reviewable as JSON.
    try:
        append_premarket_limits_artifact(
            bot,
            decision_time_str=decision_time_str,
            placed=placed,
            no_cap=no_cap,
            skipped=skipped,
            waited=waited,
            is_final=is_final,
            symbol_count=len(symbols),
            limit_order_ids=dict(bot.premarket_limit_order_ids),
        )
    except Exception:
        logger.warning("premarket_limits artifact append failed", exc_info=True)

    # If final checkpoint, mark as done
    if is_final:
        bot.premarket_dynamic_limits_done = True
        bot._save_state()
    else:
        # Save state after each checkpoint to track decided symbols
        bot._save_state()


def append_premarket_limits_artifact(
    bot,
    decision_time_str: str,
    placed: int,
    no_cap: int,
    skipped: int,
    waited: int,
    is_final: bool,
    symbol_count: int,
    limit_order_ids: Dict[str, str],
) -> None:
    """Append one checkpoint outcome to state/logs/premarket_limits_YYYY-MM-DD.json.

    File schema:
      {
        "date": "YYYY-MM-DD",
        "checkpoints": [
          {"time": "05:00", "placed": 2, "no_cap": 0, "skipped": 1, "waited": 3, ...},
          ...
        ],
        "limit_order_ids_at_end": {"AAPL": "<order_id>", ...}
      }

    Idempotent at the checkpoint level: appending the same HH:MM twice
    replaces the prior entry rather than duplicating it (defensive
    against any future retry path).
    """
    today = date.today().isoformat()
    path = os.path.join(config.LOG_DIR, f"premarket_limits_{today}.json")

    existing: Dict[str, Any] = {"date": today, "checkpoints": []}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                existing = json.load(f) or existing
            if not isinstance(existing, dict) or "checkpoints" not in existing:
                existing = {"date": today, "checkpoints": []}
        except Exception:
            logger.warning(
                f"premarket_limits artifact: could not read {path}; overwriting",
                exc_info=True,
            )
            existing = {"date": today, "checkpoints": []}

    # Replace prior entry for this checkpoint if present.
    checkpoints = [c for c in existing.get("checkpoints", []) if c.get("time") != decision_time_str]
    checkpoints.append({
        "time": decision_time_str,
        "is_final": is_final,
        "symbols_considered": symbol_count,
        "placed": placed,
        "no_cap": no_cap,
        "skipped": skipped,
        "waited": waited,
    })
    existing["checkpoints"] = sorted(checkpoints, key=lambda c: c["time"])
    existing["limit_order_ids_at_end"] = limit_order_ids
    existing["decided_symbols"] = sorted(bot.premarket_decided_symbols)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    logger.info(f"Premarket-limits artifact updated for {decision_time_str}: {path}")
