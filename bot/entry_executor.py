"""15:45 MR entry executor — extracted from integrated_main.py.

Owns the full afternoon entry pipeline: paper-allocation waterfall, execution
eligibility gate, marketable-limit pricing, concurrent submission with
client_order_id reconciliation, concurrent fill monitoring, and shortfall
diagnostics.

`step_execute_entries(bot)` is the single public entry point. It mutates
`bot` state directly (positions, exec diagnostics, entries_done flag,
sold_today, _exec_stats). State ownership stays in
`CombinedOvernightReboundBot`.

Live constants (read from `bot.config`):
    MR_MAX_TOTAL_ALLOCATION_PCT  = 0.90  -> sleeve budget = deployable * 0.90 * regime_mult
    MR_ALLOC_PER_POSITION_PCT    = 0.30  -> per-position cap (3 positions x 30% = 90%)
    MR_MAX_PRIMARY_POSITIONS     = 3     -> top 3 candidates only (overflow goes to waterfall)
    MR_ADV_CAP_PCT               = 0.003 -> 0.3% of 20-day ADV per symbol

Note: MR entries execute at 15:45 (not 15:50). SCORING_TIME and ENTRY_TIME
are both set to "15:45" in config_strategy.py.
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

import requests

from bot import config
from bot.position_manager_overnight import Position
from bot.universe_builder import ExecutionDiagnostics, filter_execution_ready

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")


@dataclass
class SleeveAllocation:
    """Per-symbol allocation produced by the waterfall + sized to shares."""
    symbol: str
    shares: int
    target_dollars: float
    rank: int
    sleeve: str
    candidate: Any


def allocate_waterfall(bot, candidates, sleeve_budget: float, equity: float,
                       sleeve_name: str, max_positions: int) -> list:
    """Two-pass waterfall allocator with effective minimum sizing.

    Pre-filters: cap checks, effective min (MIN_POSITION_DOLLARS vs MIN_SHARES*price)
    Pass 1: Equal share respecting caps (MR_ALLOC_PER_POSITION_PCT, MR_ADV_CAP_PCT, MAX_POSITION_DOLLARS)
    Pass 2: Redistribute leftover to candidates with capacity (highest ADV first)

    Returns list of dicts with symbol, target_dollars, cap_dollars, adv_dollars.

    Lifted out of `step_execute_entries` for readability and testability;
    accepts `bot` as the first arg to record per-symbol sizing diagnostics
    on `bot._exec_diag.sizing_diagnostics`.
    """
    if not candidates or sleeve_budget <= 0:
        return []

    # Per-position cap = MR_ALLOC_PER_POSITION_PCT * equity (default 30%).
    max_single_dollars = equity * config.MR_ALLOC_PER_POSITION_PCT

    # ADV multiplier for IEX data (IEX reports lower volume than composite)
    adv_multiplier = getattr(config, "ADV_DOLLAR_MULTIPLIER", 1.0)
    adv_cap_pct = config.MR_ADV_CAP_PCT

    # Pre-filter: skip candidates where cap is below effective minimum
    viable = []
    skips = {}
    for c in candidates:
        sizing_adv_dollars = c.adv_dollars * adv_multiplier if c.adv_dollars > 0 else 0.0
        adv_cap = sizing_adv_dollars * adv_cap_pct if sizing_adv_dollars > 0 else 0.0
        cap = min(
            adv_cap,
            max_single_dollars,
            config.MAX_POSITION_DOLLARS,
        )

        effective_min = max(
            config.MIN_POSITION_DOLLARS,
            config.MIN_SHARES * c.signal_price,
        )

        if cap < effective_min:
            skips[c.symbol] = {
                "reason": "cap_below_effective_min",
                "cap": round(cap, 2),
                "effective_min": round(effective_min, 2),
                "raw_adv_dollars": round(c.adv_dollars, 0),
                "adv_multiplier": adv_multiplier,
                "sizing_adv_dollars": round(sizing_adv_dollars, 0),
                "adv_cap_pct": adv_cap_pct,
                "adv_cap_dollars": round(adv_cap, 2),
                "max_single_dollars": round(max_single_dollars, 2),
                "price": round(c.signal_price, 2),
                "skip_reason": f"cap=${cap:.2f} < effective_min=${effective_min:.2f}",
            }
            continue

        viable.append(c)

    if skips:
        logger.warning(f"{sleeve_name}: skipped {len(skips)} candidates (cap below effective min)")
        for sym, reason in list(skips.items())[:5]:
            logger.warning(f"  {sym}: {reason}")

    viable = viable[:max_positions]

    if not viable:
        logger.warning(f"{sleeve_name}: no viable candidates after cap/min-share filters")
        return []

    # Calculate per-candidate caps with ADV multiplier
    caps = {}
    sizing_info = {}
    last_sizing_adv = 0.0
    for c in viable:
        sizing_adv_dollars = c.adv_dollars * adv_multiplier if c.adv_dollars > 0 else 0.0
        adv_cap = sizing_adv_dollars * adv_cap_pct
        caps[c.symbol] = min(
            adv_cap,
            max_single_dollars,
            config.MAX_POSITION_DOLLARS,
        )
        sizing_info[c.symbol] = {
            "raw_adv_dollars": round(c.adv_dollars, 0),
            "adv_multiplier": adv_multiplier,
            "sizing_adv_dollars": round(sizing_adv_dollars, 0),
            "adv_cap_pct": adv_cap_pct,
            "adv_cap_dollars": round(adv_cap, 2),
            "max_single_dollars": round(max_single_dollars, 2),
            "final_cap_dollars": round(caps[c.symbol], 2),
        }
        last_sizing_adv = sizing_adv_dollars

    allocations = {c.symbol: 0.0 for c in viable}
    base_target = sleeve_budget / len(viable)

    # Pass 1: Low ADV first - give everyone their capped equal share
    for c in sorted(viable, key=lambda x: x.adv_dollars):
        effective_min = max(config.MIN_POSITION_DOLLARS, config.MIN_SHARES * c.signal_price)
        alloc = min(base_target, caps[c.symbol])
        if alloc >= effective_min:
            allocations[c.symbol] = alloc

    leftover = sleeve_budget - sum(allocations.values())

    # Pass 2: High ADV first - push leftover into names with room
    for c in sorted(viable, key=lambda x: x.adv_dollars, reverse=True):
        if leftover < config.MIN_POSITION_DOLLARS:
            break
        room = caps[c.symbol] - allocations[c.symbol]
        if room <= 0:
            continue
        add = min(leftover, room)
        allocations[c.symbol] += add
        leftover -= add

    # Build result list (filter by effective min one more time)
    result = []
    for c in viable:
        effective_min = max(config.MIN_POSITION_DOLLARS, config.MIN_SHARES * c.signal_price)
        raw_target = allocations[c.symbol]
        if raw_target >= effective_min:
            sizing_diag = sizing_info.get(c.symbol, {})
            sizing_diag.update({
                "raw_target": round(raw_target, 2),
                "final_target": round(raw_target, 2),
                "effective_min": round(effective_min, 2),
                "min_shares": config.MIN_SHARES,
                "min_position_dollars": config.MIN_POSITION_DOLLARS,
            })
            if getattr(bot, "_exec_diag", None):
                bot._exec_diag.sizing_diagnostics[c.symbol] = sizing_diag
            result.append({
                "symbol": c.symbol,
                "target_dollars": allocations[c.symbol],
                "cap_dollars": caps[c.symbol],
                "adv_dollars": c.adv_dollars,
                "sizing_adv_dollars": last_sizing_adv,
                "candidate": c,
                "sizing": sizing_diag,
            })

    total_allocated = sum(r["target_dollars"] for r in result)

    zero_alloc = [
        c.symbol for c in viable
        if allocations.get(c.symbol, 0.0) <= 0
    ]
    if zero_alloc:
        logger.warning(f"{sleeve_name}: zero allocations after waterfall: {zero_alloc[:10]}")

    logger.info(
        f"{sleeve_name} waterfall: "
        f"candidates={len(candidates)}, viable={len(viable)}, selected={len(result)}, "
        f"budget=${sleeve_budget:,.2f}, allocated=${total_allocated:,.2f}, leftover=${leftover:,.2f}"
    )

    return result


def step_execute_entries(bot) -> None:
    """15:45: clean MR-only paper allocation -> execution-gate -> market buys.
    
    Note: MR entries execute at 15:45 (not 15:50). Both SCORING_TIME and
    ENTRY_TIME are set to "15:45" in config_strategy.py.
    """
    logger.info("=" * 50)
    logger.info("ENTRY EXECUTION: Clean MR paper-test market buy orders")
    logger.info("=" * 50)

    def mark_entries_done_and_save() -> None:
        """Flag entries_done and persist state.

        Every exit path from this function — skips and successes alike —
        must go through here so a same-day restart does not re-evaluate a
        decision that was already made (e.g. re-attempting an entry that
        was already skipped by the kill switch).
        """
        bot.entries_done = True
        bot._save_state()

    # Check MR permission — block ONLY if router actually has/had a filled position.
    # A router signal that was rejected (failed execution gate, stale quote, etc.)
    # must NOT block MR.  Only a confirmed fill is a regime conflict.
    has_etf_position = bot.etf_position is not None
    intraday_etf_filled = getattr(bot, "intraday_etf_sleeve_filled", False)
    if has_etf_position or intraday_etf_filled:
        branch = bot.router_branch or "unknown"
        logger.info(
            "MR entries BLOCKED — router/V2/P1 has a live or filled intraday ETF position "
            "(branch=%s, has_position=%s, sleeve_filled=%s)",
            branch, has_etf_position, intraday_etf_filled,
        )
        mark_entries_done_and_save()
        return

    # Check if MR is enabled in config
    if not getattr(config, "MR_OVERNIGHT_ENABLED", True):
        logger.info("MR entries DISABLED in config - skipping")
        mark_entries_done_and_save()
        return

    # Check MR subtype allowlist (Issue 2)
    # MR only runs on specific router no-trade subtypes that were profitable in backtest.
    # Source of truth is MR_ALLOWED_SUBTYPES in config_strategy.py — do NOT hardcode here.
    allowed_mr_subtypes = set(getattr(config, "MR_ALLOWED_SUBTYPES", []))
    subtype = getattr(bot, "router_no_trade_subtype", None)
    if subtype not in allowed_mr_subtypes:
        logger.info(f"MR entries skipped: subtype={subtype} not in allowed set {allowed_mr_subtypes}")
        mark_entries_done_and_save()
        return

    # Early close guard (Issue 5)
    from bot.etf_router_runtime import is_early_close
    if is_early_close() and getattr(config, "SKIP_MR_ON_EARLY_CLOSE", True):
        logger.warning("MR entries skipped: early close day with SKIP_MR_ON_EARLY_CLOSE=True")
        mark_entries_done_and_save()
        return

    exec_diag = ExecutionDiagnostics()
    bot._exec_diag = exec_diag

    try:
        # Single account fetch — was previously 2 calls here plus 1 per
        # allocation in _adaptive_qty (~22 account calls per entry pass).
        account = bot.position_mgr.get_account()
        if not account:
            logger.error("Cannot fetch account — skipping entries")
            mark_entries_done_and_save()
            return
        try:
            equity = float(account.get("equity") or 0.0)
            buying_power = float(account.get("buying_power") or 0.0)
        except (TypeError, ValueError):
            equity = 0.0
            buying_power = 0.0
        if equity <= 0:
            logger.error("Cannot determine account equity — skipping entries")
            mark_entries_done_and_save()
            return
        if buying_power <= 0:
            logger.warning("Cannot determine buying power — falling back to equity")
            buying_power = equity

        # Daily loss kill switch — global flag set by
        # check_daily_loss_kill_switch(). Trips if today's PnL is worse
        # than DAILY_LOSS_LIMIT_PCT. account['last_equity'] is yesterday's
        # close equity. Once tripped, also blocks any future entries this
        # session (and the 10:00 ETF router entry checks the same flag).
        from bot import scoring as _scoring
        if _scoring.check_daily_loss_kill_switch(bot, account=account):
            logger.critical(
                f"MR entries BLOCKED by daily-loss kill switch — {bot.kill_switch_reason}"
            )
            mark_entries_done_and_save()
            return
        try:
            last_equity = float(account.get("last_equity") or 0.0)
            if last_equity > 0:
                day_ret = (equity - last_equity) / last_equity
                logger.info(
                    f"Daily PnL check OK: {day_ret:+.2%} (limit "
                    f"-{float(getattr(config, 'DAILY_LOSS_LIMIT_PCT', 0.0)):.0%})"
                )
        except (TypeError, ValueError):
            pass

        deployable = min(buying_power, equity * config.MAX_LEVERAGE)
        logger.info(
            f"Account equity: ${equity:,.2f}, buying_power: ${buying_power:,.2f}, "
            f"deployable: ${deployable:,.2f}"
        )

        # PDT filter: remove recently-sold symbols (use a local copy so the
        # original audit list is preserved through the run).
        if equity < 50_000 and bot.sold_today:
            before_mr = len(bot.mr_candidates)
            mr_candidates_filtered = [c for c in bot.mr_candidates if c.symbol not in bot.sold_today]
            blocked_mr = before_mr - len(mr_candidates_filtered)
            if blocked_mr:
                logger.warning(
                    f"PDT guard: filtered MR={blocked_mr} "
                    f"same-day re-entry candidates (equity ${equity:,.0f} < $50k)"
                )
        else:
            mr_candidates_filtered = list(bot.mr_candidates)

        # ETF-regime sizing from the clean-cache finalist:
        # full size when 3-ETF avg is negative before entry, half size otherwise.
        from bot import scoring as _scoring
        mr_size_mult, mr_regime_info = _scoring.compute_mr_etf_regime_size_multiplier(bot)

        # Sleeve budget = deployable * MR_MAX_TOTAL_ALLOCATION_PCT (0.90) * regime_mult
        mr_budget = deployable * config.MR_MAX_TOTAL_ALLOCATION_PCT * mr_size_mult
        logger.info(
            f"MR sleeve budget: ${mr_budget:,.2f} "
            f"({config.MR_MAX_TOTAL_ALLOCATION_PCT:.0%} * regime_mult={mr_size_mult:.2f}) | "
            f"regime={mr_regime_info}"
        )

        # Execution eligibility gate FIRST (before allocation)
        # This prevents budget from being assigned to names that fail spread check.
        # Cap pool size to avoid large snapshot calls (3x max positions gives replacement depth).
        EXEC_POOL_MULTIPLIER = 3
        mr_pool = mr_candidates_filtered[:config.MR_MAX_PRIMARY_POSITIONS * EXEC_POOL_MULTIPLIER]
        candidate_symbols = [c.symbol for c in mr_pool]
        fresh_snaps = bot.alpaca.get_snapshots(candidate_symbols)
        orderable, exec_rejected = filter_execution_ready(
            candidate_symbols, fresh_snaps,
            max_spread_pct=getattr(config, "ENTRY_MAX_SPREAD_PCT", 0.05), require_quote=True,
        )
        orderable_set = set(orderable)

        if exec_rejected:
            for sym, reason in exec_rejected.items():
                logger.warning(f"Execution reject {sym}: {reason}")

        # Filter candidates to only orderable symbols (from the capped pools)
        mr_orderable = [c for c in mr_pool if c.symbol in orderable_set]

        logger.info(
            f"Post-spread-filter: MR {len(mr_candidates_filtered)} -> {len(mr_orderable)} orderable"
        )

        # Paper-test guard: require min candidates AFTER execution gate (spread/quote check)
        mr_min_candidates = int(getattr(config, "MR_MIN_CANDIDATES", 1) or 1)
        if len(mr_orderable) < mr_min_candidates:
            logger.warning(
                "MR paper test: only %d orderable candidates after execution gate, below min_candidates=%d — skipping entries",
                len(mr_orderable),
                mr_min_candidates,
            )
            mark_entries_done_and_save()
            return

        # Allocate ONLY from orderable candidates (budget flows to clean names)
        mr_results = allocate_waterfall(
            bot, mr_orderable, mr_budget, equity, "MR", config.MR_MAX_PRIMARY_POSITIONS,
        )

        # Calculate leftover from primary allocation
        mr_allocated = sum(r["target_dollars"] for r in mr_results)
        mr_leftover = mr_budget - mr_allocated

        logger.info(
            f"MR sleeve: budget=${mr_budget:,.2f}, allocated=${mr_allocated:,.2f}, "
            f"leftover=${mr_leftover:,.2f}, positions={len(mr_results)}"
        )

        # Leftover redeployment into the next-best ranked MR candidates (waterfall overflow).
        # Capped by MR_MAX_WATERFALL_POSITIONS so we never exceed the absolute slot limit.
        enable_redeployment = getattr(config, "ENABLE_LEFTOVER_REDEPLOYMENT", True)
        if not enable_redeployment and mr_leftover > config.MIN_POSITION_DOLLARS:
            logger.info(
                "Leftover redeployment DISABLED: $%.2f remains as cash.",
                mr_leftover,
            )
        elif mr_leftover > config.MIN_POSITION_DOLLARS:
            allocated_symbols = {r["symbol"] for r in mr_results}
            overflow_pool = [c for c in mr_orderable if c.symbol not in allocated_symbols]
            overflow_pool.sort(key=lambda x: getattr(x, "selection_score", 0.0), reverse=True)
            max_overflow = max(
                0,
                int(getattr(config, "MR_MAX_WATERFALL_POSITIONS", config.MR_MAX_PRIMARY_POSITIONS))
                - len(mr_results),
            )
            if overflow_pool and max_overflow > 0:
                overflow_fallback = allocate_waterfall(
                    bot, overflow_pool, mr_leftover, equity, "MR_OVERFLOW", max_overflow,
                )
                if overflow_fallback:
                    overflow_allocated = sum(r["target_dollars"] for r in overflow_fallback)
                    logger.info(
                        f"MR overflow waterfall: budget=${mr_leftover:,.2f}, "
                        f"allocated=${overflow_allocated:,.2f}, positions={len(overflow_fallback)}"
                    )
                    for r in overflow_fallback:
                        r["fallback"] = True
                    mr_results.extend(overflow_fallback)
                    mr_leftover -= overflow_allocated

            if mr_leftover > config.MIN_POSITION_DOLLARS:
                logger.warning(
                    f"Final MR leftover: ${mr_leftover:,.2f} (no more orderable candidates)"
                )

        # Build SleeveAllocation list with shares calculated from target dollars
        allocations: List[SleeveAllocation] = []
        for rank, r in enumerate(mr_results, start=1):
            c = r["candidate"]
            shares = math.floor(r["target_dollars"] / c.signal_price) if c.signal_price > 0 else 0
            sleeve_label = "MR" if not r.get("fallback") else "MR_OVERFLOW"
            allocations.append(SleeveAllocation(
                symbol=c.symbol,
                shares=shares,
                target_dollars=r["target_dollars"],
                rank=rank,
                sleeve=sleeve_label,
                candidate=c,
            ))

        # Hard cap at the absolute waterfall maximum slots.
        max_slots = int(getattr(config, "MR_MAX_WATERFALL_POSITIONS", config.MR_MAX_PRIMARY_POSITIONS))
        if len(allocations) > max_slots:
            logger.warning(
                f"Slot cap trimming allocations {len(allocations)} -> {max_slots}"
            )
            allocations = allocations[:max_slots]

        if not allocations:
            logger.warning("No positions sized — skipping entries")
            mark_entries_done_and_save()
            return

        exec_diag.selected_symbols = [a.symbol for a in allocations]
        exec_diag.orderable_symbols = [a.symbol for a in allocations]
        exec_diag.rejected_symbols = dict(exec_rejected)
        total_target = sum(a.target_dollars for a in allocations)
        logger.info(
            f"Selected {len(allocations)} MR allocations, total_target=${total_target:,.2f}"
        )
        logger.info(
            f"Execution pool metrics: pool_size={len(candidate_symbols)}, "
            f"orderable={len(orderable_set)}, rejected_spread={len(exec_rejected)}"
        )
        # Submit market buy orders concurrently with short timeout.
        total_deployed = 0.0

        # Track buying power locally to avoid hitting /v2/account once per symbol
        bp_remaining = buying_power

        def _adaptive_qty(alloc: SleeveAllocation, bp_avail: float,
                          bp_buffer: float = config.ENTRY_BP_BUFFER_PCT) -> int:
            """Return shares clamped to current buying power using target_dollars."""
            if bp_avail <= 0:
                return 0
            max_notional = bp_avail * bp_buffer
            target = min(alloc.target_dollars, max_notional)
            price_ref = alloc.candidate.signal_price
            if price_ref <= 0:
                return 0
            return math.floor(target / price_ref)

        # Hard cutoff for new buy submissions (don't chase too close to close).
        cutoff_str = getattr(config, "ENTRY_HARD_CUTOFF_TIME", "15:58:30")
        try:
            ch, cm, cs = (int(p) for p in cutoff_str.split(":"))
            entry_cutoff = dt_time(ch, cm, cs)
        except (ValueError, TypeError):
            entry_cutoff = dt_time(15, 58, 30)

        # Create deterministic client_order_id for each allocation
        # Format: BOT-YYYYMMDD-HHMMSS-SYMBOL
        timestamp = datetime.now(_ET).strftime("%Y%m%d-%H%M%S")
        submission_plans = []  # List of (alloc, qty, client_order_id, price_ref, limit_price)

        # Issue #6: marketable-limit slippage cap for MR entries.
        mr_slippage_pct = float(getattr(config, "ENTRY_MAX_SLIPPAGE_PCT", 0.02))

        for alloc in allocations:
            # Check cutoff before processing
            if datetime.now(_ET).time() >= entry_cutoff:
                logger.warning("ENTRY CUTOFF reached (15:58:30) — stopping new buy submissions")
                break

            symbol = alloc.symbol
            if symbol not in orderable_set:
                continue

            candidate = alloc.candidate
            price_ref = candidate.signal_price
            qty = _adaptive_qty(alloc, bp_remaining)

            if qty < config.MIN_SHARES:
                logger.warning(
                    f"ENTRY SKIP {symbol}: adaptive qty {qty} < {config.MIN_SHARES} min shares "
                    f"(bp=${bp_remaining:,.2f}, price={price_ref:.4f})"
                )
                exec_diag.failed_submissions[symbol] = "bp_resize_below_min"
                continue

            # Marketable-limit price (issue #6): cap at ask * (1 + slippage).
            # Falls back to market order (limit_price=None) when the ask is
            # missing from the snapshot, preserving prior behavior on
            # degraded data instead of refusing to enter.
            snap = (fresh_snaps or {}).get(symbol, {}) or {}
            ask = snap.get("ask")
            if ask and float(ask) > 0:
                limit_price = float(ask) * (1.0 + mr_slippage_pct)
            else:
                limit_price = None

            # Create deterministic client_order_id
            client_order_id = f"BOT-{timestamp}-{symbol}"
            planned_notional = qty * (limit_price if limit_price else price_ref)

            logger.info(
                f"ENTRY PLAN {symbol}: qty={qty}, price_ref={price_ref:.4f}, "
                f"ask={ask}, limit={f'{limit_price:.4f}' if limit_price else 'MARKET'}, "
                f"notional={planned_notional:,.2f}, bp_remaining={bp_remaining:,.2f}, "
                f"sleeve={alloc.sleeve}, rank={alloc.rank}, client_id={client_order_id}"
            )

            submission_plans.append((alloc, qty, client_order_id, price_ref, limit_price))
            # Decrement local BP tracker by the planned notional
            bp_remaining = max(0.0, bp_remaining - planned_notional)

        # Batch submit all orders concurrently with short timeout.
        submitted_orders = []      # List of (order_id, alloc, qty, candidate, client_order_id)
        submission_timeouts = []   # List of (alloc, qty, client_order_id, price_ref)

        def _submit_entry_order(plan):
            """Submit one buy order. Runs inside ThreadPoolExecutor."""
            alloc, qty, client_order_id, price_ref, limit_price = plan
            symbol = alloc.symbol
            submit_start = datetime.now(_ET)
            t0 = time.perf_counter()

            try:
                if limit_price is not None:
                    buy_resp, error_type = bot.position_mgr.submit_buy_order(
                        symbol,
                        qty,
                        client_order_id=client_order_id,
                        timeout=getattr(config, "ENTRY_SUBMIT_TIMEOUT_SECONDS", 2),
                        order_type="limit",
                        limit_price=limit_price,
                    )
                else:
                    buy_resp, error_type = bot.position_mgr.submit_buy_order(
                        symbol,
                        qty,
                        client_order_id=client_order_id,
                        timeout=getattr(config, "ENTRY_SUBMIT_TIMEOUT_SECONDS", 2),
                    )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                return {
                    "symbol": symbol,
                    "alloc": alloc,
                    "qty": qty,
                    "client_order_id": client_order_id,
                    "price_ref": price_ref,
                    "buy_resp": buy_resp,
                    "error_type": error_type,
                    "elapsed_ms": elapsed_ms,
                    "submit_start": submit_start,
                    "exception": None,
                }

            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return {
                    "symbol": symbol,
                    "alloc": alloc,
                    "qty": qty,
                    "client_order_id": client_order_id,
                    "price_ref": price_ref,
                    "buy_resp": None,
                    "error_type": "exception",
                    "elapsed_ms": elapsed_ms,
                    "submit_start": submit_start,
                    "exception": e,
                }

        max_workers = min(
            len(submission_plans),
            int(getattr(config, "ENTRY_SUBMIT_MAX_WORKERS", 8)),
        )

        logger.warning(
            "ENTRY CONCURRENT SUBMIT START: orders=%d workers=%d timeout=%ss",
            len(submission_plans),
            max_workers,
            getattr(config, "ENTRY_SUBMIT_TIMEOUT_SECONDS", 2),
        )

        # Collect per-order submission latencies so we can publish a
        # p50/p95/avg summary in the run_health artifact (observability).
        submit_latencies_ms: List[float] = []

        if max_workers <= 0:
            logger.warning("ENTRY CONCURRENT SUBMIT: no submission plans")
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_submit_entry_order, plan): plan
                    for plan in submission_plans
                }

                for future in as_completed(futures):
                    result = future.result()

                    symbol = result["symbol"]
                    alloc = result["alloc"]
                    qty = result["qty"]
                    client_order_id = result["client_order_id"]
                    price_ref = result["price_ref"]
                    buy_resp = result["buy_resp"]
                    error_type = result["error_type"]
                    elapsed_ms = result["elapsed_ms"]
                    submit_latencies_ms.append(float(elapsed_ms))

                    if buy_resp and buy_resp.get("id"):
                        order_id = buy_resp["id"]
                        submitted_orders.append((order_id, alloc, qty, alloc.candidate, client_order_id))
                        exec_diag.submitted_symbols.append(symbol)
                        logger.info(
                            "ENTRY SUBMITTED %s: order_id=%s client_id=%s elapsed_ms=%.1f",
                            symbol,
                            order_id,
                            client_order_id,
                            elapsed_ms,
                        )

                    elif error_type in ("timeout", "network_error", "exception"):
                        submission_timeouts.append((alloc, qty, client_order_id, price_ref))
                        logger.warning(
                            "ENTRY TIMEOUT/NETWORK %s: error=%s elapsed_ms=%.1f; will reconcile client_id=%s",
                            symbol,
                            error_type,
                            elapsed_ms,
                            client_order_id,
                        )

                    else:
                        exec_diag.failed_submissions[symbol] = f"submit_failed_{error_type}"
                        logger.error(
                            "ENTRY REJECTED %s: error=%s elapsed_ms=%.1f; no reconciliation",
                            symbol,
                            error_type,
                            elapsed_ms,
                        )

        logger.info(
            "ENTRY SUBMISSION SUMMARY: %d immediate success, %d timeout/error needing reconciliation",
            len(submitted_orders),
            len(submission_timeouts),
        )

        # Reconcile timeouts by querying orders by client_order_id
        if submission_timeouts:
            logger.info(f"ENTRY RECONCILING {len(submission_timeouts)} timeout/error submissions...")
            # Give Alpaca a moment to process orders
            time.sleep(1.0)

            for alloc, qty, client_order_id, price_ref in submission_timeouts:
                symbol = alloc.symbol
                try:
                    # Query orders by client_order_id using the correct endpoint
                    base_url = getattr(config, "ALPACA_BASE_URL", "https://api.alpaca.markets").rstrip("/")
                    url = f"{base_url}/v2/orders:by_client_order_id"
                    params = {"client_order_id": client_order_id}
                    resp = bot.position_mgr.session.get(
                        url,
                        params=params,
                        timeout=getattr(config, "ENTRY_RECONCILE_TIMEOUT_SECONDS", 3),
                    )
                    resp.raise_for_status()
                    order_data = resp.json()  # Returns a single order object, not a list

                    if order_data and order_data.get("id"):
                        # Order exists - add to submitted list
                        order_id = order_data.get("id")
                        submitted_orders.append((order_id, alloc, qty, alloc.candidate, client_order_id))
                        exec_diag.submitted_symbols.append(symbol)
                        logger.info(f"ENTRY RECONCILED {symbol}: order_id={order_id}, client_id={client_order_id}")
                    else:
                        # Order does not exist - treat as failed
                        exec_diag.failed_submissions[symbol] = "reconciliation_no_order"
                        logger.warning(f"ENTRY RECONCILE FAILED {symbol}: no order found for client_id={client_order_id}")
                except requests.exceptions.HTTPError as e:
                    # 404 means order doesn't exist - treat as failed
                    if e.response and e.response.status_code == 404:
                        exec_diag.failed_submissions[symbol] = "reconciliation_no_order"
                        logger.warning(f"ENTRY RECONCILE FAILED {symbol}: 404 no order for client_id={client_order_id}")
                    else:
                        exec_diag.failed_submissions[symbol] = f"reconciliation_http_{e.response.status_code if e.response else 'unknown'}"
                        logger.error(f"ENTRY RECONCILE ERROR {symbol}: HTTP {e}")
                except Exception as e:
                    exec_diag.failed_submissions[symbol] = f"reconciliation_error: {str(e)}"
                    logger.error(f"ENTRY RECONCILE ERROR {symbol}: {e}")

        # Pass 2: monitor fills for all submitted orders CONCURRENTLY.
        # Previously this was sequential: 3 positions x 30s worst-case = 90s
        # blocking inside the entry window. Now each order waits in its own
        # worker so worst-case is ~30s total regardless of order count.
        if submitted_orders:
            fill_workers = min(
                len(submitted_orders),
                int(getattr(config, "ENTRY_SUBMIT_MAX_WORKERS", 8)),
            )
            logger.info(
                "ENTRY FILL MONITOR START: orders=%d workers=%d",
                len(submitted_orders), fill_workers,
            )

            def _wait_fill(entry):
                order_id, alloc, qty, candidate, client_order_id = entry
                fill = bot.position_mgr.get_order_fill(order_id, max_wait=30)
                return entry, fill

            with ThreadPoolExecutor(max_workers=fill_workers) as executor:
                fill_futures = {
                    executor.submit(_wait_fill, entry): entry for entry in submitted_orders
                }
                for future in as_completed(fill_futures):
                    entry, fill = future.result()
                    order_id, alloc, qty, candidate, client_order_id = entry
                    symbol = alloc.symbol

                    if fill and int(fill["filled_qty"]) > 0:
                        filled_qty = int(fill["filled_qty"])
                        fill_price = fill["filled_avg_price"]

                        position = Position(
                            symbol=symbol,
                            entry_price=fill_price,
                            quantity=filled_qty,
                            entry_time=datetime.now(_ET),
                            adv_estimate=candidate.adv_dollars,
                            sleeve=alloc.sleeve,
                            current_price=fill_price,
                        )
                        bot.position_mgr.positions[symbol] = position
                        total_deployed += fill_price * filled_qty
                        exec_diag.filled_symbols.append(symbol)
                        exec_diag.fill_details[symbol] = {
                            "qty": filled_qty, "price": round(fill_price, 4),
                            "score": round(candidate.selection_score, 4),
                            "day_return": round(candidate.day_return, 4),
                            "sleeve": alloc.sleeve,
                            "rank": alloc.rank,
                        }

                        logger.info(
                            f"ENTRY FILLED {symbol}: sleeve={alloc.sleeve}, "
                            f"qty={filled_qty}, avg={fill_price:.4f}, "
                            f"score={candidate.selection_score:.3f}, client_id={client_order_id}"
                        )
                    else:
                        try:
                            bot.position_mgr._cancel_order(order_id)
                            logger.warning(
                                f"ENTRY NO FILL {symbol}: order canceled "
                                f"(order_id={order_id}, client_id={client_order_id})"
                            )
                        except Exception as e:
                            logger.error(f"ENTRY CANCEL ERROR {symbol}: {e}")
                        exec_diag.failed_submissions[symbol] = "no_fill"

        # Mop-up disabled (ENTRY_MOPUP_MAX_POSITIONS = 0)
        mark_entries_done_and_save()

        # Execution stats
        mr_filled = sum(
            1 for s in exec_diag.filled_symbols
            if exec_diag.fill_details.get(s, {}).get("sleeve", "").startswith("MR")
        )

        # Latency summary (observability). p50/p95/avg of per-order
        # POST /v2/orders elapsed time. Surfaces a sluggish broker
        # endpoint into the run_health artifact instead of being buried
        # in per-order log lines.
        latency_summary: Dict[str, Any] = {"count": len(submit_latencies_ms)}
        if submit_latencies_ms:
            sorted_lat = sorted(submit_latencies_ms)
            n = len(sorted_lat)
            p50 = sorted_lat[n // 2]
            p95 = sorted_lat[min(n - 1, int(n * 0.95))]
            latency_summary.update({
                "avg_ms": round(sum(sorted_lat) / n, 1),
                "p50_ms": round(p50, 1),
                "p95_ms": round(p95, 1),
                "max_ms": round(sorted_lat[-1], 1),
            })

        bot._exec_stats = {
            "selected": len(exec_diag.selected_symbols),
            "orderable": len(exec_diag.orderable_symbols),
            "exec_rejected": len(exec_diag.rejected_symbols),
            "exec_rejected_reasons": exec_diag.rejected_symbols,
            "orders_submitted": len(exec_diag.submitted_symbols),
            "entries_filled": len(exec_diag.filled_symbols),
            "mr_filled": mr_filled,
            "total_deployed": total_deployed,
            "equity": equity,
            "deployable": deployable,
            "submit_latency_ms": latency_summary,
        }

        deployment_pct = total_deployed / deployable * 100 if deployable > 0 else 0.0
        logger.info(
            f"Entry execution complete: {len(exec_diag.filled_symbols)} filled "
            f"({mr_filled} MR), "
            f"{len(exec_diag.rejected_symbols)} rejected at execution gate, "
            f"${total_deployed:,.2f} deployed "
            f"({deployment_pct:.1f}% of deployable)"
        )

        # Shortfall diagnostics
        if deployment_pct < 80.0:
            logger.warning("=== DEPLOYMENT SHORTFALL DIAGNOSTICS ===")
            if equity < 50_000 and bot.sold_today:
                logger.warning(f"PDT guard active: sold_today={bot.sold_today}")
            if exec_diag.rejected_symbols:
                logger.warning(f"Execution gate rejected: {len(exec_diag.rejected_symbols)} symbols")
            if exec_diag.failed_submissions:
                logger.warning(f"Failed submissions: {len(exec_diag.failed_submissions)} symbols")

            # Sizing/rounding issues (candidates too small)
            planned_symbols = set([a.symbol for a in allocations])
            filled_symbols = set(exec_diag.filled_symbols)
            not_filled = planned_symbols - filled_symbols
            if not_filled:
                logger.warning(f"Symbols not filled: {len(not_filled)} (e.g., {list(not_filled)[:5]})")

            # Target vs actual deployment
            target_deploy_pct = deployable / equity * 100
            logger.warning(
                f"Target deployment: ${deployable:,.2f} ({target_deploy_pct:.1f}% of equity) "
                f"-> Actual: ${total_deployed:,.2f} ({deployment_pct:.1f}%) "
                f"= Gap of ${deployable - total_deployed:,.2f}"
            )
            logger.warning("=== END SHORTFALL DIAGNOSTICS ===")

    except Exception as e:
        logger.exception(f"Error in entry execution: {e}")
        mark_entries_done_and_save()
