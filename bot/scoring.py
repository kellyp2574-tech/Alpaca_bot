"""Afternoon scoring + risk gating — extracted from integrated_main.py.

Owns the ~3:30 PM universe build, the 3:50 PM Stage C quality filter + MR/GDP
candidate construction, the MR ETF regime size multiplier, and the global
daily-loss kill switch evaluation.

Every function takes the orchestrator instance (`bot`) as its first argument
and mutates its state directly. State ownership stays with
`CombinedOvernightReboundBot`; this module only houses the logic.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional, Tuple

from bot import config
from bot.mean_reversion_scorer import (
    build_mean_reversion_candidates,
    filter_mean_reversion_candidates,
)
from bot.universe_builder import (
    build_universe,
    filter_minute_data_quality,
    save_universe_audit,
    save_candidates_audit,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Stage A/B/C: universe + candidates
# ────────────────────────────────────────────────────────────────────

def step_collect_data(bot) -> None:
    """~3:30 PM: Build universe (price/ADV/daily-bar filters). Stage C minute-quality runs at 3:50."""
    logger.info("=" * 50)
    logger.info("DATA COLLECTION: Building base universe (staged pipeline)")
    logger.info("=" * 50)

    # Free MR pipeline: skip paid/live universe build - watchlist built at scoring time
    if getattr(config, "USE_FREE_MR_PIPELINE", False):
        logger.info("FREE MR PIPELINE enabled: skipping paid/live universe build at 15:30")
        bot.universe = []
        bot._universe_diag = None
        bot._adv_cache = {}
        bot.data_collected = True
        bot._save_state()
        return

    try:
        final, diag, adv_cache = build_universe(bot.massive, bot.alpaca)

        bot.universe = final
        bot._universe_diag = diag
        bot._adv_cache = adv_cache

        if not bot.universe:
            logger.error("Empty universe after pipeline — cannot proceed")
            return

        save_universe_audit(diag, final)

        bot.data_collected = True
        bot._save_state()
        logger.info(f"Base universe ready: {len(bot.universe)} symbols (Stage C minute-quality runs at 3:50)")

    except Exception as e:
        logger.exception(f"Error in data collection: {e}")


def step_score_and_rank(bot) -> None:
    """~15:45: Fetch 9:30-15:45 bars, run Stage C quality filter, build MR candidates."""
    logger.info("=" * 50)
    logger.info("SCORING: Building MR candidates")
    logger.info("=" * 50)

    # Free MR pipeline: Massive previous-day + Alpaca live snapshots
    if getattr(config, "USE_FREE_MR_PIPELINE", False):
        from bot.mr_free_data_pipeline import (
            build_massive_prevday_watchlist,
            build_live_mr_candidates_from_free_pipeline,
            convert_live_to_existing_mr_candidate,
            previous_weekday,
        )

        watchlist = build_massive_prevday_watchlist(
            massive_api_key=config.MASSIVE_API_KEY,
            trade_date=previous_weekday(),
            min_prev_close=float(getattr(config, "MR_FREE_PREV_MIN_PRICE", 0.75)),
            max_prev_close=float(getattr(config, "MR_FREE_PREV_MAX_PRICE", 3.00)),
            min_prev_dollar_volume=float(getattr(config, "MR_FREE_PREV_MIN_DOLLAR_VOLUME", 500_000)),
            apply_prior_ret_filter=True,  # Apply prior-day return filter for overnight MR
        )

        live_candidates = build_live_mr_candidates_from_free_pipeline(
            watchlist=watchlist,
            alpaca_data_key=config.ALPACA_API_KEY,
            alpaca_data_secret=config.ALPACA_SECRET_KEY,
            feed=getattr(config, "DATA_FEED", "iex"),
        )

        bot.mr_candidates = [
            convert_live_to_existing_mr_candidate(c)
            for c in live_candidates
        ]

        bot.scoring_done = True
        bot._save_state()

        # Audit logging for debugging
        def _free_mr_dict(c):
            return {
                "symbol": c.symbol,
                "sleeve": "MR_FREE",
                "selection_score": round(c.selection_score, 4),
                "signal_price": round(c.signal_price, 4),
                "day_return": round(c.day_return, 4),
                "volume_ratio": round(c.volume_ratio, 2),
                "close_position": round(c.close_position, 3),
                "late_drop_1530_1550": round(c.late_drop_1530_1550, 4),
                "adv_dollars": round(c.adv_dollars, 0),
                "prior_ret": round(c.prior_ret, 4) if c.prior_ret is not None else None,
                "source": "massive_prevday_plus_alpaca_snapshot",
            }

        audit_dicts = {
            "mr_selected": [_free_mr_dict(c) for c in bot.mr_candidates[:getattr(config, "MR_MAX_PRIMARY_POSITIONS", 3)]],
            "mr_all_passed": [_free_mr_dict(c) for c in bot.mr_candidates],
        }
        save_candidates_audit(audit_dicts)

        logger.info(
            f"FREE MR PIPELINE: watchlist={len(watchlist)} "
            f"passed={len(bot.mr_candidates)}"
        )
        return

    try:
        today = date.today().isoformat()
        signal_end = config.ENTRY_TIME  # usually 15:45 in current paper/live config

        # 1. Fetch 9:30-15:50 minute bars for the full base universe AND
        # the MR ETF regime symbols in a single batched request (API #2:
        # avoids a separate get_intraday_bars_for_signal call inside
        # compute_mr_etf_regime_size_multiplier).
        regime_symbols = []
        if getattr(config, "ENABLE_MR_ETF_REGIME_SIZING", False):
            regime_symbols = list(
                getattr(config, "MR_ETF_REGIME_SYMBOLS", ["SPY", "IWM", "QQQ"])
            )
        fetch_symbols = list(bot.universe)
        for s in regime_symbols:
            if s not in fetch_symbols:
                fetch_symbols.append(s)

        logger.info(
            f"Fetching 9:30-{signal_end} minute bars for {len(fetch_symbols)} symbols "
            f"({len(bot.universe)} universe + {len(regime_symbols)} regime ETFs)..."
        )
        bot._minute_bars = bot.alpaca.get_intraday_bars_for_signal(
            fetch_symbols, today, start="09:30", end=signal_end,
        )

        # Log signal bar timestamps to verify data recency in live trading
        sample_last_times = []
        for sym, bars in list(bot._minute_bars.items())[:20]:
            if bars:
                sample_last_times.append((sym, bars[-1].get("t")))
        logger.info(f"Signal bar timestamp samples: {sample_last_times[:10]}")

        # 2. Stage C: minute-bar data quality filter
        pre_c_count = len(bot.universe)
        quality_passed = filter_minute_data_quality(
            bot.universe,
            bot._minute_bars,
            min_minute_bars=30,
            diag=bot._universe_diag,
        )
        logger.info(f"Stage C data quality: {pre_c_count} -> {len(quality_passed)}")
        bot.universe = quality_passed

        if not bot.universe:
            logger.error("Empty universe after Stage C data quality — cannot score")
            bot.scoring_done = True
            return

        # 3. Build raw MR candidates from minute bars
        raw_mr = build_mean_reversion_candidates(
            bot.universe,
            bot._minute_bars,
            bot._adv_cache,
        )
        filtered_mr = filter_mean_reversion_candidates(raw_mr)

        # Paper-test: enforce min ADV filter explicitly.
        # Live Alpaca IEX volume is much lower than composite volume, so apply the
        # same multiplier used for ADV sizing/caps before comparing to the liquidity floor.
        min_adv = float(getattr(config, "MR_MIN_AVG_DOLLAR_VOLUME", 0) or 0)
        adv_multiplier = float(getattr(config, "ADV_DOLLAR_MULTIPLIER", 1.0) or 1.0)

        if min_adv > 0:
            before_adv = len(filtered_mr)
            adv_passed = []
            adv_rejected_sample = []

            for c in filtered_mr:
                raw_adv = float(getattr(c, "adv_dollars", 0.0) or 0.0)
                adjusted_adv = raw_adv * adv_multiplier

                if adjusted_adv >= min_adv:
                    adv_passed.append(c)
                elif len(adv_rejected_sample) < 10:
                    adv_rejected_sample.append({
                        "symbol": c.symbol,
                        "raw_adv_dollars": round(raw_adv, 0),
                        "adjusted_adv_dollars": round(adjusted_adv, 0),
                        "min_required": round(min_adv, 0),
                    })

            filtered_mr = adv_passed

            logger.info(
                "MR ADV filter: %d -> %d using raw_adv * %.1f >= $%.0f",
                before_adv,
                len(filtered_mr),
                adv_multiplier,
                min_adv,
            )

            if adv_rejected_sample:
                logger.info("MR ADV rejected sample: %s", adv_rejected_sample)

        # Paper-test: rank by lowest close_position when configured
        if getattr(config, "MR_RANK_BY_CLOSE_LOCATION_ONLY", False):
            filtered_mr = sorted(
                filtered_mr,
                key=lambda c: (
                    getattr(c, "close_position", 999.0),
                    getattr(c, "day_return", 0.0),
                )
            )

        bot.mr_candidates = filtered_mr  # Keep ALL passed candidates for allocator

        # CRITICAL DIAGNOSTIC: MR pipeline counts
        logger.info(
            f"MR pipeline: "
            f"universe={len(bot.universe)}, "
            f"raw={len(raw_mr)}, "
            f"passed={len(filtered_mr)}"
        )

        # Better "why no MR trade" log when final candidate list is empty
        if not filtered_mr:
            logger.warning(
                "MR NO-CANDIDATES after filters: raw=%d, stage_c_universe=%d, "
                "min_adv=$%.0f, adv_multiplier=%.1f, price_range=$%.2f-$%.2f, "
                "day_ret_max=%.2f, close_pos_max=%.2f",
                len(raw_mr),
                len(bot.universe),
                float(getattr(config, "MR_MIN_AVG_DOLLAR_VOLUME", 0) or 0),
                float(getattr(config, "ADV_DOLLAR_MULTIPLIER", 1.0) or 1.0),
                float(getattr(config, "MR_MIN_PRICE", 0) or 0),
                float(getattr(config, "MR_MAX_PRICE", 0) or 0),
                float(getattr(config, "MR_DAY_RET_MAX", 0) or 0),
                float(getattr(config, "MR_CLOSE_POSITION_MAX", 0) or 0),
            )

        bot.scoring_done = True
        bot._save_state()

        logger.info(
            f"MR scoring: raw={len(raw_mr)} passed={len(filtered_mr)} "
            f"top_slots={min(len(filtered_mr), config.MR_MAX_PRIMARY_POSITIONS)}"
        )

        # Log top MR candidates (only up to max positions)
        for c in bot.mr_candidates[:config.MR_MAX_PRIMARY_POSITIONS]:
            logger.info(
                f"MR TOP {c.symbol}: price={c.signal_price:.2f}, "
                f"day_ret={c.day_return:.2%}, vol_ratio={c.volume_ratio:.2f}x, "
                f"close_pos={c.close_position:.2f}, "
                f"late_drop={c.late_drop_1530_1550:.2%}, "
                f"score={c.selection_score:.3f}"
            )

        # Save candidates audit artifact
        def _mr_dict(c):
            raw_adv = float(getattr(c, "adv_dollars", 0.0) or 0.0)
            adv_multiplier = float(getattr(config, "ADV_DOLLAR_MULTIPLIER", 1.0) or 1.0)
            adjusted_adv = raw_adv * adv_multiplier

            return {
                "symbol": c.symbol,
                "sleeve": "MR",
                "selection_score": round(c.selection_score, 4),
                "signal_price": round(c.signal_price, 4),
                "day_return": round(c.day_return, 4),
                "volume_ratio": round(c.volume_ratio, 2),
                "close_position": round(c.close_position, 3),
                "late_drop_1530_1550": round(c.late_drop_1530_1550, 4),
                "adv_dollars_raw": round(raw_adv, 0),
                "adv_multiplier": adv_multiplier,
                "adv_dollars_adjusted": round(adjusted_adv, 0),
            }

        audit_dicts = {
            "mr_selected": [_mr_dict(c) for c in bot.mr_candidates[:config.MR_MAX_PRIMARY_POSITIONS]],
            "mr_all_passed": [_mr_dict(c) for c in bot.mr_candidates],
        }
        save_candidates_audit(audit_dicts)

        if bot._universe_diag:
            top_mr = [_mr_dict(c) for c in bot.mr_candidates[:20]]
            save_universe_audit(
                bot._universe_diag, bot.universe,
                scored_top20=top_mr,
            )

    except Exception as e:
        logger.exception(f"Error in scoring: {e}")
        bot.scoring_done = True


# ────────────────────────────────────────────────────────────────────
# Risk gating
# ────────────────────────────────────────────────────────────────────

def check_daily_loss_kill_switch(bot, account: Optional[dict] = None) -> bool:
    """Evaluate the global daily-loss kill switch.

    Trips ``bot.kill_switch_tripped`` (and persists state) when
    ``(equity - last_equity) / last_equity <= -DAILY_LOSS_LIMIT_PCT``.
    Once tripped, the flag remains True for the rest of the session and
    blocks both the 10:00 ETF router entry and the 15:45 MR entries.

    Args:
        account: Optional pre-fetched account dict (avoids redundant
            API calls when the caller already has one).

    Returns:
        True if the kill switch is tripped (either previously or just
        now), False otherwise.
    """
    if bot.kill_switch_tripped:
        return True

    loss_limit = float(getattr(config, "DAILY_LOSS_LIMIT_PCT", 0.0) or 0.0)
    if loss_limit <= 0:
        return False

    try:
        acct = account if account is not None else bot.position_mgr.get_account()
    except Exception:
        logger.warning("kill-switch: account fetch failed; cannot evaluate", exc_info=True)
        return False
    if not acct:
        return False

    try:
        equity = float(acct.get("equity") or 0.0)
        last_equity = float(acct.get("last_equity") or 0.0)
    except (TypeError, ValueError):
        return False
    if equity <= 0 or last_equity <= 0:
        return False

    day_ret = (equity - last_equity) / last_equity
    if day_ret <= -loss_limit:
        bot.kill_switch_tripped = True
        bot.kill_switch_reason = (
            f"day_ret={day_ret:+.2%} <= -{loss_limit:.0%} "
            f"(equity=${equity:,.2f} vs last_equity=${last_equity:,.2f})"
        )
        logger.critical(
            f"DAILY LOSS KILL SWITCH TRIPPED — {bot.kill_switch_reason}. "
            f"All new entries (ETF router + MR) BLOCKED for the rest of the session."
        )
        try:
            bot._save_state()
        except Exception:
            logger.warning("kill-switch: state save failed", exc_info=True)
        return True

    return False


# ────────────────────────────────────────────────────────────────────
# MR ETF regime size multiplier
# ────────────────────────────────────────────────────────────────────

def compute_mr_etf_regime_size_multiplier(bot) -> Tuple[float, dict]:
    """Return MR size multiplier from SPY/IWM/QQQ avg return vs open.

    Uses the same late-day timing as the paper-test signal. If ETF data is
    unavailable, fail open at full size but log the reason so the paper test
    does not silently skip otherwise valid MR candidates.
    """
    if not getattr(config, "ENABLE_MR_ETF_REGIME_SIZING", False):
        return 1.0, {"enabled": False, "reason": "disabled"}

    symbols = list(getattr(config, "MR_ETF_REGIME_SYMBOLS", ["SPY", "IWM", "QQQ"]))
    today = date.today().isoformat()
    end_time = getattr(config, "ENTRY_TIME", "15:45")

    # Reuse the minute bars already fetched for the universe at 15:45
    # (API #2). step_score_and_rank prepends the regime ETFs to the
    # batched fetch, so SPY/IWM/QQQ should already be in bot._minute_bars.
    # Fall back to a separate fetch only if the cache is missing data
    # (e.g. step_score_and_rank was skipped or returned early).
    cached = bot._minute_bars or {}
    missing = [s for s in symbols if not cached.get(s)]
    if missing:
        logger.info(
            f"MR ETF regime sizing: {len(symbols) - len(missing)} cached, "
            f"fetching {missing} from API"
        )
        try:
            extra = bot.alpaca.get_intraday_bars_for_signal(
                missing, today, start="09:30", end=end_time,
            )
            bars_by_symbol = dict(cached)
            bars_by_symbol.update(extra)
        except Exception:
            logger.warning("MR ETF regime sizing: failed to fetch ETF bars; using full size", exc_info=True)
            return 1.0, {"enabled": True, "reason": "fetch_failed"}
    else:
        bars_by_symbol = cached

    returns = {}

    def _bar_val(bar: dict, *keys: str) -> Optional[float]:
        for key in keys:
            try:
                val = bar.get(key)
                if val is not None:
                    return float(val)
            except (TypeError, ValueError, AttributeError):
                continue
        return None

    for sym in symbols:
        bars = bars_by_symbol.get(sym, []) if bars_by_symbol else []
        if not bars:
            continue
        first = bars[0]
        last = bars[-1]
        open_px = _bar_val(first, "o", "open")
        close_px = _bar_val(last, "c", "close")
        if open_px and close_px and open_px > 0:
            returns[sym] = close_px / open_px - 1.0

    if not returns:
        logger.warning("MR ETF regime sizing: no usable ETF bars; using full size")
        return 1.0, {"enabled": True, "reason": "no_usable_bars"}

    avg_ret = sum(returns.values()) / len(returns)
    is_negative = avg_ret < 0
    mult = (
        float(getattr(config, "MR_ETF_NEGATIVE_SIZE_MULT", 1.0))
        if is_negative
        else float(getattr(config, "MR_ETF_POSITIVE_SIZE_MULT", 0.5))
    )
    info = {
        "enabled": True,
        "reason": "ok",
        "avg_return": avg_ret,
        "is_negative": is_negative,
        "returns": returns,
        "multiplier": mult,
    }
    logger.warning(
        "MR ETF REGIME SIZE: avg=%+.2f%% multiplier=%.2f returns=%s",
        avg_ret * 100.0,
        mult,
        {k: f"{v:+.2%}" for k, v in returns.items()},
    )
    return mult, info
