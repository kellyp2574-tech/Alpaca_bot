"""Universe Builder — staged filtering pipeline with diagnostics.

Stages:
  A. Eligibility: US common stock, active, tradable, no ETFs/warrants/units/OTC
  B. Liquidity: price range, ADV minimum
  C. Data quality: enough minute bars, enough daily bars, valid OHLCV
  D. Broker executability: Alpaca says tradable

Every stage logs counts and rejection reasons so you can audit
why the universe shrank on any given day.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Set, Tuple

from bot import config
from bot.market_data import AlpacaDataClient
from bot.massive_client import MassiveClient

logger = logging.getLogger(__name__)

# Symbols to always exclude (manual blacklist)
BLACKLIST: Set[str] = set()

# Asset types that are NOT common stock — reject these
_EXCLUDED_SUFFIXES = {
    ".W",   # warrants
    ".R",   # rights
    ".U",   # units
    ".WS",  # warrants (alt)
}


_MAX_REJECTION_SAMPLES = 5  # example symbols logged per rejection reason


@dataclass
class UniverseDiagnostics:
    """Tracks how many symbols survive each filter stage."""
    raw_symbols: int = 0
    after_asset_type: int = 0
    after_price: int = 0
    after_adv: int = 0
    after_data_quality: int = 0
    after_tradability: int = 0
    dropped_by_reason: Dict[str, int] = field(default_factory=dict)
    rejection_samples: Dict[str, List[str]] = field(default_factory=dict)

    def _inc(self, reason: str, count: int = 1, symbol: str = ""):
        self.dropped_by_reason[reason] = self.dropped_by_reason.get(reason, 0) + count
        if symbol:
            samples = self.rejection_samples.setdefault(reason, [])
            if len(samples) < _MAX_REJECTION_SAMPLES:
                samples.append(symbol)

    def summary(self) -> str:
        lines = [
            f"Universe pipeline: {self.raw_symbols} raw",
            f"  -> {self.after_asset_type} after asset-type filter",
            f"  -> {self.after_price} after price filter",
            f"  -> {self.after_adv} after ADV filter",
            f"  -> {self.after_tradability} after broker tradability filter",
            f"  -> {self.after_data_quality} after data-quality filter (final)",
        ]
        if self.dropped_by_reason:
            lines.append("  Rejections:")
            for reason, count in sorted(self.dropped_by_reason.items(), key=lambda x: -x[1]):
                samples = self.rejection_samples.get(reason, [])
                sample_str = f"  e.g. {', '.join(samples)}" if samples else ""
                lines.append(f"    {reason}: {count}{sample_str}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Stage A: Eligibility (asset type filtering)
# ──────────────────────────────────────────────────────────────

def filter_asset_type(
    assets: List[dict], diag: UniverseDiagnostics
) -> List[str]:
    """Filter Alpaca assets to US common stock only.

    Rejects ETFs, warrants, rights, units, preferreds, OTC names.
    Args:
        assets: Raw list from Alpaca GET /v2/assets (each is a dict).
    Returns:
        List of symbols that pass.
    """
    passed = []
    for asset in assets:
        symbol = asset.get("symbol", "")
        asset_class = asset.get("class", "")
        exchange = asset.get("exchange", "")
        status = asset.get("status", "")
        tradable = asset.get("tradable", False)
        name = (asset.get("name") or "").lower()

        # Must be active + tradable US equity
        if asset_class != "us_equity":
            diag._inc("not_us_equity", symbol=symbol)
            continue
        if status != "active":
            diag._inc("not_active", symbol=symbol)
            continue
        if not tradable:
            diag._inc("not_tradable", symbol=symbol)
            continue

        # Reject OTC / pink sheets
        if exchange in ("OTC", "OTCBB", "PINK"):
            diag._inc("otc_exchange", symbol=symbol)
            continue

        # Reject ETFs (name heuristic — Alpaca doesn't have a clean "is_etf" flag)
        if any(tag in name for tag in ("etf", "fund", "trust", "proshares", "ishares", "vanguard", "spdr")):
            diag._inc("etf_or_fund", symbol=symbol)
            continue

        # Reject warrants, rights, units by suffix
        if any(symbol.endswith(suffix) for suffix in _EXCLUDED_SUFFIXES):
            diag._inc("warrant_right_unit", symbol=symbol)
            continue

        # Reject preferred stock (e.g. "BAC-L") and class shares (e.g. "BRK.B").
        # The blanket ``len(symbol) > 5`` rule used to live here too but it
        # rejected legitimate 6-letter common-stock tickers (e.g. some IEX-only
        # microcaps). Stage D's broker tradability check is a better catch-all
        # for any genuinely non-tradeable symbol.
        if "-" in symbol or "." in symbol:
            diag._inc("preferred_or_class_share", symbol=symbol)
            continue

        # Reject blacklisted
        if symbol in BLACKLIST:
            diag._inc("blacklisted", symbol=symbol)
            continue

        passed.append(symbol)

    diag.after_asset_type = len(passed)
    return passed


# ──────────────────────────────────────────────────────────────
# Stage B: Liquidity (price + ADV)
# ──────────────────────────────────────────────────────────────

def filter_price(
    symbols: List[str],
    snapshots: Dict[str, dict],
    min_price: float,
    max_price: float,
    diag: UniverseDiagnostics,
) -> List[str]:
    """Filter by current price range using Massive snapshot data."""
    passed = []
    for symbol in symbols:
        snap = snapshots.get(symbol)
        if not snap:
            diag._inc("no_snapshot", symbol=symbol)
            continue
        price = snap.get("price") or snap.get("last_price") or 0
        if price < min_price:
            diag._inc("price_too_low", symbol=symbol)
            continue
        if price > max_price:
            diag._inc("price_too_high", symbol=symbol)
            continue
        passed.append(symbol)
    diag.after_price = len(passed)
    return passed


def filter_adv(
    symbols: List[str],
    daily_bars: Dict[str, List[dict]],
    adv_lookback: int,
    diag: UniverseDiagnostics,
) -> Tuple[List[str], Dict[str, Tuple[float, float]]]:
    """Build the per-symbol ADV cache from daily bars.

    The hard ADV-dollars gate has been removed because the live sizer
    already protects via ``ADV_CAP_PCT`` (each position is capped at
    0.3% of the symbol's ADV). Symbols with no daily bars are still
    rejected because we cannot size them.

    Returns:
        (passed_symbols, adv_cache {sym: (shares, dollars)})
    """
    passed = []
    adv_cache: Dict[str, Tuple[float, float]] = {}

    for symbol in symbols:
        bars = daily_bars.get(symbol, [])
        if not bars:
            diag._inc("no_daily_bars", symbol=symbol)
            continue

        adv_shares, adv_dollars = AlpacaDataClient.calculate_adv(bars, adv_lookback)
        adv_cache[symbol] = (adv_shares, adv_dollars)
        passed.append(symbol)

    diag.after_adv = len(passed)
    return passed, adv_cache


# ──────────────────────────────────────────────────────────────
# Stage C: Data quality
# ──────────────────────────────────────────────────────────────

def filter_minute_data_quality(
    symbols: List[str],
    minute_bars: Dict[str, List[dict]],
    min_minute_bars: int = 30,
    diag: Optional[UniverseDiagnostics] = None,
) -> List[str]:
    """Reject symbols without enough intraday minute bars or with invalid OHLCV.

    This is the primary quality gate at 3:48 PM after signal bars are fetched.
    """
    if diag is None:
        diag = UniverseDiagnostics()

    passed = []
    for symbol in symbols:
        mbars = minute_bars.get(symbol, [])

        if len(mbars) < min_minute_bars:
            diag._inc("too_few_minute_bars", symbol=symbol)
            continue

        first = mbars[0]
        last = mbars[-1]
        if not first.get("o") or first["o"] <= 0:
            diag._inc("invalid_open_price", symbol=symbol)
            continue
        if not last.get("c") or last["c"] <= 0:
            diag._inc("invalid_close_price", symbol=symbol)
            continue
        total_vol = sum(b.get("v", 0) for b in mbars)
        if total_vol <= 0:
            diag._inc("zero_intraday_volume", symbol=symbol)
            continue

        passed.append(symbol)

    diag.after_data_quality = len(passed)
    return passed


def filter_daily_data_quality(
    symbols: List[str],
    daily_bars: Dict[str, List[dict]],
    min_daily_bars: int = 10,
    diag: Optional[UniverseDiagnostics] = None,
) -> List[str]:
    """Reject symbols without enough daily bar history."""
    if diag is None:
        diag = UniverseDiagnostics()

    passed = []
    for symbol in symbols:
        dbars = daily_bars.get(symbol, [])
        if len(dbars) < min_daily_bars:
            diag._inc("too_few_daily_bars", symbol=symbol)
            continue
        passed.append(symbol)

    return passed


def filter_full_data_quality(
    symbols: List[str],
    minute_bars: Dict[str, List[dict]],
    daily_bars: Dict[str, List[dict]],
    min_minute_bars: int = 30,
    min_daily_bars: int = 10,
    diag: Optional[UniverseDiagnostics] = None,
) -> List[str]:
    """Convenience wrapper: run both minute + daily quality in one pass."""
    daily_passed = filter_daily_data_quality(
        symbols, daily_bars, min_daily_bars, diag,
    )
    return filter_minute_data_quality(
        daily_passed, minute_bars, min_minute_bars, diag,
    )


# ──────────────────────────────────────────────────────────────
# Stage D: Broker executability
# ──────────────────────────────────────────────────────────────

def filter_broker_tradable(
    symbols: List[str],
    tradable_set: Set[str],
    diag: UniverseDiagnostics,
) -> List[str]:
    """Final filter: only keep symbols the broker confirms are tradable."""
    if not tradable_set:
        logger.warning("Tradable set empty — skipping broker filter")
        diag.after_tradability = len(symbols)
        return symbols

    passed = []
    for symbol in symbols:
        if symbol in tradable_set:
            passed.append(symbol)
        else:
            diag._inc("broker_not_tradable", symbol=symbol)
    diag.after_tradability = len(passed)
    return passed


# ──────────────────────────────────────────────────────────────
# Execution eligibility (last-mile, right before order submission)
# ──────────────────────────────────────────────────────────────

def filter_execution_ready(
    symbols: List[str],
    snapshots: Dict[str, dict],
    max_spread_pct: float = 0.05,
    require_quote: bool = True,
    max_stale_seconds: float = 300.0,
    min_quote_size: int = 0,
) -> Tuple[List[str], Dict[str, str]]:
    """Last-mile filter: only keep symbols that are actually orderable *right now*.

    Checks:
      - Has snapshot with last_price > 0
      - Has latest quote (bid + ask both present) if require_quote
      - Spread not absurd (bid-ask / midpoint < max_spread_pct)
      - Last trade not stale (timestamp age < max_stale_seconds)
      - Bid/ask size >= min_quote_size (if > 0)

    Args:
        symbols: Candidate symbols after scoring/selection.
        snapshots: Fresh Alpaca snapshots keyed by symbol.
        max_spread_pct: Maximum bid-ask spread as fraction of midpoint (default 5%).
        require_quote: If True, reject symbols missing bid/ask entirely.
        max_stale_seconds: Reject if last trade timestamp > this many seconds old.
        min_quote_size: Minimum bid_size / ask_size to accept (0 = no check).

    Returns:
        (orderable_symbols, rejected {symbol: reason})
    """
    from datetime import datetime, timezone

    orderable = []
    rejected: Dict[str, str] = {}
    now_utc = datetime.now(timezone.utc)

    for symbol in symbols:
        snap = snapshots.get(symbol)
        if not snap:
            rejected[symbol] = "no_snapshot"
            continue

        last_price = snap.get("last_price") or snap.get("close") or 0
        if not last_price or last_price <= 0:
            rejected[symbol] = "no_last_price"
            continue

        # Staleness check
        ts_raw = snap.get("timestamp") or snap.get("last_trade_timestamp")
        if ts_raw and max_stale_seconds > 0:
            try:
                if isinstance(ts_raw, str):
                    # Handle ISO format with or without trailing Z
                    ts_str = ts_raw.replace("Z", "+00:00")
                    ts = datetime.fromisoformat(ts_str)
                else:
                    ts = ts_raw  # already a datetime
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age = (now_utc - ts).total_seconds()
                if age > max_stale_seconds:
                    rejected[symbol] = f"stale_quote_{age:.0f}s"
                    continue
            except (ValueError, TypeError) as e:
                # A parse failure means we cannot prove freshness either way.
                # Log explicitly so a feed-format change is visible instead of
                # silently disabling the staleness gate.
                logger.warning(
                    f"filter_execution_ready: cannot parse snapshot timestamp "
                    f"for {symbol} ({ts_raw!r}): {e}"
                )

        bid = snap.get("bid")
        ask = snap.get("ask")

        if require_quote and (not bid or not ask):
            rejected[symbol] = "missing_bid_ask"
            continue

        if bid and ask and bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            spread_pct = (ask - bid) / mid if mid > 0 else 999
            if spread_pct > max_spread_pct:
                rejected[symbol] = f"wide_spread_{spread_pct:.1%}"
                continue

        # Quote size check
        if min_quote_size > 0:
            bid_size = snap.get("bid_size") or 0
            ask_size = snap.get("ask_size") or 0
            if bid_size < min_quote_size or ask_size < min_quote_size:
                rejected[symbol] = f"thin_quote_bid{bid_size}_ask{ask_size}"
                continue

        orderable.append(symbol)

    if rejected:
        samples = list(rejected.items())[:10]
        logger.info(
            f"Execution eligibility: {len(orderable)} orderable, "
            f"{len(rejected)} rejected. Samples: {samples}"
        )
    else:
        logger.info(f"Execution eligibility: all {len(orderable)} symbols orderable")

    return orderable, rejected


# ──────────────────────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────────────────────

def build_universe(
    massive: MassiveClient,
    alpaca: AlpacaDataClient,
) -> Tuple[List[str], UniverseDiagnostics, Dict[str, Tuple[float, float]]]:
    """Run the base universe pipeline (Stages A + B + D).

    Stage C (data quality) is intentionally NOT run here because minute bars
    are not available at 3:30 PM.  The orchestrator runs Stage C at 3:48
    after fetching signal bars, using ``filter_minute_data_quality()``.

    Returns:
        (base_universe, diagnostics, adv_cache)
    """
    diag = UniverseDiagnostics()

    # Stage A: Fetch all Alpaca assets and filter by type
    logger.info("Universe Stage A: asset-type eligibility...")
    raw_assets = alpaca.get_tradable_assets_full()
    diag.raw_symbols = len(raw_assets)
    eligible = filter_asset_type(raw_assets, diag)
    logger.info(f"  Stage A: {diag.raw_symbols} -> {diag.after_asset_type}")

    # Stage B-price: Price filter using Massive snapshot
    logger.info("Universe Stage B: price filter via Massive snapshot...")
    massive_snapshots = massive.get_full_market_snapshot()
    if not massive_snapshots:
        logger.error("Massive snapshot failed — using Alpaca snapshots as fallback")
        massive_snapshots = {}
        for i in range(0, len(eligible), 1000):
            chunk = eligible[i:i + 1000]
            snaps = alpaca.get_snapshots(chunk)
            for sym, s in snaps.items():
                massive_snapshots[sym] = {"price": s.get("last_price") or s.get("close") or 0}

    price_passed = filter_price(
        eligible, massive_snapshots,
        config.MIN_PRICE, config.MAX_PRICE, diag,
    )
    logger.info(f"  Stage B-price: {diag.after_asset_type} -> {diag.after_price}")

    # Stage B-adv: build ADV cache from daily bars (sizing uses ADV_CAP_PCT)
    logger.info("Universe Stage B: ADV cache build...")
    daily_bars = alpaca.get_daily_bars(
        price_passed,
        days=config.ADV_LOOKBACK_DAYS + 5,
    )
    adv_passed, adv_cache = filter_adv(
        price_passed, daily_bars, config.ADV_LOOKBACK_DAYS, diag,
    )
    logger.info(f"  Stage B-adv: {diag.after_price} -> {diag.after_adv}")

    # Stage C: deferred — run by orchestrator at 3:48 after minute bars arrive
    diag.after_data_quality = len(adv_passed)

    # Stage D: Broker tradability check — reuses the Stage A asset list.
    # The list is at most a couple of minutes old at this point and re-fetching
    # adds an unnecessary round-trip on a payload of thousands of assets.
    logger.info("Universe Stage D: broker tradability check (reusing Stage A asset list)...")
    tradable_set: Set[str] = {
        a["symbol"] for a in raw_assets
        if a.get("tradable") and a.get("status") == "active"
    }
    final = filter_broker_tradable(adv_passed, tradable_set, diag)
    logger.info(f"  Stage D: {diag.after_adv} -> {diag.after_tradability}")

    logger.info(diag.summary())
    return final, diag, adv_cache


# ──────────────────────────────────────────────────────────────
# Audit artifact
# ──────────────────────────────────────────────────────────────

def save_universe_audit(
    diag: UniverseDiagnostics,
    final_symbols: List[str],
    scored_top20: Optional[List[dict]] = None,
    deployed_estimate: float = 0.0,
):
    """Save daily universe audit to state/logs/universe_YYYY-MM-DD.json."""
    today = date.today().isoformat()
    audit = {
        "date": today,
        "raw_symbols": diag.raw_symbols,
        "after_asset_type": diag.after_asset_type,
        "after_price": diag.after_price,
        "after_adv": diag.after_adv,
        "after_tradability": diag.after_tradability,
        "after_data_quality": diag.after_data_quality,
        "final_count": len(final_symbols),
        "dropped_by_reason": diag.dropped_by_reason,
        "rejection_samples": diag.rejection_samples,
        "deployed_estimate": deployed_estimate,
    }
    if scored_top20:
        audit["top_20_scored"] = scored_top20

    path = os.path.join(config.LOG_DIR, f"universe_{today}.json")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(audit, f, indent=2)
        logger.info(f"Universe audit saved: {path}")
    except Exception as e:
        logger.error(f"Failed to save universe audit: {e}")


def save_run_health(
    diag: Optional[UniverseDiagnostics],
    scored_count: int = 0,
    selected_count: int = 0,
    orderable_count: int = 0,
    filled_count: int = 0,
    total_deployed: float = 0.0,
    equity: float = 0.0,
    exec_rejected: Optional[Dict[str, str]] = None,
    fallback_used: bool = False,
    extra: Optional[dict] = None,
):
    """Save daily bot health report to state/logs/run_health_YYYY-MM-DD.json.

    This is the ONE file you inspect first after each session.
    """
    today = date.today().isoformat()
    report: Dict = {"date": today}

    # Universe pipeline counts
    if diag:
        report["universe"] = {
            "raw": diag.raw_symbols,
            "after_asset_type": diag.after_asset_type,
            "after_price": diag.after_price,
            "after_adv": diag.after_adv,
            "after_tradability": diag.after_tradability,
            "after_data_quality": diag.after_data_quality,
            "dropped_by_reason": diag.dropped_by_reason,
            "rejection_samples": diag.rejection_samples,
        }

    report["scoring"] = {"scored_candidates": scored_count}
    report["selection"] = {"selected": selected_count, "orderable": orderable_count}
    report["execution"] = {
        "filled": filled_count,
        "total_deployed": round(total_deployed, 2),
        "equity": round(equity, 2),
        "deployment_pct": round(total_deployed / equity * 100, 1) if equity > 0 else 0,
    }
    if exec_rejected:
        report["execution"]["rejected_reasons"] = exec_rejected
    report["fallback_used"] = fallback_used
    if extra:
        report.update(extra)

    path = os.path.join(config.LOG_DIR, f"run_health_{today}.json")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Run health report saved: {path}")
    except Exception as e:
        logger.error(f"Failed to save run health report: {e}")


def save_candidates_audit(candidates):
    """Save daily scored candidates to state/logs/candidates_YYYY-MM-DD.json.

    Args:
        candidates: Either a List[dict] (legacy, single ranked list) or a
                    dict with named ranked views, e.g.:
                    {
                      "top_20_by_head_score": [...],
                      "top_20_by_composite_score": [...],
                    }
    """
    today = date.today().isoformat()
    path = os.path.join(config.LOG_DIR, f"candidates_{today}.json")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"date": today, "candidates": candidates}, f, indent=2)
        logger.info(f"Candidates audit saved: {path}")
    except Exception as e:
        logger.error(f"Failed to save candidates audit: {e}")


# ──────────────────────────────────────────────────────────────
# Execution diagnostics
# ──────────────────────────────────────────────────────────────

@dataclass
class ExecutionDiagnostics:
    """Tracks every step from selection -> fill for post-day analysis."""
    selected_symbols: List[str] = field(default_factory=list)
    orderable_symbols: List[str] = field(default_factory=list)
    rejected_symbols: Dict[str, str] = field(default_factory=dict)
    submitted_symbols: List[str] = field(default_factory=list)
    filled_symbols: List[str] = field(default_factory=list)
    failed_submissions: Dict[str, str] = field(default_factory=dict)
    fill_details: Dict[str, dict] = field(default_factory=dict)
    sizing_diagnostics: Dict[str, dict] = field(default_factory=dict)  # Per-symbol sizing info

    def to_dict(self) -> dict:
        return {
            "selected": self.selected_symbols,
            "orderable": self.orderable_symbols,
            "rejected_at_gate": self.rejected_symbols,
            "submitted": self.submitted_symbols,
            "filled": self.filled_symbols,
            "failed_submissions": self.failed_submissions,
            "fill_details": self.fill_details,
            "sizing": self.sizing_diagnostics,
            "counts": {
                "selected": len(self.selected_symbols),
                "orderable": len(self.orderable_symbols),
                "rejected_at_gate": len(self.rejected_symbols),
                "submitted": len(self.submitted_symbols),
                "filled": len(self.filled_symbols),
                "failed_submissions": len(self.failed_submissions),
            },
        }


def save_execution_audit(exec_diag: ExecutionDiagnostics):
    """Save daily execution diagnostics to state/logs/execution_YYYY-MM-DD.json."""
    today = date.today().isoformat()
    path = os.path.join(config.LOG_DIR, f"execution_{today}.json")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {"date": today}
        payload.update(exec_diag.to_dict())
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Execution audit saved: {path}")
    except Exception as e:
        logger.error(f"Failed to save execution audit: {e}")
