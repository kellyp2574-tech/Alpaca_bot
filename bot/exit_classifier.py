"""Exit Classifier — morning exit scheduling for overnight momentum positions.

At 9:35 AM, each overnight position is classified into one of two buckets:

  ret_open_to_935 > EXIT_UP_MOVE_PCT (config)   -> exit at 9:35  (up from open)
  otherwise                                      -> exit at 11:30 (hold)

Price sourcing priority:
  open_price  : 9:30 snapshot preferred; first minute-bar open as fallback
  price_935   : last minute-bar close (9:30-9:35) preferred; snapshot fallback
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from bot import config

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
# Exit time bucket constants (ET, 24h "HH:MM")
# ═══════════════════════════════════════════════════
EXIT_BUCKET_935  = "09:35"
EXIT_BUCKET_1130 = "11:30"

# Threshold is authoritative in config_strategy.EXIT_UP_MOVE_PCT
UP_MOVE_PCT = config.EXIT_UP_MOVE_PCT


@dataclass
class ExitClassification:
    """Per-symbol exit classification result."""
    symbol: str
    exit_time: str              # EXIT_BUCKET_935 or EXIT_BUCKET_1130
    open_price: float
    price_935: float
    move_5m_pct: float
    open_price_source: str      # "snapshot" | "minute_bar" | "missing"
    price_935_source: str       # "minute_bar" | "snapshot" | "missing"
    gap_pct: Optional[float] = None


# ═══════════════════════════════════════════════════
# Core classifier
# ═══════════════════════════════════════════════════

def get_exit_time(move_5m_pct: float) -> str:
    """Return EXIT_BUCKET_935 if move > UP_MOVE_PCT, else EXIT_BUCKET_1130."""
    if move_5m_pct > UP_MOVE_PCT:
        return EXIT_BUCKET_935
    return EXIT_BUCKET_1130


# ═══════════════════════════════════════════════════
# Batch classifier
# ═══════════════════════════════════════════════════

def classify_positions(
    symbols: List[str],
    open_prices: Dict[str, float],
    open_price_sources: Dict[str, str],
    snapshots_935: Dict[str, dict],
    minute_bars_935: Dict[str, List[dict]],
    entry_prices: Optional[Dict[str, float]] = None,
) -> Dict[str, ExitClassification]:
    """Classify all overnight positions at 9:35 AM.

    Args:
        symbols: List of position symbols.
        open_prices: {symbol: open_price} — already resolved by caller.
        open_price_sources: {symbol: "snapshot"|"minute_bar"|"missing"}.
        snapshots_935: {symbol: snapshot_dict} from 9:35 Alpaca snapshot.
        minute_bars_935: {symbol: [bar, ...]} 9:30-9:35 minute bars.
        entry_prices: {symbol: entry_price} for gap_pct logging (optional).

    Returns:
        {symbol: ExitClassification}
    """
    classifications: Dict[str, ExitClassification] = {}

    for symbol in symbols:
        open_price = open_prices.get(symbol)
        open_src   = open_price_sources.get(symbol, "missing")

        # ── Resolve price_935 ────────────────────────────────────────
        # Prefer last minute-bar close (matches backtest price_935 closely)
        bars = minute_bars_935.get(symbol, [])
        price_935: Optional[float] = None
        price_935_src = "missing"
        if bars:
            last_close = bars[-1].get("c")
            if last_close and last_close > 0:
                price_935     = last_close
                price_935_src = "minute_bar"
        if price_935 is None:
            snap = snapshots_935.get(symbol, {})
            snap_price = snap.get("last_price") or snap.get("close")
            if snap_price and snap_price > 0:
                price_935     = snap_price
                price_935_src = "snapshot"

        # ── Missing data — default to 11:30 ─────────────────────────
        if not open_price or not price_935 or open_price <= 0:
            logger.warning(
                f"EXIT CLASSIFY: {symbol} missing price data "
                f"(open={open_price} [{open_src}], "
                f"price_935={price_935} [{price_935_src}]) — defaulting to 11:30"
            )
            classifications[symbol] = ExitClassification(
                symbol=symbol,
                exit_time=EXIT_BUCKET_1130,
                open_price=open_price or 0.0,
                price_935=price_935 or 0.0,
                move_5m_pct=0.0,
                open_price_source=open_src,
                price_935_source=price_935_src,
            )
            continue

        move_5m_pct = (price_935 - open_price) / open_price * 100

        gap_pct = None
        entry_price = entry_prices.get(symbol) if entry_prices else None
        if entry_price and entry_price > 0:
            gap_pct = (open_price - entry_price) / entry_price * 100

        exit_time = get_exit_time(move_5m_pct)

        classifications[symbol] = ExitClassification(
            symbol=symbol,
            exit_time=exit_time,
            open_price=open_price,
            price_935=price_935,
            move_5m_pct=move_5m_pct,
            open_price_source=open_src,
            price_935_source=price_935_src,
            gap_pct=gap_pct,
        )

        gap_str = f" gap={gap_pct:+.2f}%" if gap_pct is not None else ""
        logger.info(
            f"EXIT CLASSIFY: {symbol} | "
            f"open={open_price:.4f}[{open_src}] "
            f"price_935={price_935:.4f}[{price_935_src}] "
            f"move={move_5m_pct:+.2f}%{gap_str} -> {exit_time}"
        )

    # ── Summary ──────────────────────────────────────────────────────
    buckets: Dict[str, List[str]] = {}
    for sym, cls in classifications.items():
        buckets.setdefault(cls.exit_time, []).append(sym)
    for bucket, syms in sorted(buckets.items()):
        logger.info(f"EXIT SCHEDULE: {bucket} -> {len(syms)} positions: {syms}")

    # Source quality summary
    fallback_open  = sum(1 for c in classifications.values() if c.open_price_source  == "minute_bar")
    fallback_935   = sum(1 for c in classifications.values() if c.price_935_source   == "snapshot")
    missing_any    = sum(1 for c in classifications.values() if "missing" in (c.open_price_source, c.price_935_source))
    n = len(classifications)
    logger.info(
        f"EXIT CLASSIFY sources: {n} total | "
        f"open fallbacks={fallback_open} | "
        f"price_935 snapshot fallbacks={fallback_935} | "
        f"missing={missing_any}"
    )
    if fallback_open > n // 2:
        logger.warning(
            f"EXIT CLASSIFY: {fallback_open}/{n} symbols using minute-bar open fallback — "
            f"classification may be less reliable today"
        )

    return classifications
