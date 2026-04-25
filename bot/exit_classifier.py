"""Exit Classifier — morning exit scheduling for overnight momentum positions.

At 9:35 AM, each overnight position is classified using an entry-price comparison:

  price_935 > entry_price  ->  TRAIL   (place 1.25% trailing stop, exit by 11:30 latest)
  price_935 <= entry_price ->  EXIT_935 (exit immediately — gap faded)

Price sourcing priority:
  price_935: last minute-bar close (9:30-9:35) preferred; snapshot fallback

The open_price is still captured at 9:30 for gap_pct logging only.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
# Exit action bucket constants
# ═══════════════════════════════════════════════════
EXIT_BUCKET_935  = "09:35"    # Immediate market sell — gap faded
EXIT_BUCKET_TRAIL = "trail"   # Place trailing stop; hard fallback at 11:30
EXIT_BUCKET_1130 = "11:30"    # Hard fallback / timed exit for any remaining position


@dataclass
class ExitClassification:
    """Per-symbol exit classification result."""
    symbol: str
    exit_bucket: str            # EXIT_BUCKET_935 | EXIT_BUCKET_TRAIL | EXIT_BUCKET_1130
    entry_price: float
    price_935: float
    ret_vs_entry_pct: float     # (price_935 - entry_price) / entry_price * 100
    price_935_source: str       # "minute_bar" | "snapshot" | "missing"
    open_price: float = 0.0     # for gap_pct logging
    open_price_source: str = "missing"
    gap_pct: Optional[float] = None


# ═══════════════════════════════════════════════════
# Core classifier
# ═══════════════════════════════════════════════════

def classify_exit(price_935: float, entry_price: float) -> str:
    """Return EXIT_BUCKET_TRAIL if price_935 > entry_price, else EXIT_BUCKET_935."""
    if price_935 > entry_price:
        return EXIT_BUCKET_TRAIL
    return EXIT_BUCKET_935


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
    """Classify all overnight positions at 9:35 AM using entry-price logic.

    Args:
        symbols: List of position symbols.
        open_prices: {symbol: open_price} — resolved by caller at 9:30 (for logging only).
        open_price_sources: {symbol: source_str}.
        snapshots_935: {symbol: snapshot_dict} from 9:35 Alpaca snapshot.
        minute_bars_935: {symbol: [bar, ...]} 9:30-9:35 minute bars.
        entry_prices: {symbol: entry_price} — REQUIRED for classification.

    Returns:
        {symbol: ExitClassification}
    """
    classifications: Dict[str, ExitClassification] = {}
    entry_prices = entry_prices or {}

    for symbol in symbols:
        open_price = open_prices.get(symbol, 0.0)
        open_src   = open_price_sources.get(symbol, "missing")
        entry_price = entry_prices.get(symbol, 0.0)

        # ── Resolve price_935 ────────────────────────────────────────
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

        # ── Missing price_935 — default to EXIT_935 (conservative) ───
        if price_935 is None or price_935 <= 0:
            logger.warning(
                f"EXIT CLASSIFY: {symbol} missing price_935 "
                f"[{price_935_src}] — defaulting to immediate exit"
            )
            classifications[symbol] = ExitClassification(
                symbol=symbol,
                exit_bucket=EXIT_BUCKET_935,
                entry_price=entry_price,
                price_935=0.0,
                ret_vs_entry_pct=0.0,
                price_935_source=price_935_src,
                open_price=open_price,
                open_price_source=open_src,
            )
            continue

        # ── Missing entry_price — default to EXIT_935 ────────────────
        if entry_price <= 0:
            logger.warning(
                f"EXIT CLASSIFY: {symbol} missing entry_price — defaulting to immediate exit"
            )
            classifications[symbol] = ExitClassification(
                symbol=symbol,
                exit_bucket=EXIT_BUCKET_935,
                entry_price=0.0,
                price_935=price_935,
                ret_vs_entry_pct=0.0,
                price_935_source=price_935_src,
                open_price=open_price,
                open_price_source=open_src,
            )
            continue

        ret_vs_entry_pct = (price_935 - entry_price) / entry_price * 100

        gap_pct = None
        if open_price > 0:
            gap_pct = (open_price - entry_price) / entry_price * 100

        exit_bucket = classify_exit(price_935, entry_price)

        classifications[symbol] = ExitClassification(
            symbol=symbol,
            exit_bucket=exit_bucket,
            entry_price=entry_price,
            price_935=price_935,
            ret_vs_entry_pct=ret_vs_entry_pct,
            price_935_source=price_935_src,
            open_price=open_price,
            open_price_source=open_src,
            gap_pct=gap_pct,
        )

        gap_str = f" gap={gap_pct:+.2f}%" if gap_pct is not None else ""
        logger.info(
            f"EXIT CLASSIFY: {symbol} | "
            f"entry={entry_price:.4f} "
            f"price_935={price_935:.4f}[{price_935_src}] "
            f"ret_vs_entry={ret_vs_entry_pct:+.2f}%{gap_str} -> {exit_bucket}"
        )

    # ── Summary ──────────────────────────────────────────────────────
    buckets: Dict[str, List[str]] = {}
    for sym, cls in classifications.items():
        buckets.setdefault(cls.exit_bucket, []).append(sym)
    for bucket, syms in sorted(buckets.items()):
        logger.info(f"EXIT SCHEDULE: {bucket} -> {len(syms)} positions: {syms}")

    fallback_935   = sum(1 for c in classifications.values() if c.price_935_source == "snapshot")
    missing_any    = sum(1 for c in classifications.values() if c.price_935_source == "missing")
    n = len(classifications)
    logger.info(
        f"EXIT CLASSIFY sources: {n} total | "
        f"price_935 snapshot fallbacks={fallback_935} | missing={missing_any}"
    )

    return classifications
