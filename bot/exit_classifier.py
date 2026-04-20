"""Exit Classifier — morning exit scheduling for overnight momentum positions.

At 9:35 AM, each overnight position is classified into one of two buckets:

  ret_open_to_935 > +0.5%   -> exit at 9:35  (up from open)
  otherwise                  -> exit at 11:30 (hold)
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
# Exit time bucket constants (ET, 24h "HH:MM")
# ═══════════════════════════════════════════════════
EXIT_BUCKET_935 = "09:35"
EXIT_BUCKET_1130 = "11:30"

# Classification threshold
UP_MOVE_PCT = 0.5   # ret_open_to_935 > 0.5% -> exit immediately at 9:35


@dataclass
class ExitClassification:
    """Per-symbol exit classification result."""
    symbol: str
    exit_time: str           # EXIT_BUCKET_935 or EXIT_BUCKET_1130
    open_price: float
    price_935: float
    move_5m_pct: float
    gap_pct: Optional[float] = None  # logging only


# ═══════════════════════════════════════════════════
# Core classifier
# ═══════════════════════════════════════════════════

def get_exit_time(move_5m_pct: float) -> str:
    """Classify a position into an exit time bucket.

    Args:
        move_5m_pct: (price_935 - open_price) / open_price * 100

    Returns:
        "09:35" if up > 0.5% from open, else "11:30"
    """
    if move_5m_pct > UP_MOVE_PCT:
        return EXIT_BUCKET_935
    return EXIT_BUCKET_1130


# ═══════════════════════════════════════════════════
# Batch classifier
# ═══════════════════════════════════════════════════

def classify_positions(
    symbols: List[str],
    open_prices: Dict[str, float],
    snapshots_935: Dict[str, dict],
    entry_prices: Optional[Dict[str, float]] = None,
) -> Dict[str, ExitClassification]:
    """Classify all overnight positions at 9:35 AM.

    Args:
        symbols: List of position symbols.
        open_prices: {symbol: open_price} from 9:30 snapshot.
        snapshots_935: {symbol: snapshot_dict} from 9:35 snapshot.
        entry_prices: {symbol: entry_price} for gap_pct logging (optional).

    Returns:
        {symbol: ExitClassification}
    """
    classifications: Dict[str, ExitClassification] = {}

    for symbol in symbols:
        open_price = open_prices.get(symbol)
        snap = snapshots_935.get(symbol, {})
        price_935 = snap.get("last_price") or snap.get("close")

        if not open_price or not price_935 or open_price <= 0:
            logger.warning(
                f"EXIT CLASSIFY: {symbol} missing price data "
                f"(open={open_price}, price_935={price_935}) — defaulting to 11:30"
            )
            classifications[symbol] = ExitClassification(
                symbol=symbol,
                exit_time=EXIT_BUCKET_1130,
                open_price=open_price or 0,
                price_935=price_935 or 0,
                move_5m_pct=0.0,
            )
            continue

        move_5m_pct = (price_935 - open_price) / open_price * 100

        # Gap % for logging only
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
            gap_pct=gap_pct,
        )

        gap_str = f" gap={gap_pct:+.2f}%" if gap_pct is not None else ""
        logger.info(
            f"EXIT CLASSIFY: {symbol} | "
            f"open={open_price:.4f} price_935={price_935:.4f} "
            f"move_5m={move_5m_pct:+.2f}%{gap_str} -> EXIT @ {exit_time}"
        )

    # Summary by bucket
    buckets: Dict[str, List[str]] = {}
    for sym, cls in classifications.items():
        buckets.setdefault(cls.exit_time, []).append(sym)

    for bucket, syms in sorted(buckets.items()):
        logger.info(f"EXIT SCHEDULE: {bucket} -> {len(syms)} positions: {syms}")

    return classifications
