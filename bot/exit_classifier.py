"""V2 Exit Classifier — morning exit scheduling for overnight momentum positions.

At 9:35 AM, each overnight position is classified into an exit time bucket
based on the first 5 minutes of price action and VWAP trend:

  move_5m_pct < -1.0                       → hold to 2:00 PM  (weak / dropping)
  move_5m_pct > +1.0  AND  above_vwap      → exit at 9:35     (strong + above VWAP)
  move_5m_pct > +1.0                        → exit at 11:00 AM (strong, no trend)
  otherwise                                 → exit at 10:00 AM (default)
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
# Exit time bucket constants (ET, 24h "HH:MM")
# ═══════════════════════════════════════════════════
EXIT_BUCKET_935 = "09:35"
EXIT_BUCKET_1000 = "10:00"
EXIT_BUCKET_1100 = "11:00"
EXIT_BUCKET_1400 = "14:00"

# Classification thresholds (pct)
STRONG_MOVE_PCT = 1.0   # |move| > 1% = "strong"
WEAK_MOVE_PCT = -1.0    # move < -1% = "weak / dropping"


@dataclass
class ExitClassification:
    """Per-symbol exit classification result."""
    symbol: str
    exit_time: str           # one of EXIT_BUCKET_*
    open_price: float
    price_935: float
    move_5m_pct: float
    vwap_5m: float
    above_vwap: bool
    gap_pct: Optional[float] = None  # logging only


# ═══════════════════════════════════════════════════
# VWAP helpers
# ═══════════════════════════════════════════════════

def compute_vwap(minute_bars: List[dict]) -> float:
    """Compute VWAP from minute bars.

    VWAP = Σ(typical_price × volume) / Σ(volume)
    typical_price = (high + low + close) / 3
    """
    total_pv = 0.0
    total_v = 0.0
    for bar in minute_bars:
        h = bar.get("h", 0)
        l = bar.get("l", 0)
        c = bar.get("c", 0)
        v = bar.get("v", 0)
        if v > 0:
            typical = (h + l + c) / 3.0
            total_pv += typical * v
            total_v += v
    return total_pv / total_v if total_v > 0 else 0.0


def is_above_vwap(price_935: float, vwap: float) -> bool:
    """Price-above-VWAP condition: price at 9:35 is above the 5-min VWAP.

    Standard interpretation — if the current price sits above the
    volume-weighted average of the first 5 minutes, momentum is still
    pushing upward.
    """
    if vwap <= 0:
        return False
    return price_935 > vwap


# ═══════════════════════════════════════════════════
# Core classifier
# ═══════════════════════════════════════════════════

def get_v2_exit_time(move_5m_pct: float, above_vwap: bool) -> str:
    """Classify a position into an exit time bucket.

    Args:
        move_5m_pct: (price_935 - open_price) / open_price * 100
        above_vwap: True if price_935 > 5-min VWAP

    Returns:
        Exit time bucket string ("09:35", "10:00", "11:00", "14:00")
    """
    # 1) Weak open / drop after open → hold to 2pm
    if move_5m_pct < WEAK_MOVE_PCT:
        return EXIT_BUCKET_1400

    # 2) Strong open + above VWAP → exit early at 9:35
    if move_5m_pct > STRONG_MOVE_PCT and above_vwap:
        return EXIT_BUCKET_935

    # 3) Strong open but NOT above VWAP → hold to 11am
    if move_5m_pct > STRONG_MOVE_PCT:
        return EXIT_BUCKET_1100

    # 4) Everything else → default 10am
    return EXIT_BUCKET_1000


# ═══════════════════════════════════════════════════
# Batch classifier
# ═══════════════════════════════════════════════════

def classify_positions(
    symbols: List[str],
    open_prices: Dict[str, float],
    snapshots_935: Dict[str, dict],
    minute_bars: Dict[str, List[dict]],
    entry_prices: Optional[Dict[str, float]] = None,
) -> Dict[str, ExitClassification]:
    """Classify all overnight positions at 9:35 AM.

    Args:
        symbols: List of position symbols.
        open_prices: {symbol: open_price} from 9:30 snapshot.
        snapshots_935: {symbol: snapshot_dict} from 9:35 snapshot.
        minute_bars: {symbol: [bar, ...]} 9:30-9:35 minute bars.
        entry_prices: {symbol: entry_price} for gap_pct logging (optional).

    Returns:
        {symbol: ExitClassification}
    """
    classifications: Dict[str, ExitClassification] = {}

    for symbol in symbols:
        open_price = open_prices.get(symbol)
        snap = snapshots_935.get(symbol, {})
        price_935 = snap.get("last_price") or snap.get("close")
        bars = minute_bars.get(symbol, [])

        if not open_price or not price_935 or open_price <= 0:
            logger.warning(
                f"V2 EXIT: {symbol} missing price data (open={open_price}, "
                f"price_935={price_935}) — defaulting to 10:00 AM"
            )
            classifications[symbol] = ExitClassification(
                symbol=symbol,
                exit_time=EXIT_BUCKET_1000,
                open_price=open_price or 0,
                price_935=price_935 or 0,
                move_5m_pct=0.0,
                vwap_5m=0.0,
                above_vwap=False,
            )
            continue

        move_5m_pct = (price_935 - open_price) / open_price * 100

        # VWAP from minute bars
        vwap = compute_vwap(bars) if bars else open_price
        price_above_vwap = is_above_vwap(price_935, vwap)

        # Gap % for logging only
        gap_pct = None
        entry_price = entry_prices.get(symbol) if entry_prices else None
        if entry_price and entry_price > 0:
            gap_pct = (open_price - entry_price) / entry_price * 100

        exit_time = get_v2_exit_time(move_5m_pct, price_above_vwap)

        classifications[symbol] = ExitClassification(
            symbol=symbol,
            exit_time=exit_time,
            open_price=open_price,
            price_935=price_935,
            move_5m_pct=move_5m_pct,
            vwap_5m=vwap,
            above_vwap=price_above_vwap,
            gap_pct=gap_pct,
        )

        gap_str = f" gap={gap_pct:+.2f}%" if gap_pct is not None else ""
        logger.info(
            f"V2 EXIT CLASSIFY: {symbol} | "
            f"open={open_price:.4f} price_935={price_935:.4f} "
            f"move_5m={move_5m_pct:+.2f}% vwap={vwap:.4f} "
            f"above_vwap={price_above_vwap}{gap_str} → EXIT @ {exit_time}"
        )

    # Summary by bucket
    buckets: Dict[str, List[str]] = {}
    for sym, cls in classifications.items():
        buckets.setdefault(cls.exit_time, []).append(sym)

    for bucket, syms in sorted(buckets.items()):
        logger.info(f"V2 EXIT SCHEDULE: {bucket} → {len(syms)} positions: {syms}")

    return classifications
