"""
Intraday Mean Reversion Classifier — Morning Momentum Strategy.

Sleeve parameters are taken VERBATIM from combined_long_backtest.py (the validated
source of truth). Do not change values without re-running the backtest.

Regime:
  ACTIVE    — VIX >= 15 (README validated threshold) AND |SPY gap| > 1% AND |QQQ gap| > 1%
                -> Themes A, B, C only (no UL fallback on active days)
  DEAD_ZONE — anything else
                -> Theme D; UL fills slots only when D is empty

UL is a dead-zone-only fallback — it is never used on ACTIVE days.
This matches the router-exit portfolio version producing the 26.37x / Sharpe 2.39 baseline.

Sorting: (sleeve_rank ASC, |pm_ret| DESC) — severity ranking for tie-breaking.
Caps: min=1, max=8 per day.

Router exit rule (10:00 AM):
  If router SHORT (sqqq_goldilocks / uvxy_crash): exit non-Theme-A positions.
  If router LONG or NONE: hold all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from bot import config

logger = logging.getLogger(__name__)

REGIME_ACTIVE    = "ACTIVE"
REGIME_DEAD_ZONE = "DEAD_ZONE"

# Router actions that trigger MR exit (validated Version D study)
ROUTER_SHORT_ACTIONS = ("sqqq_goldilocks", "uvxy_crash")

# Leveraged/index ETFs excluded from universe (mirrors backtest LEVERAGED_ETFS)
LEVERAGED_ETFS = {
    "TQQQ","SQQQ","SPXL","SPXS","UPRO","SDS","QLD","QID",
    "UVXY","SVXY","VXX","VIXY","TVIX","NUGT","DUST","JNUG","JDST",
    "LABU","LABD","SOXL","SOXS","TNA","TZA","FAS","FAZ",
    "UDOW","SDOW","TECL","TECS","ERX","ERY","GUSH","DRIP",
    "FNGU","FNGD","NAIL","DRV","CURE","RETL","BOIL","KOLD",
    "ZIV","DGAZ","UGAZ","UCO","SCO","UVIX","SVIX","ZVZZT",
    "SPY","QQQ","IWM","VIX","^VIX",
}


@dataclass
class IntradayMRSleeve:
    """Per-sleeve params — exact values from combined_long_backtest.py."""
    name: str           # e.g. "LA-1", "LB-2", "LD-3"
    theme: str          # "A", "B", "C", "D", "UL"
    sleeve_rank: int    # from SLEEVE_RANK in backtest (lower = higher priority)
    prior_ret_lo: float # bin lower bound (inclusive: lo <= val)
    prior_ret_hi: float # bin upper bound (exclusive: val < hi)
    pm_ret_lo: float
    pm_ret_hi: float
    price_lo: float     # inclusive
    price_hi: float     # exclusive (mirrors backtest: plo <= open < phi)
    entry_time: str
    exit_time: str
    tp_pct: Optional[float]
    sl_pct: Optional[float]


@dataclass
class IntradayMRCandidate:
    """A symbol matched to a sleeve for 9:32-9:47 intraday MR entry."""
    symbol: str
    theme: str
    sleeve_name: str
    sleeve_rank: int
    prior_ret: float
    pm_ret: float
    severity_score: float   # |prior_ret| + |pm_ret| — tiebreak within rank
    signal_price: float     # today's open (entry proxy)
    entry_time: str
    exit_time: str
    tp_pct: Optional[float]
    sl_pct: Optional[float]
    regime: str


# ─────────────────────────────────────────────────────────────────────────────
# Sleeve table — verbatim from combined_long_backtest.py
# Columns: prior_lo, prior_hi, pm_lo, pm_hi, price_lo, price_hi,
#          name, theme, rank, entry, exit, tp, sl
# Bin convention: lo <= value < hi  (mirrors backtest bin_value)
# ─────────────────────────────────────────────────────────────────────────────

_SLEEVE_ROWS = [
    # ── Theme A (LA): prior -7% to -5%, pm -10% to -5% ──────────────────────
    (-0.07, -0.05,  -0.10, -0.05,   50, 100,  "LA-1", "A",  2, "09:32", "15:50", 0.10, 0.05),
    (-0.07, -0.05,  -0.10, -0.05,   20,  50,  "LA-2", "A",  7, "09:32", "15:50", 0.10, 0.05),
    (-0.07, -0.05,  -0.10, -0.05,   10,  20,  "LA-3", "A",  3, "09:32", "15:50", 0.10, 0.10),
    (-0.07, -0.05,  -0.10, -0.05,    5,  10,  "LA-4", "A", 10, "09:32", "15:50", 0.10, 0.10),
    (-0.07, -0.05,  -0.10, -0.05,    2,   5,  "LA-5", "A", 13, "09:32", "13:00", 0.05, 0.05),
    # ── Theme B (LB): prior -5% to -3%, pm -10% to -5% ──────────────────────
    (-0.05, -0.03,  -0.10, -0.05,   50, 100,  "LB-1", "B",  4, "09:33", "11:30", 0.05, 0.05),
    (-0.05, -0.03,  -0.10, -0.05,   20,  50,  "LB-2", "B", 11, "09:32", "11:30", 0.10, 0.10),
    (-0.05, -0.03,  -0.10, -0.05,   10,  20,  "LB-3", "B", 12, "09:33", "11:30", 0.05, 0.03),
    (-0.05, -0.03,  -0.10, -0.05,    5,  10,  "LB-4", "B",  8, "09:32", "15:50", 0.05, 0.05),
    # ── Theme C (LC): prior -10% to -7%, pm -10% to -5% ─────────────────────
    (-0.10, -0.07,  -0.10, -0.05,   50, 100,  "LC-1", "C",  1, "09:32", "11:30", 0.10, 0.10),
    (-0.10, -0.07,  -0.10, -0.05,   20,  50,  "LC-2", "C",  5, "09:33", "15:00", 0.10, 0.05),
    (-0.10, -0.07,  -0.10, -0.05,   10,  20,  "LC-3", "C",  9, "09:47", "10:02", 0.05, 0.03),
    (-0.10, -0.07,  -0.10, -0.05,    2,   5,  "LC-4", "C",  6, "09:32", "15:00", 0.10, 0.10),
    # ── Theme D (LD): dead zone ───────────────────────────────────────────────
    (-0.15, -0.10,  -0.05, -0.03,   50, 100,  "LD-1", "D", 14, "09:32", "15:00", 0.05, 0.05),
    (-0.15, -0.10,  -0.05, -0.03,   20,  50,  "LD-2", "D", 18, "09:32", "15:00", 0.10, 0.05),
    (-0.10, -0.07,  -0.05, -0.03,   50, 100,  "LD-3", "D", 16, "09:32", "15:00", 0.10, 0.10),
    (-0.10, -0.07,  -0.05, -0.03,   20,  50,  "LD-4", "D", 19, "09:47", "15:00", 0.10, 0.05),
    (-0.07, -0.05,  -0.05, -0.03,   50, 100,  "LD-5", "D", 17, "09:33", "15:50", 0.05, 0.05),
    (-99.0, -0.15, -99.0, -0.10,    10,  20,  "LD-6", "D", 15, "09:37", "10:30", 0.05, 0.03),
    # ── UL (all days — fills slots when no regime match) ─────────────────────
    ( 0.03,  0.05,   0.10, 99.0,    50, 100,  "UL-1", "UL", 20, "09:33", "15:50", 0.03, 0.03),
    (-0.05, -0.03, -99.0, -0.10,    20,  50,  "UL-3", "UL", 21, "09:33", "15:00", 0.10, 0.05),
]

SLEEVES: List[IntradayMRSleeve] = [
    IntradayMRSleeve(
        name=r[6], theme=r[7], sleeve_rank=r[8],
        prior_ret_lo=r[0], prior_ret_hi=r[1],
        pm_ret_lo=r[2],    pm_ret_hi=r[3],
        price_lo=float(r[4]), price_hi=float(r[5]),
        entry_time=r[9], exit_time=r[10],
        tp_pct=r[11],    sl_pct=r[12],
    )
    for r in _SLEEVE_ROWS
]

# Pre-split lists sorted by rank (ascending) so first match = highest-priority
ACTIVE_SLEEVES    = sorted([s for s in SLEEVES if s.theme in ("A", "B", "C")], key=lambda s: s.sleeve_rank)
DEAD_ZONE_SLEEVES = sorted([s for s in SLEEVES if s.theme == "D"],             key=lambda s: s.sleeve_rank)
UL_SLEEVES        = sorted([s for s in SLEEVES if s.theme == "UL"],            key=lambda s: s.sleeve_rank)


# ─────────────────────────────────────────────────────────
# Regime classification
# ─────────────────────────────────────────────────────────

def classify_regime(vix_open: float, spy_gap: float, qqq_gap: float) -> str:
    """
    ACTIVE: VIX >= threshold AND |SPY gap| > gap_threshold AND |QQQ gap| > gap_threshold.
    Backtest used VIX >= 20; README validated threshold is 15 (set via config).
    """
    vix_threshold = float(getattr(config, "INTRADAY_MR_VIX_THRESHOLD", 15.0))
    gap_threshold  = float(getattr(config, "INTRADAY_MR_GAP_THRESHOLD",  0.01))
    if (vix_open >= vix_threshold
            and abs(spy_gap) > gap_threshold
            and abs(qqq_gap) > gap_threshold):
        return REGIME_ACTIVE
    return REGIME_DEAD_ZONE


def _sleeve_match(s: IntradayMRSleeve, prior_ret: float, pm_ret: float, price: float) -> bool:
    """Bin convention lo <= val < hi (mirrors backtest bin_value)."""
    return (
        s.prior_ret_lo <= prior_ret < s.prior_ret_hi
        and s.pm_ret_lo <= pm_ret < s.pm_ret_hi
        and s.price_lo  <= price   < s.price_hi
    )


def build_intraday_mr_candidates(
    snapshots: Dict[str, dict],
) -> List[IntradayMRCandidate]:
    """
    Build and rank intraday MR candidates from snapshot data.

    snapshots must contain:
      - SPY, QQQ snapshots (for regime gaps)
      - '_vix_open' key injected by build_intraday_mr_watchlist (float)
      - Each stock snapshot needs: open, prev_close, prev_volume, prev2_close
        (prev2_close added by enrich_snapshots_with_prev2 before this call)

    Returns candidates sorted by (sleeve_rank ASC, |pm_ret| DESC), capped at max.
    Returns [] if fewer than min_cands.
    """
    # Extract VIX (injected by runtime under '_vix_open'; use .get to avoid mutation)
    vix_open = snapshots.get("_vix_open")
    if vix_open is None:
        logger.warning("VIX open unavailable — defaulting to DEAD_ZONE")
        vix_open = 0.0

    spy_snap = snapshots.get("SPY", {})
    qqq_snap = snapshots.get("QQQ", {})
    spy_open = spy_snap.get("open");  spy_prev = spy_snap.get("prev_close")
    qqq_open = qqq_snap.get("open");  qqq_prev = qqq_snap.get("prev_close")

    if None in (spy_open, spy_prev, qqq_open, qqq_prev):
        logger.warning("SPY/QQQ missing open or prev_close — cannot classify regime")
        return []

    spy_gap = float(spy_open) / float(spy_prev) - 1.0
    qqq_gap = float(qqq_open) / float(qqq_prev) - 1.0
    regime  = classify_regime(float(vix_open), spy_gap, qqq_gap)
    logger.info(
        f"Intraday MR regime: {regime} "
        f"(VIX={vix_open:.2f}, SPY_gap={spy_gap:.2%}, QQQ_gap={qqq_gap:.2%})"
    )

    sleeve_pool = ACTIVE_SLEEVES if regime == REGIME_ACTIVE else DEAD_ZONE_SLEEVES
    adv_min   = float(getattr(config, "INTRADAY_MR_MIN_ADV_DOLLARS", 1_000_000))
    # Apply IEX volume multiplier to compensate for IEX undercounting (same fix as overnight MR)
    adv_multiplier = float(getattr(config, "ADV_DOLLAR_MULTIPLIER", 1.0))
    min_cands = int(getattr(config,   "INTRADAY_MR_MIN_CANDIDATES",  1))
    max_cands = int(getattr(config,   "INTRADAY_MR_MAX_CANDIDATES",  8))

    abc_cands: List[IntradayMRCandidate] = []
    d_cands:   List[IntradayMRCandidate] = []
    ul_cands:  List[IntradayMRCandidate] = []

    for symbol, snap in snapshots.items():
        if not isinstance(snap, dict):
            continue
        if symbol.upper() in LEVERAGED_ETFS:
            continue

        open_px         = snap.get("open")
        prev_close      = snap.get("prev_close")
        prev2_close_raw = snap.get("prev2_close")
        prev_volume     = snap.get("prev_volume") or 0

        if not open_px or not prev_close or not prev2_close_raw:
            continue

        open_px     = float(open_px)
        prev_close  = float(prev_close)
        prev2_close = float(prev2_close_raw)
        prev_volume = float(prev_volume)

        if prev2_close <= 0 or prev_close <= 0 or open_px <= 0:
            continue

        # ADV check with IEX multiplier
        effective_adv = prev_close * prev_volume * adv_multiplier
        if effective_adv < adv_min:
            continue

        prior_ret = prev_close / prev2_close - 1.0
        pm_ret    = open_px   / prev_close   - 1.0

        if abs(prior_ret) > 0.60 or abs(pm_ret) > 0.60:
            continue

        # Pass 1: regime sleeve match (first match = highest priority rank)
        matched = None
        for s in sleeve_pool:
            if _sleeve_match(s, prior_ret, pm_ret, open_px):
                matched = s
                break

        if matched is not None:
            cand = IntradayMRCandidate(
                symbol=symbol, theme=matched.theme, sleeve_name=matched.name,
                sleeve_rank=matched.sleeve_rank, prior_ret=prior_ret, pm_ret=pm_ret,
                severity_score=abs(prior_ret) + abs(pm_ret), signal_price=open_px,
                entry_time=matched.entry_time, exit_time=matched.exit_time,
                tp_pct=matched.tp_pct, sl_pct=matched.sl_pct, regime=regime,
            )
            if matched.theme in ("A", "B", "C"):
                abc_cands.append(cand)
            else:
                d_cands.append(cand)
            continue

        # Pass 2: UL — evaluated on ALL days but only used as dead-zone fallback
        # (mirrors backtest: UL only fills when D empty on dead-zone days)
        if regime == REGIME_DEAD_ZONE:
            for s in UL_SLEEVES:
                if _sleeve_match(s, prior_ret, pm_ret, open_px):
                    ul_cands.append(IntradayMRCandidate(
                        symbol=symbol, theme=s.theme, sleeve_name=s.name,
                        sleeve_rank=s.sleeve_rank, prior_ret=prior_ret, pm_ret=pm_ret,
                        severity_score=abs(prior_ret) + abs(pm_ret), signal_price=open_px,
                        entry_time=s.entry_time, exit_time=s.exit_time,
                        tp_pct=s.tp_pct, sl_pct=s.sl_pct, regime=regime,
                    ))
                    break

    # Combine exactly as backtest:
    #   Active:    ABC only
    #   Dead zone: D candidates; fall back to UL only if D is empty
    if regime == REGIME_ACTIVE:
        combined = abc_cands
    else:
        combined = d_cands if d_cands else ul_cands

    # Sort: sleeve_rank ASC, |pm_ret| DESC  (mirrors backtest sort)
    combined.sort(key=lambda c: (c.sleeve_rank, -abs(c.pm_ret)))
    combined = combined[:max_cands]

    if len(combined) < min_cands:
        logger.info(f"Intraday MR: {len(combined)} candidates < min={min_cands} — skipping day")
        return []

    logger.info(
        f"Intraday MR: {len(combined)} candidates "
        f"(regime={regime}, VIX={vix_open:.1f})"
    )
    for c in combined:
        logger.info(
            f"  {c.symbol} [{c.sleeve_name}/Theme-{c.theme}] "
            f"prior={c.prior_ret:.2%} pm={c.pm_ret:.2%} "
            f"entry={c.entry_time} exit={c.exit_time} tp={c.tp_pct} sl={c.sl_pct}"
        )
    return combined


def _get_vix_open(snapshots: Dict[str, dict]) -> Optional[float]:
    """Extract VIX open from snapshots if a VIX proxy symbol was explicitly fetched."""
    for key in ("VIX", "^VIX", "VIXY"):
        snap = snapshots.get(key, {})
        val = snap.get("open") or snap.get("last")
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


def enrich_snapshots_with_prev2(
    snapshots: Dict[str, dict],
    daily_bars: Dict[str, List[dict]],
) -> None:
    """
    Add 'prev2_close' to each snapshot from daily_bars.
    Bars must be sorted by date ASC; bars[-2] = T-2 close.
    Mutates snapshots in place.
    """
    for symbol, bars in daily_bars.items():
        if symbol not in snapshots or len(bars) < 2:
            continue
        prev2_bar   = bars[-2]
        prev2_close = prev2_bar.get("c") or prev2_bar.get("close")
        if prev2_close is not None:
            snapshots[symbol]["prev2_close"] = float(prev2_close)


def apply_router_exit_rule(positions: List[dict], router_action: str) -> List[str]:
    """
    At 10:00: return list of symbols to exit based on router signal.
    Router SHORT → exit non-Theme-A. Router LONG/NONE → hold all.
    """
    action_lower = (router_action or "").lower()
    if action_lower not in ROUTER_SHORT_ACTIONS:
        logger.info(f"Router action={router_action!r} → hold all intraday MR positions")
        return []
    to_exit = [p["symbol"] for p in positions if p.get("theme", "").upper() != "A"]
    kept    = [p["symbol"] for p in positions if p.get("theme", "").upper() == "A"]
    logger.info(f"Router SHORT ({router_action}) → keeping Theme A {kept}; exiting {to_exit}")
    return to_exit
