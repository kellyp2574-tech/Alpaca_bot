"""
Premarket scanning pipeline (Option B):
Massive seeds universe (broad + cheap), Alpaca snapshots validate gap (truth).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .morning_config import Config
from .storage import Candidate


# ---------------------------
# Ledger tracking (audit)
# ---------------------------

@dataclass
class Drop:
    symbol: str
    stage: str
    reason: str
    details: Dict[str, object] = field(default_factory=dict)


@dataclass
class CandidateLedger:
    run_date: str
    seed_source: str = "massive_full_snapshot"
    snapshot_count_seen: int = 0         # total items in Massive snapshot
    snapshot_count_with_prev_obj: int = 0  # items with prev_day object
    seed_total: int = 0          # usable items (prev_close > 0)
    seed_selected: int = 0       # seed symbols we pass to Alpaca validation
    validated: int = 0           # symbols that passed Alpaca snapshot checks & filters
    final: int = 0               # final candidates returned
    drops: List[Drop] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["drops"] = [asdict(d) for d in self.drops]
        path.write_text(__import__("json").dumps(payload, indent=2))


# ---------------------------
# Massive: seed universe
# ---------------------------

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _get_any(obj, *names, default=None):
    """Schema-tolerant field getter - supports both dict and object attributes."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        for n in names:
            if n in obj and obj[n] is not None:
                return obj[n]
        return default
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return default


def seed_universe_massive(
    massive_client,
    *,
    max_seed: int = 600,
    include_otc: bool = False,
) -> Tuple[List[str], Dict[str, float], Dict[str, object], List[Drop]]:
    """
    Use Massive full market snapshot to get a broad, liquid seed universe.
    We DO NOT use Massive 'todaysChangePerc' for gap truth (delay/session semantics).
    We only use it to optionally pre-rank or loosen filters if you want later.

    Returns:
        seed_symbols: top max_seed by liquidity proxy
        prev_close_map: dict of symbol -> prev_close from Massive
        meta: summary stats
        drops: any drop reasons observed
    """
    drops: List[Drop] = []

    # Massive python SDK: get_snapshot_all("stocks", include_otc="false")
    snapshot = massive_client.get_snapshot_all("stocks", include_otc=str(include_otc).lower())

    rows: List[Tuple[str, float, float, float]] = []
    # (symbol, liquidity_proxy, prev_close, prev_vol)

    snapshot_count_seen = 0
    snapshot_count_with_prev_obj = 0
    usable = 0
    
    for item in snapshot:
        snapshot_count_seen += 1
        
        # Schema-tolerant field extraction
        t = _get_any(item, "ticker", "symbol")
        prev = _get_any(item, "prev_day", "prevDay", "prev_daily_bar", "prevDailyBar", "prev")

        if not t:
            continue
            
        if not prev:
            drops.append(Drop(t, "massive_seed", "no_prev_obj", {}))
            continue
        
        snapshot_count_with_prev_obj += 1

        # Schema-tolerant prev close + volume extraction
        # Massive API uses: close, volume (not c, v)
        prev_c = _get_any(prev, "close", "c", "prev_close", "closingPrice")
        prev_v = _get_any(prev, "volume", "v", "prev_volume", "vol")

        prev_close = _safe_float(prev_c)
        prev_vol = _safe_float(prev_v)

        if prev_close <= 0:
            drops.append(Drop(t, "massive_seed", "no_prev_close", {
                "prev_type": type(prev).__name__,
                "prev_c_raw": prev_c,
                "prev_v_raw": prev_v,
            }))
            continue

        # Liquidity proxy: prev_close * prev_vol
        liq = prev_close * prev_vol
        usable += 1
        rows.append((t, liq, prev_close, prev_vol))

    # Sort by liquidity and take top N
    rows.sort(key=lambda r: r[1], reverse=True)
    seed_rows = rows[:max_seed]
    seed_symbols = [r[0] for r in seed_rows]
    
    # Build prev_close map for Alpaca validation stage
    prev_close_map = {r[0]: r[2] for r in seed_rows}  # symbol -> prev_close

    meta = {
        "snapshot_count_seen": snapshot_count_seen,
        "snapshot_count_with_prev_obj": snapshot_count_with_prev_obj,
        "usable_snapshot_items": usable,
        "max_seed": max_seed,
    }
    return seed_symbols, prev_close_map, meta, drops


# ---------------------------
# Alpaca snapshots: gap truth
# ---------------------------

def build_candidates_alpaca_snapshot(
    cfg: Config,
    alpaca,
    seed_symbols: List[str],
    prev_close_map: Dict[str, float],
    now: datetime,
    *,
    max_candidates: int = 300,
    ledger: Optional[CandidateLedger] = None,
) -> List[Candidate]:
    """
    Uses Alpaca Snapshot API data to compute real gap vs previous close:
        prev_close = from Massive (prev_close_map)
        price_now  = mid(quote) or latestTrade.p from Alpaca snapshot

    Requires: alpaca.get_snapshots(symbols) -> Dict[str, snapshot]
    """
    drops: List[Drop] = []
    out: List[Candidate] = []

    if not seed_symbols:
        if ledger:
            ledger.drops.extend(drops)
        return out

    snaps = alpaca.get_snapshots(seed_symbols)  # must be implemented in AlpacaDataAdapter
    
    # Coverage logging: track snapshot availability
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Snapshot coverage: {len(snaps)}/{len(seed_symbols)} symbols returned")

    for sym in seed_symbols:
        snap = snaps.get(sym)
        if snap is None:
            drops.append(Drop(sym, "alpaca_snapshot", "no_snapshot", {}))
            continue

        # ---- prev close from Massive ----
        prev_close = prev_close_map.get(sym, 0.0)
        if prev_close <= 0:
            drops.append(Drop(sym, "alpaca_snapshot", "no_prev_close_from_massive", {}))
            continue

        # ---- price now (prefer quote mid) ----
        lq = getattr(snap, "latest_quote", None) or getattr(snap, "latestQuote", None)
        lt = getattr(snap, "latest_trade", None) or getattr(snap, "latestTrade", None)

        price_now = 0.0
        if lq is not None:
            bp = _safe_float(getattr(lq, "bp", 0) or getattr(lq, "bid_price", 0))
            ap = _safe_float(getattr(lq, "ap", 0) or getattr(lq, "ask_price", 0))
            if bp > 0 and ap > 0:
                price_now = (bp + ap) / 2.0

        if price_now <= 0 and lt is not None:
            price_now = _safe_float(getattr(lt, "p", 0) or getattr(lt, "price", 0))

        if price_now <= 0:
            drops.append(Drop(sym, "alpaca_snapshot", "no_price_now", {}))
            continue

        # ---- Filters ----
        if not (cfg.min_price <= price_now <= cfg.max_price):
            drops.append(Drop(sym, "alpaca_snapshot", "price_out_of_range", {"price": price_now}))
            continue

        gap_pct = (price_now - prev_close) / prev_close
        if not (cfg.min_gap_pct <= gap_pct <= cfg.max_gap_pct):
            drops.append(Drop(sym, "alpaca_snapshot", "gap_out_of_range", {"gap_pct": gap_pct}))
            continue

        # Daily volume baseline OFF - EntryLoop enforces real constraint (5-min $ volume)
        
        out.append(
            Candidate(
                symbol=sym,
                price=price_now,
                prev_close=prev_close,
                pm_last=price_now,   # in this architecture, pm_last means "latest snapshot price"
                pm_high=price_now,   # unknown without premarket bars; safe placeholder
                pm_volume=0,         # unknown without premarket bars
                avg_vol_30d=0,       # unknown unless you compute it elsewhere
                float_shares=0,
                gap_pct=gap_pct,
                pm_vol_float=0,
                relvol=0,
                score=gap_pct,
            )
        )

        if len(out) >= max_candidates:
            break

    out.sort(key=lambda c: c.gap_pct, reverse=True)

    if ledger is not None:
        ledger.drops.extend(drops)
        ledger.validated = len(out)

    return out


# ---------------------------
# Public entry point (matches your morning_main.py call)
# ---------------------------

def build_candidates(
    cfg: Config,
    alpaca,
    massive_client,
    date: datetime,
) -> Tuple[List[Candidate], CandidateLedger]:
    """
    Signature matches morning_main.fetch_candidates():
        build_candidates(cfg, data.alpaca, data.massive, today)

    Returns:
        (candidates, ledger)
    """
    run_date = date.date().isoformat()
    ledger = CandidateLedger(run_date=run_date)

    # Stage A: Massive seeds universe
    max_seed = getattr(cfg, "max_seed_universe", 600)
    seed_symbols, prev_close_map, meta, seed_drops = seed_universe_massive(
        massive_client,
        max_seed=max_seed,
        include_otc=False,
    )
    ledger.snapshot_count_seen = int(meta.get("snapshot_count_seen", 0))
    ledger.snapshot_count_with_prev_obj = int(meta.get("snapshot_count_with_prev_obj", 0))
    ledger.seed_total = int(meta.get("usable_snapshot_items", 0))
    ledger.seed_selected = len(seed_symbols)
    ledger.drops.extend(seed_drops)

    # Stage B: Alpaca snapshot validates gap truth
    max_candidates = getattr(cfg, "max_candidates_returned", 300)
    candidates = build_candidates_alpaca_snapshot(
        cfg,
        alpaca,
        seed_symbols,
        prev_close_map,
        date,
        max_candidates=max_candidates,
        ledger=ledger,
    )

    ledger.final = len(candidates)
    return candidates, ledger
