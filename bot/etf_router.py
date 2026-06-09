"""ETF Router — 5-strategy intraday decision system.

All qualifying strategies fire simultaneously (uncorrelated — r≈0):

  At 10:00 AM (using 9:30-10:00 30-min returns):
    1. VXX Spike Recovery  — VXX 30min >= +2.5% AND QQQ range 0.3-0.8%  → BUY TQQQ, exit 15:30
    2. VXX Collapse        — VXX 30min <= -2.0% AND QQQ 30min >= -1.0%   → BUY TQQQ, exit 15:30
    3. Momentum Sleeve     — QQQ 30min >= +0.5%
                              NORMAL regime  → BUY TQQQ, exit 15:00
                              HIGH_RISK regime (VXX >= +2% OR VXX price >= $400) → BUY SQQQ (anti-momentum)

  At 10:10 AM (only if no 10:00 trade fired — using 9:30-10:10 40-min returns):
    4. Router Long         — QQQ-SPY 40min spread >= +0.2%               → BUY TQQQ, exit 15:00
    5. SVIX Long           — SVIX 40min >= +0.2%                         → BUY SVIX, exit 15:00

Capital: 90% equity split equally across all strategies that fire.
All returns are plain percentages (not bps).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, date
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)


class RouterBranch(Enum):
    """Intraday strategy branches in priority order."""
    VXX_SPIKE_RECOVERY   = "VXX_SPIKE_RECOVERY"
    VXX_COLLAPSE         = "VXX_COLLAPSE"
    MOMENTUM_SLEEVE      = "MOMENTUM_SLEEVE"       # Normal regime: TQQQ
    MOMENTUM_SLEEVE_ANTI = "MOMENTUM_SLEEVE_ANTI"  # HIGH_RISK regime: SQQQ
    ROUTER_LONG          = "ROUTER_LONG"
    SVIX_LONG            = "SVIX_LONG"
    NO_TRADE             = "NO_TRADE"


@dataclass
class ETFTapeSnapshot:
    """Price tracking for one ETF from 9:30 open onwards."""
    symbol: str
    open_930: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    latest_price: Optional[float] = None
    latest_time: Optional[datetime] = None

    def update(self, price: float, timestamp: datetime):
        if self.open_930 is None:
            self.open_930 = price
        self.high = price if self.high is None else max(self.high, price)
        self.low  = price if self.low  is None else min(self.low,  price)
        self.latest_price = price
        self.latest_time = timestamp

    def return_pct(self) -> Optional[float]:
        """Return vs 9:30 open as a plain percentage (not bps)."""
        if self.latest_price is None or self.open_930 is None or self.open_930 == 0:
            return None
        return (self.latest_price / self.open_930 - 1.0) * 100.0

    def range_pct(self) -> Optional[float]:
        """High-low range as % of open."""
        if self.high is None or self.low is None or self.open_930 is None or self.open_930 == 0:
            return None
        return (self.high - self.low) / self.open_930 * 100.0

    def return_bps(self) -> Optional[float]:
        """Return vs 9:30 open in basis points (kept for logging compat)."""
        r = self.return_pct()
        return r * 100.0 if r is not None else None

    def is_valid(self) -> bool:
        return self.open_930 is not None and self.latest_price is not None


@dataclass
class MarketTape:
    """All ETF tape snapshots for intraday decisions."""
    qqq:  ETFTapeSnapshot = field(default_factory=lambda: ETFTapeSnapshot("QQQ"))
    spy:  ETFTapeSnapshot = field(default_factory=lambda: ETFTapeSnapshot("SPY"))
    vxx:  ETFTapeSnapshot = field(default_factory=lambda: ETFTapeSnapshot("VXX"))
    svix: ETFTapeSnapshot = field(default_factory=lambda: ETFTapeSnapshot("SVIX"))
    tqqq: ETFTapeSnapshot = field(default_factory=lambda: ETFTapeSnapshot("TQQQ"))
    sqqq: ETFTapeSnapshot = field(default_factory=lambda: ETFTapeSnapshot("SQQQ"))

    snapshot_time: Optional[datetime] = None

    def get_returns_summary(self) -> Dict[str, Optional[float]]:
        return {
            "QQQ":  self.qqq.return_pct(),
            "SPY":  self.spy.return_pct(),
            "VXX":  self.vxx.return_pct(),
            "SVIX": self.svix.return_pct(),
            "TQQQ": self.tqqq.return_pct(),
            "SQQQ": self.sqqq.return_pct(),
        }


# Keep old alias so etf_router_runtime.py references don't break during transition
MarketState0930to1000 = MarketTape


@dataclass
class RouterDecision:
    """Result of the intraday strategy evaluation."""
    branch: RouterBranch
    symbol: Optional[str] = None       # TQQQ or SVIX
    entry_time: Optional[time] = None
    exit_time: Optional[time] = None
    decision_time: Optional[datetime] = None
    market_state: Optional[MarketTape] = None
    conditions_met: Dict[str, Any] = field(default_factory=dict)

    def mr_blocked(self) -> bool:
        """Intraday ETF trade blocks overnight ETF strategies AND single-stock MR."""
        return self.branch != RouterBranch.NO_TRADE

    def to_dict(self) -> dict:
        return {
            "branch": self.branch.value,
            "symbol": self.symbol,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "decision_time": self.decision_time.isoformat() if self.decision_time else None,
            "mr_blocked": self.mr_blocked(),
            "conditions_met": self.conditions_met,
        }


class ETFRouter:
    """5-strategy intraday ETF router."""

    def __init__(self, config):
        self.config = config
        self.tape = MarketTape()
        self.decision: Optional[RouterDecision] = None
        self.recording_started = False

    def start_recording(self, opens_930: Dict[str, float], timestamp: datetime):
        """Seed the tape with 9:30 open prices."""
        for symbol, price in opens_930.items():
            snap = getattr(self.tape, symbol.lower(), None)
            if snap:
                snap.open_930 = price
                snap.high = price
                snap.low = price
                snap.latest_price = price
                snap.latest_time = timestamp
        self.recording_started = True
        logger.info(f"ETF router: tape recording started. Opens: {opens_930}")

    def update_tape(self, symbol: str, price: float, timestamp: datetime):
        """Update one symbol's tape entry."""
        if not self.recording_started:
            return
        snap = getattr(self.tape, symbol.lower(), None)
        if snap:
            snap.update(price, timestamp)

    # ── 10:00 AM decision (strategies 1-3) ───────────────────────────────────

    def make_decision(self, timestamp: datetime) -> List[RouterDecision]:
        """Evaluate strategies 1-3 at 10:00 AM. Returns all that qualify."""
        self.tape.snapshot_time = timestamp
        returns = self.tape.get_returns_summary()
        logger.info(f"ETF router 10:00 returns (%): {returns}")

        fired: List[RouterDecision] = []
        for check in (
            self._check_vxx_spike_recovery,
            self._check_vxx_collapse,
            self._check_momentum_sleeve,
        ):
            decision = check(timestamp)
            if decision:
                fired.append(decision)

        if not fired:
            logger.info("ETF router 10:00: no strategy fired (strategies 1-3)")
        else:
            logger.info(f"ETF router 10:00: {len(fired)} strategies fired: {[d.branch.value for d in fired]}")
        return fired

    # ── 10:10 AM decision (strategies 4-5) ───────────────────────────────────

    def make_decision_1010(self, timestamp: datetime) -> List[RouterDecision]:
        """Evaluate strategies 4-5 at 10:10 AM (only if no 10:00 trade fired)."""
        self.tape.snapshot_time = timestamp
        returns = self.tape.get_returns_summary()
        logger.info(f"ETF router 10:10 returns (%): {returns}")

        fired: List[RouterDecision] = []
        for check in (
            self._check_router_long,
            self._check_svix_long,
        ):
            decision = check(timestamp)
            if decision:
                fired.append(decision)

        if not fired:
            logger.info("ETF router 10:10: no strategy fired (strategies 4-5)")
        else:
            logger.info(f"ETF router 10:10: {len(fired)} strategies fired: {[d.branch.value for d in fired]}")
        return fired

    # ── Strategy implementations ──────────────────────────────────────────────

    def _check_vxx_spike_recovery(self, timestamp: datetime) -> Optional[RouterDecision]:
        """Strategy 1: VXX spiked >= +2.5% AND QQQ range 0.3-0.8% → BUY TQQQ, exit 15:30."""
        vxx_ret  = self.tape.vxx.return_pct()
        qqq_range = self.tape.qqq.range_pct()

        min_vxx  = float(getattr(self.config, "VXX_SPIKE_MIN_RETURN_PCT",      2.5))
        range_lo = float(getattr(self.config, "VXX_SPIKE_QQQ_RANGE_MIN_PCT",   0.3))
        range_hi = float(getattr(self.config, "VXX_SPIKE_QQQ_RANGE_MAX_PCT",   0.8))
        exit_t   = _parse_hhmm(getattr(self.config, "VXX_SPIKE_EXIT_TIME", "15:30"))
        vehicle  = getattr(self.config, "VXX_SPIKE_VEHICLE", "TQQQ")

        conditions = {
            f"VXX 30min >= +{min_vxx}%": vxx_ret is not None and vxx_ret >= min_vxx,
            f"QQQ range {range_lo}-{range_hi}%": (
                qqq_range is not None and range_lo <= qqq_range <= range_hi
            ),
        }
        if all(conditions.values()):
            logger.warning(f"STRATEGY 1 VXX SPIKE RECOVERY FIRED: VXX={vxx_ret:.2f}% QQQ_range={qqq_range:.2f}%")
            return RouterDecision(
                branch=RouterBranch.VXX_SPIKE_RECOVERY,
                symbol=vehicle,
                entry_time=time(10, 0),
                exit_time=exit_t,
                decision_time=timestamp,
                market_state=self.tape,
                conditions_met=conditions,
            )
        return None

    def _check_vxx_collapse(self, timestamp: datetime) -> Optional[RouterDecision]:
        """Strategy 2: VXX <= -2.0% AND QQQ >= -1.0% → BUY TQQQ, exit 15:30."""
        vxx_ret = self.tape.vxx.return_pct()
        qqq_ret = self.tape.qqq.return_pct()

        max_vxx = float(getattr(self.config, "VXX_COLLAPSE_VXX_MAX_RETURN_PCT", -2.0))
        min_qqq = float(getattr(self.config, "VXX_COLLAPSE_QQQ_MIN_RETURN_PCT", -1.0))
        exit_t  = _parse_hhmm(getattr(self.config, "VXX_COLLAPSE_EXIT_TIME", "15:30"))
        vehicle = getattr(self.config, "VXX_COLLAPSE_VEHICLE", "TQQQ")

        conditions = {
            f"VXX 30min <= {max_vxx}%": vxx_ret is not None and vxx_ret <= max_vxx,
            f"QQQ 30min >= {min_qqq}%": qqq_ret is not None and qqq_ret >= min_qqq,
        }
        if all(conditions.values()):
            logger.warning(f"STRATEGY 2 VXX COLLAPSE FIRED: VXX={vxx_ret:.2f}% QQQ={qqq_ret:.2f}%")
            return RouterDecision(
                branch=RouterBranch.VXX_COLLAPSE,
                symbol=vehicle,
                entry_time=time(10, 0),
                exit_time=exit_t,
                decision_time=timestamp,
                market_state=self.tape,
                conditions_met=conditions,
            )
        return None

    def _is_high_risk_regime(self) -> bool:
        """True if VXX 30min return >= HIGH_RISK threshold OR VXX price >= HIGH_RISK price level."""
        vxx_ret   = self.tape.vxx.return_pct()
        vxx_price = self.tape.vxx.latest_price
        hr_ret    = float(getattr(self.config, "VXX_HIGH_RISK_RETURN_PCT", 2.0))
        hr_price  = float(getattr(self.config, "VXX_HIGH_RISK_PRICE", 400.0))
        ret_trigger   = vxx_ret   is not None and vxx_ret   >= hr_ret
        price_trigger = vxx_price is not None and vxx_price >= hr_price
        return ret_trigger or price_trigger

    def _check_momentum_sleeve(self, timestamp: datetime) -> Optional[RouterDecision]:
        """Strategy 3: QQQ 30min >= +0.5%.

        NORMAL regime  → BUY TQQQ (momentum follow-through).
        HIGH_RISK regime → BUY SQQQ (anti-momentum: QQQ up but VXX spiking = fakeout).
        """
        qqq_ret = self.tape.qqq.return_pct()

        min_qqq = float(getattr(self.config, "MOMENTUM_QQQ_MIN_RETURN_PCT", 0.5))
        exit_t  = _parse_hhmm(getattr(self.config, "MOMENTUM_EXIT_TIME", "15:00"))

        conditions = {
            f"QQQ 30min >= +{min_qqq}%": qqq_ret is not None and qqq_ret >= min_qqq,
        }
        if all(conditions.values()):
            high_risk = self._is_high_risk_regime()
            vxx_ret   = self.tape.vxx.return_pct()
            vxx_price = self.tape.vxx.latest_price
            if high_risk:
                vehicle = getattr(self.config, "MOMENTUM_ANTI_VEHICLE", "SQQQ")
                branch  = RouterBranch.MOMENTUM_SLEEVE_ANTI
                logger.warning(
                    f"STRATEGY 3 MOMENTUM ANTI-MOMENTUM FIRED (HIGH_RISK): "
                    f"QQQ={qqq_ret:.2f}% VXX={vxx_ret:.2f}% VXX_price={vxx_price} "
                    f"→ BUY {vehicle}"
                )
            else:
                vehicle = getattr(self.config, "MOMENTUM_VEHICLE", "TQQQ")
                branch  = RouterBranch.MOMENTUM_SLEEVE
                logger.warning(
                    f"STRATEGY 3 MOMENTUM SLEEVE FIRED: QQQ={qqq_ret:.2f}% → BUY {vehicle}"
                )
            return RouterDecision(
                branch=branch,
                symbol=vehicle,
                entry_time=time(10, 0),
                exit_time=exit_t,
                decision_time=timestamp,
                market_state=self.tape,
                conditions_met={**conditions, "high_risk_regime": high_risk},
            )
        return None

    def _check_router_long(self, timestamp: datetime) -> Optional[RouterDecision]:
        """Strategy 4: QQQ-SPY 40min spread >= +0.2% → BUY TQQQ, exit 15:00."""
        qqq_ret = self.tape.qqq.return_pct()
        spy_ret = self.tape.spy.return_pct()

        min_spread = float(getattr(self.config, "ROUTER_LONG_SPREAD_MIN_PCT", 0.2))
        exit_t     = _parse_hhmm(getattr(self.config, "ROUTER_LONG_EXIT_TIME", "15:00"))
        vehicle    = getattr(self.config, "ROUTER_LONG_VEHICLE", "TQQQ")

        if qqq_ret is None or spy_ret is None:
            return None

        spread = qqq_ret - spy_ret
        conditions = {
            f"QQQ-SPY 40min spread >= +{min_spread}%": spread >= min_spread,
        }
        if all(conditions.values()):
            logger.warning(f"STRATEGY 4 ROUTER LONG FIRED: QQQ={qqq_ret:.2f}% SPY={spy_ret:.2f}% spread={spread:.2f}%")
            return RouterDecision(
                branch=RouterBranch.ROUTER_LONG,
                symbol=vehicle,
                entry_time=time(10, 10),
                exit_time=exit_t,
                decision_time=timestamp,
                market_state=self.tape,
                conditions_met=conditions,
            )
        return None

    def _check_svix_long(self, timestamp: datetime) -> Optional[RouterDecision]:
        """Strategy 5: SVIX 40min >= +0.2% → BUY SVIX, exit 15:00."""
        svix_ret = self.tape.svix.return_pct()

        min_svix = float(getattr(self.config, "SVIX_LONG_MIN_RETURN_PCT", 0.2))
        exit_t   = _parse_hhmm(getattr(self.config, "SVIX_LONG_EXIT_TIME", "15:00"))
        vehicle  = getattr(self.config, "SVIX_LONG_VEHICLE", "SVIX")

        conditions = {
            f"SVIX 40min >= +{min_svix}%": svix_ret is not None and svix_ret >= min_svix,
        }
        if all(conditions.values()):
            logger.warning(f"STRATEGY 5 SVIX LONG FIRED: SVIX={svix_ret:.2f}%")
            return RouterDecision(
                branch=RouterBranch.SVIX_LONG,
                symbol=vehicle,
                entry_time=time(10, 10),
                exit_time=exit_t,
                decision_time=timestamp,
                market_state=self.tape,
                conditions_met=conditions,
            )
        return None

    def reset(self):
        """Reset for new day."""
        self.tape = MarketTape()
        self.decision = None
        self.recording_started = False
        logger.info("ETF router: reset for new day")


def _parse_hhmm(value: str) -> time:
    parts = str(value).split(":")
    return time(int(parts[0]), int(parts[1]))


def parse_router_decision_from_dict(data: dict) -> Optional[RouterDecision]:
    """Rehydrate RouterDecision from persisted state."""
    if not data:
        return None
    try:
        branch = RouterBranch(data.get("branch", "NO_TRADE"))
        return RouterDecision(
            branch=branch,
            symbol=data.get("symbol"),
            entry_time=datetime.strptime(data["entry_time"], "%H:%M:%S").time() if data.get("entry_time") else None,
            exit_time=datetime.strptime(data["exit_time"], "%H:%M:%S").time() if data.get("exit_time") else None,
            decision_time=datetime.fromisoformat(data["decision_time"]) if data.get("decision_time") else None,
            conditions_met=data.get("conditions_met", {}),
        )
    except Exception as e:
        logger.error(f"Failed to parse router decision from state: {e}")
        return None
