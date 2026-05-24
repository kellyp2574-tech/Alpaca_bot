"""ETF Router - 9:30-10:00 AM market state measurement and 10:00 AM decision logic.

This module handles the ETF routing decision with 7 priority branches:
1. A++ long (TQQQ)
2. A long (TQQQ)
3. A- long (TQQQ)
4. A- weak (TQQQ)
5. SQQQ Goldilocks
6. UVXY crash
7. No trade
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, date
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)


class RouterBranch(Enum):
    """ETF router decision branches in priority order."""
    A_PLUS_PLUS_LONG = "A++ long"
    A_LONG = "A long"
    A_MINUS_LONG = "A- long"
    A_MINUS_WEAK = "A- weak"
    SQQQ_GOLDILOCKS = "SQQQ Goldilocks"
    UVXY_CRASH = "UVXY crash"
    NO_TRADE = "No trade"


@dataclass
class ETFTapeSnapshot:
    """Real-time tape data for a single ETF from 9:30-10:00."""
    symbol: str
    open_930: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    latest_price: Optional[float] = None
    latest_time: Optional[datetime] = None
    price_0945: Optional[float] = None  # Price at 09:45 for continuation check
    
    def update(self, price: float, timestamp: datetime):
        """Update snapshot with new price data.

        ``open_930`` is treated as immutable after the first seed (normally
        from ``ETFRouter.start_recording``). This guarantees a later tape
        print can never silently overwrite the recorded 9:30 open even if a
        future refactor forgets to pre-populate ``latest_price``.
        """
        # Seed open_930 only if it has never been set. Do NOT use latest_price
        # as the seed gate — it's a separate field with a separate purpose.
        if self.open_930 is None:
            self.open_930 = price

        # high/low always track the realized 9:30-10:00 range.
        self.high = price if self.high is None else max(self.high, price)
        self.low = price if self.low is None else min(self.low, price)

        # Capture 09:45 price for continuation check
        ts_time = timestamp.time() if isinstance(timestamp, datetime) else timestamp
        if ts_time >= time(9, 45) and self.price_0945 is None:
            self.price_0945 = price

        self.latest_price = price
        self.latest_time = timestamp
    
    def continued_up_from_0945(self) -> Optional[bool]:
        """Check if price continued up from 09:45 to 10:00."""
        if self.price_0945 is None or self.latest_price is None:
            return None
        return self.latest_price > self.price_0945
    
    def return_bps(self) -> Optional[float]:
        """Return vs 9:30 open in basis points."""
        if self.latest_price is None or self.open_930 is None or self.open_930 == 0:
            return None
        return (self.latest_price / self.open_930 - 1.0) * 10000
    
    def range_position(self) -> Optional[float]:
        """Position within 9:30-10:00 range (0=at low, 1=at high)."""
        if self.latest_price is None or self.low is None or self.high is None:
            return None
        range_size = self.high - self.low
        if range_size == 0:
            return 0.5
        return (self.latest_price - self.low) / range_size
    
    def is_valid(self) -> bool:
        """Check if we have valid data for decision making."""
        return (
            self.open_930 is not None 
            and self.latest_price is not None
            and self.high is not None
            and self.low is not None
        )


@dataclass
class MarketState0930to1000:
    """Complete market state snapshot at 10:00 AM decision time."""
    qqq: ETFTapeSnapshot = field(default_factory=lambda: ETFTapeSnapshot("QQQ"))
    spy: ETFTapeSnapshot = field(default_factory=lambda: ETFTapeSnapshot("SPY"))
    iwm: ETFTapeSnapshot = field(default_factory=lambda: ETFTapeSnapshot("IWM"))
    xlk: ETFTapeSnapshot = field(default_factory=lambda: ETFTapeSnapshot("XLK"))
    vxx: ETFTapeSnapshot = field(default_factory=lambda: ETFTapeSnapshot("VXX"))
    sqqq: ETFTapeSnapshot = field(default_factory=lambda: ETFTapeSnapshot("SQQQ"))
    uvxy: ETFTapeSnapshot = field(default_factory=lambda: ETFTapeSnapshot("UVXY"))
    tqqq: ETFTapeSnapshot = field(default_factory=lambda: ETFTapeSnapshot("TQQQ"))
    
    snapshot_time: Optional[datetime] = None
    
    def all_valid(self) -> bool:
        """Check if all required symbols have valid data."""
        return all([
            self.qqq.is_valid(),
            self.spy.is_valid(),
            self.iwm.is_valid(),
            self.xlk.is_valid(),
            self.vxx.is_valid(),
            self.sqqq.is_valid(),
            self.uvxy.is_valid(),
            self.tqqq.is_valid(),
        ])
    
    def get_returns_summary(self) -> Dict[str, Optional[float]]:
        """Get all returns in bps for logging."""
        return {
            "QQQ": self.qqq.return_bps(),
            "SPY": self.spy.return_bps(),
            "IWM": self.iwm.return_bps(),
            "XLK": self.xlk.return_bps(),
            "VXX": self.vxx.return_bps(),
            "SQQQ": self.sqqq.return_bps(),
            "UVXY": self.uvxy.return_bps(),
            "TQQQ": self.tqqq.return_bps(),
        }
    
    def get_range_positions(self) -> Dict[str, Optional[float]]:
        """Get all range positions for logging."""
        return {
            "QQQ": self.qqq.range_position(),
            "VXX": self.vxx.range_position(),
            "UVXY": self.uvxy.range_position(),
        }


@dataclass
class RouterDecision:
    """Result of the 10:00 AM ETF router decision."""
    branch: RouterBranch
    symbol: Optional[str] = None  # TQQQ, SQQQ, or UVXY
    entry_time: Optional[time] = None
    exit_time: Optional[time] = None
    decision_time: Optional[datetime] = None
    market_state: Optional[MarketState0930to1000] = None
    
    # Detailed conditions that fired
    conditions_met: Dict[str, Any] = field(default_factory=dict)
    
    def mr_blocked(self) -> bool:
        """MR is blocked if any ETF trade was made."""
        return self.branch != RouterBranch.NO_TRADE
    
    def to_dict(self) -> dict:
        """Serialize for state persistence."""
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
    """ETF Router implementing the 7-priority-branch decision tree."""
    
    def __init__(self, config):
        self.config = config
        self.tape = MarketState0930to1000()
        self.decision: Optional[RouterDecision] = None
        self.recording_started = False
    
    def start_recording(self, opens_930: Dict[str, float], timestamp: datetime):
        """Initialize tape with 9:30 opens."""
        for symbol, price in opens_930.items():
            snapshot = getattr(self.tape, symbol.lower(), None)
            if snapshot:
                snapshot.open_930 = price
                snapshot.high = price
                snapshot.low = price
                snapshot.latest_price = price
                snapshot.latest_time = timestamp
        
        self.recording_started = True
        logger.info(f"ETF router: Started 9:30-10:00 tape recording. Opens: {opens_930}")
    
    def update_tape(self, symbol: str, price: float, timestamp: datetime):
        """Update tape with latest price."""
        if not self.recording_started:
            return
        
        snapshot = getattr(self.tape, symbol.lower(), None)
        if snapshot:
            snapshot.update(price, timestamp)
    
    def make_decision(self, timestamp: datetime) -> RouterDecision:
        """Execute the 7-priority-branch decision tree at 10:00 AM."""
        self.tape.snapshot_time = timestamp
        
        # Validate we have all required data
        if not self.tape.all_valid():
            missing = []
            for sym in ["qqq", "spy", "iwm", "xlk", "vxx", "sqqq", "uvxy", "tqqq"]:
                snap = getattr(self.tape, sym)
                if not snap.is_valid():
                    missing.append(sym.upper())
            logger.warning(f"ETF router: Missing data for {missing}, cannot make decision")
            return RouterDecision(
                branch=RouterBranch.NO_TRADE,
                decision_time=timestamp,
                market_state=self.tape,
                conditions_met={"error": f"missing_data: {missing}"},
            )
        
        # Log market state
        returns = self.tape.get_returns_summary()
        range_pos = self.tape.get_range_positions()
        logger.info(f"ETF router 10:00 state - Returns (bps): {returns}")
        logger.info(f"ETF router 10:00 state - Range positions: {range_pos}")
        
        # Branch 1: A++ long (TQQQ)
        decision = self._check_branch_1_a_plus_plus(timestamp)
        if decision:
            return decision
        
        # Branch 2: A long (TQQQ)
        decision = self._check_branch_2_a_long(timestamp)
        if decision:
            return decision
        
        # Branch 3: A- long (TQQQ)
        decision = self._check_branch_3_a_minus_long(timestamp)
        if decision:
            return decision
        
        # Branch 4: A- weak (TQQQ)
        decision = self._check_branch_4_a_minus_weak(timestamp)
        if decision:
            return decision
        
        # Branch 5: SQQQ Goldilocks
        decision = self._check_branch_5_sqqq_goldilocks(timestamp)
        if decision:
            return decision
        
        # Branch 6: UVXY crash
        decision = self._check_branch_6_uvxy_crash(timestamp)
        if decision:
            return decision
        
        # Branch 7: No trade
        logger.info("ETF router: No branch fired - staying flat, MR allowed")
        return RouterDecision(
            branch=RouterBranch.NO_TRADE,
            decision_time=timestamp,
            market_state=self.tape,
            conditions_met={},
        )
    
    def _check_branch_1_a_plus_plus(self, timestamp: datetime) -> Optional[RouterDecision]:
        """Branch 1: A++ long - strongest bullish conditions."""
        qqq_ret = self.tape.qqq.return_bps()
        xlk_ret = self.tape.xlk.return_bps()
        spy_ret = self.tape.spy.return_bps()
        iwm_ret = self.tape.iwm.return_bps()
        vxx_ret = self.tape.vxx.return_bps()
        qqq_range_pos = self.tape.qqq.range_position()
        qqq_continued = self.tape.qqq.continued_up_from_0945()
        
        conditions = {
            "QQQ > +10 bps": qqq_ret is not None and qqq_ret > 10,
            "XLK > 0 bps": xlk_ret is not None and xlk_ret > 0,
            "SPY > 0 bps": spy_ret is not None and spy_ret > 0,
            "IWM > 0 bps": iwm_ret is not None and iwm_ret > 0,
            "VXX < 0 bps": vxx_ret is not None and vxx_ret < 0,
            "QQQ range_pos >= 0.80": qqq_range_pos is not None and qqq_range_pos >= 0.80,
            "QQQ continued up 09:45->10:00": qqq_continued is True,
        }
        
        all_met = all(conditions.values())
        
        if all_met:
            logger.info(f"ETF router: BRANCH 1 A++ LONG FIRED - Conditions: {conditions}")
            return RouterDecision(
                branch=RouterBranch.A_PLUS_PLUS_LONG,
                symbol="TQQQ",
                entry_time=time(10, 1),  # Enter strictly after 10:00
                exit_time=time(15, 0),
                decision_time=timestamp,
                market_state=self.tape,
                conditions_met=conditions,
            )
        return None
    
    def _check_branch_2_a_long(self, timestamp: datetime) -> Optional[RouterDecision]:
        """Branch 2: A long - strong bullish conditions."""
        qqq_ret = self.tape.qqq.return_bps()
        xlk_ret = self.tape.xlk.return_bps()
        vxx_ret = self.tape.vxx.return_bps()
        iwm_ret = self.tape.iwm.return_bps()
        spy_ret = self.tape.spy.return_bps()
        qqq_range_pos = self.tape.qqq.range_position()
        
        # Check continuation from 09:45 to 10:00 (price higher now than at 09:45)
        continued_up = self.tape.qqq.continued_up_from_0945() is True
        range_ok = qqq_range_pos is not None and qqq_range_pos >= 0.50
        
        conditions = {
            "QQQ > +10 bps": qqq_ret is not None and qqq_ret > 10,
            "XLK > 0 bps": xlk_ret is not None and xlk_ret > 0,
            "VXX < 0 bps": vxx_ret is not None and vxx_ret < 0,
            "IWM > 0 bps": iwm_ret is not None and iwm_ret > 0,
            "SPY > -5 bps": spy_ret is not None and spy_ret > -5,
            "QQQ continued up OR range_pos >= 0.50": continued_up or range_ok,
        }
        
        all_met = all(conditions.values())
        
        if all_met:
            logger.info(f"ETF router: BRANCH 2 A LONG FIRED - Conditions: {conditions}")
            return RouterDecision(
                branch=RouterBranch.A_LONG,
                symbol="TQQQ",
                entry_time=time(10, 1),  # Enter strictly after 10:00
                exit_time=time(15, 0),
                decision_time=timestamp,
                market_state=self.tape,
                conditions_met=conditions,
            )
        return None
    
    def _check_branch_3_a_minus_long(self, timestamp: datetime) -> Optional[RouterDecision]:
        """Branch 3: A- long - moderate bullish conditions."""
        qqq_ret = self.tape.qqq.return_bps()
        xlk_ret = self.tape.xlk.return_bps()
        vxx_ret = self.tape.vxx.return_bps()
        iwm_ret = self.tape.iwm.return_bps()
        spy_ret = self.tape.spy.return_bps()
        
        conditions = {
            "QQQ > +10 bps": qqq_ret is not None and qqq_ret > 10,
            "XLK > 0 bps": xlk_ret is not None and xlk_ret > 0,
            "VXX < 0 bps": vxx_ret is not None and vxx_ret < 0,
            "IWM > -10 bps": iwm_ret is not None and iwm_ret > -10,
            "SPY > -5 bps": spy_ret is not None and spy_ret > -5,
        }
        
        all_met = all(conditions.values())
        
        if all_met:
            logger.info(f"ETF router: BRANCH 3 A- LONG FIRED - Conditions: {conditions}")
            return RouterDecision(
                branch=RouterBranch.A_MINUS_LONG,
                symbol="TQQQ",
                entry_time=time(10, 1),
                exit_time=time(15, 0),
                decision_time=timestamp,
                market_state=self.tape,
                conditions_met=conditions,
            )
        return None
    
    def _check_branch_4_a_minus_weak(self, timestamp: datetime) -> Optional[RouterDecision]:
        """Branch 4: A- weak - weaker bullish conditions."""
        qqq_ret = self.tape.qqq.return_bps()
        xlk_ret = self.tape.xlk.return_bps()
        vxx_ret = self.tape.vxx.return_bps()
        iwm_ret = self.tape.iwm.return_bps()
        spy_ret = self.tape.spy.return_bps()
        
        conditions = {
            "QQQ between +5 and +10 bps": qqq_ret is not None and 5 <= qqq_ret <= 10,
            "XLK > 0 bps": xlk_ret is not None and xlk_ret > 0,
            "VXX < 0 bps": vxx_ret is not None and vxx_ret < 0,
            "IWM > 0 bps": iwm_ret is not None and iwm_ret > 0,
            "SPY > -5 bps": spy_ret is not None and spy_ret > -5,
        }
        
        all_met = all(conditions.values())
        
        if all_met:
            logger.info(f"ETF router: BRANCH 4 A- WEAK FIRED - Conditions: {conditions}")
            return RouterDecision(
                branch=RouterBranch.A_MINUS_WEAK,
                symbol="TQQQ",
                entry_time=time(10, 1),
                exit_time=time(15, 0),
                decision_time=timestamp,
                market_state=self.tape,
                conditions_met=conditions,
            )
        return None
    
    def _check_branch_5_sqqq_goldilocks(self, timestamp: datetime) -> Optional[RouterDecision]:
        """Branch 5: SQQQ Goldilocks - short Nasdaq conditions."""
        sqqq_ret = self.tape.sqqq.return_bps()
        qqq_ret = self.tape.qqq.return_bps()
        
        conditions = {
            "SQQQ > 0 bps": sqqq_ret is not None and sqqq_ret > 0,
            "QQQ between -30 and -60 bps": qqq_ret is not None and -60 <= qqq_ret <= -30,
            "SQQQ between +100 and +150 bps": sqqq_ret is not None and 100 <= sqqq_ret <= 150,
            "QQQ > -80 bps (safety)": qqq_ret is not None and qqq_ret > -80,
            "SQQQ < +250 bps (safety)": sqqq_ret is not None and sqqq_ret < 250,
        }
        
        all_met = all(conditions.values())
        
        if all_met:
            logger.info(f"ETF router: BRANCH 5 SQQQ GOLDILOCKS FIRED - Conditions: {conditions}")
            return RouterDecision(
                branch=RouterBranch.SQQQ_GOLDILOCKS,
                symbol="SQQQ",
                entry_time=time(10, 1),
                exit_time=time(14, 0),
                decision_time=timestamp,
                market_state=self.tape,
                conditions_met=conditions,
            )
        return None
    
    def _check_branch_6_uvxy_crash(self, timestamp: datetime) -> Optional[RouterDecision]:
        """Branch 6: UVXY crash - volatility spike conditions."""
        qqq_ret = self.tape.qqq.return_bps()
        vxx_ret = self.tape.vxx.return_bps()
        vxx_range_pos = self.tape.vxx.range_position()
        uvxy_ret = self.tape.uvxy.return_bps()
        uvxy_range_pos = self.tape.uvxy.range_position()
        
        conditions = {
            "QQQ < -60 bps": qqq_ret is not None and qqq_ret < -60,
            "VXX > 0 bps": vxx_ret is not None and vxx_ret > 0,
            "VXX range_pos >= 0.80": vxx_range_pos is not None and vxx_range_pos >= 0.80,
            "UVXY > 0 bps": uvxy_ret is not None and uvxy_ret > 0,
            "UVXY range_pos >= 0.80": uvxy_range_pos is not None and uvxy_range_pos >= 0.80,
            "UVXY < +500 bps (safety)": uvxy_ret is not None and uvxy_ret < 500,
        }
        
        all_met = all(conditions.values())
        
        if all_met:
            logger.info(f"ETF router: BRANCH 6 UVXY CRASH FIRED - Conditions: {conditions}")
            return RouterDecision(
                branch=RouterBranch.UVXY_CRASH,
                symbol="UVXY",
                entry_time=time(10, 1),
                exit_time=time(11, 0),
                decision_time=timestamp,
                market_state=self.tape,
                conditions_met=conditions,
            )
        return None
    
    def reset(self):
        """Reset router for new day."""
        self.tape = MarketState0930to1000()
        self.decision = None
        self.recording_started = False
        logger.info("ETF router: Reset for new day")


def parse_router_decision_from_dict(data: dict) -> Optional[RouterDecision]:
    """Rehydrate RouterDecision from persisted state."""
    if not data:
        return None
    
    try:
        branch = RouterBranch(data.get("branch", "No trade"))
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
