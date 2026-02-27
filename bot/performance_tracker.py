"""Performance metrics tracking for adaptive allocation and monitoring."""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional

from .clock import market_now

logger = logging.getLogger(__name__)


@dataclass
class TradeMetrics:
    """Metrics for a single trade."""
    symbol: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    realized_slippage_pct: float
    participation_pct: float  # % of 5-min volume
    max_adverse_excursion_pct: float  # Max drawdown during trade
    hold_duration_minutes: float


@dataclass
class DailyMetrics:
    """Daily aggregate metrics."""
    date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy: float
    avg_slippage_pct: float
    avg_participation_pct: float
    max_adverse_excursion_pct: float
    largest_win: float
    largest_loss: float
    
    # Rolling metrics (30-day)
    rolling_30d_expectancy: Optional[float] = None
    rolling_30d_win_rate: Optional[float] = None
    rolling_30d_avg_slippage: Optional[float] = None


class PerformanceTracker:
    """Tracks performance metrics and manages adaptive allocation."""
    
    def __init__(self, config):
        self.config = config
        self.metrics_file = Path(config.metrics_log_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Rolling windows for adaptive allocation
        self.recent_slippages: Deque[float] = deque(maxlen=config.slippage_lookback_trades)
        self.recent_trades: Deque[TradeMetrics] = deque(maxlen=100)  # Last 100 trades
        
        # Daily tracking
        self.daily_trades: List[TradeMetrics] = []
        self.current_date: Optional[str] = None
        
        # Allocation state
        self.allocation_reduced = False
        self.current_allocation_pct = config.daily_deploy_pct
        
    def log_trade(
        self,
        symbol: str,
        entry_time: datetime,
        exit_time: datetime,
        entry_price: float,
        exit_price: float,
        qty: float,
        expected_entry_price: float,
        five_min_volume: float,
        max_adverse_excursion_pct: float = 0.0,
    ) -> None:
        """Log a completed trade and update metrics."""
        
        # Calculate metrics
        pnl = (exit_price - entry_price) * qty
        pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
        
        # Realized slippage (entry only, as exit is market conditions)
        realized_slippage_pct = abs(entry_price - expected_entry_price) / expected_entry_price if expected_entry_price > 0 else 0.0
        
        # Participation rate
        trade_dollar_volume = qty * entry_price
        participation_pct = trade_dollar_volume / five_min_volume if five_min_volume > 0 else 0.0
        
        # Hold duration
        hold_duration_minutes = (exit_time - entry_time).total_seconds() / 60.0
        
        trade_metrics = TradeMetrics(
            symbol=symbol,
            entry_time=entry_time.isoformat(),
            exit_time=exit_time.isoformat(),
            entry_price=entry_price,
            exit_price=exit_price,
            qty=qty,
            pnl=pnl,
            pnl_pct=pnl_pct,
            realized_slippage_pct=realized_slippage_pct,
            participation_pct=participation_pct,
            max_adverse_excursion_pct=max_adverse_excursion_pct,
            hold_duration_minutes=hold_duration_minutes,
        )
        
        # Update rolling windows
        self.recent_slippages.append(realized_slippage_pct)
        self.recent_trades.append(trade_metrics)
        
        # Update daily tracking
        today = market_now().date().isoformat()
        if self.current_date != today:
            if self.current_date is not None and self.daily_trades:
                self._save_daily_metrics()
            self.current_date = today
            self.daily_trades = []
        
        self.daily_trades.append(trade_metrics)
        
        # Check for allocation adjustment
        self._check_allocation_adjustment()
        
        logger.info(
            f"Trade logged: {symbol} PnL=${pnl:.2f} ({pnl_pct*100:.2f}%) "
            f"Slippage={realized_slippage_pct*100:.3f}% Participation={participation_pct*100:.2f}%"
        )
    
    def _check_allocation_adjustment(self) -> None:
        """Check if allocation should be reduced based on slippage."""
        if len(self.recent_slippages) < self.config.slippage_lookback_trades:
            return  # Not enough data yet
        
        avg_slippage = sum(self.recent_slippages) / len(self.recent_slippages)
        
        if avg_slippage > self.config.slippage_threshold_pct and not self.allocation_reduced:
            logger.warning(
                f"⚠️ ALLOCATION REDUCED: Avg slippage {avg_slippage*100:.2f}% > "
                f"threshold {self.config.slippage_threshold_pct*100:.2f}% "
                f"over last {len(self.recent_slippages)} trades. "
                f"Reducing allocation from {self.current_allocation_pct*100:.0f}% to "
                f"{self.config.reduced_allocation_pct*100:.0f}%"
            )
            self.allocation_reduced = True
            self.current_allocation_pct = self.config.reduced_allocation_pct
            
        elif avg_slippage <= self.config.slippage_threshold_pct * 0.75 and self.allocation_reduced:
            # Restore allocation if slippage improves significantly
            logger.info(
                f"✅ ALLOCATION RESTORED: Avg slippage {avg_slippage*100:.2f}% improved. "
                f"Restoring allocation to {self.config.daily_deploy_pct*100:.0f}%"
            )
            self.allocation_reduced = False
            self.current_allocation_pct = self.config.daily_deploy_pct
    
    def get_current_allocation_pct(self) -> float:
        """Get current allocation percentage (may be reduced due to slippage)."""
        return self.current_allocation_pct
    
    def _save_daily_metrics(self) -> None:
        """Save daily aggregate metrics to log file."""
        if not self.daily_trades:
            return
        
        # Calculate daily aggregates
        total_trades = len(self.daily_trades)
        winning_trades = sum(1 for t in self.daily_trades if t.pnl > 0)
        losing_trades = sum(1 for t in self.daily_trades if t.pnl < 0)
        
        total_pnl = sum(t.pnl for t in self.daily_trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        wins = [t.pnl for t in self.daily_trades if t.pnl > 0]
        losses = [t.pnl for t in self.daily_trades if t.pnl < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        avg_slippage_pct = sum(t.realized_slippage_pct for t in self.daily_trades) / total_trades
        avg_participation_pct = sum(t.participation_pct for t in self.daily_trades) / total_trades
        max_adverse_excursion_pct = max(t.max_adverse_excursion_pct for t in self.daily_trades)
        
        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0
        
        # Calculate 30-day rolling metrics
        rolling_30d_expectancy, rolling_30d_win_rate, rolling_30d_avg_slippage = self._calculate_rolling_metrics()
        
        daily_metrics = DailyMetrics(
            date=self.current_date,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            expectancy=expectancy,
            avg_slippage_pct=avg_slippage_pct,
            avg_participation_pct=avg_participation_pct,
            max_adverse_excursion_pct=max_adverse_excursion_pct,
            largest_win=largest_win,
            largest_loss=largest_loss,
            rolling_30d_expectancy=rolling_30d_expectancy,
            rolling_30d_win_rate=rolling_30d_win_rate,
            rolling_30d_avg_slippage=rolling_30d_avg_slippage,
        )
        
        # Append to JSONL file
        try:
            with self.metrics_file.open("a") as f:
                f.write(json.dumps(asdict(daily_metrics)) + "\n")
            logger.info(f"📊 Daily metrics saved: {self.current_date} - {total_trades} trades, PnL=${total_pnl:.2f}")
        except Exception as e:
            logger.error(f"Failed to save daily metrics: {e}")
    
    def _calculate_rolling_metrics(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate 30-day rolling metrics from recent trades."""
        if len(self.recent_trades) < 10:
            return None, None, None
        
        # Use last 30 days of trades (approximated by recent_trades deque)
        trades = list(self.recent_trades)
        
        total = len(trades)
        winners = sum(1 for t in trades if t.pnl > 0)
        
        win_rate = winners / total if total > 0 else 0.0
        
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        avg_slippage = sum(t.realized_slippage_pct for t in trades) / total
        
        return expectancy, win_rate, avg_slippage
    
    def finalize_day(self) -> None:
        """Finalize and save metrics at end of day."""
        if self.daily_trades:
            self._save_daily_metrics()
