"""
Trade Reporting System - Tracks all trades and generates statistics
Automatically updates reports after each sell
"""
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from bot import config

logger = logging.getLogger("trade_reporter")


@dataclass
class TradeRecord:
    """Individual trade record"""
    symbol: str
    action: str  # "BUY" or "SELL"
    quantity: float
    price: float
    timestamp: str
    strategy: str  # "morning_momentum" or "etf_rotation"
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class CompletedTrade:
    """Completed buy-sell pair with P&L"""
    symbol: str
    strategy: str
    buy_quantity: float
    buy_price: float
    buy_timestamp: str
    sell_quantity: float
    sell_price: float
    sell_timestamp: str
    buy_order_id: Optional[str] = None
    sell_order_id: Optional[str] = None
    notes: Optional[str] = None
    
    @property
    def buy_value(self) -> float:
        return self.buy_quantity * self.buy_price
    
    @property
    def sell_value(self) -> float:
        return self.sell_quantity * self.sell_price
    
    @property
    def pnl_dollars(self) -> float:
        return self.sell_value - self.buy_value
    
    @property
    def pnl_percentage(self) -> float:
        if self.buy_value == 0:
            return 0.0
        return (self.pnl_dollars / self.buy_value) * 100
    
    @property
    def is_win(self) -> bool:
        return self.pnl_dollars > 0
    
    @property
    def hold_days(self) -> float:
        """Calculate holding period in days"""
        buy_dt = datetime.fromisoformat(self.buy_timestamp.replace('Z', '+00:00'))
        sell_dt = datetime.fromisoformat(self.sell_timestamp.replace('Z', '+00:00'))
        return (sell_dt - buy_dt).total_seconds() / (24 * 3600)


class TradeReporter:
    """Manages trade tracking and statistics generation"""
    
    def __init__(self, report_dir: str = None):
        self.report_dir = Path(report_dir or config.STATE_DIR) / "reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        self.trades_file = self.report_dir / "trades.json"
        self.completed_trades_file = self.report_dir / "completed_trades.json"
        self.stats_file = self.report_dir / "statistics.txt"
        
        self.trades: List[TradeRecord] = []
        self.completed_trades: List[CompletedTrade] = []
        
        self._load_data()
    
    def _load_data(self):
        """Load existing trade data"""
        try:
            if self.trades_file.exists():
                with open(self.trades_file, 'r') as f:
                    trades_data = json.load(f)
                    self.trades = [TradeRecord(**t) for t in trades_data]
            
            if self.completed_trades_file.exists():
                with open(self.completed_trades_file, 'r') as f:
                    completed_data = json.load(f)
                    self.completed_trades = [CompletedTrade(**t) for t in completed_data]
                    
        except Exception as e:
            logger.error(f"Error loading trade data: {e}")
            self.trades = []
            self.completed_trades = []
    
    def _save_data(self):
        """Save trade data to files"""
        try:
            # Save trades
            trades_data = [asdict(t) for t in self.trades]
            with open(self.trades_file, 'w') as f:
                json.dump(trades_data, f, indent=2)
            
            # Save completed trades
            completed_data = [asdict(t) for t in self.completed_trades]
            with open(self.completed_trades_file, 'w') as f:
                json.dump(completed_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving trade data: {e}")
    
    def add_trade(self, symbol: str, action: str, quantity: float, price: float, 
                  strategy: str, order_id: str = None, client_order_id: str = None, 
                  notes: str = None):
        """Add a new trade record"""
        trade = TradeRecord(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            timestamp=datetime.now().isoformat(),
            strategy=strategy,
            order_id=order_id,
            client_order_id=client_order_id,
            notes=notes
        )
        
        self.trades.append(trade)
        self._save_data()
        
        logger.info(f"Trade recorded: {action} {quantity} {symbol} @ ${price:.2f}")
        
        # If this is a sell, try to complete a pair
        if action == "SELL":
            self._try_complete_trade(trade)
    
    def _try_complete_trade(self, sell_trade: TradeRecord):
        """Try to find matching buy trade and complete the pair"""
        # Find unmatched buys for this symbol and strategy
        unmatched_buys = [
            t for t in self.trades 
            if t.action == "BUY" and t.symbol == sell_trade.symbol and 
               t.strategy == sell_trade.strategy and
               not self._is_buy_matched(t)
        ]
        
        if not unmatched_buys:
            logger.warning(f"No matching buy found for sell: {sell_trade.symbol}")
            return
        
        # Use FIFO (first in, first out) - match oldest buy
        buy_trade = unmatched_buys[0]
        
        completed = CompletedTrade(
            symbol=sell_trade.symbol,
            strategy=sell_trade.strategy,
            buy_quantity=buy_trade.quantity,
            buy_price=buy_trade.price,
            buy_timestamp=buy_trade.timestamp,
            sell_quantity=sell_trade.quantity,
            sell_price=sell_trade.price,
            sell_timestamp=sell_trade.timestamp,
            buy_order_id=buy_trade.order_id,
            sell_order_id=sell_trade.order_id,
            notes=f"Buy: {buy_trade.notes or 'N/A'} | Sell: {sell_trade.notes or 'N/A'}"
        )
        
        self.completed_trades.append(completed)
        self._save_data()
        
        # Generate updated statistics
        self.generate_report()
        
        logger.info(f"Completed trade: {sell_trade.symbol} P&L: ${completed.pnl_dollars:+.2f} ({completed.pnl_percentage:+.2f}%)")
    
    def _is_buy_matched(self, buy_trade: TradeRecord) -> bool:
        """Check if a buy trade has already been matched with a sell"""
        for completed in self.completed_trades:
            if (completed.symbol == buy_trade.symbol and 
                completed.strategy == buy_trade.strategy and
                completed.buy_timestamp == buy_trade.timestamp):
                return True
        return False
    
    def calculate_statistics(self) -> Dict:
        """Calculate comprehensive trading statistics"""
        if not self.completed_trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "total_pnl": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "avg_hold_days": 0.0
            }
        
        wins = [t for t in self.completed_trades if t.is_win]
        losses = [t for t in self.completed_trades if not t.is_win]
        
        win_pcts = [t.pnl_percentage for t in wins]
        loss_pcts = [t.pnl_percentage for t in losses]
        
        total_pnl = sum(t.pnl_dollars for t in self.completed_trades)
        
        stats = {
            "total_trades": len(self.completed_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(self.completed_trades) * 100 if self.completed_trades else 0,
            "avg_win_pct": sum(win_pcts) / len(win_pcts) if win_pcts else 0,
            "avg_loss_pct": sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0,
            "total_pnl": total_pnl,
            "largest_win": max((t.pnl_dollars for t in wins), default=0),
            "largest_loss": min((t.pnl_dollars for t in losses), default=0),
            "avg_hold_days": sum(t.hold_days for t in self.completed_trades) / len(self.completed_trades),
            "strategy_breakdown": self._get_strategy_breakdown()
        }
        
        return stats
    
    def _get_strategy_breakdown(self) -> Dict:
        """Get statistics broken down by strategy"""
        breakdown = {}
        
        for strategy in ["morning_momentum", "etf_rotation"]:
            strategy_trades = [t for t in self.completed_trades if t.strategy == strategy]
            if strategy_trades:
                wins = [t for t in strategy_trades if t.is_win]
                breakdown[strategy] = {
                    "total_trades": len(strategy_trades),
                    "wins": len(wins),
                    "win_rate": len(wins) / len(strategy_trades) * 100,
                    "avg_pct": sum(t.pnl_percentage for t in strategy_trades) / len(strategy_trades),
                    "total_pnl": sum(t.pnl_dollars for t in strategy_trades)
                }
            else:
                breakdown[strategy] = {
                    "total_trades": 0,
                    "wins": 0,
                    "win_rate": 0,
                    "avg_pct": 0,
                    "total_pnl": 0
                }
        
        return breakdown
    
    def generate_report(self):
        """Generate and save comprehensive statistics report"""
        stats = self.calculate_statistics()
        
        report = []
        report.append("=" * 60)
        report.append("TRADE PERFORMANCE REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        # Overall Statistics
        report.append("\nOVERALL PERFORMANCE:")
        report.append(f"Total Trades: {stats['total_trades']}")
        report.append(f"Win Rate: {stats['win_rate']:.1f}%")
        report.append(f"Winning Trades: {stats['winning_trades']}")
        report.append(f"Losing Trades: {stats['losing_trades']}")
        report.append(f"Average Win %: {stats['avg_win_pct']:.2f}%")
        report.append(f"Average Loss %: {stats['avg_loss_pct']:.2f}%")
        report.append(f"Total P&L: ${stats['total_pnl']:,.2f}")
        report.append(f"Largest Win: ${stats['largest_win']:,.2f}")
        report.append(f"Largest Loss: ${stats['largest_loss']:,.2f}")
        report.append(f"Average Hold Days: {stats['avg_hold_days']:.1f}")
        
        # Strategy Breakdown
        report.append("\nSTRATEGY BREAKDOWN:")
        for strategy, data in stats['strategy_breakdown'].items():
            report.append(f"\n{strategy.replace('_', ' ').title()}:")
            report.append(f"  Trades: {data['total_trades']}")
            report.append(f"  Win Rate: {data['win_rate']:.1f}%")
            report.append(f"  Avg %: {data['avg_pct']:.2f}%")
            report.append(f"  P&L: ${data['total_pnl']:,.2f}")
        
        # Recent Trades
        report.append("\nRECENT COMPLETED TRADES (Last 10):")
        recent_trades = sorted(self.completed_trades, key=lambda x: x.sell_timestamp, reverse=True)[:10]
        
        for trade in recent_trades:
            report.append(
                f"{trade.sell_timestamp[:10]} {trade.symbol:<6} "
                f"${trade.pnl_dollars:+8.2f} ({trade.pnl_percentage:+6.2f}%) "
                f"{trade.strategy.replace('_', ' '):<15} "
                f"{trade.hold_days:.1f}d"
            )
        
        # Open Positions
        open_positions = self._get_open_positions()
        if open_positions:
            report.append("\nOPEN POSITIONS:")
            for symbol, pos in open_positions.items():
                report.append(f"{symbol}: {pos['quantity']} @ ${pos['price']:.2f} ({pos['strategy']})")
        
        report.append("\n" + "=" * 60)
        
        # Save report
        report_text = "\n".join(report)
        try:
            with open(self.stats_file, 'w') as f:
                f.write(report_text)
        except Exception as e:
            logger.error(f"Error saving report: {e}")
        
        # Log to console
        logger.info("Trade report updated")
        for line in report[:15]:  # Log first 15 lines to console
            logger.info(line)
    
    def _get_open_positions(self) -> Dict:
        """Get current open positions (unmatched buys)"""
        open_positions = {}
        
        for trade in self.trades:
            if trade.action == "BUY" and not self._is_buy_matched(trade):
                if trade.symbol not in open_positions:
                    open_positions[trade.symbol] = {
                        "quantity": trade.quantity,
                        "price": trade.price,
                        "strategy": trade.strategy,
                        "timestamp": trade.timestamp
                    }
                else:
                    # Aggregate multiple buys for same symbol
                    existing = open_positions[trade.symbol]
                    total_value = (existing["quantity"] * existing["price"] + 
                                 trade.quantity * trade.price)
                    total_quantity = existing["quantity"] + trade.quantity
                    existing["quantity"] = total_quantity
                    existing["price"] = total_value / total_quantity if total_quantity > 0 else 0
        
        return open_positions
    
    def get_recent_trades(self, days: int = 30) -> List[CompletedTrade]:
        """Get trades from last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            t for t in self.completed_trades 
            if datetime.fromisoformat(t.sell_timestamp.replace('Z', '+00:00')) > cutoff_date
        ]


# Global instance for easy access
_trade_reporter = None

def get_trade_reporter() -> TradeReporter:
    """Get global trade reporter instance"""
    global _trade_reporter
    if _trade_reporter is None:
        _trade_reporter = TradeReporter()
    return _trade_reporter


def log_trade_with_reporting(symbol: str, action: str, quantity: float, price: float, 
                            strategy: str, order_id: str = None, client_order_id: str = None, 
                            notes: str = None):
    """Convenience function to log trade and update reporting"""
    reporter = get_trade_reporter()
    reporter.add_trade(
        symbol=symbol,
        action=action,
        quantity=quantity,
        price=price,
        strategy=strategy,
        order_id=order_id,
        client_order_id=client_order_id,
        notes=notes
    )
