"""
Enhanced Position Manager with Trade Reporting Integration
Wraps the morning momentum position manager to add trade reporting
"""
import logging
from typing import Optional
from bot.trade_reporter import log_trade_with_reporting

logger = logging.getLogger(__name__)


class ReportingPositionManager:
    """Wrapper for PositionManager that adds trade reporting"""
    
    def __init__(self, original_pm, strategy_name="morning_momentum"):
        self.original_pm = original_pm
        self.strategy_name = strategy_name
    
    def __getattr__(self, name):
        """Delegate all other attributes to original position manager"""
        return getattr(self.original_pm, name)
    
    def open_position(self, symbol, qty, price, stop_pct, *,
                     entry_time=None, entry_order_id=None, entry_client_order_id=None):
        """Open position with trade reporting"""
        # Call original method
        result = self.original_pm.open_position(
            symbol, qty, price, stop_pct,
            entry_time=entry_time,
            entry_order_id=entry_order_id,
            entry_client_order_id=entry_client_order_id,
        )
        
        # Log trade to reporting system
        try:
            log_trade_with_reporting(
                symbol=symbol,
                action="BUY",
                quantity=qty,
                price=price,
                strategy=self.strategy_name,
                order_id=entry_order_id,
                client_order_id=entry_client_order_id,
                notes=f"Entry with stop: {stop_pct*100:.1f}%"
            )
        except Exception:
            logger.exception("Reporting failed for %s BUY; continuing without reporting", symbol)
        
        return result
    
    def exit_position(self, symbol, price, timestamp, *, reason=""):
        """Exit position with trade reporting"""
        # Get position info before exit
        position = self.original_pm.positions.get(symbol)
        exit_qty = 0
        entry_price = 0
        
        if position:
            exit_qty = position.qty
            entry_price = position.entry_price
        
        # Call original method
        result = self.original_pm.exit_position(symbol, price, timestamp, reason=reason)
        
        # Log trade to reporting system
        if exit_qty > 0:
            try:
                log_trade_with_reporting(
                    symbol=symbol,
                    action="SELL",
                    quantity=exit_qty,
                    price=price,
                    strategy=self.strategy_name,
                    notes=f"Exit: {reason}"
                )
            except Exception:
                logger.exception("Reporting failed for %s SELL; continuing without reporting", symbol)
        
        return result
    
    def force_exit_all(self, prices, *, reason=""):
        """Force exit all positions with trade reporting"""
        # Log all exits before calling original
        for symbol, position in self.original_pm.positions.items():
            if not position.exit_pending and symbol in prices:
                try:
                    log_trade_with_reporting(
                        symbol=symbol,
                        action="SELL",
                        quantity=position.qty,
                        price=prices[symbol],
                        strategy=self.strategy_name,
                        notes=f"Force exit: {reason}"
                    )
                except Exception:
                    logger.exception("Reporting failed for %s force exit; continuing", symbol)
        
        # Call original method
        return self.original_pm.force_exit_all(prices, reason=reason)


def create_reporting_position_manager(original_pm, strategy_name="morning_momentum"):
    """Factory function to create wrapped position manager"""
    return ReportingPositionManager(original_pm, strategy_name)
