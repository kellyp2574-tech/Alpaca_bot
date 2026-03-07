"""
Enhanced Position Manager with Trade Reporting Integration
Wraps the morning momentum position manager to add trade reporting
"""
import logging
from typing import Optional
from bot.trade_reporter import log_trade_with_reporting
from bot.monitoring import get_session_monitor

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
                     entry_time=None, entry_order_id=None, entry_client_order_id=None,
                     gap_at_entry=0.0, first_5min_volume=0.0, fill_pct=100.0, entry_slippage_bps=0.0):
        """Open position with trade reporting"""
        # Call original method
        result = self.original_pm.open_position(
            symbol, qty, price, stop_pct,
            entry_time=entry_time,
            entry_order_id=entry_order_id,
            entry_client_order_id=entry_client_order_id,
            gap_at_entry=gap_at_entry,
            first_5min_volume=first_5min_volume,
            fill_pct=fill_pct,
            entry_slippage_bps=entry_slippage_bps,
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
        entry_time_str = ""
        gap_at_entry = 0.0
        first_5min_volume = 0.0
        fill_pct = 100.0
        entry_slippage_bps = 0.0
        peak_price = 0.0
        stop_price = 0.0
        
        if position:
            exit_qty = position.qty
            entry_price = position.entry_price
            entry_time_str = position.entry_time.isoformat() if position.entry_time else ""
            gap_at_entry = getattr(position, 'gap_at_entry', 0.0)
            first_5min_volume = getattr(position, 'first_5min_volume', 0.0)
            fill_pct = getattr(position, 'fill_pct', 100.0)
            entry_slippage_bps = getattr(position, 'entry_slippage_bps', 0.0)
            peak_price = getattr(position, 'peak_price', entry_price)
            stop_price = getattr(position, 'stop_price', 0.0)
        
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
            
            # Record trade outcome to monitoring system
            try:
                monitor = get_session_monitor()
                mfe = ((peak_price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0
                mae = ((stop_price - entry_price) / entry_price * 100) if entry_price > 0 and stop_price > 0 else 0.0
                monitor.record_trade_outcome(
                    symbol=symbol,
                    entry_time=entry_time_str,
                    exit_time=timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                    entry_price=entry_price,
                    exit_price=price,
                    qty=exit_qty,
                    exit_reason=reason,
                    gap_at_entry=gap_at_entry,
                    first_5min_volume=first_5min_volume,
                    max_favorable_excursion=mfe,
                    max_adverse_excursion=mae,
                    fill_pct=fill_pct,
                    entry_slippage_bps=entry_slippage_bps,
                )
            except Exception:
                logger.exception("Monitoring trade outcome failed for %s; continuing", symbol)
        
        return result
    
    def force_exit_all(self, prices, *, reason=""):
        """Force exit all positions with trade reporting"""
        # Capture position state before exits for monitoring
        pre_exit_positions = {}
        for symbol, position in self.original_pm.positions.items():
            if not position.exit_pending and symbol in prices:
                pre_exit_positions[symbol] = {
                    "qty": position.qty,
                    "entry_price": position.entry_price,
                    "entry_time": position.entry_time.isoformat() if position.entry_time else "",
                    "gap_at_entry": getattr(position, 'gap_at_entry', 0.0),
                    "first_5min_volume": getattr(position, 'first_5min_volume', 0.0),
                    "fill_pct": getattr(position, 'fill_pct', 100.0),
                    "entry_slippage_bps": getattr(position, 'entry_slippage_bps', 0.0),
                    "peak_price": getattr(position, 'peak_price', position.entry_price),
                    "stop_price": getattr(position, 'stop_price', 0.0),
                }
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
        result = self.original_pm.force_exit_all(prices, reason=reason)
        
        # Record trade outcomes to monitoring
        try:
            monitor = get_session_monitor()
            from datetime import datetime
            from zoneinfo import ZoneInfo
            now_str = datetime.now(ZoneInfo("America/New_York")).isoformat()
            for symbol, info in pre_exit_positions.items():
                entry_price = info["entry_price"]
                exit_price = prices.get(symbol, entry_price)
                mfe = ((info["peak_price"] - entry_price) / entry_price * 100) if entry_price > 0 else 0.0
                mae = ((info["stop_price"] - entry_price) / entry_price * 100) if entry_price > 0 and info["stop_price"] > 0 else 0.0
                monitor.record_trade_outcome(
                    symbol=symbol,
                    entry_time=info["entry_time"],
                    exit_time=now_str,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    qty=info["qty"],
                    exit_reason=reason,
                    gap_at_entry=info["gap_at_entry"],
                    first_5min_volume=info["first_5min_volume"],
                    max_favorable_excursion=mfe,
                    max_adverse_excursion=mae,
                    fill_pct=info["fill_pct"],
                    entry_slippage_bps=info["entry_slippage_bps"],
                )
        except Exception:
            logger.exception("Monitoring trade outcomes failed during force exit; continuing")
        
        return result


def create_reporting_position_manager(original_pm, strategy_name="morning_momentum"):
    """Factory function to create wrapped position manager"""
    return ReportingPositionManager(original_pm, strategy_name)
