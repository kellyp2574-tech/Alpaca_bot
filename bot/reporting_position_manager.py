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
                     gap_at_entry=0.0, first_5min_volume=0.0, fill_pct=100.0,
                     entry_slippage_bps=0.0, report_trade=False):
        """Open position.  BUY reporting is NOT done here.

        BUY trades are reported explicitly at the confirmed-fill call site
        (e.g. _execute_entry) so that reconciliation/adoption paths never
        produce duplicate BUY records.  The report_trade kwarg is retained
        for backward compatibility but defaults to False and is ignored.
        """
        return self.original_pm.open_position(
            symbol, qty, price, stop_pct,
            entry_time=entry_time,
            entry_order_id=entry_order_id,
            entry_client_order_id=entry_client_order_id,
            gap_at_entry=gap_at_entry,
            first_5min_volume=first_5min_volume,
            fill_pct=fill_pct,
            entry_slippage_bps=entry_slippage_bps,
        )
    
    def _read_realized_exit(self, symbol, fallback_price):
        """Read and consume actual fill metadata from PositionManager._realized_exits.

        Returns a dict with exit_price, exit_qty, and full metadata.
        The record is popped (removed) after reading to prevent memory leak
        over long-running sessions.
        Falls back to None if no realized data exists.
        """
        exits = getattr(self.original_pm, "_realized_exits", {})
        return exits.pop(symbol, None)

    def exit_position(self, symbol, price, timestamp, *, reason=""):
        """Exit position with trade reporting.

        IMPORTANT: Only records the sell AFTER confirming the position was
        actually removed by the underlying position manager.  Uses the actual
        broker fill price from _realized_exits, not the requested price.
        """
        # Capture pre-exit metadata (position may be removed after call)
        position = self.original_pm.positions.get(symbol)
        if not position:
            return self.original_pm.exit_position(symbol, price, timestamp, reason=reason)

        pre_exit = {
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

        result = self.original_pm.exit_position(symbol, price, timestamp, reason=reason)

        # Only record if the position was actually removed (confirmed exit)
        if symbol in self.original_pm.positions:
            logger.debug("exit_position(%s): position still tracked -- skipping sell report", symbol)
            return result

        # Read actual fill price from realized exits (set by _record_fill)
        realized = self._read_realized_exit(symbol, price)
        actual_price = realized["exit_price"] if realized else price
        actual_qty = realized["exit_qty"] if realized else pre_exit["qty"]
        info = realized if realized else pre_exit

        try:
            log_trade_with_reporting(
                symbol=symbol,
                action="SELL",
                quantity=actual_qty,
                price=actual_price,
                strategy=self.strategy_name,
                notes=f"Exit: {reason}"
            )
        except Exception:
            logger.exception("Reporting failed for %s SELL; continuing without reporting", symbol)

        # Record trade outcome to monitoring system
        try:
            monitor = get_session_monitor()
            entry_price = info.get("entry_price", pre_exit["entry_price"])
            peak_price = info.get("peak_price", pre_exit["peak_price"])
            stop_price = info.get("stop_price", pre_exit["stop_price"])
            mfe = ((peak_price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0
            mae = ((stop_price - entry_price) / entry_price * 100) if entry_price > 0 and stop_price > 0 else 0.0
            exit_time_str = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
            monitor.record_trade_outcome(
                symbol=symbol,
                entry_time=info.get("entry_time", pre_exit["entry_time"]),
                exit_time=exit_time_str,
                entry_price=entry_price,
                exit_price=actual_price,
                qty=actual_qty,
                exit_reason=reason,
                gap_at_entry=info.get("gap_at_entry", pre_exit["gap_at_entry"]),
                first_5min_volume=info.get("first_5min_volume", pre_exit["first_5min_volume"]),
                max_favorable_excursion=mfe,
                max_adverse_excursion=mae,
                fill_pct=info.get("fill_pct", pre_exit["fill_pct"]),
                entry_slippage_bps=info.get("entry_slippage_bps", pre_exit["entry_slippage_bps"]),
            )
        except Exception:
            logger.exception("Monitoring trade outcome failed for %s; continuing", symbol)

        return result

    def force_exit_all(self, prices, *, reason=""):
        """Force exit all positions with trade reporting.

        IMPORTANT: Records sells AFTER the exit, only for positions that were
        actually removed.  Uses actual broker fill prices from _realized_exits.
        """
        # Capture pre-exit symbol set to detect which were removed
        pre_exit_symbols = set(self.original_pm.positions.keys())

        # Call original method (this is where broker exits actually happen)
        result = self.original_pm.force_exit_all(prices, reason=reason)

        # Only record sells for positions that were ACTUALLY removed
        remaining_symbols = set(self.original_pm.positions.keys())
        confirmed_symbols = pre_exit_symbols - remaining_symbols

        if not confirmed_symbols:
            if pre_exit_symbols:
                logger.warning(
                    "force_exit_all: %d positions attempted but NONE confirmed removed",
                    len(pre_exit_symbols),
                )
            return result

        # Pop realized exits ONCE into a local cache (pop is destructive)
        realized_cache = {}
        for symbol in confirmed_symbols:
            realized_cache[symbol] = self._read_realized_exit(symbol, prices.get(symbol, 0.0))

        for symbol in confirmed_symbols:
            realized = realized_cache.get(symbol)
            actual_price = realized["exit_price"] if realized else prices.get(symbol, 0.0)
            actual_qty = realized["exit_qty"] if realized else 0.0
            try:
                log_trade_with_reporting(
                    symbol=symbol,
                    action="SELL",
                    quantity=actual_qty,
                    price=actual_price,
                    strategy=self.strategy_name,
                    notes=f"Force exit: {reason}"
                )
            except Exception:
                logger.exception("Reporting failed for %s force exit; continuing", symbol)

        # Record trade outcomes to monitoring (reuse cached realized data)
        try:
            monitor = get_session_monitor()
            from datetime import datetime
            from zoneinfo import ZoneInfo
            now_str = datetime.now(ZoneInfo("America/New_York")).isoformat()
            for symbol in confirmed_symbols:
                realized = realized_cache.get(symbol)
                if realized:
                    entry_price = realized.get("entry_price", 0.0)
                    exit_price = realized["exit_price"]
                    exit_qty = realized["exit_qty"]
                    peak_price = realized.get("peak_price", entry_price)
                    stop_price = realized.get("stop_price", 0.0)
                    entry_time = realized.get("entry_time", "")
                    if hasattr(entry_time, "isoformat"):
                        entry_time = entry_time.isoformat()
                else:
                    entry_price = 0.0
                    exit_price = prices.get(symbol, 0.0)
                    exit_qty = 0.0
                    peak_price = 0.0
                    stop_price = 0.0
                    entry_time = ""
                mfe = ((peak_price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0
                mae = ((stop_price - entry_price) / entry_price * 100) if entry_price > 0 and stop_price > 0 else 0.0
                monitor.record_trade_outcome(
                    symbol=symbol,
                    entry_time=entry_time,
                    exit_time=now_str,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    qty=exit_qty,
                    exit_reason=reason,
                    gap_at_entry=realized.get("gap_at_entry", 0.0) if realized else 0.0,
                    first_5min_volume=realized.get("first_5min_volume", 0.0) if realized else 0.0,
                    max_favorable_excursion=mfe,
                    max_adverse_excursion=mae,
                    fill_pct=realized.get("fill_pct", 100.0) if realized else 100.0,
                    entry_slippage_bps=realized.get("entry_slippage_bps", 0.0) if realized else 0.0,
                )
        except Exception:
            logger.exception("Monitoring trade outcomes failed during force exit; continuing")

        return result


def create_reporting_position_manager(original_pm, strategy_name="morning_momentum"):
    """Factory function to create wrapped position manager"""
    return ReportingPositionManager(original_pm, strategy_name)
