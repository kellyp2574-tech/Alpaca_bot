"""
Live test script: Fetch 20 candidates, monitor for 2 min, buy on price increase,
trail 0.4%, sell after 15 min if not stopped out.

This is a standalone test script that does NOT modify any files in bot/.
"""
import logging
import time
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Optional

from bot.morning_config import Config
from bot.morning_main import fetch_candidates
from bot.data_sources import init_data_stack
from bot.execution import ExecutionClient, ExecutionConfig
from bot.clock import market_now

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Test parameters
NUM_CANDIDATES = 20
MONITOR_DURATION_SEC = 120  # 2 minutes
BUY_ON_PRICE_INCREASE = True
TRAIL_PCT = 0.004  # 0.4%
MAX_HOLD_DURATION_SEC = 900  # 15 minutes

class LiveTestTrader:
    def __init__(self, dry_run: bool = True):
        self.cfg = Config()
        self.data = init_data_stack()
        
        exec_cfg = ExecutionConfig(
            buy_slippage_pct=self.cfg.exec_slippage_buy_pct,
            sell_slippage_pct=self.cfg.exec_slippage_sell_pct,
        )
        self.execution = ExecutionClient(dry_run=dry_run, cfg=exec_cfg)
        
        self.positions: Dict[str, dict] = {}  # symbol -> position info
        self.monitoring: Dict[str, dict] = {}  # symbol -> monitoring info
        
    def fetch_test_candidates(self):
        """Fetch candidates for testing"""
        logger.info("Fetching candidates...")
        candidates, stats = fetch_candidates(self.cfg, self.data)
        
        # Take first NUM_CANDIDATES
        test_candidates = candidates[:NUM_CANDIDATES]
        logger.info(f"Selected {len(test_candidates)} candidates for testing")
        
        for c in test_candidates[:5]:
            logger.info(f"  {c.symbol}: prev_close=${c.prev_close:.2f}, gap={c.gap_pct:.1%}, price=${c.price:.2f}")
        
        return test_candidates
    
    def start_monitoring(self, candidates):
        """Start monitoring candidates for 2 minutes"""
        logger.info(f"\n{'='*80}")
        logger.info(f"MONITORING PHASE: Tracking {len(candidates)} symbols for {MONITOR_DURATION_SEC}s")
        logger.info(f"{'='*80}")
        
        # Get initial prices
        symbols = [c.symbol for c in candidates]
        initial_quotes = self.data.alpaca.get_latest_quotes(symbols)
        
        for c in candidates:
            quote = initial_quotes.get(c.symbol)
            if quote and quote.ask_price > 0:
                initial_price = quote.ask_price
            else:
                initial_price = c.price
            
            self.monitoring[c.symbol] = {
                'initial_price': initial_price,
                'current_price': initial_price,
                'start_time': time.time(),
                'candidate': c,
            }
            logger.info(f"  {c.symbol}: initial_price=${initial_price:.2f}")
        
        # Monitor for MONITOR_DURATION_SEC
        start_time = time.time()
        check_interval = 10  # Check every 10 seconds
        
        while time.time() - start_time < MONITOR_DURATION_SEC:
            elapsed = time.time() - start_time
            remaining = MONITOR_DURATION_SEC - elapsed
            logger.info(f"\nMonitoring... {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
            
            # Get current prices
            current_quotes = self.data.alpaca.get_latest_quotes(symbols)
            
            for symbol, info in self.monitoring.items():
                quote = current_quotes.get(symbol)
                if quote and quote.ask_price > 0:
                    info['current_price'] = quote.ask_price
                    price_change_pct = (info['current_price'] - info['initial_price']) / info['initial_price']
                    logger.info(f"  {symbol}: ${info['current_price']:.2f} ({price_change_pct:+.2%})")
            
            time.sleep(check_interval)
        
        logger.info(f"\n{'='*80}")
        logger.info("MONITORING PHASE COMPLETE")
        logger.info(f"{'='*80}")
    
    def execute_buys(self):
        """Buy symbols that increased in price during monitoring"""
        logger.info(f"\n{'='*80}")
        logger.info("BUY PHASE: Executing buys for symbols with price increase")
        logger.info(f"{'='*80}")
        
        buy_count = 0
        for symbol, info in self.monitoring.items():
            initial_price = info['initial_price']
            current_price = info['current_price']
            price_change_pct = (current_price - initial_price) / initial_price
            
            if current_price > initial_price:
                logger.info(f"\n{symbol}: Price increased {price_change_pct:+.2%} (${initial_price:.2f} → ${current_price:.2f})")
                
                # Calculate position size (simple: $1000 per position)
                target_notional = 1000.0
                qty = target_notional / current_price
                
                # Round to 2 decimals for fractional shares
                qty = round(qty, 2)
                
                logger.info(f"  Buying {qty} shares @ ${current_price:.2f}")
                
                # Place buy order
                fill = self.execution.place_entry(
                    symbol,
                    qty,
                    current_price,
                    client_order_id=f"TEST_BUY_{symbol}_{int(time.time())}"
                )
                
                if fill and fill.status in {"filled", "dry_run"}:
                    self.positions[symbol] = {
                        'qty': fill.filled_qty,
                        'entry_price': fill.avg_price,
                        'entry_time': time.time(),
                        'peak_price': fill.avg_price,
                        'trail_stop': fill.avg_price * (1 - TRAIL_PCT),
                        'candidate': info['candidate'],
                    }
                    logger.info(f"  ✓ BUY FILLED: {fill.filled_qty} shares @ ${fill.avg_price:.2f}")
                    buy_count += 1
                else:
                    logger.warning(f"  ✗ BUY FAILED: {fill.status if fill else 'No fill'}")
            else:
                logger.info(f"{symbol}: No price increase ({price_change_pct:+.2%}), skipping")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"BUY PHASE COMPLETE: {buy_count} positions opened")
        logger.info(f"{'='*80}")
    
    def monitor_positions(self):
        """Monitor positions with 0.4% trailing stop, max 15 min hold"""
        if not self.positions:
            logger.info("\nNo positions to monitor")
            return
        
        logger.info(f"\n{'='*80}")
        logger.info(f"POSITION MONITORING: {len(self.positions)} positions, 0.4% trail, 15 min max hold")
        logger.info(f"{'='*80}")
        
        check_interval = 5  # Check every 5 seconds
        
        while self.positions:
            logger.info(f"\n--- Position Check ({len(self.positions)} open) ---")
            
            symbols = list(self.positions.keys())
            current_quotes = self.data.alpaca.get_latest_quotes(symbols)
            
            for symbol in list(self.positions.keys()):
                pos = self.positions[symbol]
                quote = current_quotes.get(symbol)
                
                if not quote or quote.bid_price <= 0:
                    logger.warning(f"{symbol}: No valid quote, skipping")
                    continue
                
                current_price = quote.bid_price
                elapsed_sec = time.time() - pos['entry_time']
                elapsed_min = elapsed_sec / 60
                
                # Update peak and trailing stop
                if current_price > pos['peak_price']:
                    pos['peak_price'] = current_price
                    pos['trail_stop'] = current_price * (1 - TRAIL_PCT)
                
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
                trail_distance_pct = (current_price - pos['trail_stop']) / current_price
                
                logger.info(
                    f"{symbol}: ${current_price:.2f} | P&L: {pnl_pct:+.2%} | "
                    f"Trail: ${pos['trail_stop']:.2f} ({trail_distance_pct:.2%} away) | "
                    f"Hold: {elapsed_min:.1f}min"
                )
                
                # Exit conditions
                exit_reason = None
                
                # 1. Trailing stop hit
                if current_price <= pos['trail_stop']:
                    exit_reason = f"trail_stop (${pos['trail_stop']:.2f})"
                
                # 2. Max hold time (15 min)
                elif elapsed_sec >= MAX_HOLD_DURATION_SEC:
                    exit_reason = f"max_hold ({MAX_HOLD_DURATION_SEC/60:.0f} min)"
                
                if exit_reason:
                    logger.info(f"  → SELLING {symbol}: {exit_reason}")
                    
                    # Place sell order
                    fill = self.execution.place_exit(
                        symbol,
                        pos['qty'],
                        current_price,
                        client_order_id=f"TEST_SELL_{symbol}_{int(time.time())}"
                    )
                    
                    if fill and fill.status in {"filled", "dry_run"}:
                        pnl_dollars = (fill.avg_price - pos['entry_price']) * pos['qty']
                        pnl_pct_final = (fill.avg_price - pos['entry_price']) / pos['entry_price']
                        logger.info(
                            f"  ✓ SELL FILLED: {fill.filled_qty} shares @ ${fill.avg_price:.2f} | "
                            f"P&L: ${pnl_dollars:+.2f} ({pnl_pct_final:+.2%})"
                        )
                        del self.positions[symbol]
                    else:
                        logger.warning(f"  ✗ SELL FAILED: {fill.status if fill else 'No fill'}")
            
            if self.positions:
                time.sleep(check_interval)
        
        logger.info(f"\n{'='*80}")
        logger.info("POSITION MONITORING COMPLETE: All positions closed")
        logger.info(f"{'='*80}")
    
    def cleanup_existing_positions(self):
        """Close all existing positions to start with clean slate"""
        logger.info(f"\n{'='*80}")
        logger.info("CLEANUP: Closing all existing positions")
        logger.info(f"{'='*80}")
        
        try:
            if self.execution.dry_run or not self.execution.client:
                logger.info("Dry-run mode: skipping position cleanup")
                return
            
            # Get all open positions
            positions = self.execution.client.get_all_positions()
            
            if not positions:
                logger.info("No existing positions to close")
                return
            
            logger.info(f"Found {len(positions)} open positions to close:")
            
            for pos in positions:
                symbol = pos.symbol
                qty = abs(float(pos.qty))
                current_price = float(pos.current_price)
                
                logger.info(f"  Closing {symbol}: {qty} shares @ ${current_price:.2f}")
                
                # Place market sell order to close
                fill = self.execution.place_exit(
                    symbol,
                    qty,
                    current_price,
                    client_order_id=f"CLEANUP_{symbol}_{int(time.time())}"
                )
                
                if fill and fill.status in {"filled", "dry_run"}:
                    logger.info(f"    ✓ Closed {symbol}: {fill.filled_qty} shares @ ${fill.avg_price:.2f}")
                else:
                    logger.warning(f"    ✗ Failed to close {symbol}: {fill.status if fill else 'No fill'}")
            
            logger.info("Cleanup complete\n")
            
        except Exception as e:
            logger.exception(f"Cleanup failed: {e}")
    
    def run(self):
        """Run the full test cycle"""
        logger.info(f"\n{'='*80}")
        logger.info("LIVE TEST TRADE SCRIPT")
        logger.info(f"{'='*80}")
        logger.info(f"Parameters:")
        logger.info(f"  Candidates: {NUM_CANDIDATES}")
        logger.info(f"  Monitor duration: {MONITOR_DURATION_SEC}s ({MONITOR_DURATION_SEC/60:.1f} min)")
        logger.info(f"  Buy on price increase: {BUY_ON_PRICE_INCREASE}")
        logger.info(f"  Trailing stop: {TRAIL_PCT:.2%}")
        logger.info(f"  Max hold: {MAX_HOLD_DURATION_SEC}s ({MAX_HOLD_DURATION_SEC/60:.0f} min)")
        logger.info(f"{'='*80}\n")
        
        try:
            # Step 0: Clean up existing positions
            self.cleanup_existing_positions()
            # Step 1: Fetch candidates
            candidates = self.fetch_test_candidates()
            
            if not candidates:
                logger.error("No candidates found, exiting")
                return
            
            # Step 2: Monitor for 2 minutes
            self.start_monitoring(candidates)
            
            # Step 3: Execute buys for symbols with price increase
            self.execute_buys()
            
            # Step 4: Monitor positions with trailing stop
            self.monitor_positions()
            
            logger.info("\n✅ TEST COMPLETE")
            
        except KeyboardInterrupt:
            logger.info("\n\n⚠️  Test interrupted by user")
            if self.positions:
                logger.warning(f"Closing {len(self.positions)} open positions...")
                symbols = list(self.positions.keys())
                quotes = self.data.alpaca.get_latest_quotes(symbols)
                for symbol, pos in self.positions.items():
                    quote = quotes.get(symbol)
                    price = quote.bid_price if quote and quote.bid_price > 0 else pos['entry_price']
                    self.execution.place_exit(symbol, pos['qty'], price)
                logger.info("All positions closed")
        except Exception as e:
            logger.exception(f"Test failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Live test trade script")
    parser.add_argument("--live", action="store_true", help="Run with real orders (default: dry-run)")
    args = parser.parse_args()
    
    dry_run = not args.live
    
    if dry_run:
        logger.info("🔵 DRY RUN MODE (no real orders)")
    else:
        logger.warning("🔴 LIVE MODE (real orders will be placed!)")
        confirm = input("Are you sure you want to place real orders? (yes/no): ")
        if confirm.lower() != "yes":
            logger.info("Exiting")
            exit(0)
    
    trader = LiveTestTrader(dry_run=dry_run)
    trader.run()
