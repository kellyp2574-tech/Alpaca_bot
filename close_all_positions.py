"""Close all open positions in paper account"""
from bot.execution import ExecutionClient, ExecutionConfig
from bot.morning_config import Config

cfg = Config()
exec_cfg = ExecutionConfig(
    buy_slippage_pct=cfg.exec_slippage_buy_pct,
    sell_slippage_pct=cfg.exec_slippage_sell_pct,
)
execution = ExecutionClient(dry_run=False, cfg=exec_cfg)

print("=" * 80)
print("CLOSING ALL POSITIONS")
print("=" * 80)

try:
    positions = execution.client.get_all_positions()
    
    if not positions:
        print("\nNo open positions to close")
    else:
        print(f"\nFound {len(positions)} open positions:")
        
        for pos in positions:
            symbol = pos.symbol
            qty = abs(float(pos.qty))
            current_price = float(pos.current_price)
            
            print(f"\n  Closing {symbol}: {qty} shares @ ${current_price:.2f}")
            
            # Place market sell order
            fill = execution.place_exit(
                symbol,
                qty,
                current_price,
                client_order_id=f"CLOSE_ALL_{symbol}_{int(__import__('time').time())}"
            )
            
            if fill and fill.status in {"filled", "dry_run"}:
                print(f"    ✓ Closed {symbol}: {fill.filled_qty} shares @ ${fill.avg_price:.2f}")
            else:
                print(f"    ✗ Failed to close {symbol}: {fill.status if fill else 'No fill'}")
        
        print("\n" + "=" * 80)
        print("DONE")
        print("=" * 80)
        
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
