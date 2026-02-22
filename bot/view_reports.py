"""
Trade Report Viewer - Simple script to view trading statistics
"""
import argparse
from bot.trade_reporter import get_trade_reporter


def main():
    parser = argparse.ArgumentParser(description="View trade performance reports")
    parser.add_argument("--recent", type=int, help="Show trades from last N days")
    args = parser.parse_args()
    
    reporter = get_trade_reporter()
    
    # Generate fresh report
    reporter.generate_report()
    
    # Show recent trades if requested
    if args.recent:
        recent_trades = reporter.get_recent_trades(args.recent)
        if recent_trades:
            print(f"\nRECENT TRADES (Last {args.recent} days):")
            print("-" * 80)
            for trade in recent_trades:
                print(f"{trade.sell_timestamp[:19]} {trade.symbol:<6} "
                      f"${trade.pnl_dollars:+8.2f} ({trade.pnl_percentage:+6.2f}%) "
                      f"{trade.strategy.replace('_', ' '):<15} "
                      f"{trade.hold_days:.1f}d")
        else:
            print(f"\nNo trades found in the last {args.recent} days")


if __name__ == "__main__":
    main()
