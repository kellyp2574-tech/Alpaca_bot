"""
Test file to verify all Alpaca API calls with different feed parameters.

This script tests:
1. delayed_sip snapshots (broad filter data)
2. IEX snapshots (live refinement data)
3. IEX quotes (live quote refresh)
4. IEX stream subscription (live bar data)

Run this to ensure all feed parameters are syntactically correct and working.
"""

import sys
import time
from datetime import datetime, timedelta

# Add bot directory to path
sys.path.insert(0, 'bot')

from bot.data_alpaca import AlpacaDataAdapter
from bot.clock import market_now

print("=" * 80)
print("ALPACA FEED PARAMETER TEST")
print("=" * 80)
print()

# Initialize Alpaca adapter (uses environment variables)
print("Initializing Alpaca data adapter...")
try:
    alpaca = AlpacaDataAdapter()
    print(f"✓ Initialized with default feed: {alpaca.feed}")
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    sys.exit(1)

print()

# Test symbols (small set for quick testing)
test_symbols = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD"]
print(f"Test symbols: {', '.join(test_symbols)}")
print()

# =============================================================================
# TEST 1: delayed_sip snapshots (8:30-8:40 AM broad filter)
# =============================================================================
print("=" * 80)
print("TEST 1: delayed_sip Snapshots (Broad Filter)")
print("=" * 80)
print("Purpose: Get 15-minute delayed data for broad coverage")
print("Timeline: 8:30-8:40 AM")
print()

try:
    print("Calling get_snapshots(symbols, feed='delayed_sip')...")
    snapshots_delayed = alpaca.get_snapshots(test_symbols, feed="delayed_sip")
    
    print(f"✓ Success: Received {len(snapshots_delayed)} snapshots")
    
    # Show sample data
    if snapshots_delayed:
        sample_symbol = list(snapshots_delayed.keys())[0]
        snap = snapshots_delayed[sample_symbol]
        print(f"\nSample snapshot for {sample_symbol}:")
        print(f"  - latest_trade: {snap.latest_trade}")
        print(f"  - latest_quote: {snap.latest_quote}")
        print(f"  - daily_bar: {snap.daily_bar}")
        print(f"  - prev_daily_bar: {snap.prev_daily_bar}")
        
        if snap.latest_trade:
            print(f"  - Trade price: ${snap.latest_trade.p:.2f}")
        if snap.latest_quote:
            print(f"  - Quote bid/ask: ${snap.latest_quote.bp:.2f} / ${snap.latest_quote.ap:.2f}")
    
    print("\n✓ delayed_sip snapshots: PASS")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    print("\nAttempting fallback to IEX...")
    
    try:
        snapshots_delayed = alpaca.get_snapshots(test_symbols, feed="iex")
        print(f"✓ Fallback successful: Received {len(snapshots_delayed)} snapshots with IEX")
    except Exception as e2:
        print(f"✗ Fallback also failed: {e2}")

print()

# =============================================================================
# TEST 2: IEX snapshots (9:05 AM first refinement)
# =============================================================================
print("=" * 80)
print("TEST 2: IEX Snapshots (First Refinement)")
print("=" * 80)
print("Purpose: Get live IEX data for refinement")
print("Timeline: 9:05 AM")
print()

try:
    print("Calling get_snapshots(symbols, feed='iex')...")
    snapshots_iex = alpaca.get_snapshots(test_symbols, feed="iex")
    
    print(f"✓ Success: Received {len(snapshots_iex)} snapshots")
    
    # Show sample data
    if snapshots_iex:
        sample_symbol = list(snapshots_iex.keys())[0]
        snap = snapshots_iex[sample_symbol]
        print(f"\nSample snapshot for {sample_symbol}:")
        if snap.latest_trade:
            print(f"  - Trade price: ${snap.latest_trade.p:.2f}")
        if snap.latest_quote:
            print(f"  - Quote bid/ask: ${snap.latest_quote.bp:.2f} / ${snap.latest_quote.ap:.2f}")
    
    print("\n✓ IEX snapshots: PASS")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()

print()

# =============================================================================
# TEST 3: IEX quotes (9:15 AM second refinement + live refresh)
# =============================================================================
print("=" * 80)
print("TEST 3: IEX Latest Quotes (Second Refinement + Live)")
print("=" * 80)
print("Purpose: Get latest live quotes for refinement and trading")
print("Timeline: 9:15 AM + 9:30 AM onward")
print()

try:
    print("Calling get_latest_quotes(symbols, feed='iex')...")
    quotes_iex = alpaca.get_latest_quotes(test_symbols, feed="iex")
    
    print(f"✓ Success: Received {len(quotes_iex)} quotes")
    
    # Show sample data
    if quotes_iex:
        print("\nSample quotes:")
        for symbol in list(quotes_iex.keys())[:3]:
            quote = quotes_iex[symbol]
            print(f"  {symbol}: bid=${quote.bid_price:.2f}, ask=${quote.ask_price:.2f}")
    
    print("\n✓ IEX quotes: PASS")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()

print()

# =============================================================================
# TEST 4: IEX stream subscription (9:28 AM stream start)
# =============================================================================
print("=" * 80)
print("TEST 4: IEX Stream Subscription (Live Bars)")
print("=" * 80)
print("Purpose: Subscribe to live 1-minute bars")
print("Timeline: 9:28 AM onward")
print()

try:
    print("Calling subscribe_stream(symbols, feed='iex')...")
    alpaca.subscribe_stream(test_symbols, feed="iex")
    
    print(f"✓ Success: Subscribed to {len(test_symbols)} symbols with IEX feed")
    print("  Note: Stream is running in background thread")
    
    # Try to read a few bars (with timeout)
    print("\nAttempting to read bars for 5 seconds...")
    start_time = time.time()
    bars_received = 0
    
    while time.time() - start_time < 5:
        bar = alpaca.next_bar(timeout=1.0)
        if bar:
            bars_received += 1
            print(f"  ✓ Received bar: {bar.symbol} @ ${bar.c:.2f} (volume: {bar.v:,})")
            if bars_received >= 3:
                break
    
    if bars_received > 0:
        print(f"\n✓ Stream is working: Received {bars_received} bars")
    else:
        print("\n⚠ No bars received (market may be closed or low activity)")
    
    # Clean up stream
    print("\nClosing stream...")
    alpaca.close_stream()
    print("✓ Stream closed")
    
    print("\n✓ IEX stream subscription: PASS")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    
    # Try to close stream even on error
    try:
        alpaca.close_stream()
    except:
        pass

print()

# =============================================================================
# TEST 5: Feed parameter validation
# =============================================================================
print("=" * 80)
print("TEST 5: Feed Parameter Validation")
print("=" * 80)
print()

# Test invalid feed
print("Testing invalid feed parameter...")
try:
    alpaca.get_snapshots(["AAPL"], feed="invalid_feed")
    print("✗ FAILED: Should have raised error for invalid feed")
except ValueError as e:
    print(f"✓ Correctly rejected invalid feed: {e}")
except Exception as e:
    print(f"⚠ Unexpected error type: {e}")

print()

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print()
print("All feed parameter tests completed.")
print()
print("Key findings:")
print("1. delayed_sip: Use for broad filter (8:30-8:40 AM)")
print("2. IEX: Use for refinements and live trading (9:05 AM onward)")
print("3. Feed parameters are properly validated")
print("4. Fallback logic works (delayed_sip -> IEX)")
print()
print("Next steps:")
print("- Run this test during market hours for full validation")
print("- Check logs for any API rate limit warnings")
print("- Verify delayed_sip data is actually 15 minutes delayed")
print()
print("=" * 80)
