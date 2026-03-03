# Trade Logging with Comprehensive Metrics

## Overview
The trade reporter now captures comprehensive entry/exit metrics and auto-classifies trades for detailed analysis.

## Usage in Morning Momentum Strategy

### When Entering a Trade (BUY)
```python
from bot.trade_reporter import log_trade_with_reporting

# Prepare entry metadata from candidate
metadata = {
    'security_type': 'etp',  # or 'stock', 'etf'
    'gap_pct': candidate.gap_pct,  # e.g., 0.18 for 18%
    'prev_close': candidate.prev_close,
    'price_now': entry_price,
    'first_5min_volume': dollar_vol_5min,  # First 5-min dollar volume
    'daily_dollar_volume': daily_vol * prev_close,  # Optional
    'spread_at_entry': ask - bid,  # Bid-ask spread at entry
}

# Log the buy with metadata
log_trade_with_reporting(
    symbol=symbol,
    action='BUY',
    quantity=qty,
    price=entry_price,
    strategy='morning_momentum',
    order_id=order.id,
    notes=f"Gap {candidate.gap_pct:.1%}",
    metadata=metadata
)
```

### When Exiting a Trade (SELL)
```python
# Prepare exit metadata
metadata = {
    'exit_reason': 'TP',  # or 'trail', 'stop', 'time', 'manual'
    'r_multiple': pnl_dollars / risk_dollars,  # Profit/risk ratio
}

# Log the sell with metadata
log_trade_with_reporting(
    symbol=symbol,
    action='SELL',
    quantity=qty,
    price=exit_price,
    strategy='morning_momentum',
    order_id=order.id,
    notes=f"Exit: {exit_reason}",
    metadata=metadata
)
```

## Captured Metrics

### Entry Metrics (from BUY)
- `security_type`: "stock", "etf", or "etp"
- `gap_pct`: Gap percentage (e.g., 0.18 for 18%)
- `prev_close`: Previous day's close price
- `price_now`: Entry price
- `first_5min_volume`: Dollar volume in first 5 minutes
- `daily_dollar_volume`: Previous day's dollar volume
- `spread_at_entry`: Bid-ask spread at entry

### Exit Metrics (from SELL)
- `exit_reason`: "TP", "trail", "stop", "time", "manual"
- `r_multiple`: R multiple (profit/risk ratio)

### Auto-Classified Fields
The system automatically classifies each trade:
- `is_etp`: True if symbol is an exchange-traded product
- `is_leveraged`: True if symbol is a known leveraged ETF
- `is_vol_product`: True if symbol is a volatility product (UVXY, VXX, etc.)
- `price_bucket`: "2-5", "5-20", "20-100", or "other"
- `gap_bucket`: "5-10", "10-15", "15-20", "20-25", or "other"

## Output Files

### state/reports/completed_trades.json
Full JSON with all metrics for each completed trade.

### state/reports/statistics.txt
Human-readable summary with:
- Overall performance stats
- Strategy breakdown
- Recent trades with entry/exit prices
- Open positions

## Example Integration in EntryLoop

```python
# In EntryLoop when entering position
candidate = self.ctx.candidate_map[symbol]
first_5min_bars = self.ctx.bar_tracker.get(symbol, [])
dollar_vol_5min = sum(bar.v * bar.c for bar in first_5min_bars)

# Get spread from latest quote
quote = self.ctx.data.alpaca.get_latest_quote(symbol)
spread = quote.ask_price - quote.bid_price if quote else 0

metadata = {
    'security_type': 'etp',
    'gap_pct': candidate.gap_pct,
    'prev_close': candidate.prev_close,
    'price_now': entry_price,
    'first_5min_volume': dollar_vol_5min,
    'spread_at_entry': spread,
}

log_trade_with_reporting(
    symbol=symbol,
    action='BUY',
    quantity=qty,
    price=entry_price,
    strategy='morning_momentum',
    metadata=metadata
)
```

```python
# When exiting position
# Calculate R multiple
risk_dollars = entry_price * stop_pct * qty
pnl_dollars = (exit_price - entry_price) * qty
r_multiple = pnl_dollars / risk_dollars if risk_dollars > 0 else 0

metadata = {
    'exit_reason': 'TP',  # Determine from exit logic
    'r_multiple': r_multiple,
}

log_trade_with_reporting(
    symbol=symbol,
    action='SELL',
    quantity=qty,
    price=exit_price,
    strategy='morning_momentum',
    metadata=metadata
)
```

## Analysis Benefits

With these metrics, you can analyze:
- Which gap ranges perform best (5-10% vs 20-25%)
- Performance by price bucket (low vs high priced stocks)
- Leveraged vs non-leveraged performance
- Volatility products vs regular stocks
- Spread impact on profitability
- R multiple distribution
- Exit reason effectiveness (TP vs trail vs stop)
