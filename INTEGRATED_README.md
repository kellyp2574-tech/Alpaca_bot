# Integrated Alpaca Bot

This bot combines the 3 ETF rotation strategy with the morning momentum strategy into a single integrated system.

## Overview

The bot runs from **8:30 AM to 3:30 PM** ET with two distinct phases:

1. **Morning Momentum (8:30 AM - 11:00 AM)**: Executes gap-based momentum trading
2. **3 ETF Rotation (11:00 AM - 3:30 PM)**: Performs hourly checks of the MA crossover strategy

## Key Features

- **Single execution**: No more hourly scheduling - runs continuously once started
- **Strategy separation**: Morning momentum uses cash, ETF rotation preserves position value
- **Manual rebalancing**: Separate script for 50/50 portfolio allocation
- **Comprehensive trade reporting**: Automatic win/loss tracking and statistics generation

## Files

### Main Bot
- `bot/integrated_main.py` - Main integrated bot (NEW)
- `bot/rebalance.py` - Manual 50/50 rebalancing script (NEW)

### Trade Reporting
- `bot/trade_reporter.py` - Core trade tracking and statistics system (NEW)
- `bot/reporting_position_manager.py` - Position manager wrapper for trade logging (NEW)
- `bot/view_reports.py` - Simple report viewer script (NEW)

### Existing Components (Updated)
- `bot/strategies.py` - Now only contains 3 ETF rotation logic
- `bot/config.py` - Removed Monday Dip and BB bands configurations
- `bot/main.py` - Original bot (preserved for reference)

## Usage

### Running the Integrated Bot

```bash
# Normal trading
python -m bot.integrated_main

# Dry run (no actual trades)
python -m bot.integrated_main --dry-run
```

### Manual Rebalancing

```bash
# Check if rebalancing needed
python -m bot.rebalance --dry-run

# Execute rebalancing (includes safety check)
python -m bot.rebalance

# Force rebalancing (even within threshold)
python -m bot.rebalance --force

# Check for open Morning Momentum positions only
python -m bot.rebalance --check-mm
```

⚠️ **Safety Feature**: The rebalance script automatically checks for open Morning Momentum positions and will abort if any are detected. This prevents conflicts between strategies.

### Trade Reporting

```bash
# View current trade statistics
python -m bot.view_reports

# View trades from last 30 days
python -m bot.view_reports --recent 30
```

## Strategy Details

### Morning Momentum (8:30 AM - 11:00 AM)
- Scans for gapped-up stocks with strong volume
- Enters positions between 9:35 AM - 10:30 AM
- Uses trailing stops and risk management
- Exits all positions by 10:30 AM
- **Position Sizing**: 50% of actual cash divided by qualified positions at 9:35
- **Volume Constraint**: Maximum 5% of first 5-minute trading volume
- **Consistent Sizing**: All positions use same size calculated at 9:35 mark
- **Fractional Handling**: Rounds down for non-fractional shares

### 3 ETF Rotation (11:00 AM - 3:30 PM)
- Checks MA crossover signals hourly
- Rotates between QLD (growth), UBT (bonds), DBMF (alternative)
- Preserves position value when rotating
- Uses 50% of portfolio equity

## Trade Reporting

The bot automatically tracks all trades and generates comprehensive statistics:

### Tracked Metrics
- **Win Rate**: Percentage of profitable trades
- **Average Win/Loss %**: Average percentage gain/loss per trade
- **Total P&L**: Overall profit/loss in dollars
- **Largest Win/Loss**: Best and worst individual trades
- **Average Hold Days**: How long trades are held on average
- **Strategy Breakdown**: Performance by strategy type

### Automatic Features
- **Real-time logging**: Every buy/sell is recorded immediately
- **Automatic pairing**: Buys and sells are automatically matched
- **Instant statistics**: Reports update after each sell
- **Strategy tracking**: Separate metrics for morning momentum vs ETF rotation

### Report Files
- `state/reports/statistics.txt` - Main performance report
- `state/reports/trades.json` - Raw trade data
- `state/reports/completed_trades.json` - Completed buy-sell pairs

### Sample Report Output
```
============================================================
TRADE PERFORMANCE REPORT
Generated: 2025-02-22 15:30:00
============================================================

OVERALL PERFORMANCE:
Total Trades: 45
Win Rate: 62.2%
Winning Trades: 28
Losing Trades: 17
Average Win %: 3.45%
Average Loss %: -1.82%
Total P&L: $2,845.67
Largest Win: $245.30
Largest Loss: -$89.15
Average Hold Days: 1.2

STRATEGY BREAKDOWN:
morning_momentum:
  Trades: 38
  Win Rate: 63.2%
  Avg %: 2.98%
  P&L: $2,156.30

etf_rotation:
  Trades: 7
  Win Rate: 57.1%
  Avg %: 4.12%
  P&L: $689.37
```

## Position Sizing Logic

### Morning Momentum Strategy

The morning momentum strategy uses a sophisticated position sizing approach:

1. **Cash Allocation**: Uses 50% of actual available cash (not portfolio equity)
2. **Position Count**: Counts qualified positions at 9:35 AM mark
3. **Base Size**: Divides cash allocation by number of qualified positions
4. **Volume Cap**: Limits position size to 5% of first 5-minute volume
5. **Consistency**: All entries use same size calculated at 9:35
6. **Rounding**: Rounds down for non-fractional shares

#### Example Calculation:
```
Account Cash: $100,000
50% Allocation: $50,000
Qualified Positions at 9:35: 8 stocks
Base Position Size: $50,000 ÷ 8 = $6,250 per position

Volume Constraint:
- Stock A: 5-min volume = $2,000,000
- 5% cap = $100,000 (no constraint)
- Final size: $6,250

- Stock B: 5-min volume = $50,000  
- 5% cap = $2,500 (constraint active)
- Final size: $2,500 (smaller of base and cap)
```

#### Key Features:
- **Calculated Once**: Position size calculated at first entry (9:35) and reused
- **Volume Safety**: Prevents large positions in low-volume stocks
- **Cash Preservation**: Never uses more than 50% of available cash
- **Fractional Support**: Automatically handles fractional vs whole shares

## Portfolio Allocation

The bot maintains a **50/50 split**:
- **50%** in 3 ETF rotation strategy
- **50%** cash for morning momentum trading

## Configuration

### Environment Variables
```
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_PAPER=true  # or false for live trading
```

### Key Settings (config.py)
```python
# 3 ETF Rotation
MA_TRADE_GROWTH = "QLD"     # 2x leveraged growth ETF
MA_TRADE_SAFE = "UBT"       # 2x leveraged bond ETF  
MA_TRADE_ALT = "DBMF"       # Alternative ETF
MA_ALLOC_PCT = 0.50         # 50% allocation

# Morning Momentum (from Alpaca_Morning_Momentum/config.py)
entry_start = "09:35"       # Entry window start
entry_cutoff = "10:30"      # Entry window end
hard_exit = "10:30"         # Force exit time
```

## Migration from Old Bot

### What's Kept
- ✅ 3 ETF rotation strategy (MA crossover)
- ✅ State management and logging
- ✅ Alpaca integration

### What's Removed
- ❌ Tuesday rebound strategy
- ❌ Bollinger Bands strategy
- ❌ Hourly scheduling requirement

### What's Added
- ✅ Morning momentum integration
- ✅ Continuous execution (8:30 AM - 3:30 PM)
- ✅ Manual rebalancing script

## Daily Workflow

1. **Before Market**: Run `python -m bot.rebalance` if needed to maintain 50/50 allocation
2. **8:30 AM**: Start integrated bot `python -m bot.integrated_main`
3. **8:30 AM - 11:00 AM**: Bot runs morning momentum strategy
4. **11:00 AM - 3:30 PM**: Bot checks 3 ETF rotation hourly
5. **3:30 PM**: Bot automatically exits

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `Alpaca_Morning_Momentum` folder exists and is accessible
2. **State conflicts**: The bot uses separate state files for each strategy
3. **Market hours**: Bot automatically checks if market is open

### Logs

- `state/logs/bot.log` - Main bot logs
- `state/mm_positions.json` - Morning momentum state
- `state/bot_state.json` - 3 ETF rotation state

## Risk Management

- Morning momentum has built-in position limits and stop losses
- 3 ETF rotation uses systematic MA crossover signals
- 50/50 allocation limits exposure to any single strategy
- Dry-run mode available for testing

## Support

For issues with:
- **Morning momentum**: Check `Alpaca_Morning_Momentum` bot documentation
- **3 ETF rotation**: Review existing bot logs and state
- **Integration**: Check this file's logs and configuration
