# Alpaca Trading Bot - Morning Gap Momentum Strategy

Production-ready gap momentum trading bot that trades pre-market gappers during the morning session. Trades via Alpaca API with dynamic position sizing, volume-aware allocation, and robust risk management.

## Strategy Overview

### Morning Gap Momentum (75% of equity deployed, 9:00 AM - 11:00 AM)
- **Signal:** Pre-market gap (5-25%) + opening breakout + volume confirmation
- **Universe:** 4,000 stocks screened → $10M+ ADV, $1-$100 price range
- **Screening:** $250K+ first 5-min volume, 5-25% gap, opening breakout filter
- **Entry:** DAY limit orders at 9:40-11:00 AM with dynamic position sizing
- **Exit:** 1.2% trailing activation, 1% trailing stop, 5% hard stop
- **Risk:** 25 max concurrent positions, 25 max trades/day, daily R limit -3R
- **Guarantee:** Hard exit at 11:00 AM under ALL conditions

## Dynamic Position Sizing

### Three-Constraint Allocation System
Each position sized using **minimum of three constraints**:
1. **Equal-weight base:** `remaining_cash / remaining_positions`
2. **Concentration cap:** 25% max per ticker
3. **Volume limit:** 1% of stock's first 5-min dollar volume

### Allocation Flow (Low → High Liquidity)
- Candidates sorted by 5-min dollar volume (ascending)
- Small-cap names processed first, large-cap names last
- Stocks hitting volume cap leave unused cash for other positions
- Second pass distributes leftover to high-liquidity names (respecting both caps)

### Example ($10,000 deploy, 10 candidates)
- Low-liq stock ($50K vol): gets $500 (1% vol cap hit)
- Mid-liq stock ($200K vol): gets $1,000 (equal-weight)
- High-liq stock ($1M vol): gets $1,200 (equal-weight + spillover)
- Leftover distributed to highest liquidity names up to caps


## Key Features

### 🚀 Production-Grade Reliability
- **Hard Exit Guarantee:** All positions flat by 11:00 AM regardless of crashes
- **Supervision System:** Orchestrator monitors positions even if EntryLoop fails
- **DAY Orders:** Fractional share support with async fill reconciliation
- **Emergency Fallbacks:** Market order escalation, broker position cleanup

### 🔒 Risk Management
- **Duplicate Prevention:** Each symbol max one entry attempt per day
- **Position Sizing:** Dynamic allocation with 25% concentration cap, 1% volume participation
- **Stop Logic:** 1% trailing stop after 1.2% gain, 5% hard stop, breakeven protection
- **Time-Based Exits:** Automatic reconciliation, DAY order fill tracking

### 📊 Market Coverage
- **Pre-market Scanning:** 9:00 AM - 9:29 AM (every 5 minutes)
- **Entry Window:** 9:40 AM - 11:00 AM (gap momentum)
- **Hard Exit:** 11:00 AM (all positions closed)
- **Data Sources:** Alpaca Market Data API + Massive API for screening
- **Quote Handling:** Real-time 1-min bars with quote mid for entries

## Project Structure

```
Alpaca_bot/
├── bot/                           # Trading bot
│   ├── integrated_main.py         # Main orchestrator (8:30 AM - 11:00 AM)
│   ├── morning_main.py           # Gap momentum entry logic with dynamic sizing
│   ├── position_manager.py       # Position management with trailing stops
│   ├── execution.py              # Order execution with fractional share support
│   ├── morning_config.py         # Strategy parameters
│   ├── premarket_scan.py         # Candidate screening via Massive + Alpaca
│   ├── state_manager.py          # State persistence
│   ├── risk_manager.py           # Risk controls & daily limits
│   ├── data_alpaca.py            # Alpaca data adapter
│   ├── clock.py                  # Market time utilities
│   └── storage.py                # Data structures (Candidate, Position, etc.)
├── state/                        # Runtime state & logs (gitignored)
├── .env.example                  # API key template
├── close_all_positions.py        # Utility to close all positions
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your Alpaca API key and secret
```

## Usage

```bash
# Run gap momentum bot (main production mode)
python -m bot.integrated_main

# Dry run — show signals without trading
python -m bot.integrated_main --dry-run

# Close all open positions (cleanup utility)
python close_all_positions.py
```

## Daily Schedule

| Time | Activity |
|------|----------|
| 8:30 AM | Bot starts, initialization |
| 9:00-9:29 AM | Pre-market scanning (every 5 min) |
| 9:29 AM | Final scan, lock watchlist (35 candidates) |
| 9:30 AM | Market open, collect first 5 bars |
| 9:40 AM | Entry window opens, dynamic allocation calculated |
| 9:40-11:00 AM | Active trading with trailing stops |
| 11:00 AM | **Hard exit guarantee** - all positions flat |
| 11:00-11:30 AM | Cleanup, reconciliation, reporting |

## Critical Guarantees

### ✅ Hard Exit Guarantee
- **All positions flat by 11:00 AM** regardless of:
  - EntryLoop crashes or exceptions
  - Network connectivity issues
  - Stream data interruptions
  - Manual intervention

### ✅ No Duplicate Entries
- **Maximum one entry attempt per symbol per day**
- Failed entries marked "done for today"
- Persistent attempt tracking with audit trail

### ✅ Fractional Share Support
- **DAY limit orders** for fractional shares
- Async fill reconciliation for "unknown" status
- Auto-floor to whole shares if not fractionable

### ✅ Position Supervision
- **Orchestrator-level monitoring** continues after EntryLoop
- Stop loss checks every 30 seconds
- Broker position reconciliation

## Data Sources

- **Screening:** Massive API (4,000 stock universe sorted by liquidity)
- **Market Data:** Alpaca Market Data API (1-min bars, quotes, snapshots)
- **Execution:** Alpaca Trading API (paper or live)

## Configuration

### Gap Momentum Strategy (`bot/morning_config.py`)
```python
# Universe Filters
min_price: float = 1.0                      # $1 min price
max_price: float = 100.0                    # $100 max price
min_dollar_volume: float = 10_000_000       # $10M ADV floor
min_5min_volume: float = 250_000            # $250K first 5-min volume
min_gap_pct: float = 0.05                   # 5% min gap
max_gap_pct: float = 0.25                   # 25% max gap
opening_breakout: bool = True               # Price > first 1-min bar high
max_seed_universe: int = 4000               # 4,000 stocks screened
max_candidates_monitored: int = 35          # 35 final candidates

# Position Sizing
daily_deploy_pct: float = 0.75              # 75% of equity deployed
max_per_ticker_pct: float = 0.25            # 25% max per position
max_position_pct_of_5min_vol: float = 0.01  # 1% of 5-min volume
min_order_dollars: float = 25.0             # $25 minimum order

# Risk Guardrails
max_concurrent: int = 25                    # Max positions
max_trades_per_day: int = 25                # Max trades/day
daily_kill_r: float = -3.0                  # Stop at -3R daily

# Exit Rules
take_profit_pct: float = 0.012              # 1.2% trailing activation
trail_pct: float = 0.01                     # 1% trailing stop
stop_loss_pct: float = 0.05                 # 5% hard stop
```

## Production Deployment

```bash
# Recommended cron schedule (run at 8:25 AM)
25 8 * * 1-5 cd /path/to/Alpaca_bot && python -m bot.integrated_main

# The bot handles all timing internally:
# - 8:30 AM: Bot starts
# - 9:00-9:29 AM: Pre-market screening
# - 9:40 AM: Entry window opens
# - 11:00 AM: Hard exit guarantee
```

## Development Notes

- **No margin used** - all trades cash-settled
- **Fractional shares supported** for precise position sizing
- **Timezone aware** - all times in market timezone (ET)
- **Crash resilient** - state persistence and recovery
- **Production tested** - hard exit guarantees verified

## Key Algorithms

### Dynamic Position Allocation
```python
for each candidate (sorted low → high liquidity):
    base = remaining_cash / remaining_positions
    vol_limit = candidate.liq_5m_dollar * 0.01  # 1% of 5-min volume
    target = min(base, 25% cap, vol_limit)
    if target >= $25:
        allocate target
        remaining_cash -= target

# Second pass: distribute leftover to high-liq names (respecting both caps)
```

### Entry Reconciliation
- DAY orders return "unknown" status immediately
- Reconciliation loop checks order status every cycle
- Fills detected via `get_order_by_id()` polling
- Position opened when status changes to "filled"

### Fractional Share Handling
- Check if asset is fractionable via `is_fractionable()`
- Auto-floor to whole shares if not fractionable
- Use DAY time_in_force for fractional quantities
