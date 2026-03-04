# Integrated Trading Bot - Morning Momentum + ETF Rotation

Production-ready integrated trading bot that combines two complementary strategies for comprehensive market coverage. Trades via Alpaca API with robust risk management and position supervision.

## Strategies

### A. Morning Momentum (50% of equity, 8:30-10:30 AM)
- **Signal:** Pre-market gap + 5-minute volume spike + opening strength
- **Screening:** $1M+ first 5min volume, 7-15% gap, green opening candle
- **Entry:** IOC orders at 9:30-10:30 AM with strict duplicate prevention
- **Exit:** 1.2% trailing activation, 1% trailing stop, 5% hard stop
- **Risk:** 2% per trade, max 12 concurrent positions, daily R limit -3R
- **Guarantee:** Hard exit at 10:30 AM under ALL conditions

### B. ETF Rotation (50% of equity, 11:00 AM-3:30 PM)
- **Signal:** 100-day SMA crossover on QQQ/TLT with 3% hysteresis
- **Trades:** QLD (2x QQQ) or UBT (2x TLT), DBMF fallback
- **Check:** Hourly rotation checks with position synchronization
- **Risk:** Broker position reconciliation, orphan detection

## Backtested Performance (0.2% slippage)

| Period | CAGR | Max Drawdown | $10k → |
|--------|------|-------------|--------|
| 14.9yr (2009-2024) | +20.5% | -32.7% | $162,601 |
| 5.6yr w/ DBMF (2019-2024) | +26.8% | -28.5% | $38,102 |

Zero negative 5-year rolling windows. Worst 3-year annualized: +5.2%.

## Key Features

### 🚀 Production-Grade Reliability
- **Hard Exit Guarantee:** All MM positions flat by 10:30 AM regardless of crashes
- **Supervision System:** Orchestrator monitors positions even if EntryLoop fails
- **IOC Orders:** No hanging orders - immediate fill or cancel with retry logic
- **Emergency Fallbacks:** Market order escalation, broker position cleanup

### 🔒 Risk Management
- **Duplicate Prevention:** Each symbol max one entry attempt per day
- **Position Sizing:** 50% allocation per strategy, daily deploy caps
- **Stop Logic:** ATR-based stops with min/max bounds, breakeven protection
- **Time-Based Exits:** Automatic reconciliation, timeout handling

### 📊 Market Coverage
- **Morning Session:** 8:30 AM - 10:30 AM (momentum plays)
- **Afternoon Session:** 11:00 AM - 3:30 PM (ETF rotation)
- **Data Sources:** Alpaca primary, Yahoo fallback
- **Quote Handling:** Real-time quotes with stale quote protection

## Project Structure

```
Alpaca_bot/
├── bot/                           # Integrated trading bot
│   ├── integrated_main.py         # Main orchestrator (8:30 AM - 3:30 PM)
│   ├── morning_main.py           # Morning momentum entry logic
│   ├── position_manager.py       # Position management with exit logic
│   ├── execution.py              # Order execution with marketable limits
│   ├── morning_config.py         # MM strategy parameters
│   ├── config.py                 # ETF rotation parameters
│   ├── premarket_scan.py         # Candidate screening (9:30-9:35)
│   ├── state_manager.py          # State persistence
│   ├── risk_manager.py           # Risk controls & daily limits
│   ├── data_sources.py           # Market data abstraction
│   ├── clock.py                  # Market time utilities
│   └── main.py                   # Legacy standalone modes
├── state/                        # Runtime state & logs (gitignored)
├── .env.example                  # API key template
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
# Run integrated bot (main production mode)
python -m bot.integrated_main

# Check current state and positions
python -m bot.main --status

# Dry run — show signals without trading
python -m bot.integrated_main --dry-run

# Legacy standalone modes
python -m bot.main --dry-run
```

## Daily Schedule

| Time | Strategy | Activity |
|------|----------|----------|
| 8:30 AM | MM | Pre-market candidate screening |
| 9:25 AM | MM | Candidate evaluation & preparation |
| 9:30-10:30 AM | MM | Entry window with IOC orders |
| 10:30 AM | MM | **Hard exit guarantee** - all positions flat |
| 11:00 AM-3:30 PM | ETF | Hourly rotation checks |
| 3:30 PM | ETF | Final rotation check |
| 4:00 PM | Both | End-of-day reconciliation |

## Critical Guarantees

### ✅ Hard Exit Guarantee
- **All MM positions flat by 10:30 AM** regardless of:
  - EntryLoop crashes or exceptions
  - Network connectivity issues
  - Stream data interruptions
  - Manual intervention

### ✅ No Duplicate Entries
- **Maximum one entry attempt per symbol per day**
- IOC unfilled entries marked "done for today"
- Persistent attempt tracking with audit trail

### ✅ No Hanging Orders
- **All orders use IOC (Immediate or Cancel)**
- Retry logic with progressive aggressiveness
- Emergency market order escalation

### ✅ Position Supervision
- **Orchestrator-level monitoring** continues after EntryLoop
- Stop loss checks every 30 seconds
- Broker position reconciliation

## Data Sources

- **Primary:** Alpaca Market Data API (SIP feed)
- **Fallback:** Yahoo Finance (automatic if Alpaca returns empty)

## Configuration

### Morning Momentum (`bot/morning_config.py`)
```python
# Screening
min_5min_volume: float = 1_000_000  # $1M first 5min volume
min_gap_pct: float = 0.07            # 7% min gap
max_gap_pct: float = 0.15            # 15% max gap
opening_breakout: bool = True         # Price > first 1-min bar high

# Risk
max_concurrent: int = 12             # Max positions
risk_per_trade: float = 0.02         # 2% risk per trade
daily_kill_r: float = -3.0           # Stop at -3R daily

# Exits
take_profit_pct: float = 0.012       # 1.2% trailing activation
trail_pct: float = 0.01               # 1% trailing stop
stop_loss_pct: float = 0.05           # 5% hard stop
```

### ETF Rotation (`bot/config.py`)
```python
# Allocation
MA_ALLOC_PCT: float = 0.50           # 50% to ETF rotation
MA_HYSTERESIS_PCT: float = 0.03      # 3% hysteresis buffer

# ETFs
QLD: str = "QLD"                      # 2x QQQ
UBT: str = "UBT"                      # 2x TLT
DBMF: str = "DBMF"                    # Money market fallback
```

## Production Deployment

```bash
# Recommended cron schedule (run at 8:25 AM)
25 8 * * 1-5 cd /path/to/Alpaca_bot && python -m bot.integrated_main

# The bot handles all timing internally:
# - 8:30 AM: Pre-market screening
# - 9:30 AM: Entry window opens
# - 10:30 AM: Hard exit guarantee
# - 11:00 AM-3:30 PM: ETF rotation
```

## Development Notes

- **No margin used** - all trades cash-settled
- **Fractional shares supported** for precise position sizing
- **Timezone aware** - all times in market timezone (ET)
- **Crash resilient** - state persistence and recovery
- **Production tested** - hard exit guarantees verified

## Legacy Support

The original standalone modes remain available:
- `python -m bot.main` for ETF rotation only
- `python -m bot.scheduler` for cron-based execution

However, `integrated_main.py` is the recommended production approach for complete strategy coverage.
