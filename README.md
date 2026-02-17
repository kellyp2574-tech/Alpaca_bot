# Leveraged ETF Trading Bot

Automated trading bot for leveraged ETFs using three complementary strategies. Trades via Alpaca API with fractional share support.

## Strategies

### A. MA Crossover (50% of equity)
- **Signal:** 100-day SMA on QQQ/TLT with 3% hysteresis buffer
- **Confirmation:** 2 days entry, 5 days exit (asymmetric)
- **Trades:** QLD (2x QQQ) or UBT (2x TLT), DBMF fallback
- **Check:** Daily near close

### B. Monday Dip (30% cap)
- **Signal:** SPY Monday close < Friday close, dip ≥ 0.5%
- **Trade:** Buy UPRO (3x SPY) at close, hold 2 days
- **Take profit:** 2.5%
- **Priority:** Takes precedence over BB Reversion

### C. Bollinger Band Reversion (20% cap)
- **Signal:** SPY closes below 20-day Bollinger Band (2σ)
- **Trade:** Buy UPRO at close, exit when SPY returns to 20-day SMA
- **Stop loss:** 10% | **Take profit:** 5%
- **Mutual exclusion:** Only one dip trade (B or C) open at a time

## Backtested Performance (0.2% slippage)

| Period | CAGR | Max Drawdown | $10k → |
|--------|------|-------------|--------|
| 14.9yr (2009-2024) | +20.5% | -32.7% | $162,601 |
| 5.6yr w/ DBMF (2019-2024) | +26.8% | -28.5% | $38,102 |

Zero negative 5-year rolling windows. Worst 3-year annualized: +5.2%.

## Project Structure

```
Alpaca_bot/
├── bot/                    # Live trading bot
│   ├── config.py           # Strategy parameters & API config
│   ├── alpaca_client.py    # Alpaca API wrapper (orders, positions)
│   ├── data.py             # Market data (Alpaca primary, Yahoo fallback)
│   ├── strategies.py       # Signal logic (pure functions, no side effects)
│   ├── state_manager.py    # JSON state persistence
│   ├── main.py             # Bot orchestrator (--status, --dry-run)
│   └── scheduler.py        # Hourly cron entry point
├── backtest/               # Strategy research & backtesting scripts
├── state/                  # Runtime state & logs (gitignored)
├── .env.example            # API key template
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
# Check current state and positions
python -m bot.main --status

# Dry run — show signals without trading
python -m bot.main --dry-run

# Force a full run (ignores time-of-day checks)
python -m bot.scheduler --force

# Hourly cron entry point (normal operation)
python -m bot.scheduler
```

## Scheduler Modes & Cron Setup

Three cron lines, all at :30 to avoid open/close bells:

| Mode | Schedule | What it does |
|------|----------|-------------|
| `intraday` | Mon–Fri 10:30–2:30 | Live-price stop loss & take profit checks only |
| `monday_close` | **Monday** 3:30 PM | Monday dip buy + MA crossover + BB reversion + exits |
| `daily_close` | **Tue–Fri** 3:30 PM | MA crossover + BB reversion + dip exits |

```
# Intraday exits — Mon-Fri, 10:30 AM to 2:30 PM ET
30 10-14 * * 1-5 cd /path/to/Alpaca_bot && python -m bot.scheduler --mode intraday

# Monday close — Monday 3:30 PM (tuesday recovery + MA + BB)
30 15 * * 1 cd /path/to/Alpaca_bot && python -m bot.scheduler --mode monday_close

# Daily close — Tue-Fri 3:30 PM (MA rotation + BB entries + dip exits)
30 15 * * 2-5 cd /path/to/Alpaca_bot && python -m bot.scheduler --mode daily_close
```

Manual override: `python -m bot.scheduler --force [--dry-run]`

## Data Sources

- **Primary:** Alpaca Market Data API (SIP feed)
- **Fallback:** Yahoo Finance (automatic if Alpaca returns empty)

## Configuration

All strategy parameters are in `bot/config.py`. No margin is used — margin is controlled at the Alpaca account level.
