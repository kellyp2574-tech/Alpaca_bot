# Alpaca Trading Bot - Morning Gap Momentum Strategy

Production-ready gap momentum trading bot that trades pre-market gappers during the morning session. Features multi-stage candidate scanning with intelligent data feed selection, smart entry order management, comprehensive monitoring system, and automated liquidity ranking.

## Strategy Overview

### Morning Gap Momentum (85% of equity deployed, 8:30 AM - 2:30 PM)
- **Signal:** Pre-market gap (5-25%) + opening breakout + volume confirmation
- **Universe:** 4,000 stocks (liquidity-ranked) → $10M+ ADV, $1-$100 price range
- **Staged Scanning:** Multi-stage refinement (8:30 AM, 9:05 AM, 9:15 AM, 9:25 AM freeze)
- **Data Feeds:** Delayed SIP for broad filtering, IEX for live refinement and trading
- **Entry:** Smart limit orders with 60s age / 1% price cancellation logic
- **Partial Fills:** Converted to positions (not discarded)
- **Exit:** 5% trailing activation, 1.5% trailing stop, 5% hard stop
- **Risk:** 25 max concurrent positions, 25 max trades/day, daily R limit -3R
- **Guarantee:** Hard exit at 2:30 PM under ALL conditions
- **Post-Market:** Automated liquidity ranking generation at 4:05 PM

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
- **Hard Exit Guarantee:** All positions flat by 2:30 PM regardless of crashes
- **Supervision System:** Orchestrator monitors positions even if EntryLoop fails
- **Smart Order Management:** 60s age OR 1% price movement cancellation logic
- **Partial Fill Handling:** Partial fills converted to positions (not discarded)
- **DAY Orders:** Fractional share support with async fill reconciliation
- **Emergency Fallbacks:** Market order escalation, broker position cleanup

### 🔒 Risk Management
- **Duplicate Prevention:** Each symbol max one entry attempt per day
- **Position Sizing:** Dynamic allocation with 25% concentration cap, 1% volume participation
- **Stop Logic:** 1.5% trailing stop after 5% gain, 5% hard stop, breakeven protection
- **Time-Based Exits:** Automatic reconciliation, DAY order fill tracking

### 📊 Multi-Stage Candidate Scanning
- **8:30 AM:** Build 4,000-symbol universe (liquidity-ranked from prior day)
- **8:30-8:40 AM:** Broad filter using delayed_sip snapshots (batched)
- **9:05 AM:** First live refinement using IEX snapshots
- **9:15 AM:** Second IEX refinement (narrow to final watchlist)
- **9:25 AM:** Freeze candidates, prepare sizing
- **9:28 AM:** Start IEX live stream
- **9:30 AM onward:** Trade from IEX live data

### 📈 Comprehensive Monitoring System
- **Live Dashboard:** Real-time P&L, positions, fill rates, slippage tracking
- **Entry Quality Metrics:** Slippage, fill rates, time-to-fill, cancellation reasons
- **Exit Quality Metrics:** Exit slippage, force-flat tracking, partial exits
- **Trade Outcomes:** Win rate, expectancy, profit factor, MFE/MAE analysis
- **Risk Exposure:** Drawdown tracking, concentration metrics, capital deployment
- **4-Layer Reporting:** Live console, EOD report, trade ledger CSV, rolling stats JSON
- **Automated Alerts:** Data quality, execution quality, risk threshold warnings

### 🔄 Automated Liquidity Ranking
- **Post-Market Job:** Runs at 4:05 PM after all positions closed
- **Prior-Day Data:** Fetches last trading day's bars (handles weekends/holidays)
- **Dollar Volume Ranking:** Calculates prev_close × prev_volume for all symbols
- **Next-Day Universe:** Top 4,000 most liquid symbols selected for scanning
- **Resilient Execution:** Runs even if position state has issues

## Project Structure

```
Alpaca_bot/
├── bot/                           # Trading bot
│   ├── integrated_main.py         # Main orchestrator (8:30 AM - 4:05 PM)
│   ├── morning_main.py            # Gap momentum entry logic with smart order management
│   ├── morning_main_staged.py     # Staged candidate fetching orchestration
│   ├── premarket_scan_staged.py   # Multi-stage scanning (delayed_sip → IEX)
│   ├── position_manager.py        # Position management with trailing stops
│   ├── execution.py               # Order execution with fractional share support
│   ├── morning_config.py          # Strategy parameters and timeline constants
│   ├── state_manager.py           # State persistence
│   ├── risk_manager.py            # Risk controls & daily limits
│   ├── monitoring.py              # Comprehensive monitoring system (11 sections)
│   ├── monitor_reports.py         # 4-layer reporting (live, EOD, CSV, JSON)
│   ├── universe_loader.py         # Alpaca Assets API universe builder
│   ├── liquidity_ranker.py        # Post-market liquidity ranking (4:05 PM)
│   ├── data_alpaca.py             # Alpaca data adapter (delayed_sip, IEX feeds)
│   ├── clock.py                   # Market time utilities
│   └── storage.py                 # Data structures (Candidate, Position, etc.)
├── state/                         # Runtime state & logs (gitignored)
│   ├── universe/                  # Universe and liquidity ranking cache
│   │   ├── alpaca_assets_us_equity.json
│   │   └── liquidity_ranking.json
│   └── reports/                   # Monitoring reports
│       ├── eod_report_YYYY-MM-DD.txt
│       ├── trade_ledger.csv
│       └── rolling_stats.json
├── .env.example                   # API key template
├── close_all_positions.py         # Utility to close all positions
├── test_liquidity_ranker.py       # Manual test for liquidity ranking
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
| 8:30 AM | Bot starts, build 4,000-symbol universe (liquidity-ranked) |
| 8:30-8:40 AM | Stage 1: Broad filter using delayed_sip snapshots |
| 9:05 AM | Stage 2: First live refinement using IEX snapshots |
| 9:15 AM | Stage 3: Second IEX refinement (narrow to final watchlist) |
| 9:25 AM | Freeze candidates, prepare sizing |
| 9:28 AM | Start IEX live stream |
| 9:30 AM | Market open, collect first 5 bars |
| 9:40 AM | Entry window opens, dynamic allocation calculated |
| 9:40-2:30 PM | Active trading with smart order management & trailing stops |
| 2:30 PM | **Hard exit guarantee** - all positions flat |
| 2:30-4:05 PM | Position supervision, cleanup, reconciliation |
| 4:05 PM | Generate liquidity ranking for next day, EOD reports |
| 4:10 PM | Bot shutdown |

## Critical Guarantees

### ✅ Hard Exit Guarantee
- **All positions flat by 2:30 PM** regardless of:
  - EntryLoop crashes or exceptions
  - Network connectivity issues
  - Stream data interruptions
  - Manual intervention

### ✅ Smart Entry Order Management
- **60-second age OR 1% price movement** cancellation logic
- Orders given time to fill before cancellation
- Prevents premature cancellations on slower executions
- Cancels if price runs away (>1% above limit)

### ✅ Partial Fill Handling
- **Partial fills converted to positions** (not discarded)
- Consistent behavior: immediate execution and reconciliation
- Proper deployment tracking and stats recording
- Prevents re-entry attempts on partial fills

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

- **Universe:** Alpaca Assets API (master asset catalog)
- **Liquidity Ranking:** Prior-day bars from Alpaca (dollar volume calculation)
- **Broad Filtering:** Alpaca delayed_sip feed (15-min delayed, broad coverage)
- **Live Refinement:** Alpaca IEX feed (free live data)
- **Market Data:** Alpaca Market Data API (1-min bars, quotes, snapshots)
- **Execution:** Alpaca Trading API (paper or live)

## Monitoring & Reporting

### 11-Section Monitoring System
1. **Daily Dashboard:** Top-line metrics (equity, P&L, positions, trades)
2. **Funnel Metrics:** Candidate pipeline tracking with drop reasons
3. **Entry Execution Quality:** Slippage, fill rates, time-to-fill, cancellations
4. **Exit Execution Quality:** Exit slippage, force-flat tracking, partial exits
5. **Trade Outcomes:** Win rate, expectancy, profit factor, MFE/MAE
6. **Running Tallies:** Intraday, daily, weekly, MTD, all-time, rolling stats
7. **Risk & Exposure:** Drawdown, concentration, capital deployment
8. **Data Integrity:** Missing data, API failures, stale quotes
9. **Broker Integrity:** Order rejections, position mismatches, reconciliations
10. **Strategy Drift:** Returns by day-of-week, gap bucket, fill time, etc.
11. **Alerts:** Automated warnings for data quality, execution, risk thresholds

### 4-Layer Reporting
1. **Live Console Summary:** Compact dashboard printed every 5 minutes during market hours
2. **End-of-Day Report:** Comprehensive text report saved to `state/reports/eod_report_YYYY-MM-DD.txt`
3. **Trade Ledger CSV:** One row per order event in `state/reports/trade_ledger.csv`
4. **Rolling Stats JSON:** Persistent cumulative stats in `state/reports/rolling_stats.json`

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
daily_deploy_pct: float = 0.85              # 85% of equity deployed
max_per_ticker_pct: float = 0.25            # 25% max per position
max_position_pct_of_5min_vol: float = 0.01  # 1% of 5-min volume
min_order_dollars: float = 25.0             # $25 minimum order

# Risk Guardrails
max_concurrent: int = 25                    # Max positions
max_trades_per_day: int = 25                # Max trades/day
daily_kill_r: float = -3.0                  # Stop at -3R daily

# Exit Rules
take_profit_pct: float = 0.05               # 5% trailing activation
trail_pct: float = 0.015                    # 1.5% trailing stop
stop_loss_pct: float = 0.05                 # 5% hard stop
```

## Production Deployment

```bash
# Recommended cron schedule (run at 8:25 AM)
25 8 * * 1-5 cd /path/to/Alpaca_bot && python -m bot.integrated_main

# The bot handles all timing internally:
# - 8:30 AM: Bot starts, universe build
# - 8:30-9:25 AM: Multi-stage candidate scanning
# - 9:40 AM: Entry window opens
# - 2:30 PM: Hard exit guarantee
# - 4:05 PM: Liquidity ranking generation
# - 4:10 PM: Bot shutdown
```

## Recent Updates (March 2026)

### 🎯 Multi-Stage Candidate Scanning Timeline
**Problem:** Previous single-stage scanning at 9:00 AM was inefficient and used expensive data feeds unnecessarily.

**Solution:** Implemented 4-stage refinement pipeline with intelligent feed selection:
- **Stage 1 (8:30 AM):** Broad filter using delayed_sip snapshots (15-min delayed, free)
- **Stage 2 (9:05 AM):** First live refinement using IEX snapshots (free live data)
- **Stage 3 (9:15 AM):** Second IEX refinement (narrow to final watchlist)
- **Stage 4 (9:25 AM):** Freeze candidates, prepare sizing, start stream at 9:28 AM

**Impact:** Reduced API costs, improved candidate quality, better data freshness at entry time.

### 🔄 Automated Liquidity Ranking System
**Problem:** Universe selection was arbitrary (alphabetical), not based on tradability.

**Solution:** Implemented daily post-market job (4:05 PM) that:
- Fetches prior-day bars for all tradable symbols
- Calculates dollar volume (prev_close × prev_volume)
- Ranks and selects top 4,000 most liquid symbols
- Handles weekends/holidays by fetching last 5 days and using most recent bar
- Runs even if position state has issues (resilient execution)

**Impact:** Universe now consists of most liquid, tradable stocks. Better fill rates and tighter spreads.

### 🎯 Smart Entry Order Management
**Problem:** Previous logic canceled ANY open entry order immediately (too aggressive).

**Solution:** Implemented intelligent cancellation logic:
- Only cancel if **age >= 60 seconds** OR **price moved >= 1% above limit**
- Uses submitted_ts from pending state for accurate age tracking
- Uses broker's actual limit_price (fallback to intended_price)
- Uses latest quote cache for price movement check
- Logs cancellation reason (time vs price trigger)

**Impact:** Orders given time to fill (60s window), better fill rates, prevents premature cancellations.

### ✅ Partial Fill Position Handling
**Problem:** Partial fills were being discarded (marked done, no position opened).

**Solution:** Fixed both immediate execution and reconciliation paths:
- **Immediate partial fills:** Open position with filled quantity, mark done for day
- **Reconciled partial fills:** Same behavior, consistent across all code paths
- Proper deployment tracking (on_deploy called)
- Proper stats recording (record_entry called)

**Impact:** Partial fills now create real positions, proper risk tracking, no lost capital.

### 🐛 Critical Bug Fixes
**Reconciliation Iteration Bug:**
- **Before:** `for pending in pending_entries:` (iterates dict keys, not values)
- **After:** `for client_order_id, pending in pending_entries.items()`
- **Impact:** Would have crashed on first reconciliation attempt

**Partial Fill Deployment Tracking:**
- **Before:** Reconciled partial fills didn't call `on_deploy()` or `record_entry()`
- **After:** Proper deployment and stats tracking for all partial fills
- **Impact:** Accurate risk management and reporting

### 📊 Comprehensive Monitoring System
**Problem:** No visibility into execution quality, slippage, or strategy performance.

**Solution:** Implemented 11-section monitoring system with 4-layer reporting:
- **11 Monitoring Sections:** Dashboard, funnel, entry quality, exit quality, trade outcomes, tallies, risk, data integrity, broker integrity, drift diagnostics, alerts
- **Live Console:** Compact dashboard every 5 minutes during market hours
- **EOD Report:** Full daily summary saved to text file
- **Trade Ledger CSV:** One row per order event for analysis
- **Rolling Stats JSON:** Persistent cumulative performance metrics

**Impact:** Full visibility into bot performance, execution quality, and risk metrics.

### 🔧 Logging Optimization
**Problem:** Excessive INFO-level logging cluttered logs during production.

**Solution:** Downgraded verbose logs to DEBUG level:
- Position sizing details
- Volume cap calculations
- Breakeven/trail activation messages
- Exit reconciliation details
- Supervision loop status messages

**Impact:** Cleaner production logs, easier to spot important events and errors.

## Development Notes

- **No margin used** - all trades cash-settled
- **Fractional shares supported** for precise position sizing
- **Timezone aware** - all times in market timezone (ET)
- **Crash resilient** - state persistence and recovery
- **Production tested** - hard exit guarantees verified
- **Comprehensive monitoring** - 11-section metrics with 4-layer reporting
- **Smart order management** - 60s age / 1% price cancellation logic
- **Liquidity-ranked universe** - Top 4,000 symbols by dollar volume

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
