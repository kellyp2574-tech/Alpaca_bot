# Gap Momentum Day Trading Bot

Automated equity day-trading strategy on Alpaca that captures overnight gap momentum in low-priced stocks ($0.50-$5.00). Uses a staged entry system with Market-On-Open orders followed by aggressive post-open rescue passes, and VIX-conditioned exit timing with trailing stops.

---

## Strategy Overview

### Signal: Overnight Gap Momentum

The bot identifies stocks with significant overnight gaps (3%+ minimum) and strong liquidity, then enters long positions at the market open to capture intraday momentum continuation.

**Core Candidates (4%+ gaps):**
- Primary allocation target
- Highest conviction trades
- Deployed first with full capital allocation

**Filler Candidates (3-4% gaps):**
- Secondary allocation (conditional)
- Only used if core deployment <80% and capital remains
- Fills remaining position slots up to MAX_POSITIONS

### Entry: Staged 3-Phase Execution

**Phase 1: MOO Slice (9:27 AM)**
- Submit 25% of target position as Market-On-Open orders
- Executes at 9:30 AM opening auction
- Minimizes slippage vs. post-open market orders

**Phase 2: First Rescue Pass (9:30:10 AM)**
- Reconcile MOO fills
- Submit aggressive marketable limits for remaining 75%
- Uses live quotes with 50 bps buffer above ask
- Chase guard: skip if price moved >3% from expected open

**Phase 3: Second Rescue Pass (9:30:30 AM)**
- Final fill attempt for any remaining unfilled size
- Finalize positions with weighted average entry price

### Exit: VIX-Conditioned Timing

**Low VIX (<12):** Exit at 2:30 PM
- Sliced over 10 minutes (3 slices)
- Low volatility = take profits early

**Middle VIX (12-22):** Trailing stop or 3:30 PM exit
- Trailing stop: Activate at +15% gain, trail by 3%
- If no trailing stop triggered: exit at 3:30 PM (sliced over 8 minutes)

**High VIX (>22):** Exit at 3:30 PM
- Sliced over 6 minutes (3 slices)
- High volatility = hold longer for momentum continuation

### Failsafe Flatten System

Independent broker-based position checks at **3:30 PM, 3:45 PM, and 3:58 PM** to ensure all positions are closed before market close, regardless of local bot state.

---

## Daily Schedule

| Time (ET) | Activity | Details |
|-----------|----------|---------|
| **9:00 AM** | Universe Building | Pull full market snapshot from Massive API, filter by price ($0.50-$5.00) |
| **9:25 AM** | Candidate Selection | Compute gaps, split into core (4%+) and filler (3-4%), rank by liquidity |
| **9:27 AM** | MOO Slice Submission | Submit 25% of target position as Market-On-Open orders |
| **9:30:10 AM** | Rescue Pass 1 | Reconcile MOO fills, submit aggressive limits for remaining size |
| **9:30:30 AM** | Rescue Pass 2 | Final rescue pass, finalize positions |
| **9:30-4:00 PM** | Exit Monitoring | VIX-conditioned exits with trailing stops or time-based sliced exits |
| **3:30 PM** | Failsafe Flatten 1 | Broker-based position check and flatten |
| **3:45 PM** | Failsafe Flatten 2 | Second broker-based flatten sweep |
| **3:58 PM** | Failsafe Flatten 3 | Final pre-close flatten sweep |
| **4:00 PM** | Market Close | Final broker flatten if needed, save state, shut down |

---

## Position Sizing & Risk Management

### Capital Allocation

**Dynamic per-position budget:**
```
remaining_capital = total_buying_power - allocated_capital
per_position_budget = remaining_capital / remaining_slots
```

**Liquidity cap:**
```
max_position_size = min(per_position_budget, ADV × 0.003) / current_price
```

Each position is capped at **0.3% of Average Daily Volume** to ensure liquid exits.

### Two-Phase Allocation

**Phase 1: Core Candidates (4%+ gaps)**
- Deploy to highest-conviction trades first
- Calculate deployment ratio = used_capital / total_capital
- Track remaining capital and position slots

**Phase 2: Filler Candidates (3-4% gaps) - Conditional**
- Only triggered if:
  - Core deployment ratio < 80%
  - Remaining capital > $1,000
  - Remaining position slots available
- Uses only remaining capital (prevents over-allocation)
- Limited to remaining slots (respects MAX_POSITIONS cap)

### Configuration Parameters

```python
MIN_PRICE = 0.50              # Minimum stock price
MAX_PRICE = 5.00              # Maximum stock price
MIN_GAP_PCT = 3.0             # Minimum gap percentage
MAX_GAP_PCT = 50.0            # Maximum gap percentage (sanity check)
MIN_ADV_DOLLARS = 5_000_000   # $5M minimum average daily dollar volume
LIQUIDITY_CAP_PCT = 0.003     # 0.3% of ADV max position size
MAX_POSITIONS = 100           # Maximum concurrent positions
```

---

## Data Sources

### Universe Building (Step 1)
**Primary:** Massive API (Polygon)
- Full market snapshot at 9:00 AM
- Filter by price range ($0.50-$5.00)
- ~4,000-8,000 symbols typically

**Fallback:** Alpaca Assets API
- Used if Massive API fails after 3 retries
- Fetches tradable US equities
- Batched snapshot requests (1,000 symbols per batch)

### Gap Calculation (Step 2)
**Massive API snapshot at 9:25 AM:**
- Fresh snapshot (not stale 9:00 data)
- Provides: open, prev_close, volume, prev_volume
- Gap % = (open - prev_close) / prev_close × 100
- ADV estimate = prev_volume × prev_close

### Live Execution Data
**Alpaca IEX feed:**
- Real-time quotes for rescue pass pricing
- Position monitoring during trading hours
- VIX level from Alpaca snapshot (fallback: VIXY ETF × 10)

---

## Order Execution Details

### MOO Orders (Market-On-Open)
```python
order_type = "market"
time_in_force = "opg"  # Executes at 9:30 AM auction
```
- Must submit before 9:28 AM cutoff (9:27:30 safety buffer)
- Polls for fills with 5-minute timeout
- Does NOT aggressively cancel on partial fills (lets auction complete)

### Rescue Pass Orders (Marketable Limits)
```python
order_type = "limit"
time_in_force = "day"
limit_price = ask × 1.005  # 50 bps above ask for aggressive fill
```
- Parallel submission: all orders submitted first, then polled
- 10-second fill timeout with partial fill cancellation
- Buying power scaling: reduces quantities if capital insufficient
- Chase guard: skips if price moved >3% from expected open

### Exit Orders (Market)
```python
order_type = "market"
time_in_force = "day"
```
- Trailing stops: instant full exit (no slicing)
- Time-based exits: sliced over 6-10 minutes (3 slices)
- Failsafe flattens: broker-based market orders at 3:30/3:45/3:58 PM

---

## State Persistence & Recovery

### State Files

**`state/positions.json`** - Active positions
- Symbol, entry price, quantity, entry time
- Gap %, ADV estimate
- Peak price, trailing stop price, trailing stop active flag

**`state/bot_state.json`** - Bot execution state
- Stage flags (universe, candidates, entry, exit)
- VIX level
- Staged entry state (stage1_done, stage2_done, entry_submission_locked)
- Entry plans (MOO/rescue fill tracking)

**`state/pre_trade_state.json`** - Pre-trade data for recovery
- Universe symbols
- Massive snapshots
- Candidates (core + filler)
- Date stamp for validation

### Recovery Logic

**Startup before 9:27:30:**
- Restore positions, entry plans, and pre-trade state
- Resume from last completed stage

**Startup 9:27:30-9:30 with no entry plans:**
- Disable new entries (missed MOO cutoff)
- Only run exit/failsafe logic

**Startup after 9:30:30 with no positions:**
- Disable entries and exits
- Only run failsafe flatten sweeps

**Broker reconciliation:**
- On startup, fetch live broker positions
- Rebuild missing local Position objects
- Prevents silent broker/local mismatch

---

## Architecture

```
run.py                      # Entry point
bot/
  integrated_main.py        # GapMomentumBot orchestrator — daily schedule, event loop
  config.py                 # API keys (.env), state paths, logging, strategy params
  position_manager.py       # Position & order management, staged entry, exits
  gap_calculator.py         # Gap calculation, candidate filtering, liquidity ranking
  massive_client.py         # Massive API client for universe building
  market_data.py            # Alpaca data client (IEX feed)
  vix_fetcher.py            # VIX level fetching (Alpaca snapshot or VIXY fallback)
  state_manager.py          # State persistence and recovery
state/
  positions.json            # Active positions (auto-created)
  bot_state.json            # Bot execution state (auto-created)
  pre_trade_state.json      # Pre-trade data for recovery (auto-created)
  daily_log.json            # Daily summary log (auto-created)
  logs/
    bot.log                 # Main log file
    trades.log              # Trade-specific log
```

---

## Setup

### Requirements
- Python 3.11+
- Alpaca trading account (paper or live)
- Massive API key (Polygon) for universe building

### Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys:
#   ALPACA_API_KEY=your_alpaca_key
#   ALPACA_SECRET_KEY=your_alpaca_secret
#   ALPACA_PAPER=true  # or false for live trading
#   MASSIVE_API_KEY=your_massive_key
```

### Dependencies
```
requests>=2.31.0
python-dotenv>=1.0.0
```

---

## Usage

### Basic Usage
```bash
python run.py
```

Start the bot at or before **9:00 AM ET**. It will:
1. Wait for each scheduled event
2. Execute trades automatically
3. Shut down at 4:00 PM after market close

### Late Start Behavior

**Started after 9:00 but before 9:25:**
- Immediately runs universe building
- Continues normally

**Started after 9:25 but before 9:27:30:**
- Immediately runs candidate selection
- Continues to MOO submission

**Started after 9:27:30 but before 9:30:**
- Skips entry stage (missed MOO cutoff)
- Only runs exit/failsafe logic

**Started after 9:30:30:**
- Checks for existing positions
- If positions exist: runs exit/failsafe logic
- If no positions: only runs failsafe flatten sweeps

### Monitoring

**Log files:**
- `state/logs/bot.log` - Main execution log
- `state/logs/trades.log` - Trade-specific events

**State files:**
- `state/positions.json` - Current positions
- `state/bot_state.json` - Execution state
- `state/daily_log.json` - Daily summaries

---

## Configuration

All parameters in `bot/config.py`:

### Price & Gap Filters
```python
MIN_PRICE = 0.50              # Minimum stock price
MAX_PRICE = 5.00              # Maximum stock price
MIN_GAP_PCT = 3.0             # Minimum gap percentage
MAX_GAP_PCT = 50.0            # Maximum gap percentage
```

### Volume & Liquidity
```python
MIN_ADV_DOLLARS = 5_000_000   # $5M minimum ADV
LIQUIDITY_CAP_PCT = 0.003     # 0.3% of ADV max position
MAX_POSITIONS = 100           # Max concurrent positions
```

### VIX Exit Thresholds
```python
VIX_LOW_THRESHOLD = 12.0      # Below = early exit (2:30 PM)
VIX_HIGH_THRESHOLD = 22.0     # Above = late exit (3:30 PM)
EXIT_TIME_LOW_VIX = "14:30"   # 2:30 PM
EXIT_TIME_MIDDLE_VIX = "15:30"  # 3:30 PM
EXIT_TIME_HIGH_VIX = "15:30"  # 3:30 PM
```

### Trailing Stop (Middle VIX Regime)
```python
TRAILING_STOP_ACTIVATION = 0.15  # 15% gain to activate
TRAILING_STOP_PCT = 0.03         # 3% trail
```

### Staged Entry
```python
USE_STAGED_OPEN_ENTRY = True     # Enable staged entry
MOO_ENTRY_PCT = 0.25             # 25% MOO slice
POST_OPEN_ENTRY_TIME_1 = "09:30:10"  # First rescue pass
POST_OPEN_ENTRY_TIME_2 = "09:30:30"  # Second rescue pass
POST_OPEN_BUY_LIMIT_BUFFER = 0.005   # 50 bps above ask
MAX_CHASE_FROM_OPEN_PCT = 0.03       # 3% max chase
MIN_RESCUE_NOTIONAL = 100.0          # Skip tiny orders
MIN_RESCUE_SHARES = 1                # Minimum share count
```

---

## Known Limitations

1. **Massive API dependency** - Universe building requires Massive API. Alpaca fallback is slower and may miss symbols.

2. **No intraday monitoring system** - No real-time performance tracking, alerts, or trade ledgers (monitoring system referenced in code memories but not implemented).

3. **Polling-based execution** - 1-second event loop. Very fast market moves between polls could be missed.

4. **No maximum position size cap** - Liquidity cap could theoretically produce very large positions for high-ADV stocks. Consider adding absolute dollar/share caps.

5. **Sliced exits use time-based execution** - Not VWAP-aware. Could improve with volume-weighted slicing.

6. **No backtesting framework** - Strategy parameters tuned manually without systematic backtesting.

7. **VIX fallback estimation** - Uses VIXY × 10 approximation if VIX unavailable. Could be inaccurate.

8. **State recovery race conditions** - Startup in 9:27:30-9:30 window could abandon live MOO orders if no entry plans saved.

9. **No duplicate protection between core/filler** - Same symbol could theoretically appear in both core and filler candidate lists.

10. **Massive API endpoint may be Polygon-specific** - Code uses Polygon v2 endpoint structure. Verify compatibility with actual Massive.com API.

---

## Risk Warnings

⚠️ **This is a high-frequency day-trading strategy with significant risks:**

- **Gap fade risk:** Overnight gaps can reverse intraday, causing losses
- **Low-priced stock volatility:** $0.50-$5.00 stocks are highly volatile
- **Liquidity risk:** 0.3% ADV cap may not prevent slippage in fast markets
- **Technology risk:** API failures, network issues, or bugs could cause losses
- **Regulatory risk:** Pattern Day Trader rules apply (requires $25k+ for margin accounts)

**Always test in paper trading first. Never risk capital you cannot afford to lose.**
