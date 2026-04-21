# Overnight Momentum Trading Bot

Automated equity trading strategy on Alpaca that captures overnight momentum by entering positions at 3:50 PM and exiting the next morning. Uses a composite scoring model to identify high-momentum small-cap stocks with strong volume profiles.

---

## Strategy Overview

### Core Concept: T-1 Entry / T+1 Exit

The bot operates on a **two-day cycle**:
- **Afternoon (T-1)**: Score and enter positions at 3:50 PM for overnight hold
- **Morning (T+1)**: Exit positions at 9:35 AM or 11:30 AM with hard-stop protection at 9:30 AM

### Signal: 3:50 PM Momentum Model

The bot scores stocks based on 6 metrics computed from 9:30 AM - 3:50 PM intraday bars:

1. **Intraday Return** (20% weight) - Price appreciation from 9:30 open to 3:50 close
2. **Proximity to High** (15% weight) - How close current price is to day's high
3. **Volume vs Average** (20% weight) - Last 60min volume vs average 60min volume
4. **Volume Trend** (10% weight) - Recent volume acceleration
5. **Vs Market** (25% weight) - Return relative to SPY benchmark
6. **ATR Percent** (-10% weight) - Volatility penalty (negative weight)

**Composite Score**: Normalized 0-1 score, assigned to buckets 1-5 (5 = highest quality)

**Selection Criteria**: Only bucket 4+ positions selected (top-tier momentum)

---

## Daily Schedule

### Morning Session (T+1 - Exit Day)

| Time (ET) | Activity | Details |
|-----------|----------|---------|
| **9:00 AM** | Startup | Detect overnight positions from broker, reconcile local state |
| **9:30 AM** | Hard Stop Check | Exit if open ≤ entry × 0.95 (-5% stop) |
| **9:35 AM** | Classification + Exits | Classify by open-to-9:35 return; exit if move > +0.5% |
| **11:30 AM** | Hold Bucket Exit | Exit all remaining positions |
| **11:35 AM** | Post-Exit Failsafe | Verify broker flat, run failsafe if needed |

### Afternoon Session (T-1 - Entry Day)

| Time (ET) | Activity | Details |
|-----------|----------|---------|
| **3:30 PM** | Data Collection | Build universe (Massive + Alpaca), fetch daily bars for ADV/ATR |
| **3:48 PM** | Scoring | Fetch 9:30-3:50 minute bars, compute 6 metrics, score & bucket |
| **3:50 PM** | Entry Execution | HEAD/TAIL allocation -> execution gate -> market buy orders |
| **4:00 PM** | Market Close | Save positions to state, hold overnight |

---

## Exit Strategy (Morning T+1)

### Hard Stop (-5% from entry)
**Trigger:** 9:30 AM market open  
**Condition:** `open_price <= entry_price × 0.95`  
**Action:** Immediate market sell  
**Purpose:** Protects against overnight gap-down

---

### Morning Classification (9:35 AM)

At 9:35 AM, all remaining positions are classified into one of two exit buckets based on the open-to-9:35 return.

**Classification Rule:**
```
ret_open_to_935 = (price_935 - open_price) / open_price × 100

if ret_open_to_935 > +0.5%  ->  exit immediately at 9:35 AM
else                         ->  hold to 11:30 AM
```

**Price sourcing (in priority order):**
- `open_price`: 9:30 AM snapshot preferred; first minute-bar open as fallback
- `price_935`: last 9:30-9:35 minute-bar close preferred; snapshot as fallback

Each symbol's price source is logged and persisted in the classification audit so data quality issues are visible after the fact.

**Threshold:** Configured via `EXIT_UP_MOVE_PCT = 0.5` in `config_strategy.py`

**Partial Fill Re-Routing:**  
If a 9:35 exit doesn't fully fill, the position is automatically re-scheduled to the 11:30 bucket.

---

### Failsafe (11:35 AM)
**Trigger:** 11:35 AM  
**Condition:** Broker still shows open positions  
**Action:** Multi-layer flatten (market -> limit -3% -> limit -5%)

---

## Position Sizing & Selection

### HEAD/TAIL Allocation System

Capital is split into two pools:

**Capital Split:**
- **HEAD**: 70% of deployable capital -> top 10 positions (equal-weight)
- **TAIL**: 30% of deployable capital + HEAD leftover -> remaining positions (waterfall)

**Deployable Capital:**
```python
deployable = equity × MAX_LEVERAGE  # 1.0 = cash account, no margin
```

**Minimum deployment target:** Allocator keeps walking down the ranked candidate list until 80% of capital is deployed or all valid candidates are exhausted. The tier `max_positions` cap is treated as a soft limit — it will be exceeded if needed to reach the 80% target.

---

### HEAD Allocation (Top 10 - Equal Weight)

```python
slot_size = (deployable × 0.70) / 10
max_dollars = min(slot_size, ADV × 0.003)  # ADV cap
shares = floor(max_dollars / price)

if shares < 25:  # MIN_SHARES gate
    skip position, roll full slot to TAIL
```

---

### TAIL Allocation (Positions 11-30 - Waterfall)

```python
tail_capital = (deployable × 0.30) + HEAD_leftover

for each candidate (ranked 11+):
    max_dollars = min(remaining_cash, ADV × 0.003)
    shares = floor(max_dollars / price)

    if shares < 25:
        skip (do NOT consume cash)

    deploy shares, subtract cost from remaining_cash

    stop when:
        - 80% deployment target reached
        - OR no more viable candidates
```

---

### Sizing Constraints

```python
MIN_SHARES = 25               # Skip if below this threshold
ADV_CAP_PCT = 0.003           # Max position = 0.3% of 20-day ADV
MAX_TOTAL_POSITIONS = 30      # Soft cap (exceeded to reach 80% deployment target)
```

### Universe Filters

```python
UNIVERSE_PRESET = "expanded_smallcap"
MIN_PRICE = $1.00
MAX_PRICE = $10.00
MIN_ADV_DOLLARS = $2,000,000  # $2M minimum daily volume
```

---

## Data Pipeline (Afternoon 3:30-3:50 PM)

### Stage A: Universe Building (3:30 PM)
**Massive API:**
- Full market snapshot
- Filter by price ($1-$10), tradability
- ADV filter removes illiquid names

**Fallback:** Alpaca Assets API if Massive fails

### Stage B: Daily Bars & ADV/ATR (3:30 PM)
**Alpaca daily bars (20-day lookback):**
- Computes ADV (20-day average dollar volume)
- Computes ATR (14-day average true range)
- Filters: ADV ≥ $2M

### Stage C: Minute Bar Quality (3:48 PM)
**Alpaca minute bars (9:30-3:50):**
- Fetches intraday bars for signal generation
- Filters: ≥30 minute bars required
- Removes symbols with poor data quality

### Stage D: Execution Gate (3:50 PM)
**Fresh snapshots before order submission:**
- Rejects wide spreads (>5%)
- Requires valid bid/ask quotes
- Final tradability check

---

## Order Execution

### Entry Orders (3:50 PM - Market Buy)
```python
order_type = "market"
time_in_force = "day"
```
1. PDT guard filters out same-day re-entries when equity < $50k
2. Submit market buy orders for all allocated positions
3. Poll for fills (30-second timeout)
4. Create Position objects with fill price as entry_price
5. Save positions to state for overnight hold

### Exit Orders (Morning - Market Sell)
```python
order_type = "market"
time_in_force = "day"
```
1. Submit market sell order
2. Poll for fills (30-second timeout)
3. If market sell fails -> limit sell at -3% of last price

**Failsafe Multi-Layer Flatten:**
1. Market sell (full qty)
2. Limit sell at -3% (full qty)
3. Limit sell at -5% (half qty, then remainder)
4. Manual intervention flag if all layers fail

---

## State Persistence & Recovery

### State Files

**`state/positions.json`** - Overnight positions
- Symbol, entry_price, quantity, entry_time
- ADV estimate, peak_price, current_price
- Persists across days for T+1 exit

**`state/bot_state.json`** - Same-day recovery state
- Date stamp (only restores if same day)
- Morning exit flags: `hard_stops_checked`, `v2_classified`, `exits_1130_done`, `post_exit_failsafe_done`
- Afternoon entry flags: `data_collected`, `scoring_done`, `entries_done`
- Open prices captured at 9:30 AM
- Exit schedule (`{symbol: exit_bucket}`)
- Classification audit (`{symbol: {open_price, price_935, move_5m_pct, exit_time, open_price_source, price_935_source}}`)

### Recovery Logic

**Same-day restart:**
- Restores `bot_state.json` if date matches today
- Resumes from last completed stage
- Prevents duplicate work

**New day startup:**
- Ignores stale `bot_state.json` from previous day
- Loads `positions.json` (overnight holds from yesterday's entries)
- Reconciles with broker positions as ground truth

**Broker reconciliation:**
- Fetches live broker positions at startup
- Rebuilds missing local Position objects
- Syncs quantities if mismatch
- Removes local positions broker no longer holds

---

## Architecture

```
run.py                              # Entry point
bot/
  integrated_main.py                # OvernightMomentumBot orchestrator
  position_manager_overnight.py     # Position & order management, exits, failsafe
  exit_classifier.py                # Morning exit classification (9:35 AM)
  momentum_scorer.py                # Scoring model, bucket assignment, HEAD/TAIL allocation
  universe_builder.py               # 4-stage pipeline, diagnostics, audit trails
  massive_client.py                 # Massive API client
  market_data.py                    # Alpaca data client
  state_manager.py                  # State persistence
  config.py                         # Unified config re-export
  config_broker.py                  # API credentials
  config_runtime.py                 # Paths, logging
  config_universe.py                # Price/ADV filters, universe preset
  config_strategy.py                # Scoring weights, tiers, exit rules, timing
state/
  positions.json                    # Overnight positions
  bot_state.json                    # Same-day recovery state
  audit/                            # Daily diagnostic reports
    universe_YYYY-MM-DD.json
    candidates_YYYY-MM-DD.json
    execution_YYYY-MM-DD.json
    health_YYYY-MM-DD.json
  logs/
    bot_YYYY-MM-DD.log              # Daily log file
```

---

## Setup

### Requirements
- Python 3.11+
- Alpaca trading account (paper or live)
- Massive API key for universe building

### Installation

```bash
# Clone repository
git clone https://github.com/kellyp2574-tech/Alpaca_bot.git
cd Alpaca_bot

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

Start the bot at **9:00 AM ET**. It will:
1. Manage morning exits (9:30-11:35 AM) if overnight positions exist
2. Wait for the afternoon entry window (3:30-4:00 PM)
3. Score the universe and enter new positions at 3:50 PM
4. Shut down at 4:00 PM, holding positions overnight

### Late Start Behavior

**Started after 11:35 AM with positions:**
- Immediately runs failsafe flatten
- Exits all positions at market

**Started after 4:00 PM:**
- Logs error and exits (nothing to do)

**Started between 11:35 AM - 3:30 PM:**
- Waits for 3:30 PM data collection
- Continues with afternoon entry flow

### Monitoring

**Log files:**
- `state/logs/bot_YYYY-MM-DD.log` - Daily execution log

**State files:**
- `state/positions.json` - Overnight positions
- `state/bot_state.json` - Same-day recovery state (includes classification audit)

**Audit reports:**
- `state/audit/universe_YYYY-MM-DD.json` - Universe pipeline diagnostics
- `state/audit/candidates_YYYY-MM-DD.json` - Top scored candidates
- `state/audit/execution_YYYY-MM-DD.json` - Entry execution details
- `state/audit/health_YYYY-MM-DD.json` - Daily run health metrics

---

## Configuration

All parameters split across config files:

### Universe Filters (`config_universe.py`)
```python
UNIVERSE_PRESET = "expanded_smallcap"
MIN_PRICE = 1.00              # Minimum stock price
MAX_PRICE = 10.00             # Maximum stock price
MIN_ADV_DOLLARS = 2_000_000  # $2M minimum ADV
ADV_LOOKBACK_DAYS = 20        # Days for ADV calculation
ATR_LOOKBACK_DAYS = 14        # Days for ATR calculation
```

### Scoring Weights (`config_strategy.py`)
```python
SCORE_WEIGHT_INTRADAY_RETURN = 0.20
SCORE_WEIGHT_PROXIMITY_HIGH  = 0.15
SCORE_WEIGHT_VOLUME_VS_AVG   = 0.20
SCORE_WEIGHT_VOLUME_TREND    = 0.10
SCORE_WEIGHT_VS_MARKET       = 0.25
SCORE_WEIGHT_ATR_PCT         = -0.10  # Negative = volatility penalty
```

### Position Sizing (`config_strategy.py`)
```python
MAX_LEVERAGE = 1.0            # No margin (cash account)
ADV_CAP_PCT  = 0.003          # 0.3% of ADV max position
MIN_SHARES   = 25             # Minimum shares per position

HEAD_PCT             = 0.70   # 70% capital to top 10 (equal-weight)
TAIL_PCT             = 0.30   # 30% capital to waterfall positions
MAX_HEAD_POSITIONS   = 10
MAX_TOTAL_POSITIONS  = 30     # Soft cap (exceeded to reach 80% deployment)
```

### Exit Rules (`config_strategy.py`)
```python
HARD_STOP_PCT    = -0.05      # -5% from entry at 9:30 open
EXIT_UP_MOVE_PCT =  0.5       # ret_open_to_935 > 0.5% -> exit at 9:35
V2_FAILSAFE_TIME = "11:35"    # Post-exit failsafe

V2_CLASSIFY_TIME      = "09:35"
EXIT_BUCKET_1130_TIME = "11:30"
```

### Timing (`config_strategy.py`)
```python
DATA_COLLECTION_TIME = "15:30"
SCORING_TIME         = "15:48"
ENTRY_TIME           = "15:50"
MARKET_OPEN_TIME     = "09:30"
```

---

## Known Limitations

1. **Massive API dependency** - Universe building requires Massive API. Alpaca fallback is slower and may miss symbols.

2. **Polling-based execution** - 1-second event loop. Very fast market moves between polls could be missed.

3. **Single-day holding period** - Strategy assumes overnight hold with next-morning exit. Extended holds not supported.

4. **No sector diversification** - Selection is purely momentum-based. Could concentrate in a single sector during rotations.

5. **Broker API dependency** - Failsafe system relies on broker API. If broker API fails, position state may be incorrect.

6. **No real-time monitoring alerts** - Bot logs to file but doesn't send alerts for critical events.

7. **Fixed stop level** - Hard stop (-5%) is not adaptive to per-symbol volatility.

8. **Minute bar data quality** - Requires ≥30 minute bars for scoring. Symbols with poor data are excluded.

9. **No maximum drawdown protection** - Bot will continue entering positions after consecutive losing days.

---

## Risk Warnings

⚠️ **This is an overnight momentum strategy with significant risks:**

- **Overnight gap risk:** Positions held overnight are exposed to gap risk from after-hours news/events
- **Stop-loss execution risk:** Market opens can gap through stop levels, resulting in larger losses than expected
- **Momentum reversal risk:** Strong afternoon momentum can reverse overnight or at the open
- **Liquidity risk:** 0.3% ADV cap may not prevent slippage in fast-moving small-cap names
- **Technology risk:** API failures, network issues, or bugs could cause losses or missed exits
- **Data quality risk:** Poor minute bar data can result in incorrect scoring and bad position selection
- **PDT risk:** Accounts under $25,000 equity are subject to Pattern Day Trader rules; same-day re-entries are blocked when equity < $50,000

**Always test in paper trading first. Never risk capital you cannot afford to lose.**
