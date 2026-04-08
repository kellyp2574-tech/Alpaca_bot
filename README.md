# Overnight Momentum Trading Bot

Automated equity trading strategy on Alpaca that captures overnight momentum by entering positions at 3:50 PM and exiting the next morning with stop-loss protection. Uses a 350-model scoring system to identify high-momentum stocks with strong volume profiles.

---

## Strategy Overview

### Core Concept: T-1 Entry / T+1 Exit

The bot operates on a **two-day cycle**:
- **Afternoon (T-1)**: Score and enter positions at 3:50 PM for overnight hold
- **Morning (T+1)**: Exit positions at 9:30 AM, 9:35 AM, or 11:00 AM with stop-loss protection

### Signal: 350 Model (9:30-3:50 Momentum)

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
|-----------|----------|---------||
| **9:00 AM** | Startup | Detect overnight positions from broker, reconcile local state |
| **9:30 AM** | Hard Stop Check | Exit if open ≤ entry × 0.95 (-5% stop) |
| **9:35 AM** | Drop Stop Check | Exit if price dropped ≥6% from open high |
| **11:00 AM** | Final Exit | Exit ALL remaining positions at market |
| **11:05 AM** | Post-Exit Failsafe | Verify broker flat, run failsafe if needed |

### Afternoon Session (T-1 - Entry Day)

| Time (ET) | Activity | Details |
|-----------|----------|---------||
| **3:30 PM** | Data Collection | Build universe (Massive + Alpaca), fetch daily bars for ADV/ATR |
| **3:48 PM** | Scoring | Fetch 9:30-3:50 minute bars, compute 6 metrics, score & bucket |
| **3:50 PM** | Entry Execution | Select positions by tier, size, submit market buy orders |
| **4:00 PM** | Market Close | Save positions to state, hold overnight |

---

## Exit Strategy (Morning T+1)

### Hard Stop (-5% from entry)
**Trigger:** 9:30 AM market open  
**Condition:** `open_price ≤ entry_price × 0.95`  
**Action:** Immediate market sell

**Purpose:** Protects against overnight gap-down or weak opening

### Drop Stop (6% from open high)
**Trigger:** 9:35 AM  
**Condition:** `current_price drops ≥6% from open_high`  
**Calculation:** `open_high = max(open_price, price_at_935)`  
**Action:** Immediate market sell

**Purpose:** Protects against early morning reversal after strong open

### Time Stop (11:00 AM)
**Trigger:** 11:00 AM  
**Condition:** Unconditional  
**Action:** Exit ALL remaining positions at market

**Purpose:** Captures overnight momentum before midday chop

### Failsafe (11:05 AM)
**Trigger:** 11:05 AM  
**Condition:** Broker still shows positions  
**Action:** Multi-layer flatten (market → limit -3% → limit -5%)

---

## Position Sizing & Selection

### Account Tier System

Positions selected based on account equity:

| Equity Range | Selection Mode | Min Bucket | Max Positions |
|--------------|----------------|------------|---------------|
| < $25,000 | Top 10 | 4 | 10 |
| $25,000 - $100,000 | Top 20 | 4 | 20 |
| > $100,000 | All bucket ≥4 | 4 | 100 |

### Position Sizing

**Equal weight allocation:**
```python
per_position_budget = equity / num_positions
```

**Liquidity cap (0.3% of ADV):**
```python
max_shares = (ADV × 0.003) / current_price
```

**Absolute cap:**
```python
max_position_dollars = $50,000
```

**Final quantity:**
```python
qty = min(
    per_position_budget / price,
    ADV × 0.003 / price,
    $50,000 / price
)
```

### Universe Filters

```python
MIN_PRICE = $0.50
MAX_PRICE = $50.00
MIN_ADV_DOLLARS = $500,000  # $500K minimum daily volume
```

---

## Data Pipeline (Afternoon 3:30-3:50 PM)

### Stage A: Universe Building (3:30 PM)
**Massive API:**
- Full market snapshot
- Filter by price ($0.50-$50), tradability
- ~4,000-8,000 symbols typically

**Fallback:** Alpaca Assets API if Massive fails

### Stage B: Daily Bars & ADV/ATR (3:30 PM)
**Alpaca daily bars:**
- Fetches last 20 days of daily bars
- Computes ADV (20-day average dollar volume)
- Computes ATR (20-day average true range)
- Filters: ADV ≥ $500K

### Stage C: Minute Bar Quality (3:48 PM)
**Alpaca minute bars (9:30-3:50):**
- Fetches 380 minute bars for signal generation
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
**Process:**
1. Submit market buy orders for all selected positions
2. Poll for fills (30-second timeout)
3. Create Position objects with fill price as entry_price
4. Save positions to state for overnight hold

**Partial Fill Handling:**
- 3-second grace period before canceling partials
- Re-checks order status during grace period
- Immediate resubmit of residual qty if still partial

### Exit Orders (Morning - Market Sell)
```python
order_type = "market"
time_in_force = "day"
```
**Process:**
1. Submit market sell order
2. Poll for fills (30-second timeout)
3. If market sell fails → limit sell at -3% of last price

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
- Morning exit flags (hard_stops_checked, drop_stops_checked, final_exit_done)
- Afternoon entry flags (data_collected, scoring_done, entries_done)
- Open prices captured at 9:30 AM

### Recovery Logic

**Same-day restart (bot crashes and restarts):**
- Restores bot_state.json if date matches today
- Resumes from last completed stage
- Prevents duplicate work

**New day startup:**
- Ignores stale bot_state.json from previous day
- Loads positions.json (overnight holds from yesterday's 3:50 PM entries)
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
  momentum_scorer.py                # 350-model scoring, bucket assignment, selection
  universe_builder.py               # 4-stage pipeline, diagnostics, audit trails
  massive_client.py                 # Massive API client
  market_data.py                    # Alpaca data client
  state_manager.py                  # State persistence
  config.py                         # Unified config re-export
  config_broker.py                  # API credentials
  config_runtime.py                 # Paths, logging
  config_universe.py                # Price/ADV filters
  config_strategy.py                # Scoring weights, tiers, exit rules
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
- Massive API key (Polygon) for universe building

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
1. Manage morning exits (9:30-11:05 AM) if overnight positions exist
2. Wait for afternoon entry window (3:30-4:00 PM)
3. Score universe and enter new positions at 3:50 PM
4. Shut down at 4:00 PM, holding positions overnight

### Late Start Behavior

**Started after 11:05 AM with positions:**
- Immediately runs failsafe flatten
- Exits all positions at market

**Started after 4:00 PM:**
- Logs error and exits (nothing to do)

**Started between 11:05 AM - 3:30 PM:**
- Waits for 3:30 PM data collection
- Continues with afternoon entry flow

### Monitoring

**Log files:**
- `state/logs/bot_YYYY-MM-DD.log` - Daily execution log

**State files:**
- `state/positions.json` - Overnight positions
- `state/bot_state.json` - Same-day recovery state

**Audit reports:**
- `state/audit/universe_YYYY-MM-DD.json` - Universe pipeline diagnostics
- `state/audit/candidates_YYYY-MM-DD.json` - Top 20 scored candidates
- `state/audit/execution_YYYY-MM-DD.json` - Entry execution details
- `state/audit/health_YYYY-MM-DD.json` - Daily run health metrics

---

## Configuration

All parameters split across config files:

### Universe Filters (`config_universe.py`)
```python
MIN_PRICE = 0.50              # Minimum stock price
MAX_PRICE = 50.00             # Maximum stock price
MIN_ADV_DOLLARS = 500_000     # $500K minimum ADV
ADV_LOOKBACK_DAYS = 20        # Days for ADV calculation
ATR_LOOKBACK_DAYS = 20        # Days for ATR calculation
```

### Scoring Weights (`config_strategy.py`)
```python
SCORE_WEIGHT_INTRADAY_RETURN = 0.20
SCORE_WEIGHT_PROXIMITY_HIGH = 0.15
SCORE_WEIGHT_VOLUME_VS_AVG = 0.20
SCORE_WEIGHT_VOLUME_TREND = 0.10
SCORE_WEIGHT_VS_MARKET = 0.25
SCORE_WEIGHT_ATR_PCT = -0.10  # Negative = volatility penalty
```

### Position Sizing (`config_strategy.py`)
```python
MAX_LEVERAGE = 1.0            # No margin (cash account)
ADV_CAP_PCT = 0.003           # 0.3% of ADV max position
MAX_POSITION_DOLLARS = 50_000 # Absolute dollar cap
```

### Exit Rules (`config_strategy.py`)
```python
HARD_STOP_PCT = -0.05         # -5% from entry
DROP_STOP_PCT = 0.06          # 6% drop from open high
EXIT_TIME = "11:00"           # Final exit time
```

### Timing (`config_strategy.py`)
```python
DATA_COLLECTION_TIME = "15:30"
SCORING_TIME = "15:48"
ENTRY_TIME = "15:50"
MARKET_OPEN_TIME = "09:30"
FIRST_CHECKPOINT_TIME = "09:35"
```

---

## Known Limitations

1. **Massive API dependency** - Universe building requires Massive API. Alpaca fallback is slower and may miss symbols.

2. **No backtesting framework** - Strategy parameters tuned manually without systematic backtesting.

3. **Polling-based execution** - 1-second event loop. Very fast market moves between polls could be missed.

4. **Single-day holding period** - Strategy assumes overnight hold with next-morning exit. Extended holds not supported.

5. **No sector diversification** - Selection is purely momentum-based. Could concentrate in single sector during sector rotations.

6. **Broker API dependency** - Failsafe system relies on broker API. If broker API fails, position state may be incorrect.

7. **No real-time monitoring alerts** - Bot logs to file but doesn't send alerts for critical events.

8. **Fixed stop levels** - Hard stop (-5%) and drop stop (6%) are not adaptive to volatility.

9. **Minute bar data quality** - Requires ≥30 minute bars for scoring. Symbols with poor data quality are excluded.

10. **No maximum drawdown protection** - Bot will continue entering positions even after consecutive losing days.

---

## Risk Warnings

⚠️ **This is an overnight momentum strategy with significant risks:**

- **Overnight gap risk:** Positions held overnight are exposed to gap risk from after-hours news/events
- **Stop-loss execution risk:** Market opens can gap through stop levels, resulting in larger losses
- **Momentum reversal risk:** Strong afternoon momentum can reverse overnight or at the open
- **Liquidity risk:** 0.3% ADV cap may not prevent slippage in fast-moving markets
- **Technology risk:** API failures, network issues, or bugs could cause losses or missed exits
- **Broker API risk:** Failsafe system depends on broker API availability
- **Data quality risk:** Poor minute bar data can result in incorrect scoring and bad position selection

**Pattern Day Trader Rules:** This strategy does NOT trigger PDT rules (positions held overnight, not day-traded).

**Always test in paper trading first. Never risk capital you cannot afford to lose.**
