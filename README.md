# Overnight Momentum Trading Bot

Automated equity trading strategy on Alpaca that captures overnight momentum by entering positions at 3:50 PM and exiting the next morning. Uses a dual-scoring model — a **HEAD score** for late-day continuation candidates and a **TAIL score** for broader momentum — with a waterfall capital allocator. All positions are exited unconditionally via market sell at 9:40 AM.

---

## Strategy Overview

### Core Concept

The bot runs a strict two-day cycle:

- **T-1 afternoon (3:30–4:00 PM):** Build a tradeable universe, compute both HEAD and TAIL scores, allocate capital using a waterfall system (HEAD then TAIL), and submit market buy orders at 3:50 PM. Positions held overnight.
- **T+1 morning (9:00–9:45 AM):** At 9:30 positions fill at the open. At 9:40 AM **all positions are market-sold unconditionally** — no classification, no conditions. A failsafe sweep runs at 9:45 AM.

The bot never holds past 9:45 AM intentionally.

---

## Execution Trace — What Happens Line by Line

This traces `OvernightMomentumBot.run()` from startup through end of day.

### 9:00 AM — Bot Startup (`run()`)

1. Logging initialised (file + stdout).
2. `_load_state()` called:
   - If `bot_state.json` is dated **today**: all stage flags restored (`v2_classified`, `post_exit_failsafe_done`, `data_collected`, `scoring_done`, `entries_done`, `sold_today`).
   - If from a **prior day**: ignored. `positions.json` loaded for overnight holds.
3. `get_broker_positions()` fetches live Alpaca positions:
   - Positions exist → `reconcile_local_positions_from_broker()` syncs local to broker truth. `morning_exits_done = False`.
   - No positions → `morning_exits_done = True` (skip all morning exit logic).
4. Schedule times pre-computed: `09:30`, `09:40`, `09:45`, `15:30`, `15:48`, `15:50`, `16:00`.
5. **Late-start guard:** if time ≥ 09:45 and positions exist → `_run_failsafe_flatten()` immediately. `morning_exits_done = True`.
6. If time ≥ 16:00 → log error and return.
7. **Main `while True` loop begins** — ticks every 1 second.

---

### 9:00–9:40 AM — Waiting Loop

Morning logic runs each tick. All time guards still `False`. If local positions empty but broker holds positions (crash recovery), `reconcile_local_positions_from_broker()` fires to rebuild local state.

---

### 9:40 AM — Market Sell All (`_exit_all_940()`)

**Guard:** `not self.v2_classified and current_time >= 09:40`

**No classification. No conditions.** Every held position is market-sold unconditionally.

1. Iterates `self.position_mgr.positions` — submits `_exit_single_position(symbol, "9:40 AM market sell")` for each.
2. `_exit_position()` handles: broker check → market sell → 30s poll → limit fallback → partial fill retry.
3. After all exits: `reconcile_local_positions_from_broker()` syncs local state.
4. Any partial fills not resolved are caught by the 9:45 failsafe.

`v2_classified = True`. State saved.

---

### 9:45 AM — Post-Exit Failsafe

**Guard:** `self.v2_classified and not self.post_exit_failsafe_done and current_time >= 09:45`

1. `broker_position_count()`:
   - **0 positions:** logs confirmed flat. Done.
   - **>0 positions:** `_run_failsafe_flatten()`:
     - Market sell → 30s poll → limit at -3% → limit at -5% → `CRITICAL` if still open.
2. `post_exit_failsafe_done = True`, `morning_exits_done = True`. State saved.

---

### 9:45 AM–3:30 PM — Idle

`morning_exits_done = True`. Morning block no longer runs. Loop sleeps 1 second per tick.

---

### 3:30 PM — Data Collection (`_step_collect_data()`)

**Guard:** `not self.data_collected and current_time >= 15:30`

Calls `build_universe(massive, alpaca)` — 4-stage pipeline:

**Stage A — Eligibility (Alpaca `GET /v2/assets`):**
US equity, active, tradable, non-OTC, non-ETF/fund, no warrants/rights/units/preferred.

**Stage B — Price + Liquidity:**
Massive API prices → $1–$10 filter. Alpaca 20-day OHLCV → ADV (≥$2M) and ATR. Falls back to Alpaca snapshots if Massive unavailable.

**Stage D — Broker executability re-check:**
Second `GET /v2/assets` for any symbols that became restricted since Stage A.

`universe_audit` written to `state/logs/`. `data_collected = True`. State saved.

---

### 3:48 PM — Scoring (`_step_score_and_rank()`)

**Guard:** `self.data_collected and not self.scoring_done and current_time >= 15:48`

#### 1. Fetch signal bars
`get_intraday_bars_for_signal(universe, today, "09:30", "15:50")` — full-day 1-minute bars.

#### 2. Stage C — Data quality filter
Remove symbols with < 30 minute bars.

#### 3. SPY return
`(SPY_last - SPY_open) / SPY_open` — market benchmark for `vs_market` metric.

#### 4. 60-min volume profiles
Last-60-minute volume and average-60-minute volume per symbol.

#### 5. Build candidates (`build_signal_candidates_350()`)
Creates `MomentumCandidate` objects: `signal_price`, `adv_dollars`, `atr_14d`, bar highs/lows/volumes.

#### 6. Compute TAIL score metrics (`compute_raw_metrics_350()`)

| Metric | Weight | Formula |
|--------|--------|---------|
| `intraday_return` | +20% | `(signal_price − open) / open` |
| `proximity_to_high` | +15% | `signal_price / day_high` (1.0 = at the high) |
| `volume_vs_avg` | +20% | `last_60min_vol / avg_60min_vol` |
| `volume_trend` | +10% | Volume acceleration in last 60 min |
| `vs_market` | +25% | `intraday_return − spy_return` |
| `atr_percent` | −10% | `ATR_14d / signal_price` (volatility penalty) |

Z-scored across the candidate pool → `composite_score = Σ(z × weight)`.

#### 7. Compute HEAD score (`compute_head_score()`)

```
late_day_share = dollar_volume_last_60min / dollar_volume_930_to_350

head_score = 0.40 × late_day_share
           − 0.30 × proximity_to_high
           − 0.10 × atr_percent
```

Higher `head_score` = volume concentrating late in the day, not extended at the high, low volatility.

#### 8. Bucket assignment (`assign_buckets()`)
Decile buckets 1–10 by `composite_score` (10 = top 10% of today's pool). Only bucket ≥ 4 considered for allocation.

Candidates saved to `state/logs/candidates_YYYY-MM-DD.json` with **two ranked views**:
- `top_20_by_head_score` — actual HEAD selection pool
- `top_20_by_composite_score` — TAIL ranking pool

`scoring_done = True`. State saved.

---

### 3:50 PM — Entry Execution (`_step_execute_entries()`)

**Guard:** `self.scoring_done and not self.entries_done and current_time >= 15:50`

#### 1. Capital calculation
```python
equity      = get_account_equity()
buying_power = get_total_capital()          # Alpaca buying_power field
deployable  = min(buying_power, equity × MAX_LEVERAGE)
```

#### 2. PDT guard
If `equity < $50,000` and `sold_today` non-empty: same-day sold symbols removed from candidates before allocation.

#### 3. Waterfall allocation (`allocate_head_tail()`)

**HEAD — top 10 by `head_score`, equal-weight:**
```
slot_size   = deployable / HEAD_COUNT             # deployable / 10
adv_cap     = adv_dollars × ADV_CAP_PCT           # adv × 0.003
max_dollars = min(slot_size, adv_cap)
shares      = floor(max_dollars / signal_price)
if shares < MIN_SHARES (25): skip, full slot → leftover
leftover capital cascades into TAIL
```

**TAIL — remaining candidates by `composite_score`, waterfall:**
```
pool: all candidates NOT in HEAD, sorted by composite_score desc, up to TAIL_MAX_POSITIONS (30)
for each candidate:
    dynamic_slice = remaining_cash / candidates_remaining
    adv_cap       = adv_dollars × ADV_CAP_PCT
    max_dollars   = min(dynamic_slice, adv_cap, remaining_cash)
    shares        = floor(max_dollars / signal_price)
    if shares < 25: skip (cash not consumed)
    deploy, subtract cost, stop at MAX_TOTAL_POSITIONS (40)
```

#### 4. Execution gate (`filter_execution_ready()`)
Fresh snapshots for all allocated symbols. Rejects: spread > 5%, missing/stale quote.

#### 5. Submit market buy orders
- `submit_buy_order(symbol, qty)` → market buy `time_in_force="day"`.
- `get_order_fill(order_id, max_wait=30)` polls up to 30s.
- On fill: `Position` stored in `position_mgr.positions` with `entry_price = fill_price`.
- On Alpaca buying-power rejection: retry once with `qty × 0.98` (2% haircut).
- On any other failure: logged, continue to next candidate.

`entries_done = True`. State saved.

---

### 4:00 PM — Market Close

**Guard:** `current_time >= 16:00`

End-of-day audit reports saved (`execution_YYYY-MM-DD.json`, `run_health_YYYY-MM-DD.json`). Loop exits. Positions held overnight.

---

## State Persistence & Recovery

### State Files

**`state/positions.json`**
- Symbol, entry_price, quantity, entry_time, adv_estimate, peak_price, current_price
- Persists across days — loaded at next-day 9:00 AM startup

**`state/bot_state.json`** — date-stamped; ignored if from a prior day
- Stage flags: `v2_classified` (9:40 sell fired), `post_exit_failsafe_done`, `data_collected`, `scoring_done`, `entries_done`
- `sold_today` — symbols sold this session (PDT guard)

### Same-Day Crash Recovery

`_load_state()` restores all stage flags. The event loop resumes from the last completed stage — no duplicate exits or entries. If the bot crashes after 9:40 but before 9:45, the failsafe will catch any remaining positions on restart.

### Broker Reconciliation

On startup and after exit waves:
- Positions in broker but not local → rebuilt from broker data
- Positions local but not in broker → removed (already filled or closed)
- Quantities mismatched → synced to broker value

---

## Architecture

```
run.py                              # Entry point
bot/
  integrated_main.py                # OvernightMomentumBot — main orchestrator
  position_manager_overnight.py     # Position tracking, order submission, failsafe flatten
  exit_classifier.py                # Unused — kept for reference only
  momentum_scorer.py                # HEAD score, TAIL score, waterfall allocator
  universe_builder.py               # 4-stage universe pipeline, audit writers
  massive_client.py                 # Massive API (universe snapshot)
  market_data.py                    # Alpaca data client (bars, snapshots)
  state_manager.py                  # JSON state persistence
  rate_limiter.py                   # API call tracking
  config.py                         # Unified config re-export
  config_broker.py                  # API credentials, endpoints
  config_runtime.py                 # Paths, logging format
  config_universe.py                # Price/ADV filters, preset
  config_strategy.py                # Scoring weights, tiers, exit rules, timing
state/
  positions.json                    # Overnight positions
  bot_state.json                    # Same-day recovery state
  logs/
    universe_YYYY-MM-DD.json        # Universe pipeline diagnostics
    candidates_YYYY-MM-DD.json      # Top 20 HEAD + top 20 TAIL scored candidates
    execution_YYYY-MM-DD.json       # Entry fill funnel details
    run_health_YYYY-MM-DD.json      # Daily run health summary
    bot_YYYY-MM-DD.log              # Full daily log
```

---

## Configuration Reference

### Universe (`config_universe.py`)
```python
UNIVERSE_PRESET   = "expanded_smallcap"
MIN_PRICE         = 1.00
MAX_PRICE         = 10.00
MIN_ADV_DOLLARS   = 2_000_000    # $2M minimum 20-day ADV
ADV_LOOKBACK_DAYS = 20
ATR_LOOKBACK_DAYS = 14
```

### TAIL composite score weights (`config_strategy.py`)
```python
SCORE_WEIGHT_INTRADAY_RETURN = 0.20
SCORE_WEIGHT_PROXIMITY_HIGH  = 0.15
SCORE_WEIGHT_VOLUME_VS_AVG   = 0.20
SCORE_WEIGHT_VOLUME_TREND    = 0.10
SCORE_WEIGHT_VS_MARKET       = 0.25
SCORE_WEIGHT_ATR_PCT         = -0.10
```

### HEAD score formula
```python
# late_day_share = dollar_volume_last_60min / dollar_volume_930_to_350
head_score = 0.40 * late_day_share - 0.30 * proximity_to_high - 0.10 * atr_percent
```

### Position sizing (`config_strategy.py`)
```python
MAX_LEVERAGE         = 1.0     # Cash account, no margin
ADV_CAP_PCT          = 0.003   # Max 0.3% of 20-day ADV per position
MIN_SHARES           = 25      # Skip if rounding yields fewer shares
HEAD_COUNT           = 10      # Fixed equal-weight HEAD slots
TAIL_MAX_POSITIONS   = 15      # Max TAIL candidates (non-HEAD pool)
MAX_TOTAL_POSITIONS  = 25      # Hard cap: HEAD + TAIL combined
```

### Exit rules (`config_strategy.py`)
```python
EXIT_940_TIME    = "09:40"    # Market sell ALL positions — no conditions
V2_FAILSAFE_TIME = "09:45"    # Post-exit broker verification
```

### Timing (`config_strategy.py`)
```python
MARKET_OPEN_TIME     = "09:30"
DATA_COLLECTION_TIME = "15:30"
SCORING_TIME         = "15:48"
ENTRY_TIME           = "15:50"
```

---

## Setup

### Requirements
- Python 3.11+
- Alpaca trading account (paper or live)
- Massive API key for universe building

### Installation

```bash
git clone https://github.com/kellyp2574-tech/Alpaca_bot.git
cd Alpaca_bot
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
cp .env.example .env
# Fill in: ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER, MASSIVE_API_KEY
```

### Dependencies
```
requests>=2.31.0
python-dotenv>=1.0.0
```

---

## Usage

```bash
python run.py
```

Start at **9:00 AM ET**. The bot manages its own schedule. Runs until 4:00 PM then exits, holding positions overnight.

### Late Start Behavior

| Start time | Has positions | Action |
|------------|--------------|--------|
| Before 9:40 AM | Any | Normal startup |
| 9:40–9:45 AM | Yes | Resumes from last saved state; re-runs any incomplete exits |
| After 9:45 AM | Yes | Immediate failsafe flatten, then continues to afternoon |
| After 4:00 PM | Any | Logs error, exits |
| 9:45 AM–3:30 PM | No | Waits for 3:30 PM data collection |

---

## Known Limitations

1. **Massive API dependency** — Universe building requires Massive API; Alpaca fallback is slower and may miss symbols.
2. **Polling-based execution** — 1-second event loop; very fast intraday moves between polls could be missed.
3. **Single-day holding period** — Strategy assumes next-morning exit; extended holds not supported.
4. **No sector diversification** — Selection is purely momentum-based; can concentrate in one sector.
5. **No real-time alerts** — Bot logs to file only; no push notifications for critical events.
6. **No drawdown protection** — Bot enters positions regardless of recent P&L history.

---

## Risk Warnings

⚠️ **This is an overnight momentum strategy with significant risks:**

- **Overnight gap risk** — No stop-loss. A gap-down at the open is held until 9:40 AM and sold unconditionally at market. The loss is fully realised at the 9:40 market fill price.
- **Momentum reversal risk** — Late-day strength used as a selection signal can fully reverse overnight.
- **Liquidity risk** — 0.3% ADV cap limits position size but does not eliminate slippage in thin small-cap names.
- **Technology risk** — API failures or bugs could cause missed exits. Failsafe layer mitigates but does not eliminate this.
- **PDT risk** — Same-day re-entries blocked when equity < $50,000; accounts < $25,000 subject to PDT rules.

**Always test in paper trading first. Never risk capital you cannot afford to lose.**
