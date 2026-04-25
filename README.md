# Overnight Momentum Trading Bot

Automated equity trading strategy on Alpaca that captures overnight momentum by entering positions at 3:50 PM and exiting the next morning. Uses a dual-scoring model — a new **HEAD score** for late-day continuation candidates and an original **TAIL score** for broader momentum — with a waterfall capital allocator. Exits are driven by an entry-price comparison at 9:35 AM, placing trailing stops on continuation names and immediately selling gap-faded positions.

---

## Strategy Overview

### Core Concept

The bot runs a strict two-day cycle:

- **T-1 afternoon (3:30–4:00 PM):** Build a tradeable universe, compute both HEAD and TAIL scores, allocate capital using a waterfall system (HEAD then TAIL), and submit market buy orders at 3:50 PM. Positions held overnight.
- **T+1 morning (9:00–11:35 AM):** At 9:30 open prices are captured. At 9:35 each position is classified vs. its entry price — continuation names get a **1.25% trailing stop**, faded names are **sold immediately**. At 11:30 any live trailing stops are cancelled and remaining positions market-sold. Failsafe sweep at 11:35 AM.

The bot never holds past 11:35 AM intentionally.

---

## Execution Trace — What Happens Line by Line

This traces `OvernightMomentumBot.run()` from startup through end of day.

### 9:00 AM — Bot Startup (`run()`)

1. Logging initialised (file + stdout).
2. `_load_state()` called:
   - If `bot_state.json` is dated **today**: all stage flags restored (`v2_classified`, `exits_1130_done`, `post_exit_failsafe_done`, `data_collected`, `scoring_done`, `entries_done`, `open_prices`, `exit_schedule`, `classification_audit`, `trailing_order_ids`).
   - If from a **prior day**: ignored. `positions.json` loaded for overnight holds.
3. `get_broker_positions()` fetches live Alpaca positions:
   - Positions exist → `reconcile_local_positions_from_broker()` syncs local to broker truth. `morning_exits_done = False`.
   - No positions → `morning_exits_done = True` (skip all morning exit logic).
4. Schedule times pre-computed: `09:30`, `09:35`, `11:30`, `11:35`, `15:30`, `15:48`, `15:50`, `16:00`.
5. **Late-start guard:** if time ≥ 11:35 and positions exist → `_run_failsafe_flatten()` immediately. `morning_exits_done = True`.
6. If time ≥ 16:00 → log error and return.
7. **Main `while True` loop begins** — ticks every 1 second.

---

### 9:00–9:30 AM — Waiting Loop

Morning logic runs each tick. All time guards still `False`. If local positions empty but broker holds positions (crash recovery), `reconcile_local_positions_from_broker()` fires to rebuild local state before 9:30.

---

### 9:30 AM — Open Price Capture (`_capture_open_prices()`)

**Guard:** `not self.hard_stops_checked and current_time >= 09:30`

**No positions exited here.** All overnight holds kept.

1. Fetches Alpaca **snapshots** for all held symbols.
2. Reads `snapshot["open"]` — the RTH opening print.
   - Valid → stored in `self.open_prices[symbol]`, `position.current_price` updated.
   - Missing/zero → deferred to 9:35 minute-bar fallback.
3. State saved.

> Open prices are used **only for gap logging** at 9:35 — they do not drive the exit decision.

---

### 9:35 AM — Classification + Exits (`_classify_and_exit_v2()`)

**Guard:** `not self.v2_classified and current_time >= 09:35`

#### Step 1: Fetch data
- `get_snapshots(symbols)` — fallback source for `price_935`.
- `get_minute_bars(symbols, "09:30", "09:35")` — primary source for `price_935`.

#### Step 2: Resolve open prices (logging only)
For each symbol, if no snapshot open from 9:30: tries `minute_bars[symbol][0]["o"]` as fallback.

#### Step 3: Classify (`classify_positions()` in `exit_classifier.py`)

```
price_935:
  1. minute_bars[-1]["c"]          -> source = "minute_bar"   (preferred)
  2. snapshot["last_price"]        -> source = "snapshot"     (fallback)
  3. neither available             -> source = "missing"      -> EXIT immediately (conservative)

Classification rule (entry-price based):
  price_935 > entry_price   ->  EXIT_BUCKET_TRAIL  ("trail")
  price_935 <= entry_price  ->  EXIT_BUCKET_935    ("09:35")
  missing entry_price       ->  EXIT_BUCKET_935    (conservative default)
```

#### Step 4: Store schedule and audit
- `self.exit_schedule = {symbol: exit_bucket}` for every symbol.
- `self.classification_audit` includes: `entry_price`, `open_price`, `price_935`, `ret_vs_entry_pct`, `exit_bucket`, source fields, `gap_pct`. Persisted in `bot_state.json`.

#### Step 5: Execute immediate exits (faded positions)
Every symbol in `EXIT_BUCKET_935` → `_exit_single_position(symbol, "price_935 <= entry_price -> 9:35 exit")`.

**Partial fill re-routing:** incomplete 9:35 exits are re-routed to `EXIT_BUCKET_1130`.

#### Step 6: Place trailing stops (continuation positions)
Every symbol in `EXIT_BUCKET_TRAIL`:
- `submit_trailing_stop_sell(symbol, qty, trail_percent=1.25)` → Alpaca `type=trailing_stop`.
- Order ID stored in `self.trailing_order_ids[symbol]` (persisted to state).
- If trailing stop placement fails: logged as warning; position falls back to 11:30 market sell.

`v2_classified = True`. State saved.

---

### 9:35 AM–11:30 AM — Waiting

Trailing stops are live on Alpaca's servers. If a trailing stop triggers and fills, Alpaca removes the position automatically — local state is cleaned up at 11:30 reconciliation. No polling required.

If all 9:35 exits completed and no positions remain, `morning_exits_done = True` immediately (early completion).

---

### 11:30 AM — Hard Exit (`_exit_bucket(EXIT_BUCKET_1130, ...)`)

**Guard:** `self.v2_classified and not self.exits_1130_done and current_time >= 11:30`

1. **Cancel all live trailing stop orders** — iterates `self.trailing_order_ids`, calls `_cancel_order(order_id)` for each symbol still held locally. Clears the dict.
2. **Reconcile with broker** — `reconcile_local_positions_from_broker()` removes any symbols whose trailing stop already filled (broker no longer holds them).
3. **Market-sell all remaining positions** — every symbol left in `self.position_mgr.positions` gets `_exit_single_position(symbol, "11:30 forced exit")`.
4. Post-exit reconciliation. `exits_1130_done = True`. State saved.

> The 11:30 exit is a **catch-all** — it handles trailing-stop survivors, faded-position partial fills, and any unscheduled leftovers.

---

### 11:35 AM — Post-Exit Failsafe

**Guard:** `not self.post_exit_failsafe_done and current_time >= 11:35`

1. `broker_position_count()`:
   - **0 positions:** logs confirmed flat. Done.
   - **>0 positions:** `_run_failsafe_flatten()`:
     - Market sell → 30s poll → limit at -3% → limit at -5% → `CRITICAL` if still open.
2. `post_exit_failsafe_done = True`, `morning_exits_done = True`. State saved.

---

### 11:35 AM–3:30 PM — Idle

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
- Stage flags: `v2_classified`, `exits_1130_done`, `post_exit_failsafe_done`, `data_collected`, `scoring_done`, `entries_done`
- `open_prices` — `{symbol: open_price}` captured at 9:30 (for gap logging)
- `exit_schedule` — `{symbol: "09:35" | "trail" | "11:30"}`
- `classification_audit` — `{symbol: {entry_price, open_price, price_935, ret_vs_entry_pct, exit_bucket, ...}}`
- `trailing_order_ids` — `{symbol: alpaca_order_id}` for live trailing stops
- `sold_today` — symbols sold this session (PDT guard)

### Same-Day Crash Recovery

`_load_state()` restores all flags including `trailing_order_ids`. The event loop resumes from the last completed stage — no duplicate exits or entries. Trailing stops already placed on Alpaca remain active through a crash/restart.

### Broker Reconciliation

On startup and after exit waves:
- Positions in broker but not local → rebuilt from broker data
- Positions local but not in broker → removed (trailing stop may have filled)
- Quantities mismatched → synced to broker value

---

## Architecture

```
run.py                              # Entry point
bot/
  integrated_main.py                # OvernightMomentumBot — main orchestrator
  position_manager_overnight.py     # Position tracking, order submission, trailing stops, failsafe
  exit_classifier.py                # 9:35 AM entry-price classification (trail vs immediate exit)
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

### TAIL score weights (`config_strategy.py`)
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
TAIL_MAX_POSITIONS   = 30      # Max TAIL candidates (non-HEAD pool)
MAX_TOTAL_POSITIONS  = 40      # Hard cap: HEAD + TAIL combined
```

### Exit rules (`config_strategy.py`)
```python
TRAILING_STOP_PCT     = 1.25       # % trail from high-water mark for continuation names
V2_CLASSIFY_TIME      = "09:35"    # Entry-price classification + trailing stop placement
EXIT_BUCKET_1130_TIME = "11:30"    # Hard fallback: cancel trailing stops + market sell all
V2_FAILSAFE_TIME      = "11:35"    # Post-exit broker verification
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
| Before 9:30 AM | Any | Normal startup |
| 9:30–11:35 AM | Yes | Resumes from last saved state (crash recovery) |
| After 11:35 AM | Yes | Immediate failsafe flatten, then continues to afternoon |
| After 4:00 PM | Any | Logs error, exits |
| 11:35 AM–3:30 PM | No | Waits for 3:30 PM data collection |

---

## Known Limitations

1. **Massive API dependency** — Universe building requires Massive API; Alpaca fallback is slower and may miss symbols.
2. **Polling-based execution** — 1-second event loop; very fast intraday moves between polls could be missed.
3. **Single-day holding period** — Strategy assumes next-morning exit; extended holds not supported.
4. **No sector diversification** — Selection is purely momentum-based; can concentrate in one sector.
5. **Trailing stop market conversion** — Alpaca trailing stops convert to market orders on trigger; fill price may differ from stop price in fast markets.
6. **Trailing stops inactive outside RTH** — Alpaca trailing stops do not trigger in extended hours; they activate at 9:30 AM the following day if still open (not expected in normal operation).
7. **No real-time alerts** — Bot logs to file only; no push notifications for critical events.
8. **No drawdown protection** — Bot enters positions regardless of recent P&L history.

---

## Risk Warnings

⚠️ **This is an overnight momentum strategy with significant risks:**

- **Overnight gap risk** — No hard stop-loss. A large gap-down at open is held through 9:35 classification; if `price_935 ≤ entry_price` the position is sold immediately at 9:35, but the loss is already realised at the open price.
- **Trailing stop slippage** — In fast or gapping markets the execution price of a triggered trailing stop can be significantly below the stop trigger price.
- **Momentum reversal risk** — Late-day strength used as a selection signal can fully reverse overnight.
- **Liquidity risk** — 0.3% ADV cap limits position size but does not eliminate slippage in thin small-cap names.
- **Technology risk** — API failures or bugs could cause missed exits. Failsafe layer mitigates but does not eliminate this.
- **PDT risk** — Same-day re-entries blocked when equity < $50,000; accounts < $25,000 subject to PDT rules.

**Always test in paper trading first. Never risk capital you cannot afford to lose.**
