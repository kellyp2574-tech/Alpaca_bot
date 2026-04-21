# Overnight Momentum Trading Bot

Automated equity trading strategy on Alpaca that captures overnight momentum by entering positions at 3:50 PM and exiting the next morning. Uses a composite scoring model to identify high-momentum small-cap stocks ($1–$10, $2M+ ADV) with strong late-day volume and relative strength.

---

## Strategy Overview

### Core Concept

The bot runs a strict two-day cycle:

- **T-1 afternoon (3:30–4:00 PM):** Build a tradeable universe, score candidates on 6 intraday metrics, allocate capital, and submit market buy orders at 3:50 PM. Positions are held overnight.
- **T+1 morning (9:00–11:35 AM):** Exit every position. At 9:30 the opening price is captured. At 9:35 each position is classified by its open-to-9:35 return — winners (>+0.5%) exit immediately, all others hold to 11:30 AM. A failsafe sweep runs at 11:35 AM.

The bot never holds past 11:35 AM intentionally.

---

## Execution Trace — What Happens Line by Line

This traces `OvernightMomentumBot.run()` from startup through end of day.

### 9:00 AM — Bot Startup (`run()`)

1. Logging is initialised (file + stdout).
2. `_load_state()` is called:
   - If `bot_state.json` exists and is dated **today**, all stage flags are restored (`v2_classified`, `exits_1130_done`, `post_exit_failsafe_done`, `data_collected`, `scoring_done`, `entries_done`, `open_prices`, `exit_schedule`, `classification_audit`).
   - If it is from a **prior day**, it is ignored. `positions.json` is loaded for overnight holds.
3. `get_broker_positions()` fetches live Alpaca positions:
   - If positions exist → `reconcile_local_positions_from_broker()` syncs local state to broker truth. `morning_exits_done = False`.
   - If no positions → `morning_exits_done = True` (skip all morning exit logic).
4. All schedule times are pre-computed from config strings into `datetime.time` objects:
   - `t_market_open = 09:30`, `t_v2_classify = 09:35`, `t_bucket_1130 = 11:30`, `t_failsafe = 11:35`, `t_data_collect = 15:30`, `t_scoring = 15:48`, `t_entry = 15:50`, `t_market_close = 16:00`.
5. **Late-start guard:** if current time ≥ 11:35 and positions exist → `_run_failsafe_flatten()` is called immediately. Positions are cleared. `morning_exits_done = True`.
6. If current time ≥ 16:00 → log error and return (nothing to do).
7. **Main `while True` loop begins** — ticks every 1 second.

---

### 9:00–9:30 AM — Waiting Loop

Each second the loop checks `morning_exits_done`. It is `False` so morning logic runs. All time guards are still `False` (before 9:30) so nothing executes yet. The loop sleeps 1 second and repeats.

If local positions are empty but broker still shows positions (e.g. crash recovery), `reconcile_local_positions_from_broker()` fires to rebuild local state before the 9:30 open price capture.

---

### 9:30 AM — Open Price Capture (`_capture_open_prices()`)

**Guard:** `not self.hard_stops_checked and current_time >= 09:30`

**No positions are exited here.** All overnight holds are kept regardless of the opening print.

1. Fetches live Alpaca **snapshots** for all held symbols.
2. For each position, reads `snapshot["open"]` — the RTH opening print.
   - If `open` is missing or zero: logs a warning, defers open price to the 9:35 minute-bar fallback.
   - If `open` is valid: stores it in `self.open_prices[symbol]`. Updates `position.current_price`.
3. State saved.

**After this step:** `self.open_prices` contains the snapshot open for every symbol that had one. No positions have been touched.

---

### 9:35 AM — Classification + Immediate Exits (`_classify_and_exit_v2()`)

**Guard:** `not self.v2_classified and current_time >= 09:35`

#### Step 1: Fetch 9:35 snapshots
`get_snapshots(symbols)` — used as `price_935` fallback only.

#### Step 2: Fetch 9:30–9:35 minute bars
`get_minute_bars(symbols, "09:30", "09:35")` — primary source for both open fallback and `price_935`.

#### Step 3: Resolve open prices
For each symbol:
- If already in `self.open_prices` (from 9:30 snapshot) → source tagged `"snapshot"`.
- Else → tries `minute_bars[symbol][0]["o"]` (first bar open):
  - Valid → stored in `self.open_prices`, source tagged `"minute_bar"`.
  - Missing or zero → source tagged `"missing"`, symbol will default to 11:30.

#### Step 4: Classify (`classify_positions()` in `exit_classifier.py`)
For each symbol:

```
price_935:
  1. last minute-bar close:  bars[-1]["c"]   -> source = "minute_bar"
  2. snapshot fallback:      snap["last_price"] or snap["close"]  -> source = "snapshot"
  3. neither available       -> source = "missing", default to 11:30

ret_open_to_935 = (price_935 - open_price) / open_price × 100

if ret_open_to_935 > EXIT_UP_MOVE_PCT (0.5%)  ->  EXIT_BUCKET_935  ("09:35")
else                                            ->  EXIT_BUCKET_1130 ("11:30")
```

Missing price data always defaults to `"11:30"`.

Source quality is summarised in logs: how many symbols used minute-bar opens, how many used snapshot fallback for `price_935`, how many were missing. A warning fires if >50% of symbols used minute-bar open fallbacks.

#### Step 5: Store schedule and audit
- `self.exit_schedule = {symbol: exit_time}` for every symbol.
- `self.classification_audit = {symbol: {open_price, price_935, move_5m_pct, exit_time, open_price_source, price_935_source, gap_pct}}` — persisted to `bot_state.json`.

#### Step 6: Execute immediate 9:35 exits
Every symbol assigned `"09:35"` is passed to `_exit_single_position(symbol, "move > 0.5% from open -> 9:35 exit")`.

**Partial fill re-routing:** if a 9:35 exit doesn't fully fill, `exit_schedule[symbol]` is updated to `"11:30"` so the 11:30 bucket catches the remainder automatically.

`v2_classified = True`. State saved.

---

### 9:35 AM–11:30 AM — Waiting Loop

The loop continues ticking every second. `morning_exits_done` is still `False`. No time guards fire yet. If all 9:35 exits completed and no positions remain, `morning_exits_done` is set `True` immediately (early completion path).

---

### 11:30 AM — Hold Bucket Exit (`_exit_bucket(EXIT_BUCKET_1130, ...)`)

**Guard:** `self.v2_classified and not self.exits_1130_done and current_time >= 11:30`

1. Collects all symbols in `exit_schedule` whose bucket is `"11:30"` AND that still exist in `self.position_mgr.positions`.
2. Calls `_exit_single_position(symbol, "scheduled 11:30 AM exit")` for each.
3. After all exits: `reconcile_local_positions_from_broker()` — syncs local state in case any fills were partial.
4. `exits_1130_done = True`. State saved.

---

### 11:35 AM — Post-Exit Failsafe

**Guard:** `not self.post_exit_failsafe_done and current_time >= 11:35`

1. Calls `broker_position_count()`.
   - **0 positions:** logs "broker confirmed flat". Done.
   - **>0 positions:** calls `_run_failsafe_flatten()`:
     - Fetches all broker positions directly.
     - For each: submits market sell → polls 30s for fill → if unfilled, limit sell at -3% → if still unfilled, limit sell at -5% → logs `CRITICAL` if still open after all layers.
2. `post_exit_failsafe_done = True`, `morning_exits_done = True`. State saved.

---

### 11:35 AM–3:30 PM — Idle

`morning_exits_done = True`. The morning block no longer runs. The afternoon block guards are all `False` until 3:30 PM. Loop sleeps 1 second each tick.

---

### 3:30 PM — Data Collection (`_step_collect_data()`)

**Guard:** `not self.data_collected and current_time >= 15:30`

Calls `build_universe(massive, alpaca)` — a 4-stage pipeline:

**Stage A — Alpaca asset list (eligibility):**
- Calls Alpaca `GET /v2/assets` — this is the authoritative eligibility source.
- Filters to: US equity, active, tradable, non-OTC, non-ETF/fund, no warrants/rights/units/preferred.

**Stage B — Price filter (Massive) + ADV/ATR (Alpaca):**
- Calls Massive API `get_full_market_snapshot()` for current prices. Filters: $1–$10.
- **If Massive fails**, falls back to Alpaca snapshots automatically.
- Fetches 20 days of daily OHLCV from Alpaca. Computes **ADV** (≥$2M gate) and **ATR**.

**Stage D — Fresh Alpaca tradability re-check:**
- Second `GET /v2/assets` call to catch any symbols that became restricted after Stage A.

Results saved to `self.universe` (list of symbols). `universe_audit` written to `state/audit/`. `data_collected = True`. State saved.

---

### 3:48 PM — Scoring (`_step_score_and_rank()`)

**Guard:** `self.data_collected and not self.scoring_done and current_time >= 15:48`

#### 1. Fetch signal bars
`get_intraday_bars_for_signal(universe, today, start="09:30", end="15:50")` — full-day minute bars for every universe symbol.

#### 2. Stage C — Minute bar data quality filter
Removes symbols with fewer than 30 minute bars. Logs the before/after count.

#### 3. Fetch SPY return
Snapshot of `SPY` → `(last - open) / open`. Used as market benchmark in scoring.

#### 4. Build 60-min volume profiles
For each symbol: last 60 minutes of volume vs. average 60-minute volume across the day.

#### 5. Build candidates (`build_signal_candidates_350()`)
Extracts per-symbol: `signal_price` (3:50 close), `adv_dollars`, `atr_percent`, first/last bar prices, volume data. Returns a list of `MomentumCandidate` objects.

#### 6. Compute raw metrics (`compute_raw_metrics_350()`)
Six metrics per candidate:

| Metric | Weight | Calculation |
|--------|--------|-------------|
| `intraday_return` | +20% | `(signal_price - open) / open` |
| `proximity_to_high` | +15% | `signal_price / day_high` |
| `volume_vs_avg` | +20% | `last_60min_volume / avg_60min_volume` |
| `volume_trend` | +10% | Recent volume acceleration |
| `vs_market` | +25% | `intraday_return − spy_return` (difference, stable when SPY near zero) |
| `atr_percent` | -10% | ATR as % of price (volatility penalty) |

#### 7. Normalize, score, bucket (`normalize_and_score_350()`, `assign_buckets()`)
- Each metric is **z-scored** across the daily candidate pool (mean 0, std 1).
- `composite_score = Σ(z_metric × weight)`.
- Candidates assigned to **decile buckets 1–10** (10 = highest score in today's pool, relative ranking). Only bucket ≥ 4 selected for allocation.
- Sorted descending by `composite_score`.

`scoring_done = True`. Top 20 saved to `state/audit/candidates_YYYY-MM-DD.json`. State saved.

---

### 3:50 PM — Entry Execution (`_step_execute_entries()`)

**Guard:** `self.scoring_done and not self.entries_done and current_time >= 15:50`

#### 1. Get equity and tier config
`get_account_equity()` → `get_selection_config(equity)` → returns `min_bucket` (≥4), `max_positions` (30), `max_leverage`.

Note: `selection_mode` has been removed — behavior is determined entirely by `min_bucket`, `max_head_positions`, and `max_positions`.

```
deployable = equity × max_leverage  (1.0 for cash accounts)
```

#### 2. PDT guard (pre-allocation)
If `equity < $50,000` and `sold_today` is non-empty: removes any same-day sold symbols from the candidate list **before** allocation. Logs how many were blocked. If nothing remains, skips entries entirely.

#### 3. HEAD/TAIL allocation (`allocate_head_tail()`)
Capital is split into two pools and allocated in ranked order:

**HEAD pool (70% of deployable, top 10 equal-weight):**
```
slot_size = deployable × 0.70 / 10
per_position_max = min(slot_size, ADV × 0.003)
shares = floor(per_position_max / signal_price)
if shares < 25: skip, unspent slot rolls to TAIL
```

**TAIL pool (30% of deployable + HEAD leftovers, waterfall):**
```
for each remaining candidate (ranked 11+):
    max_dollars = min(remaining_cash, ADV × 0.003)
    shares = floor(max_dollars / signal_price)
    if shares < 25: skip (cash not consumed)
    deploy, subtract cost
    stop when 80% deployment target reached OR no more candidates
```

The 30-position cap is a soft limit — the allocator will exceed it to reach the 80% deployment target if enough candidates exist.

#### 4. Execution gate (`filter_execution_ready()`)
Fresh snapshots fetched for all allocated symbols immediately before order submission. Rejects:
- Bid/ask spread > 5%
- Missing or invalid quote

Rejected symbols are logged with their reason.

#### 5. Submit market buy orders
For each orderable allocation:
- `submit_buy_order(symbol, qty)` → market buy, `time_in_force="day"`.
- `get_order_fill(order_id, max_wait=30)` polls up to 30 seconds.
- On fill: creates a `Position(symbol, entry_price=fill_price, quantity=filled_qty, entry_time=now, adv_estimate, ...)` and stores it in `position_mgr.positions`.
- Logs: `ENTRY AAPL: 150 @ 8.4200 [HEAD #1] (score=0.812, bucket=5)`

#### 6. Deployment shortfall diagnostics
If `total_deployed / equity < 80%`, a detailed breakdown is logged:
- PDT-blocked candidates count
- Execution gate rejection reasons
- Failed submissions / no-fill count
- Gap between target and actual deployed dollars

`entries_done = True`. State saved.

---

### 4:00 PM — Market Close

**Guard:** `current_time >= 16:00`

- If `entries_done` or positions exist: saves end-of-day audit reports (`execution_YYYY-MM-DD.json`, `health_YYYY-MM-DD.json`). State saved. **Loop exits.**
- If no entries were made: `_finalize_day()` runs cleanup, loop exits.

Positions are now held overnight. The bot process terminates.

---

## State Persistence & Recovery

### State Files

**`state/positions.json`**
- Symbol, entry_price, quantity, entry_time, adv_estimate, peak_price, current_price
- Persists across days — loaded at next-day 9:00 AM startup

**`state/bot_state.json`** — date-stamped; ignored if from a prior day
- `hard_stops_checked` (open-price capture flag), `v2_classified`, `exits_1130_done`, `post_exit_failsafe_done`
- `data_collected`, `scoring_done`, `entries_done`
- `open_prices` — `{symbol: open_price}` captured at 9:30
- `exit_schedule` — `{symbol: "09:35" | "11:30"}`
- `classification_audit` — `{symbol: {open_price, price_935, move_5m_pct, exit_time, open_price_source, price_935_source, gap_pct}}`
- `sold_today` — symbols sold this session (PDT guard)

### Same-Day Crash Recovery

If the bot restarts mid-day, `_load_state()` restores all flags. The event loop resumes from the last completed stage — no duplicate exits, no duplicate entries.

### Broker Reconciliation

On startup, live broker positions are fetched and compared to `positions.json`:
- Positions in broker but not local → rebuilt from broker data
- Positions local but not in broker → removed
- Quantities mismatched → synced to broker value

---

## Architecture

```
run.py                              # Entry point
bot/
  integrated_main.py                # OvernightMomentumBot — main orchestrator
  position_manager_overnight.py     # Position tracking, order submission, failsafe flatten
  exit_classifier.py                # 9:35 AM classification (open-to-935 return rule)
  momentum_scorer.py                # 350-model scoring, HEAD/TAIL allocation
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
  audit/
    universe_YYYY-MM-DD.json        # Universe pipeline diagnostics
    candidates_YYYY-MM-DD.json      # Top 20 scored candidates
    execution_YYYY-MM-DD.json       # Entry fill details
    health_YYYY-MM-DD.json          # Daily run health metrics
  logs/
    bot_YYYY-MM-DD.log              # Full daily log
```

---

## Configuration Reference

### Universe (`config_universe.py`)
```python
UNIVERSE_PRESET  = "expanded_smallcap"
MIN_PRICE        = 1.00
MAX_PRICE        = 10.00
MIN_ADV_DOLLARS  = 2_000_000   # $2M minimum 20-day ADV
ADV_LOOKBACK_DAYS = 20
ATR_LOOKBACK_DAYS = 14
```

### Scoring weights (`config_strategy.py`)
```python
SCORE_WEIGHT_INTRADAY_RETURN = 0.20
SCORE_WEIGHT_PROXIMITY_HIGH  = 0.15
SCORE_WEIGHT_VOLUME_VS_AVG   = 0.20
SCORE_WEIGHT_VOLUME_TREND    = 0.10
SCORE_WEIGHT_VS_MARKET       = 0.25
SCORE_WEIGHT_ATR_PCT         = -0.10
```

### Position sizing (`config_strategy.py`)
```python
MAX_LEVERAGE         = 1.0     # Cash account, no margin
ADV_CAP_PCT          = 0.003   # Max = 0.3% of 20-day ADV per position
MIN_SHARES           = 25      # Skip if rounding yields fewer
HEAD_PCT             = 0.70
TAIL_PCT             = 0.30
MAX_HEAD_POSITIONS   = 10
MAX_TOTAL_POSITIONS  = 30      # Soft cap; exceeded to reach 80% deployment
```

### Exit rules (`config_strategy.py`)
```python
EXIT_UP_MOVE_PCT =  0.5       # ret_open_to_935 > 0.5% -> exit at 9:35
V2_CLASSIFY_TIME      = "09:35"
EXIT_BUCKET_1130_TIME = "11:30"
V2_FAILSAFE_TIME      = "11:35"
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
| After 11:35 AM | Yes | Immediate failsafe flatten, then continues to afternoon |
| After 4:00 PM | Any | Logs error, exits |
| 11:35 AM–3:30 PM | No | Waits for 3:30 PM data collection |

---

## Known Limitations

1. **Massive API dependency** — Universe building requires Massive API. Alpaca fallback is slower and may miss symbols.
2. **Polling-based execution** — 1-second event loop. Very fast market moves between polls could be missed.
3. **Single-day holding period** — Strategy assumes next-morning exit. Extended holds are not supported.
4. **No sector diversification** — Selection is purely momentum-based; can concentrate in a single sector.
5. **Broker API dependency** — Failsafe and reconciliation rely on broker API availability.
6. **No real-time alerts** — Bot logs to file only; no push notifications for critical events.
7. **Minute bar data quality** — Symbols with fewer than 30 bars are excluded from scoring.
8. **No drawdown protection** — Bot enters positions regardless of recent P&L history.

---

## Risk Warnings

⚠️ **This is an overnight momentum strategy with significant risks:**

- **Overnight gap risk (no hard stop)** — There is no stop-loss. A large gap-down at the open is held through the open and not exited until 9:35 (if it gapped up enough to trigger the rule) or 11:30
- **Momentum reversal risk** — Late-day strength can reverse overnight
- **Liquidity risk** — 0.3% ADV cap may not prevent slippage in thin small-cap names
- **Technology risk** — API failures or bugs could cause missed exits
- **PDT risk** — Same-day re-entries blocked when equity < $50,000; accounts < $25,000 subject to PDT rules

**Always test in paper trading first. Never risk capital you cannot afford to lose.**
