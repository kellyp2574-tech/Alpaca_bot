# Combined Overnight Rebound Bot

Automated equity trading bot on Alpaca. Runs a two-sleeve overnight hold:

- **MR_WIDE** — mean reversion on $1–5 stocks that sold off hard with elevated volume and closed near the day's low
- **GDP_BASE** — green-day pullback on $1–10 stocks that pulled back below intraday VWAP with decelerating late momentum

Static **70/30 MR/GDP** allocation, max 20 combined positions, 10% single-name cap, 0.3% ADV liquidity cap.

For the full strategy spec see `bot/STRATEGY.md`.

---

## Daily Schedule (ET)

```
T-1 entry day:
  15:30  Build universe (asset → price → ADV cache → tradability)
  15:50  Fetch 9:30→15:50 minute bars, score MR + GDP, run execution gate
  15:50  Submit market buys (single account fetch, BP-aware sizing,
         daily-loss circuit breaker check)
  16:00  Save EOD reports, hold positions overnight

T+1 exit day:
  05:00  Bot starts, reconciles broker, restores state
  05:00–06:00  Premarket "decisive" classification at 15-min checkpoints
               (each HH:MM runs once, dedup'd)
  09:25  Cancel all open orders, freeze broker-position exit plan
  09:30  Submit batched market sells (ENABLE_FAST_OPEN_MARKET_EXIT=True;
         red-trail mode is mutually exclusive and disabled by default)
  09:45  V2 failsafe flatten (market → -3% → -5%, with broker price
         fallback if snapshot fails)
  16:00  Day complete; restart cycle next afternoon at 15:30
```

The main loop sleeps adaptively: 1s during the three hot windows above, 30s
otherwise.

---

## Run

```powershell
# Paper trading (default)
python run.py
```

Configure credentials in `.env`:

```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_PAPER=true
MASSIVE_API_KEY=...
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Architecture

```
run.py                              # entry point
bot/
  integrated_main.py                # CombinedOvernightReboundBot orchestrator
  position_manager_overnight.py     # entry/exit/failsafe + REST polling
  fill_stream.py                    # /stream trade_updates websocket (push fills)
  market_data.py                    # AlpacaDataClient — snapshots, bars, ADV
  massive_client.py                 # Massive full-market snapshot
  universe_builder.py               # 4-stage universe pipeline + diagnostics
  mean_reversion_scorer.py          # MR_WIDE candidate build + filter
  green_day_pullback_scorer.py      # GDP_BASE candidate build + filter
  scorer_utils.py                   # shared bar/VWAP/intraday metric helpers
  rate_limiter.py                   # shared sliding-window rate limiter (80/min)
  state_manager.py                  # atomic JSON state writes
  config.py + config_*.py           # broker / runtime / universe / strategy
```

### Key reliability features

- **Push fills via Alpaca `trade_updates` websocket** (`bot/fill_stream.py`).
  `get_order_fill` short-circuits with sub-second latency when the stream
  has already cached a terminal event. Falls back to REST polling
  automatically if the stream is down.
- **Atomic state writes** — tmp-file + `os.replace` prevents corruption on
  mid-write crash.
- **Adaptive main-loop sleep** — 1s during open / close / premarket
  checkpoints, 30s otherwise. ~1.5k ticks/day instead of ~36k.
- **Daily-loss circuit breaker** — aborts 15:50 entries if today's PnL ≤
  `-DAILY_LOSS_LIMIT_PCT` (default 5%) vs `account.last_equity`.
- **Per-checkpoint dedup** — premarket classification runs once per HH:MM
  (was firing ~60×/checkpoint before the audit fix).
- **Single-fetch account caching** — entry pass uses one `/v2/account` call
  with a locally tracked `bp_remaining` (was up to ~22 calls).
- **Multi-layer failsafe flatten** — market → limit −3% → half-size limit
  −5% → market escalation if no reference price is available.
- **Trust-broker-over-local** — every reconciliation path treats broker API
  errors as "unknown" rather than "flat" so a transient outage cannot
  mistakenly clear local state.
- **Circuit breaker per symbol** — 60s cooldown after 3 consecutive sell
  rejections.

---

## Daily Artifacts (`state/logs/`)

| File | Purpose |
|---|---|
| `universe_YYYY-MM-DD.json` | Full universe pipeline counts + rejection reasons |
| `candidates_YYYY-MM-DD.json` | Top scored MR + GDP candidates |
| `execution_YYYY-MM-DD.json` | Selection → submit → fill funnel |
| `run_health_YYYY-MM-DD.json` | One-glance daily health report |
| `bot.log` | Continuous text log |

---

## Tests

```powershell
.\venv\Scripts\python.exe -m pytest test_audit_fixes.py -v
```

29 tests covering: FillStream message handling, terminal cache + waiter
notification, REST short-circuit, atomic writes, hot-window detection,
snapshot staleness logging, symbol-filter rules, premarket dedup, and
daily-loss math.

---

## Configuration knobs (most-edited)

In `bot/config_strategy.py`:

| Knob | Default | Purpose |
|---|---|---|
| `MR_ALLOCATION_PCT` / `GDP_ALLOCATION_PCT` | 0.70 / 0.30 | Sleeve budget split |
| `COMBINED_MAX_POSITIONS` | 20 | Hard cap across both sleeves |
| `MAX_SINGLE_POSITION_PCT` | 0.10 | Max equity per name |
| `ADV_CAP_PCT` | 0.003 | Max position notional as fraction of 20-day ADV$ |
| `MIN_SHARES` | 25 | Minimum order size |
| `DAILY_LOSS_LIMIT_PCT` | 0.05 | Skip entries if today PnL ≤ −5% (0 disables) |
| `ENABLE_FAST_OPEN_MARKET_EXIT` | True | 09:30 batched market sells (production mode) |
| `ENABLE_RED_OPEN_TRAIL_EXIT` | False | 09:30 trailing-stop on red opens (mutually exclusive with fast-exit) |
| `RED_OPEN_TRAIL_PCT` | 1.0 | Trailing stop percent (only used when red-trail mode is on) |
| `ENABLE_PREMARKET_DYNAMIC_LIMIT_SELLS` | True | 05:00→06:00 IEX classification |
| `MR_*` filters | see file | Mean-reversion candidate gates |
| `GDP_*` filters | see file | Green-day pullback gates |

In `bot/config_universe.py`:

| Knob | Default | Purpose |
|---|---|---|
| `UNIVERSE_PRESET` | `"expanded_smallcap"` | Selects price band ($1–10) |
| `ADV_LOOKBACK_DAYS` | 20 | Days for ADV calc |

---

## Maintenance Notes

- The bot keeps state in `state/` (positions, bot flags, daily logs).
- `state_manager.clear_bot_state()` is called automatically on no-entry
  end-of-day to keep flags fresh for the next session.
- Same-day restart restores all stage flags from `state/bot_state.json`.
- Multi-day restarts ignore yesterday's bot_state but still load any
  overnight positions from `state/positions.json`.

For the full strategy and rationale see `bot/STRATEGY.md`.
