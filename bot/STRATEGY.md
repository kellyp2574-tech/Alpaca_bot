# Combined Overnight Rebound Strategy

**Status:** Live (paper)
**Strategy:** Two-sleeve overnight hold — Mean Reversion (MR_WIDE) + Green-Day Pullback (GDP_BASE)
**Allocation:** Static 70% MR / 30% GDP
**Single-name cap:** 10% of equity (`MAX_SINGLE_POSITION_PCT`)
**Liquidity cap per name:** 0.3% of 20-day ADV (`ADV_CAP_PCT`)

---

## Sleeves

### Sleeve 1 — MR_WIDE (Mean Reversion)
Buy stocks that sold off hard with elevated volume and closed near the day's low.

| Filter | Value | Source |
|---|---|---|
| Price | $1.00 – $5.00 | `MR_MIN_PRICE` / `MR_MAX_PRICE` |
| Day return (9:30 → 15:50) | ≤ −3% | `MR_DAY_RET_MAX` |
| Volume ratio vs ADV | ≥ 1.5× | `MR_VOLUME_RATIO_MIN` |
| Close position in day range | ≤ 0.20 (bottom 20%) | `MR_CLOSE_POSITION_MAX` |
| Late drop 15:30 → 15:50 | optional, off | `MR_LATE_DROP_MAX` |
| Max slots | 12 | `MR_MAX_POSITIONS` |

**Score** = 0.50·(1 − close_position) + 0.30·min(volume_ratio/3, 1) + 0.20·min(|day_return|/0.10, 1)

### Sleeve 2 — GDP_BASE (Green-Day Pullback)
Buy green stocks that pulled back below VWAP with decelerating late momentum.

| Filter | Value | Source |
|---|---|---|
| Price | $1.00 – $10.00 | `GDP_MIN_PRICE` / `GDP_MAX_PRICE` |
| Day return | +1% to +10% | `GDP_DAY_RET_MIN` / `GDP_DAY_RET_MAX` |
| Below intraday VWAP | required | `GDP_REQUIRE_BELOW_VWAP` |
| Late momentum 15:30 → 15:50 | ≤ 0 (decelerating) | `GDP_LATE_MOM_MAX` |
| Close position | optional, off | `GDP_MAX_CLOSE_POSITION` |
| Max slots | 8 | `GDP_MAX_POSITIONS` |

**Score** = 0.35·vwap_pullback + 0.25·deceleration + 0.20·volume_ratio + 0.20·(1 − close_position)

GDP candidates that overlap with MR are removed (MR takes priority).

Combined position cap: `COMBINED_MAX_POSITIONS = 20`.

---

## Sizing

Two-pass waterfall allocator per sleeve, against the sleeve's dollar budget
(`MR_ALLOCATION_PCT` × deployable; `GDP_ALLOCATION_PCT` × deployable).

1. **Pre-filter** candidates whose ADV cap would be below the effective minimum
   (`max(MIN_POSITION_DOLLARS, MIN_SHARES × signal_price)`).
2. **Pass 1 (low-ADV first):** equal-share `sleeve_budget / N`, capped per name
   by `min(ADV_CAP_PCT × adv_dollars, MAX_SINGLE_POSITION_PCT × equity, MAX_POSITION_DOLLARS)`.
3. **Pass 2 (high-ADV first):** push leftover budget into names with remaining
   capacity.

Deployable = `min(buying_power, equity × MAX_LEVERAGE)`.
Per-order BP buffer = `ENTRY_BP_BUFFER_PCT` (98%).

---

## Daily Schedule (ET)

```
T-1 entry day:
  15:30  Build base universe (asset → price → ADV → tradability)
  15:50  Fetch 9:30→15:50 minute bars, run Stage C minute-quality filter
  15:50  Score MR + GDP, allocate budgets, run execution-eligibility gate
  15:50  Submit market buys (BP-aware sizing, single account fetch)
  16:00  Save EOD reports, hold positions overnight

T+1 exit day:
  05:00  Bot starts, reconciles broker positions, restores state
  05:00 / 05:15 / 05:30 / 05:45  Premarket "decisive" classification
                                  (each checkpoint runs once via dedup set)
  06:00  Final premarket classification for all unresolved symbols
  09:25  Cancel all open orders; freeze broker-position exit plan
  09:30  Submit market sells in batch (ENABLE_FAST_OPEN_MARKET_EXIT=True).
         The alternative red-trail mode is mutually exclusive and
         currently disabled.
  09:45  V2 failsafe: force-flatten any stragglers with multi-layer retry
         (market → limit −3% → limit −5% half-then-rest)
  16:00  Day complete; restart cycle next afternoon at 15:30
```

Adaptive main-loop sleep: 1s during hot windows (05:00–06:02, 09:24–10:05,
15:29–16:01), 30s otherwise.

---

## Daily Artifacts (`state/logs/`)

- `universe_YYYY-MM-DD.json` — full universe pipeline counts and rejection reasons
- `candidates_YYYY-MM-DD.json` — top scored MR + GDP candidates
- `execution_YYYY-MM-DD.json` — selection → submit → fill funnel
- `run_health_YYYY-MM-DD.json` — single-glance daily health report

---

## Failsafe Layers

1. **09:25 cancel-all** — every open order is canceled before exit logic.
2. **09:30 batch sells** — broker positions, frozen at 09:25, sold market.
3. **Red-trail (off in production)** — `ENABLE_RED_OPEN_TRAIL_EXIT=False`.
   Mutually exclusive with fast-exit; `_validate_config` raises if both are
   accidentally enabled together.
4. **09:45 V2 failsafe** — `force_flatten_broker_positions` retries
   market → limit −3% → limit −5% half-then-rest, falls back to broker
   `current_price` / `lastday_price` / `avg_entry_price` if the live
   snapshot is unavailable, escalates to a market sell if no reference
   price can be resolved, and flags manual on residual.
5. **Circuit breaker** — per-symbol 60s cooldown after 3 consecutive
   sell rejections, applied via `_is_exit_blocked`.
6. **Trust broker > local** — every exit path treats broker API errors as
   "unknown" (None) rather than "flat", preserving local state on glitches.

---

## Key Config Files

- `bot/config_strategy.py` — sleeve filters, sizing, timing, failsafe modes
- `bot/config_universe.py` — price range presets, ADV/ATR lookback
- `bot/config_runtime.py` — state and log paths
- `bot/config_broker.py` — Alpaca/Massive credentials, data feed

All re-exported by `bot/config.py` for `from bot import config; config.X` use.

---

*Last updated: bot audit + cleanup pass; see git history for the prior VIX-conditioned
gap strategy (deleted) and head/tail allocator (deleted).*
