# Combined MR + Router Strategy

**Status:** Live (paper)  
**Strategy:** Combined overnight Mean Reversion + intraday ETF Router  
**MR Allocation:** ADV-capped, up to 33% of capital per position  
**Liquidity cap per name:** 0.3% of 20-day ADV (`ADV_CAP_PCT`)

---

## Strategy 1 — MR (Mean Reversion, Overnight Hold)

### Overview
- **Entry Time:** 15:45 ET (overnight hold)
- **Exit Time:** Next day 09:30 ET
- **Position Limit:** Top 3 candidates per day

### Stock Filtering Criteria

| Filter | Value | Config |
|---|---:|---|
| Price | $1.00 – $2.00 | `MR_MIN_PRICE` / `MR_MAX_PRICE` |
| Day return vs prior close | ≤ −4.0% | `MR_DAY_RET_MAX` |
| Close position in day range | ≤ 0.25 (bottom 25%) | `MR_CLOSE_POSITION_MAX` |
| 20D average dollar volume | ≥ $1,000,000 | `MR_MIN_AVG_DOLLAR_VOLUME` |
| Max slots | 3 | `MR_MAX_POSITIONS` |

### Candidate Selection
- Apply all filters → rank by lowest `close_position` first
- Top 3 candidates selected for entry

### Position Sizing
```python
max_shares_adv = int(adv_20d * 0.3 / 100)    # ADV constraint
max_shares_capital = int(available_capital * 0.33 / entry_price)  # Capital constraint
shares = min(max_shares_adv, max_shares_capital)
```

### Regime Sizing
The bot checks SPY, IWM, and QQQ before entry and sizes the MR sleeve
based on the average ETF move versus open:

| Market regime before 15:45 | Size multiplier |
|---|---:|
| 3-ETF average < 0 | 1.0× |
| 3-ETF average ≥ 0 | 0.5× |

### Return Calculation
```python
gross_return = (exit_price / entry_price - 1) * 100
net_return = gross_return - 0.50  # 50 bps round-trip cost
pnl = position_value * net_return / 100
```

---

## Strategy 2 — ETF Router (Intraday, Same-Day Entry/Exit)

### Overview
Pure intraday decision tree. One trade per day, routed to a leveraged ETF
based on market conditions at ~10:00 ET.

### Decision Tree Priority

#### 1. A++ Long (TQQQ, exit 15:00)
| Condition | Threshold |
|---|---|
| QQQ | > +10 bps |
| XLK green, SPY green, IWM green, VXX red | all required |
| QQQ near high | ≥ 80% of daily range |
| QQQ continues up 09:45 → 10:00 | required |

#### 2. A Long (TQQQ, exit 15:00)
| Condition | Threshold |
|---|---|
| QQQ | > +10 bps |
| XLK green, VXX red, IWM green | all required |
| QQQ continues up OR QQQ ≥ 50% of range | either |
| SPY | > −5 bps |

#### 3. A− Long (TQQQ, exit 15:00)
| Condition | Threshold |
|---|---|
| QQQ | > +10 bps |
| XLK green, VXX red | required |
| IWM | > −10 bps |
| SPY | > −5 bps |

#### 4. A− Weak (TQQQ, exit 15:00)
| Condition | Threshold |
|---|---|
| QQQ | +5 to +10 bps |
| XLK green, VXX red, IWM green | all required |
| SPY | > −5 bps |

#### 5. Goldilocks (SQQQ, exit 14:00)
| Condition | Threshold |
|---|---|
| SQQQ | green |
| QQQ | −30 to −60 bps |
| SQQQ | +100 to +150 bps |
| QQQ floor | not below −80 bps |
| SQQQ ceiling | not above +250 bps |

#### 6. UVXY Crash (UVXY, exit 11:00)
| Condition | Threshold |
|---|---|
| QQQ | < −60 bps |
| VXX green, near high (≥ 80%) | required |
| UVXY green, near high (≥ 80%) | required |
| UVXY cap | < +500 bps |

#### 7. No Trade
If none of the above conditions are met.

### Router Position Sizing
```python
router_capital = portfolio_value * ROUTER_ALLOCATION_PCT
shares = int(router_capital / entry_price)
```

### Return Calculation
```python
net_return = row["net_ret"]  # Already net of 5 bps cost
pnl = position_value * net_return
```

---

## Combined Strategy Logic

### Non-Overlap Principle
- **MR:** Overnight holds (15:45 → next day 09:30)
- **Router:** Pure intraday (same-day entry/exit, 10:00 → exit time)
- **Capital:** Both sleeves can use available capital since their holding
  periods do not overlap

### Daily Execution Flow
```
for each trading day:
    # 1. Router intraday (10:00 decision)
    router_signal = evaluate_router_decision_tree(market_data)
    if router_signal != "none":
        execute_router_trade(router_signal)

    # 2. MR overnight entry (15:45)
    mr_candidates = filter_mr_candidates(mr_data)
    mr_selected = select_top_3_by_close_location(mr_candidates)
    execute_mr_trades(mr_selected)
```

---

## Daily Schedule (ET)

```
T-1 entry day:
  09:00  Bot starts, reconciles broker positions
  10:00  ETF Router decision + entry (if signal fires)
  11:00–15:00  Router exit at configured time (depends on tier)
  15:30  Build MR universe (asset → price → ADV → tradability)
  15:45  Score MR candidates, apply ETF regime sizing
  15:45  Submit up to 3 market buys (BP-aware, ADV-capped)
  16:00  Save EOD reports, hold MR positions overnight

T+1 exit day:
  05:00  Bot starts, reconciles broker positions, restores state
  05:00–06:00  Premarket "decisive" classification at 15-min checkpoints
               (each HH:MM runs once, dedup'd)
  09:25  Cancel all open orders; freeze broker-position exit plan
  09:30  Submit batched market sells for all remaining broker positions
  09:31  Broker-native rescue pass for any remaining positions
  09:45  V2 failsafe: force-flatten any stragglers with multi-layer retry
         (market → limit −3% → limit −5% half-then-rest)
  16:00  Day complete; restart cycle
```

Adaptive main-loop sleep: 1s during hot windows (05:00–06:02, 09:24–10:05,
15:29–16:01), 30s otherwise.

---

## Daily Artifacts (`state/logs/`)

- `universe_YYYY-MM-DD.json` — full universe pipeline counts and rejection reasons
- `candidates_YYYY-MM-DD.json` — top scored MR candidates
- `execution_YYYY-MM-DD.json` — selection → submit → fill funnel
- `run_health_YYYY-MM-DD.json` — single-glance daily health report

---

## Failsafe Layers

1. **09:25 cancel-all** — every open order is canceled before exit logic.
2. **Batched open market exit** — 09:30 batch sells of frozen 09:25 positions.
3. **09:31 broker-native rescue** — direct broker position fetch and market sell.
4. **09:45 V2 failsafe** — `force_flatten_broker_positions` retries
   market → limit −3% → limit −5% half-then-rest, falls back to broker
   `current_price` / `lastday_price` / `avg_entry_price` if live snapshot
   is unavailable, escalates to market sell if no reference price resolves.
5. **Circuit breaker** — per-symbol 60s cooldown after 3 consecutive
   sell rejections, applied via `_is_exit_blocked`.
6. **Trust broker > local** — every exit path treats broker API errors as
   "unknown" (None) rather than "flat", preserving local state on glitches.

---

## Key Config Files

- `bot/config_strategy.py` — MR filters, sizing, timing, router config, failsafe modes
- `bot/config_universe.py` — price range presets, ADV/ATR lookback
- `bot/config_runtime.py` — state and log paths
- `bot/config_broker.py` — Alpaca/Massive credentials, data feed

All re-exported by `bot/config.py` for `from bot import config; config.X` use.

---

*Last updated: Combined MR + Router strategy consolidation.*
