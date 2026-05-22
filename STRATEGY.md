# Clean Overnight MR Paper-Test Strategy

**Status:** Paper test  
**Strategy:** Single-sleeve overnight mean reversion — cheap late-day washout  
**Entry:** 15:45 ET  
**Exit:** 09:30 ET next trading day  
**Allocation:** MR only; GDP/MOM disabled for this paper test  
**Max positions:** 3  

---

## Sleeve — CLEAN_OVERNIGHT_MR

Buy cheap stocks that sold off hard into the lower part of the day's range by the late-day entry window.

| Filter | Value | Config |
|---|---:|---|
| Entry price | $1.00 – $2.00 | `MR_MIN_PRICE` / `MR_MAX_PRICE` |
| Day return / signal return | ≤ −5% | `MR_DAY_RET_MAX` |
| Close position in day range | ≤ 0.25 | `MR_CLOSE_POSITION_MAX` |
| 20D average dollar volume | ≥ $1,000,000 | `MR_MIN_AVG_DOLLAR_VOLUME` |
| Volume ratio requirement | Off | `MR_VOLUME_RATIO_MIN = 0.0` |
| Max slots | 3 | `MR_MAX_POSITIONS` |
| Minimum candidate count | 2 | `MR_MIN_CANDIDATES` |

**Rank:** lowest `close_position` first.

This paper test intentionally disables the old GDP/MOM sleeve so the live paper results isolate the new clean-cache MR signal.

---

## Regime Sizing

The bot checks SPY, IWM, and QQQ before entry and sizes the MR sleeve based on the average ETF move versus open.

| Market regime before 15:45 | Size multiplier |
|---|---:|
| 3-ETF average < 0 | 1.0x |
| 3-ETF average ≥ 0 | 0.5x |

Config:

```python
ENABLE_MR_ETF_REGIME_SIZING = True
MR_ETF_REGIME_SYMBOLS = ["SPY", "IWM", "QQQ"]
MR_ETF_NEGATIVE_SIZE_MULT = 1.0
MR_ETF_POSITIVE_SIZE_MULT = 0.5
```

---

## Daily Schedule (ET)

```text
T-1 entry day:
  15:30  Build base universe
  15:45  Fetch 09:30→15:45 minute bars, score CLEAN_OVERNIGHT_MR
  15:45  Require at least 2 candidates, rank by lowest close position
  15:45  Apply ETF regime sizing and submit up to 3 market buys
  16:00  Save EOD reports, hold positions overnight

T+1 exit day:
  05:00  Bot starts, reconciles broker positions, restores state
  05:00 / 05:15 / 05:30 / 05:45  Optional rolling premarket limit classification
  06:00  Final premarket classification for unresolved symbols
  09:25  Cancel all open orders; freeze broker-position exit plan
  09:30  Submit batched market sells for all remaining broker positions
  09:31  Broker-native rescue pass for any remaining positions
  09:45  V2 failsafe: force-flatten any stragglers
```

---

## Failsafe Layers Preserved

The entry sleeve changed, but the morning risk controls remain the same:

1. 09:25 cancel-all before normal exit logic.
2. Batched 09:30 market sells using the frozen broker-position plan.
3. 09:31 broker-native rescue.
4. 09:45 force-flatten failsafe.
5. Per-symbol sell circuit breaker after repeated rejections.
6. Broker state is treated as ground truth over local state.

---

## Files

- `bot/config_strategy.py` — clean MR paper-test config
- `bot/integrated_main.py` — MR-only scoring/allocation overlay and ETF regime sizing
- `bot/position_manager_overnight.py` — unchanged order/exits/failsafes
