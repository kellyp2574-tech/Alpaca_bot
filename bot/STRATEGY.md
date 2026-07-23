# Combined Bot Strategy

**Status:** Live (paper)
**Architecture:** 3 sleeves orchestrated by `integrated_main.py`, ground truth lives in `config_strategy.py`.

| Sleeve | Window | Holding |
|---|---|---|
| Intraday ETF Router | one trade/day at 10:00 or 10:10 | same day, flat by 15:30 |
| Intraday MR (morning momentum) | entries 09:32–10:00 | same day, flat by 15:40 |
| Overnight TQQQ | entry 15:45 | overnight, sold 09:30 next day |

---

## Sleeve 1 — Intraday ETF Router (one trade per day)

Tape of `QQQ, SPY, VXX, SVIX, TQQQ, SQQQ` is recorded 09:30–10:10. Strategies are
checked in order; the first to fire takes the only trade of the day. A fill blocks
the rest of the intraday ETF sleeve (it does **not** block the overnight sleeves).

Checked at **10:00** (30-minute returns):

| # | Strategy | Trigger | Vehicle | Exit | SL / TP |
|---|---|---|---|---|---|
| 1 | VXX Spike Recovery | VXX ≥ +2.5% AND QQQ 09:30–10:00 range 0.3–0.8% | TQQQ | 15:30 | none |
| 2 | VXX Collapse | VXX ≤ −2.0% AND QQQ ≥ −1.0% | TQQQ | 15:30 | SL −1.0% (arm 13:00) |
| 3 | Momentum Sleeve | QQQ ≥ +0.5% | TQQQ (NORMAL) / SQQQ (HIGH_RISK) | 15:00 | SL −0.5% (arm 13:00), TP +2% |

Checked at **10:10** (40-minute returns, only if no 10:00 trade):

| # | Strategy | Trigger | Vehicle | Exit | SL / TP |
|---|---|---|---|---|---|
| 4 | Router Long | QQQ−SPY spread ≥ +0.2% | TQQQ | 15:30 | SL −0.5% (arm 13:30), TP +3% |
| 5 | SVIX Long | SVIX ≥ +0.2% | SVIX | 15:00 | no SL, TP +3% |

**HIGH_RISK regime** (used by Momentum): VXX 30-min return ≥ +2.0% OR VXX price ≥ $400.
In HIGH_RISK, Momentum flips to the anti-vehicle (`SQQQ`, branch `MOMENTUM_SLEEVE_ANTI`).

**Sizing:** up to `INTRADAY_ETF_ALLOCATION_PCT` (1.00 = 100% of equity) minus whatever
the morning MR sleeve has already deployed, capped by buying power and an entry buffer.

Per-strategy SL/TP and exit times are the table `ETF_SL_TP` in `config_strategy.py`
(the operative source for `check_etf_exits`).

---

## Sleeve 2 — Intraday MR (Morning Momentum)

Gap-reversal longs entered in the morning, flat the same day.

- **Active day** when `VIX ≥ 15` AND `|SPY gap| > 0.5%` AND `|QQQ gap| > 0.5%`.
- **Universe:** price $2–$100, prev-day dollar volume ≥ $1M.
- **Candidates:** 1–8 per day, classified into themes by `intraday_mr_classifier.py`.
- **Positions:** up to 3 live positions (`INTRADAY_MR_MAX_POSITIONS`). The bot submits
  the highest-ranked due candidates first; if one fails execution, the next-ranked
  due candidate takes its slot (walk-down).
- **Budget:** 50% of equity (`INTRADAY_MR_BUDGET_PCT`) split equally across the target
  3 positions.
- **Timing:** Stage 1 universe + bar cache at 09:00; Stage 2 (opens, VIX, finalize) at
  09:30; entries 09:32 until the 10:00 cutoff; hard flatten at **15:40**.
- **10:00 router-exit rule:** if the ETF router is SHORT (`sqqq_goldilocks`, i.e.
  `MOMENTUM_SLEEVE_ANTI`), exit all non-Theme-A positions.
- **10:10 reallocation:** if the router produced no qualifying signal, the unused
  router budget is redistributed to open MR winners (`INTRADAY_REALLOC_*`).

---

## Sleeve 3 — Overnight TQQQ (15:45)

Sole active overnight strategy. Single-stock MR overnight is disabled (`MR_OVERNIGHT_ENABLED=False`).

### Conditional TQQQ

Implemented in `overnight_etf_runner_conditional.py`.

| Parameter | Value | Config |
|---|---:|---|
| Allocation when signal fires | 75% of equity | `TQQQ_CONDITIONAL_ALLOCATION_PCT` |
| Max overnight exposure | 75% of equity | `OVERNIGHT_COMBINED_MAX_ALLOCATION_PCT` |

TQQQ fires when the signal is positive (blends VIX regime, QQQ/VXX/SPY day returns;
requires `expected_return > 0.2%`). All overnight positions are sold at 09:30 the next morning.

---

## Daily Schedule (ET)

```
T-1 (entry) / T+1 (exit) are the same calendar run:

  09:00  Start, load state, detect overnight positions from broker
  09:00  Intraday MR Stage 1 (universe + T-1/T-2 bar cache)
  09:25  Cancel all open orders; freeze broker-position exit plan
  09:30  Submit batched market sells for remaining overnight positions
  09:30  Begin ETF tape recording; Intraday MR Stage 2 (opens, VIX, finalize)
  09:31  Broker-native rescue pass for any remaining positions
  09:32  Intraday MR entries begin (until 10:00 cutoff)
  09:45  Post-exit failsafe — verify flat or force-flatten stragglers
  10:00  ETF router strategies 1–3; intraday MR router-exit rule
  10:10  ETF router strategies 4–5; intraday MR budget reallocation
  15:00  Intraday ETF exit for strategies 3 & 5
  15:30  Intraday ETF hard flatten for strategies 1, 2 & 4
  15:40  Intraday MR hard flatten
  15:45  Score single-stock MR; conditional TQQQ; enter MR positions
  16:00  Save EOD reports, hold overnight positions, done
```

Adaptive main-loop sleep: 1s during hot windows (see `_HOT_WINDOWS_HHMM` in
`integrated_main.py`), 30s otherwise.

---

## Morning Exit Failsafe Layers

1. **09:25 cancel-all** — every open order canceled before exit logic.
2. **09:30 batched market sells** — frozen 09:25 broker snapshot.
3. **09:31 broker-native rescue** — direct broker fetch + market sell of any remainder.
4. **09:45 failsafe flatten** — force-flatten any stragglers.
5. **Trust broker > local** — broker API errors are treated as "unknown" (None), never
   "flat", so local state is preserved on transient glitches.

`morning_liquidation_confirmed` is a one-way latch set only once the broker confirms
zero positions; entries stay blocked until it flips.

---

## Daily Artifacts (`state/logs/`)

- `universe_YYYY-MM-DD.json` — universe pipeline counts + rejection reasons
- `candidates_YYYY-MM-DD.json` — top scored MR candidates
- `execution_YYYY-MM-DD.json` — selection → submit → fill funnel
- `run_health_YYYY-MM-DD.json` — single-glance daily health report

---

## Config Files

- `config_broker.py` — Alpaca/Massive credentials, endpoints, data feed
- `config_runtime.py` — state and log paths
- `config_universe.py` — price/ADV filters, presets, lookback periods
- `config_strategy.py` — sleeve configs, exit rules, SL/TP table, timing

All re-exported by `config.py` for `from bot import config; config.X`.
