# Combined MR + Router Strategy — Quick Reference

**Status:** Live (paper)  
**Strategy:** Combined overnight Mean Reversion + intraday ETF Router  

For the full specification see [`bot/STRATEGY.md`](bot/STRATEGY.md).

---

## MR (Overnight Hold)

| Parameter | Value |
|---|---|
| Entry | 15:45 ET |
| Exit | Next day 09:30 ET |
| Max positions | 3 |
| Price range | $1.00 – $2.00 |
| Day return | ≤ −4.0% |
| Close position | ≤ 0.25 (bottom 25% of range) |
| ADV requirement | ≥ $1M (20-day avg dollar volume) |
| Sizing | min(0.3% ADV, 33% capital) |
| Regime gate | Half-size when SPY/IWM/QQQ average ≥ 0 |

---

## ETF Router (Intraday)

One trade per day, ~10:00 ET decision. Priority waterfall:

| Tier | Instrument | Exit | Core Condition |
|---|---|---|---|
| A++ Long | TQQQ | 15:00 | QQQ >10bp, all sectors green, near high, continuing up |
| A Long | TQQQ | 15:00 | QQQ >10bp, XLK/VXX/IWM confirm, SPY >−5bp |
| A− Long | TQQQ | 15:00 | QQQ >10bp, XLK/VXX confirm, IWM >−10bp |
| A− Weak | TQQQ | 15:00 | QQQ 5–10bp, XLK/VXX/IWM green |
| Goldilocks | SQQQ | 14:00 | QQQ −30 to −60bp, SQQQ +100–150bp |
| UVXY Crash | UVXY | 11:00 | QQQ <−60bp, VXX/UVXY green near highs |
| No Trade | — | — | None of the above met |

---

## Non-Overlap Principle

- MR holds overnight → exits before market open
- Router trades intraday → exits same day
- Both can use full available capital (no capital contention)
