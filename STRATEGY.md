# Combined Bot Strategy — Quick Reference

**Status:** Live (paper)
**Architecture:** 3 sleeves (Intraday ETF Router + Intraday MR + Overnight MR/TQQQ).

For the full specification see [`bot/STRATEGY.md`](bot/STRATEGY.md).

---

## Sleeves at a glance

| Sleeve | Trigger window | Holding |
|---|---|---|
| Intraday ETF Router | one trade/day at 10:00 (strats 1–3) or 10:10 (strats 4–5) | same day, flat by 15:30 |
| Intraday MR | entries 09:32–10:00 | same day, flat by 15:40 |
| Overnight | entry 15:45 | overnight, sold 09:30 next day |

## Overnight single-stock MR

| Parameter | Value |
|---|---|
| Entry / Exit | 15:45 ET → next day 09:30 ET |
| Max positions | 3 (`MR_MAX_PRIMARY_POSITIONS`) |
| Per-position size | 30% of equity |
| Price range | $1.00 – $2.00 |
| Day return | ≤ −4.0% |
| Close position | ≤ 0.25 (bottom 25% of range) |
| ADV requirement | ≥ $1M (20-day avg dollar volume) |

**Conditional TQQQ** is added on top of MR (does not block it) when favorable, capped
so combined overnight exposure stays within 90% (`OVERNIGHT_COMBINED_MAX_ALLOCATION_PCT`).

## Intraday ETF Router strategies

1. VXX Spike Recovery → TQQQ (exit 15:30)
2. VXX Collapse → TQQQ (exit 15:30)
3. Momentum Sleeve → TQQQ / SQQQ in HIGH_RISK (exit 15:00)
4. Router Long → TQQQ (exit 15:30)
5. SVIX Long → SVIX (exit 15:00)
