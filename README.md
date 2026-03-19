# 0DTE Options Bot — XSP Iron Condor + XND Directional

Automated 0DTE options strategy on Alpaca running two independent sleeves: an always-on XSP iron condor and a conditional XND directional play. Both use European-style, cash-settled index options — no early assignment risk.

---

## Strategy Overview

### Sleeve 1: XSP Iron Condor (Every Day)

- **Instrument:** XSP 0DTE options (Mini-SPX, European, cash-settled)
- **Underlying proxy:** SPY price action (XSP settles off its own index calculation)
- **Structure:** Sell call spread + sell put spread (iron condor) as a single mleg order
- **Short strikes:** 0.90% OTM from anchor price (SPY at 11:30 AM)
- **Wing width:** 1.00% (max loss = wing width - credit per contract)
- **Target credit:** ~$1.30 per contract
- **Max loss per contract:** ~$3.70 (this is the margin requirement)
- **Defense trigger:** If SPY moves 1.00% from anchor (using post-anchor max excursion), close all legs immediately
- **Entry:** 11:30 AM ET
- **Exit:** Hold to 4:00 PM cash settlement, or defense close if triggered

### Sleeve 2: XND Directional (Conditional)

- **Instrument:** XND 0DTE options (Mini-NDX, European, cash-settled)
- **Underlying proxy:** QQQ price action (XND strikes are in NDX index points; QQQ × 40 ≈ NDX)
- **Structure:** Buy a single call or put in the direction of the morning trend
- **Strike selection:** ATM relative to estimated NDX level (QQQ × 40), not raw QQQ price
- **Entry:** 10:45 AM ET
- **Exit:** Hold to 4:00 PM cash settlement

**Entry filters (all three must be true at 10:30 AM):**
- Previous day VIX close >= 18
- QQQ morning range >= 0.40%
- |QQQ morning directional move| > 0.30%

If QQQ's morning direction is up, buy a call. If down, buy a put.

---

## Sizing

### Condor Sizing

```
contracts = floor(buying_power / max_loss_per_contract)
max_loss_per_contract = wing_width - credit ≈ $3.70 × 100 = $370
```

`buying_power` comes directly from the Alpaca API. Example: $20,000 / $370 = **54 contracts**.

### Directional Sizing

Both sleeves size off **live available buying power** at their respective entry times. The directional sleeve goes first (10:45 AM), and the condor uses whatever BP remains (11:30 AM).

```
premium_budget = buying_power × directional_bp_pct × directional_leverage_multiplier
               = buying_power × 0.0125 × 5.0
qty = floor(premium_budget / (premium_per_contract × 100))
```

Premium is estimated from a **live option snapshot** (bid/ask mid via Alpaca's option snapshot API). Falls back to prior close price, then a conservative $2.00 if no data is available. For 0DTE options, stale pricing can cause significant over- or under-sizing — the live snapshot mitigates this.

---

## Daily Schedule

| Time (ET) | Activity |
|-----------|----------|
| 9:00 AM | Bot starts, pull account info, VIX previous close |
| 9:30 AM | Market opens, begin tracking SPY/QQQ prices via Alpaca snapshots |
| 10:30 AM | Morning assessment — evaluate directional filters |
| 10:45 AM | Directional entry (if qualified) — submit order, poll for fill |
| 11:30 AM | Condor entry — record anchor, compute strikes, place mleg order, poll for fill |
| 11:30–4:00 PM | Defense monitoring — check SPY max excursion vs anchor every ~60 s |
| 4:00 PM | Settlement — compute estimated P&L from proxy underlying prices |
| 4:15 PM | Log P&L, save daily report, shut down |

---

## Order Lifecycle

Orders go through a clear state machine:

1. **Order submitted** — `entry_order_id` is set, `is_open = False`, `is_filled = False`
2. **Fill confirmed** — `is_filled = True`, `is_open = True` (position is now live; recorded in PDT ledger)
3. **Terminal failure** — if the order is cancelled/rejected/expired, `entry_order_dead = True` and the bot stops polling it

Defense close orders follow the same pattern. If a defense close order terminally fails, the bot retries up to 2 times (strategy level), then escalates to `cancel_all_orders()`. The orchestrator caps total escalation attempts at 3 before logging `MANUAL INTERVENTION REQUIRED`.

Every exit fill is also recorded in the PDT ledger with an `exit_reason` (`scheduled_close`, `defense`, `emergency`, or `discretionary_early_exit`).

---

## PDT Protection

The bot includes a Pattern Day Trader (PDT) guard to prevent accidental PDT classification.

**Decision tree for discretionary early exits:**

1. Is this a mandatory exit (defense, emergency, risk-reduction)? → **Always allowed**
2. Is account equity > $30,000? → If no, **block early exit**
3. Are there < 3 day trades in the rolling 5-business-day window? → If no, **block early exit**
4. All checks pass → **Allow early exit**

**What counts as a day trade:** A same-day round trip where both the entry fill and exit fill occur on the same trading day. Only confirmed fills are counted — cancelled orders are ignored.

**Persistent ledger:** Day trades are recorded in `state/day_trade_log.json` and survive process restarts. Each record includes the trade date, strategy sleeve, symbol, timestamps, and exit reason.

**Mandatory exits are never blocked:**
- Condor defense close (SPY breach)
- Emergency flatten
- Broker/order failure cleanup

---

## Defense Monitoring

After the condor is filled, the bot monitors SPY for a 1.00% move from the anchor price.

**Post-anchor tracking:** The bot maintains separate `spy_post_anchor_high` / `spy_post_anchor_low` fields that track only last-trade prices sampled after the anchor is set. This prevents pre-anchor session extremes (from the daily bar) from falsely triggering defense.

**Max excursion check:** Defense uses `spy_max_move_from_anchor_pct()` which examines the worst post-anchor excursion — catching breaches that occurred between polling intervals even if the current price has recovered.

**Polling limitation:** Defense relies on sampled last-trade prices refreshed every ~30 seconds. A very fast sub-second excursion that fully reverses between refreshes could be missed. True tick-level defense would require a streaming WebSocket connection.

---

## Settlement & P&L

Both sleeves compute estimated P&L at 4:00 PM from proxy underlying prices:

- **Condor:** Uses `tracker.spy_last` as a proxy for XSP settlement. Computes put-spread and call-spread intrinsic values against actual leg strikes. If defense was triggered, uses the actual fill price from the close order.
- **Directional:** Uses `QQQ × 40` as an NDX proxy. Computes call/put intrinsic value against the strike price.

**Important:** These are *estimates*. XSP settles off a special index settlement value (SET), not literally SPY last trade. XND settles off an NDX-derived value, not exactly QQQ × 40. Reported P&L may differ slightly from the broker's end-of-day statement. For precise accounting, reconcile against broker reports.

---

## Architecture

```
run.py                      # Entry point
bot/
  integrated_main.py        # CondorBot orchestrator — daily schedule, main loop
  config.py                 # API keys (.env), state paths, logging
  condor_config.py          # Strategy parameters (strikes, sizing, schedule, filters)
  options_client.py         # Alpaca options REST API — contracts, mleg orders,
                            #   single-leg orders, option snapshots, account info
  market_data.py            # MorningTracker — SPY/QQQ/VIX price tracking,
                            #   session-wide daily bar stats + post-anchor defense
  condor_strategy.py        # Iron condor: enter, check_defense, close_defense,
                            #   check_defense_fill, on_settlement
  directional_strategy.py   # XND directional: assess_filters, enter, on_settlement
  pdt_guard.py              # PDT protection — persistent day-trade ledger,
                            #   rolling 5-business-day count, discretionary exit gate
state/
  day_trade_log.json        # Persistent PDT day-trade ledger (auto-created)
  reports/                  # Daily JSON reports with PDT status
```

Legacy momentum files (`morning_main.py`, `position_manager.py`, etc.) are preserved in `bot/` but are not used by the condor system.

**Data flow:**
- SPY/QQQ prices from Alpaca stock snapshots (daily bar high/low + last trade)
- VIX previous close from yfinance
- Option contracts from `/v2/options/contracts` (0DTE strike lookup)
- Live option premium from `/v1beta1/options/snapshots/{symbol}` (directional sizing)
- Iron condor placed as a single `order_class: "mleg"` order with 4 legs
- All legs use `position_intent` (`buy_to_open`, `sell_to_open`, etc.)

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
cp .env.example .env
# Fill in Alpaca keys, set ALPACA_PAPER=true for paper trading
```

**Requirements:** Python 3.11+, Alpaca Options Level 3 trading enabled.

## Usage

```bash
python run.py               # Live/paper trading
python run.py --dry-run     # Log signals without submitting orders
```

Start the bot at or before 9:00 AM ET. It will wait for each scheduled event and shut down automatically at 4:15 PM.

If started after 9:30 AM, the bot detects the late start and backfills open prices from the Alpaca daily bar. Morning range calculations will still use exchange-reported session extremes.

**Daily reports** are saved to `state/reports/daily_report_YYYY-MM-DD.json`.

## Configuration

All strategy parameters are in `bot/condor_config.py`:

```python
# Condor strikes
short_strike_pct = 0.009    # 0.90% OTM from anchor
wing_width_pct = 0.010      # 1.00% wing width
target_credit = 1.30        # Target net credit per contract
max_loss_per_contract = 3.70

# Defense
defense_trigger_pct = 0.010 # 1.00% move from anchor triggers close

# Directional filters
vix_threshold = 18.0
morning_range_pct = 0.004   # 0.40%
morning_direction_pct = 0.003  # 0.30%

# Directional sizing (based on current available buying power)
directional_bp_pct = 0.0125             # Percent of current BP for premium budget
directional_leverage_multiplier = 5.0   # Premium budget scaled by this multiplier
```

## Known Limitations

- **Settlement P&L is estimated.** XSP and XND settle off special index values, not SPY/QQQ last trades. Reported P&L should be reconciled against broker statements.
- **Defense is polling-based.** Sub-second excursions between ~30 s refresh intervals could be missed. Streaming WebSocket defense is not implemented.
- **Directional sizing falls back to stale data.** If the live option snapshot is unavailable, sizing uses prior close or a $2.00 fallback, which can be far off for 0DTE.
- **NDX proxy.** XND strike selection and settlement use QQQ × 40 as an NDX approximation. The actual ratio fluctuates slightly.
- **PDT guard uses local business-day count.** Does not account for market holidays. The 5-business-day window is based on Mon–Fri, not the exchange calendar.
