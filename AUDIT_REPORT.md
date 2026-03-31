# Full Bot Audit Report — Gap Momentum Bot
**Date:** March 30, 2026  
**Scope:** `integrated_main.py` → all called modules, traced as a 9:00 AM cold start through EOD

---

## Daily Timeline (Normal Flow)

| Time | What Happens | Files Involved |
|------|-------------|----------------|
| **9:00:00** | `run.py` → `main()` → `GapMomentumBot.__init__()` → `bot.run()` | `run.py`, `integrated_main.py` |
| **9:00:00** | `_load_state()`: load positions.json, bot_state.json, pre_trade_state.json, reconcile broker | `integrated_main.py:569-671`, `state_manager.py` |
| **9:00:00** | Startup guards: check time, disable entries if in dead zones | `integrated_main.py:93-109` |
| **9:00:00** | **Main loop begins** (`while True`, 1s sleep) | `integrated_main.py:112-166` |
| **9:00–9:25** | **Step 1**: `_step1_build_universe()` — Massive API snapshot, filter $0.50–$5.00 | `integrated_main.py:168-249`, `massive_client.py` |
| **9:25–9:27** | **Step 2**: `_step2_find_candidates()` — Refresh Massive, compute gaps, split core/filler, fetch VIX | `integrated_main.py:250-312`, `gap_calculator.py`, `vix_fetcher.py` |
| **9:27–9:30:10** | **Step 3 Phase 1**: Build entry plans (core + filler), submit MOO orders | `integrated_main.py:413-513`, `position_manager.py:517-608` |
| **9:30:10–9:30:30** | **Step 3 Phase 2**: Reconcile MOO fills, refresh prices, rescue pass 1 (marketable limits) | `integrated_main.py:515-524`, `position_manager.py:610-861` |
| **9:30:30** | **Step 3 Phase 3**: Rescue pass 2, finalize positions → `stage_entry_done = True` | `integrated_main.py:527-538`, `position_manager.py:862-898` |
| **9:30:31–15:29** | **Step 4**: `_step4_manage_exits()` every 1s — broker count check, price update, trailing stop + timed slicer | `integrated_main.py:543-567`, `position_manager.py:900-1060` |
| **15:30:00** | **Failsafe 1**: `_run_failsafe_flatten("3:30 PM")` — cancel all orders, market sell all broker positions | `integrated_main.py:140-142`, `position_manager.py:1238-1320` |
| **15:45:00** | **Failsafe 2**: `_run_failsafe_flatten("3:45 PM")` | `integrated_main.py:144-145` |
| **15:58:00** | **Failsafe 3**: `_run_failsafe_flatten("3:58 PM")` | `integrated_main.py:148-150` |
| **16:00:00** | Day complete check → `_finalize_day()` → clear state → exit | `integrated_main.py:161-164, 794-832` |

---

## CRITICAL BUGS (Will Crash or Infinite-Loop)

### BUG 1: Infinite loop + order spam after 16:00 if positions remain
**Location:** `integrated_main.py:152-158`  
**Severity:** 🔴 CRITICAL

After 16:00 PM, if `stage_exit_done` is still `False` (any position wasn't closed), the main loop hits this branch **every 1 second**:

```python
# Line 153-158
if self.stage_entry_done and not self.stage_exit_done:
    if current_time < self.market_close:
        self._step4_manage_exits(current_time)
    else:
        logger.warning("Market close reached - running final broker-based flatten")
        self._run_failsafe_flatten("4:00 PM market-close failsafe")
```

`_run_failsafe_flatten()` cancels all open orders, then submits new `time_in_force: "day"` market sell orders. **After 4:00 PM, day orders are rejected by Alpaca** (market closed). So:
1. Flatten submits orders → rejected → no fills
2. `broker_position_count()` still > 0 → `stage_exit_done` stays `False`
3. Loop repeats next second → infinite order-spam loop
4. Bot never reaches `_finalize_day()`, runs forever

**Fix needed:** Add a max-retry counter or deadline after 16:00. After N failed attempts (or 16:05), log CRITICAL, set `stage_exit_done = True`, and proceed to finalize. Or use `time_in_force: "gtc"` for post-close orders, or use the close-position DELETE endpoint.

---

### BUG 2: State recovery crashes — `PositionState.get()` doesn't exist
**Location:** `state_manager.py:88` → `position_manager.py:85-103`  
**Severity:** 🔴 CRITICAL (on any restart with saved positions)

`StateManager.load_positions()` returns `Dict[str, PositionState]` (dataclass objects):
```python
# state_manager.py:88
positions[symbol] = PositionState(**pos_data)
```

But `PositionManager.load_positions()` calls `.get()` on each value, which dataclasses don't have:
```python
# position_manager.py:87-88
position = Position(
    symbol=data.get("symbol", symbol),  # ← AttributeError: 'PositionState' has no method 'get'
```

**Every bot restart with saved positions will crash** with `AttributeError`.

**Fix:** Either have `StateManager.load_positions()` return raw dicts, or change `PositionManager.load_positions()` to use `getattr(data, "symbol", symbol)` etc.

---

### BUG 3: Division by zero in `update_positions()` if entry_price is 0
**Location:** `position_manager.py:925`  
**Severity:** 🔴 HIGH (crashes exit management loop)

```python
gain_pct = (current_price - entry_price) / entry_price
```

`reconcile_local_positions_from_broker()` (line 1335) can set `entry_price = 0.0`:
```python
avg_entry = float(pos.get("avg_entry_price", 0) or 0)
```

If Alpaca returns `avg_entry_price: null` or `"0"`, entry_price = 0.0, and `update_positions()` crashes with `ZeroDivisionError`. Since `update_positions()` is called every second in Step 4, **this kills the entire exit management loop**.

**Fix:** Guard: `if entry_price > 0:` before the division.

---

## SIGNIFICANT BUGS (Incorrect Behavior)

### BUG 4: VIX is NEVER fetched — exit regime is always "middle"
**Location:** `vix_fetcher.py:31`  
**Severity:** 🟡 HIGH (entire VIX-based exit system is dead)

```python
url = f"{self.data_url}/v2/stocks/VIX/snapshot"
```

VIX is an **index**, not a stock. Alpaca doesn't serve VIX via the stocks endpoint. This always returns a non-200 response, falling through to the default `return 15.0`.

**Consequence:** `self.vix_level` is always 15.0 (middle regime). The low-VIX early exit (2:30 PM) and high-VIX behavior never trigger. Trailing stops always activate (middle regime = 12–22, and 15.0 is in range).

**Fix:** Use `yfinance` for VIX (like the condor bot does), or Alpaca's indices endpoint if available.

---

### BUG 5: VIX is never refreshed after 9:25 AM
**Location:** `integrated_main.py:300`  
**Severity:** 🟡 MEDIUM

VIX is fetched exactly once in Step 2 (`self.vix_level = self.vix_fetcher.get_vix_level() or 15.0`). It's never updated again. Even if Bug 4 were fixed, the exit regime would use a 6-hour-stale VIX reading.

**Fix:** Refresh VIX periodically (e.g., every 15 minutes) during exit management.

---

### BUG 6: 4:00 PM failsafe flatten + timed exit slicer overlap at 15:30
**Location:** `integrated_main.py:140-142` vs `position_manager.py:966-980`  
**Severity:** 🟡 MEDIUM

For middle/high VIX (target exit = 15:30), the timed exit slicer starts in the 15:29–15:31 window. At 15:30:00, the first failsafe **also** fires. The failsafe calls `cancel_all_open_orders()`, which kills any slicer sell orders in flight, then resubmits its own market sells for the full broker position. This can cause:
- Double sell attempts on the same shares
- Slicer state becomes inconsistent (it thinks it has remaining slices but broker is flat)

**Fix:** Skip the 15:30 failsafe if slicer is actively running, or cancel slicers before running failsafe.

---

### BUG 7: `_step4_manage_exits` calls `_run_failsafe_flatten` every second on broker/local mismatch
**Location:** `integrated_main.py:553-559`  
**Severity:** 🟡 MEDIUM

When `local_count == 0` but `broker_count > 0`, the code runs `_run_failsafe_flatten()` and returns. Next second, same condition, same flatten. Each flatten cancels all orders then resubmits — creating a cancel→submit→cancel→submit cycle every second until broker reports flat.

**Fix:** Add a flag or cooldown (e.g., only run mismatch flatten once per 30 seconds).

---

### BUG 8: `massive_client.py` shadows variables silently
**Location:** `massive_client.py:53-54 → 63-64`  
**Severity:** 🟢 LOW (currently harmless but confusing)

```python
# Lines 53-54 (first declaration)
day_data = item.get("day", {})
prev_day = item.get("prevDay", {})

# Lines 63-64 (re-declared identically)
day_data = item.get("day", {})
prev_day = item.get("prevDay", {})
```

Redundant re-declarations. Delete lines 63-64.

---

## INFINITE LOOP / HANG RISKS

### RISK 1: `get_order_fill()` blocks main loop for up to 5 minutes
**Location:** `position_manager.py:390`  
**Severity:** 🟡 MEDIUM

MOO fill polling uses `max_wait=300` (5 minutes). During this time, the entire bot is blocked — no exit checks, no failsafes, no state saves. If Alpaca is slow or the order is stuck in `partially_filled`, the bot is unresponsive for 5 minutes.

This happens during Phase 2 (MOO reconciliation at 9:30:10) when `get_order_fill()` is called per-symbol sequentially. With 100 positions, worst case is 100 × 5 min = 500 min blockage (won't happen in practice, but 10 slow fills = 50 min).

---

### RISK 2: Exit slicer retries indefinitely on order failure
**Location:** `position_manager.py:1009-1060`  
**Severity:** 🟡 MEDIUM

If `_execute_exit_slice()` fails (order rejected, no fill), `slices_remaining` is NOT decremented. The slicer retries the same slice every second. It's only saved by the failsafe flattens at 15:30/15:45/15:58. But between the exit window (e.g., 14:30 for low VIX) and 15:30, that's **60 minutes of retry spam**.

**Fix:** Add max retries per slice, or mark slicer as failed after N attempts.

---

### RISK 3: Step 2 retries with sleep(5) if universe is empty
**Location:** `integrated_main.py:254-257`  
**Severity:** 🟢 LOW (bounded by time)

If `self.universe` is empty, Step 2 sleeps 5s and returns without setting `stage_candidates_done`. It retries until 9:27 when the time guard skips it. Not infinite, but wastes the entire 9:25–9:27 window on futile retries.

---

## PERFORMANCE ISSUES

### PERF 1: `broker_position_count()` makes an API call every 1 second
**Location:** `position_manager.py:1234-1236` called from `integrated_main.py:547`  
**Impact:** ~1 unnecessary API call/second for the entire trading day (6.5 hours = ~23,400 calls)

`_step4_manage_exits()` calls `broker_position_count()` at the top, which calls `get_broker_positions()` (HTTP GET). Combined with `update_positions()` (another API call), that's **2+ API calls per second** during exit management.

**Fix:** Cache broker position count with a TTL (e.g., 30 seconds). Or only check broker count every N iterations.

---

### PERF 2: `_save_state()` writes 3 JSON files every 1 second
**Location:** `integrated_main.py:567, 672-716`  
**Impact:** 3 file writes/second during exit management

`_step4_manage_exits()` calls `_save_state()` every iteration. Each call writes:
1. `positions.json`
2. `bot_state.json`  
3. `pre_trade_state.json` (includes full Massive snapshots — potentially large)

This is excessive I/O. State only changes when a position is exited.

**Fix:** Only call `_save_state()` when `exited` is non-empty, or use a dirty flag.

---

### PERF 3: Redundant API calls in `build_entry_plans()` and `enter_positions_moo()`
**Location:** `position_manager.py:295-311, 522-537`  
**Impact:** 2 wasted API calls per method invocation (4-6 total during entry)

Both methods call `get_account_equity()` (1 API call) AND then immediately call the account endpoint again for `buying_power` (2nd API call). The equity call is never used when `capital_override` is provided.

**Fix:** Skip equity/buying_power fetch when `capital_override` is provided. Combine the two calls into one.

---

### PERF 4: Time string parsing every loop iteration
**Location:** `integrated_main.py:416-417`  

```python
t1 = datetime.strptime(config.POST_OPEN_ENTRY_TIME_1, "%H:%M:%S").time()
t2 = datetime.strptime(config.POST_OPEN_ENTRY_TIME_2, "%H:%M:%S").time()
```

Parsed every second during the staged entry window. Should be parsed once in `__init__` or at class level.

---

### PERF 5: Massive client logs at INFO for every symbol missing `day.o`
**Location:** `massive_client.py:72-75`  
**Impact:** Thousands of INFO log lines per snapshot call

Before market open, most symbols won't have `day.o` populated. Each one logs:
```
INFO: AAPL: using last_trade as open proxy (no day.o yet)
```

With 4,000+ symbols, this floods the log.

**Fix:** Change to DEBUG, or count and log summary ("N symbols used last_trade as open proxy").

---

### PERF 6: Hardcoded diagnostic symbols in `market_data.py`
**Location:** `market_data.py:89`

```python
if symbol in ["SNAP", "FLY", "AAPL", "TSLA"] or open_price is None or prev_close is None:
```

Logs full snapshot diagnostics for these symbols on every `get_snapshots()` call. Left-over debug code.

**Fix:** Remove the hardcoded symbol list; keep only the `or open_price is None` condition at DEBUG level.

---

## SIMPLIFICATION OPPORTUNITIES

### SIMPLIFY 1: `state_manager.py` fragile path derivation
**Location:** `state_manager.py:102, 108, 117`

```python
bot_state_file = self.positions_file.replace("positions.json", "bot_state.json")
```

If `positions_file` ever changes, this silently breaks. Use `config.STATE_FILE` directly or derive from `STATE_DIR`:
```python
bot_state_file = os.path.join(self.state_dir, "bot_state.json")
```

---

### SIMPLIFY 2: `select_by_liquidity_and_gap()` called twice with full MAX_POSITIONS
**Location:** `integrated_main.py:287-292`

Core and filler are each filtered to `max_positions=100`, then combined to up to 200 candidates. But position sizing caps at 100 total. The filler filtering should use `remaining_slots` not `MAX_POSITIONS`.

---

### SIMPLIFY 3: Consolidate account API calls
**Location:** `position_manager.py`

`get_account_equity()`, `get_total_capital()`, and inline account fetches in `enter_positions_moo()` / `build_entry_plans()` all hit the same `/v2/account` endpoint. Could be a single `_get_account()` method with short TTL caching.

---

### SIMPLIFY 4: `_run_failsafe_flatten` post-16:00 should use close-position endpoint
**Location:** `position_manager.py:1280`

Instead of submitting market day orders (which fail after close), use Alpaca's `DELETE /v2/positions/{symbol}` endpoint which works regardless of market hours for closing positions.

---

### SIMPLIFY 5: Bare `except:` in state_manager
**Location:** `state_manager.py:141`

```python
except:
    logs = []
```

Should be `except (json.JSONDecodeError, IOError):` at minimum. Bare except catches KeyboardInterrupt and SystemExit.

---

## EDGE CASES & RISKS

| # | Edge Case | Impact | Location |
|---|-----------|--------|----------|
| E1 | Bot started between 9:27:30 and 9:30 with no state | Entry disabled for the day (by design) | `integrated_main.py:101-103` |
| E2 | Massive API down for entire 9:00–9:25 window | Alpaca fallback builds universe; if that also fails, empty universe → no candidates → no trades | `integrated_main.py:168-249` |
| E3 | Step 2 Massive refresh fails at 9:25 | Retries until 9:27, then skipped. No candidates = no trades for the day | `integrated_main.py:262-266` |
| E4 | All MOO orders get 0 fills | Rescue passes attempt to fill remaining. If those also fail, `finalize_entry_positions()` creates no positions. `stage_entry_done` is set but no positions to exit | `position_manager.py:862-898` |
| E5 | Partial exception in Phase 1 (some MOO orders submitted, then crash) | Rollback logic checks if any orders were submitted. If yes, keeps lock. Correct but orders may be orphaned on next restart | `integrated_main.py:494-511` |
| E6 | KeyboardInterrupt during MOO fill polling (9:30–9:35) | MOO orders are live at broker but local state incomplete. Next restart: `_load_state` cancels orphaned OPG orders (but these are already filled at this point, so cancel returns 422) | `integrated_main.py:839-854` |
| E7 | Non-atomic state file writes | If crash between writing positions.json and bot_state.json, state is inconsistent on next load | `integrated_main.py:672-716` |
| E8 | Alpaca returns stale broker positions after flatten | `broker_position_count() > 0` even though orders filled → failsafe re-triggers | `integrated_main.py:786` |
| E9 | Exit slicer `slices_remaining` reaches 0 but position still has shares | Won't happen: last slice always sells `position.quantity` (all remaining) | `position_manager.py:1019-1020` |

---

## SUMMARY OF FINDINGS

| Category | Count | Severity |
|----------|-------|----------|
| **Critical Bugs** | 3 | Will crash or infinite-loop |
| **Significant Bugs** | 5 | Incorrect behavior |
| **Infinite Loop Risks** | 3 | Potential hangs |
| **Performance Issues** | 6 | Wasteful but functional |
| **Simplifications** | 5 | Code quality |
| **Edge Cases** | 9 | Risk awareness |

### Top 3 Priorities
1. **BUG 1** — Post-16:00 infinite loop + order spam (will run forever if any position isn't closed)
2. **BUG 2** — State recovery crash on restart (blocks all recovery scenarios)
3. **BUG 3** — Division by zero in exit management (kills exit loop for reconciled positions)
