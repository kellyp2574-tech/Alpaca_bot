"""Unit tests for the audit-pass fixes:

1. FillStream message handling, terminal cache, and waiter notification.
2. Daily-loss circuit breaker logic in _step_execute_entries.
3. Premarket checkpoint dedup.
4. Atomic state writes.
5. Adaptive hot-window detection.
6. Snapshot staleness logging.
7. Symbol filter changes (no len>5 reject).

Run: pytest test_audit_fixes.py -v
"""
import json
import os
import threading
import time
from datetime import time as dt_time, timezone, datetime
from unittest.mock import MagicMock

import pytest


# ──────────────────────────────────────────────────────────────
# 1. FillStream
# ──────────────────────────────────────────────────────────────

def test_fill_stream_handles_terminal_fill_message():
    from bot.fill_stream import FillStream

    fs = FillStream("https://paper-api.alpaca.markets", "k", "s")
    fs._handle_message(json.dumps({
        "stream": "trade_updates",
        "data": {
            "event": "fill",
            "order": {
                "id": "ORD1",
                "status": "filled",
                "filled_qty": "100",
                "filled_avg_price": "5.43",
            },
        },
    }))
    evt = fs.get_terminal_event("ORD1")
    assert evt is not None
    assert evt["status"] == "filled"
    assert evt["filled_qty"] == 100.0
    assert evt["filled_avg_price"] == 5.43


def test_fill_stream_ignores_non_trade_updates_streams():
    from bot.fill_stream import FillStream

    fs = FillStream("https://paper-api.alpaca.markets", "k", "s")
    fs._handle_message(json.dumps({
        "stream": "account_updates",
        "data": {"event": "fill", "order": {"id": "X", "status": "filled"}},
    }))
    assert fs.get_terminal_event("X") is None


def test_fill_stream_partial_fill_not_terminal():
    from bot.fill_stream import FillStream

    fs = FillStream("https://paper-api.alpaca.markets", "k", "s")
    fs._handle_message(json.dumps({
        "stream": "trade_updates",
        "data": {
            "event": "partial_fill",
            "order": {"id": "P1", "status": "partially_filled",
                       "filled_qty": "50", "filled_avg_price": "5.00"},
        },
    }))
    # partial_fill is NOT in _TERMINAL_EVENTS
    assert fs.get_terminal_event("P1") is None


def test_fill_stream_canceled_terminal_no_fill():
    from bot.fill_stream import FillStream

    fs = FillStream("https://paper-api.alpaca.markets", "k", "s")
    fs._handle_message(json.dumps({
        "stream": "trade_updates",
        "data": {
            "event": "canceled",
            "order": {"id": "C1", "status": "canceled",
                       "filled_qty": "0", "filled_avg_price": None},
        },
    }))
    evt = fs.get_terminal_event("C1")
    assert evt is not None
    assert evt["status"] == "canceled"
    assert evt["filled_qty"] == 0.0


def test_fill_stream_wait_for_terminal_returns_when_event_arrives():
    from bot.fill_stream import FillStream

    fs = FillStream("https://paper-api.alpaca.markets", "k", "s")
    result = []

    def waiter():
        evt = fs.wait_for_terminal_event("ORD2", timeout=2.0)
        result.append(evt)

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.05)  # ensure waiter is parked on the condvar

    fs._handle_message(json.dumps({
        "stream": "trade_updates",
        "data": {
            "event": "fill",
            "order": {"id": "ORD2", "status": "filled",
                       "filled_qty": "10", "filled_avg_price": "1.23"},
        },
    }))
    t.join(timeout=3.0)
    assert result and result[0] is not None
    assert result[0]["filled_qty"] == 10.0


def test_fill_stream_wait_for_terminal_times_out():
    from bot.fill_stream import FillStream

    fs = FillStream("https://paper-api.alpaca.markets", "k", "s")
    start = time.monotonic()
    evt = fs.wait_for_terminal_event("NEVER", timeout=0.2)
    elapsed = time.monotonic() - start
    assert evt is None
    assert 0.15 < elapsed < 0.6


# ──────────────────────────────────────────────────────────────
# 2. get_order_fill stream short-circuit
# ──────────────────────────────────────────────────────────────

def test_get_order_fill_short_circuits_on_cached_terminal():
    from bot.fill_stream import FillStream
    from bot.position_manager_overnight import PositionManager

    pm = PositionManager.__new__(PositionManager)
    pm.fill_stream = FillStream("https://paper-api.alpaca.markets", "k", "s")
    pm.fill_stream._handle_message(json.dumps({
        "stream": "trade_updates",
        "data": {
            "event": "fill",
            "order": {"id": "ORD3", "status": "filled",
                       "filled_qty": "25", "filled_avg_price": "9.99"},
        },
    }))
    pm.base_url = "https://x"
    pm.session = MagicMock()

    fill = pm.get_order_fill("ORD3", max_wait=10)
    assert fill is not None
    assert fill["filled_qty"] == 25.0
    assert fill["filled_avg_price"] == 9.99
    assert fill["status"] == "filled"
    # Most importantly: we never hit REST.
    assert pm.session.get.call_count == 0


def test_get_order_fill_returns_none_on_cached_canceled_no_fill():
    from bot.fill_stream import FillStream
    from bot.position_manager_overnight import PositionManager

    pm = PositionManager.__new__(PositionManager)
    pm.fill_stream = FillStream("https://paper-api.alpaca.markets", "k", "s")
    pm.fill_stream._handle_message(json.dumps({
        "stream": "trade_updates",
        "data": {
            "event": "canceled",
            "order": {"id": "ORD4", "status": "canceled",
                       "filled_qty": "0", "filled_avg_price": None},
        },
    }))
    pm.base_url = "https://x"
    pm.session = MagicMock()

    fill = pm.get_order_fill("ORD4", max_wait=10)
    assert fill is None
    assert pm.session.get.call_count == 0


# ──────────────────────────────────────────────────────────────
# 3. Daily-loss circuit breaker (logic check via direct calc)
# ──────────────────────────────────────────────────────────────

def test_daily_loss_pct_calculation():
    """Integration is exercised inside _step_execute_entries; here we just
    verify the math used to gate it."""
    last_equity = 100_000.0
    equity = 94_500.0
    day_ret = (equity - last_equity) / last_equity
    assert pytest.approx(day_ret, abs=1e-6) == -0.055
    assert day_ret <= -0.05  # would trip the breaker


def test_daily_loss_disabled_when_zero():
    from bot import config
    # Disabled is just `loss_limit > 0` check; verify default is enabled.
    assert config.DAILY_LOSS_LIMIT_PCT > 0


# ──────────────────────────────────────────────────────────────
# 4. Atomic state writes
# ──────────────────────────────────────────────────────────────

def test_atomic_write_creates_no_tmp_leftovers(tmp_path):
    from bot.state_manager import _atomic_write_json

    target = str(tmp_path / "state.json")
    _atomic_write_json(target, {"a": 1, "b": [2, 3]})
    with open(target) as f:
        assert json.load(f) == {"a": 1, "b": [2, 3]}
    leftovers = [p for p in os.listdir(tmp_path) if p.startswith(".tmp_")]
    assert leftovers == []


def test_atomic_write_overwrites_existing(tmp_path):
    from bot.state_manager import _atomic_write_json

    target = str(tmp_path / "state.json")
    _atomic_write_json(target, {"v": 1})
    _atomic_write_json(target, {"v": 2})
    with open(target) as f:
        assert json.load(f)["v"] == 2


# ──────────────────────────────────────────────────────────────
# 5. Adaptive hot-window detection
# ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("h,m,expected", [
    (5, 0, True),    # premarket start
    (5, 30, True),   # mid-premarket
    (6, 1, True),    # final premarket trip
    (6, 5, False),   # after premarket window
    (9, 24, True),   # cancel-orders entrance
    (9, 30, True),   # market open exit batch
    (10, 4, True),   # failsafe window edge (inclusive)
    (10, 5, False),  # after failsafe
    (12, 0, False),  # midday quiet
    (15, 29, True),  # data-collection start
    (15, 50, True),  # entry execution
    (16, 0, True),   # EOD reports edge (inclusive of 16:00)
    (16, 1, False),  # past close
])
def test_hot_window_table(h, m, expected):
    from bot.integrated_main import _is_hot_window
    assert _is_hot_window(dt_time(h, m)) is expected


# ──────────────────────────────────────────────────────────────
# 6. Snapshot staleness logging on parse failure
# ──────────────────────────────────────────────────────────────

def test_filter_execution_ready_logs_unparseable_timestamp(caplog):
    import logging
    from bot.universe_builder import filter_execution_ready

    snaps = {
        "AAA": {
            "last_price": 5.0,
            "bid": 4.99,
            "ask": 5.01,
            "timestamp": "this-is-not-iso-format",
        },
    }
    with caplog.at_level(logging.WARNING, logger="bot.universe_builder"):
        orderable, rejected = filter_execution_ready(
            ["AAA"], snaps, max_spread_pct=0.05, require_quote=True,
        )
    # Symbol still passes (we don't reject on parse failure), but a warning is logged.
    assert "AAA" in orderable
    assert "AAA" not in rejected
    assert any("cannot parse snapshot timestamp" in r.message for r in caplog.records)


def test_filter_execution_ready_rejects_stale_quote():
    from bot.universe_builder import filter_execution_ready
    # 2 hours old in ISO UTC
    old_ts = "2020-01-01T00:00:00+00:00"
    snaps = {
        "AAA": {"last_price": 5.0, "bid": 4.99, "ask": 5.01, "timestamp": old_ts},
    }
    orderable, rejected = filter_execution_ready(
        ["AAA"], snaps, max_spread_pct=0.05, max_stale_seconds=300,
    )
    assert orderable == []
    assert rejected.get("AAA", "").startswith("stale_quote_")


# ──────────────────────────────────────────────────────────────
# 7. Symbol filter — no len>5 reject
# ──────────────────────────────────────────────────────────────

def test_filter_asset_type_accepts_six_letter_common_stock():
    from bot.universe_builder import filter_asset_type, UniverseDiagnostics

    assets = [
        {"symbol": "ABCDEF", "class": "us_equity", "exchange": "NASDAQ",
         "status": "active", "tradable": True, "name": "Some Common Co"},
        {"symbol": "BRK.B", "class": "us_equity", "exchange": "NYSE",
         "status": "active", "tradable": True, "name": "Berkshire B"},
        {"symbol": "BAC-L", "class": "us_equity", "exchange": "NYSE",
         "status": "active", "tradable": True, "name": "BAC Pref L"},
        {"symbol": "AAPL", "class": "us_equity", "exchange": "NASDAQ",
         "status": "active", "tradable": True, "name": "Apple Inc"},
    ]
    diag = UniverseDiagnostics()
    passed = filter_asset_type(assets, diag)
    assert "ABCDEF" in passed   # 6-letter common stock now allowed
    assert "AAPL" in passed
    assert "BRK.B" not in passed   # still rejected (dot)
    assert "BAC-L" not in passed   # still rejected (dash)


# ──────────────────────────────────────────────────────────────
# 8. Premarket checkpoint dedup (set semantics)
# ──────────────────────────────────────────────────────────────

def test_premarket_checkpoint_dedup_set_membership():
    """The fix is a simple ``set in`` check; just sanity-test the data
    structure used by integrated_main has the expected behaviour."""
    done = set()
    cp = "05:30"
    if cp not in done:
        done.add(cp)
    # Second tick at the same second/minute must skip.
    assert cp in done
    # Different checkpoint slot still runs.
    cp2 = "05:45"
    assert cp2 not in done


def _resolve_premarket_checkpoint(now_t: dt_time):
    """Replicates the integrated_main checkpoint-resolution logic so we can
    test the 06:00-final-runs-once invariant without spinning up the bot."""
    start = dt_time(5, 0)
    final = dt_time(6, 0)
    cutoff = dt_time(final.hour, final.minute + 2)
    interval = 15

    if now_t < start or now_t >= cutoff:
        return None  # outside the active window
    minutes_since_start = (now_t.hour * 60 + now_t.minute) - (start.hour * 60 + start.minute)
    cp_num = minutes_since_start // interval
    cp_minutes = (start.hour * 60 + start.minute) + cp_num * interval
    final_minutes = final.hour * 60 + final.minute
    cp_minutes = min(cp_minutes, final_minutes)
    return dt_time(cp_minutes // 60, cp_minutes % 60).strftime("%H:%M")


@pytest.mark.parametrize("h,m,expected", [
    (4, 59, None),    # before window
    (5, 0,  "05:00"),
    (5, 14, "05:00"),
    (5, 15, "05:15"),
    (5, 44, "05:30"),
    (5, 45, "05:45"),
    (5, 59, "05:45"),
    (6, 0,  "06:00"),  # *** the bug-fix case: 06:00 must resolve to 06:00 ***
    (6, 1,  "06:00"),  # ticks during the cutoff still hit the final slot
    (6, 2,  None),     # past cutoff
    (6, 30, None),
])
def test_premarket_checkpoint_resolution_includes_final(h, m, expected):
    assert _resolve_premarket_checkpoint(dt_time(h, m)) == expected


def test_validate_config_rejects_both_exit_modes_enabled(monkeypatch):
    """The new guard should fail-fast if fast-exit and red-trail are both on."""
    from bot.integrated_main import CombinedOvernightReboundBot
    from bot import config

    monkeypatch.setattr(config, "ENABLE_FAST_OPEN_MARKET_EXIT", True)
    monkeypatch.setattr(config, "ENABLE_RED_OPEN_TRAIL_EXIT", True)

    bot = CombinedOvernightReboundBot.__new__(CombinedOvernightReboundBot)
    with pytest.raises(ValueError, match="mutually exclusive"):
        bot._validate_config()


def test_default_config_has_red_trail_disabled():
    """Default production config should have fast-exit on, red-trail off."""
    from bot import config
    assert config.ENABLE_FAST_OPEN_MARKET_EXIT is True
    assert config.ENABLE_RED_OPEN_TRAIL_EXIT is False
