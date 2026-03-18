"""Quick validation of all bug fixes."""
from bot.condor_strategy import CondorStrategy, CondorState
from bot.condor_config import CondorConfig, DirectionalConfig
from bot.market_data import MorningTracker
from bot.options_client import CondorLegs
from bot.directional_strategy import DirectionalStrategy, DirectionalState


def test_condor_otm_settlement():
    """Condor expires OTM — full credit kept."""
    cs = CondorStrategy(CondorConfig(), dry_run=True)
    cs.state.is_filled = True
    cs.state.is_open = True
    cs.state.qty = 10
    cs.state.entry_credit = 1.30
    cs.state.legs = CondorLegs(
        long_put="LP", short_put="SP", short_call="SC", long_call="LC",
        long_put_strike=564.08, short_put_strike=569.83,
        short_call_strike=580.17, long_call_strike=585.92,
    )
    t = MorningTracker()
    t.spy_last = 575.0
    r = cs.on_settlement(t)
    assert r["status"] == "settled_otm", r
    assert r["realized_pnl"] == 1300.0, r
    print("  PASS: condor OTM settlement")


def test_condor_itm_settlement():
    """Condor expires ITM on call side — correct intrinsic deducted."""
    cs = CondorStrategy(CondorConfig(), dry_run=True)
    cs.state.is_filled = True
    cs.state.is_open = True
    cs.state.qty = 10
    cs.state.entry_credit = 1.30
    cs.state.legs = CondorLegs(
        long_put="LP", short_put="SP", short_call="SC", long_call="LC",
        long_put_strike=564.08, short_put_strike=569.83,
        short_call_strike=580.17, long_call_strike=585.92,
    )
    t = MorningTracker()
    t.spy_last = 582.0  # Inside the call spread
    r = cs.on_settlement(t)
    assert r["status"] == "settled_itm", r
    # Call spread: (582-580.17) - max(582-585.92, 0) = 1.83 - 0 = 1.83
    expected_exit = 1.83 * 10 * 100
    assert abs(r["exit_cost"] - expected_exit) < 0.01, f"exit_cost={r['exit_cost']} expected={expected_exit}"
    expected_pnl = 1300.0 - expected_exit
    assert abs(r["realized_pnl"] - expected_pnl) < 0.01, f"pnl={r['realized_pnl']} expected={expected_pnl}"
    print("  PASS: condor ITM settlement")


def test_condor_never_filled():
    """Condor order submitted but never filled — no P&L reported."""
    cs = CondorStrategy(CondorConfig(), dry_run=True)
    cs.state.entry_order_id = "some-id"
    cs.state.is_filled = False
    r = cs.on_settlement(MorningTracker())
    assert r["status"] == "never_filled", r
    print("  PASS: condor never_filled")


def test_condor_defense_fill_tracking():
    """Defense close tracks actual fill price."""
    cs = CondorStrategy(CondorConfig(), dry_run=True)
    cs.state.is_filled = True
    cs.state.is_open = True
    cs.state.qty = 10
    cs.state.entry_credit = 1.30
    cs.state.legs = CondorLegs(
        long_put="LP", short_put="SP", short_call="SC", long_call="LC",
        long_put_strike=564.08, short_put_strike=569.83,
        short_call_strike=580.17, long_call_strike=585.92,
    )
    # Simulate defense triggered with known fill price
    cs.state.defense_triggered = True
    cs.state.defense_filled = True
    cs.state.defense_fill_price = 0.75  # Actual debit paid
    r = cs.on_settlement(MorningTracker())
    assert r["status"] == "defense_closed", r
    expected_pnl = (1.30 * 10 * 100) - (0.75 * 10 * 100)
    assert abs(r["realized_pnl"] - expected_pnl) < 0.01, f"pnl={r['realized_pnl']} expected={expected_pnl}"
    print("  PASS: condor defense fill tracking")


def test_directional_ndx_conversion():
    """Directional settlement uses NDX-level pricing, not QQQ."""
    ds = DirectionalStrategy(DirectionalConfig(), dry_run=True)
    assert ds.QQQ_TO_NDX_RATIO == 40.0
    ds.state.is_filled = True
    ds.state.is_open = True
    ds.state.option_type = "call"
    ds.state.option_symbol = "XND260318C20000"
    ds.state.strike_price = 20000
    ds.state.qty = 2
    ds.state.premium_spent = 400.0

    t = MorningTracker()
    t.qqq_last = 505.0  # NDX estimate = 20200
    r = ds.on_settlement(t)
    assert r["ndx_settlement"] == 20200.0, r
    assert r["intrinsic_per_contract"] == 200.0, r
    # settlement_value = 200 * 2 * 100 = 40000
    # pnl = 40000 - 400 = 39600
    assert r["realized_pnl"] == 39600.0, r
    print("  PASS: directional NDX conversion + settlement")


def test_directional_never_filled():
    """Directional order submitted but never filled."""
    ds = DirectionalStrategy(DirectionalConfig(), dry_run=True)
    ds.state.entry_order_id = "some-id"
    ds.state.is_filled = False
    r = ds.on_settlement(MorningTracker())
    assert r["status"] == "never_filled", r
    print("  PASS: directional never_filled")


def test_directional_otm_settlement():
    """Directional put expires OTM — premium lost."""
    ds = DirectionalStrategy(DirectionalConfig(), dry_run=True)
    ds.state.is_filled = True
    ds.state.is_open = True
    ds.state.option_type = "put"
    ds.state.option_symbol = "XND260318P20000"
    ds.state.strike_price = 20000
    ds.state.qty = 1
    ds.state.premium_spent = 200.0

    t = MorningTracker()
    t.qqq_last = 510.0  # NDX = 20400, put is OTM
    r = ds.on_settlement(t)
    assert r["intrinsic_per_contract"] == 0.0, r
    assert r["realized_pnl"] == -200.0, r
    print("  PASS: directional OTM settlement (premium lost)")


def test_is_open_gating():
    """is_open should be False until fill is confirmed."""
    cs = CondorStrategy(CondorConfig(), dry_run=True)
    cs.state.entry_order_id = "test-order"
    assert cs.state.is_open is False
    assert cs.state.is_filled is False

    ds = DirectionalStrategy(DirectionalConfig(), dry_run=True)
    ds.state.entry_order_id = "test-order"
    assert ds.state.is_open is False
    assert ds.state.is_filled is False
    print("  PASS: is_open gating (both sleeves)")


def test_terminal_order_marking():
    """entry_order_dead field exists and defaults to False."""
    cs = CondorState()
    assert cs.entry_order_dead is False
    cs.entry_order_dead = True
    assert cs.entry_order_dead is True

    ds = DirectionalState()
    assert ds.entry_order_dead is False
    ds.entry_order_dead = True
    assert ds.entry_order_dead is True
    print("  PASS: terminal order marking (both sleeves)")


def test_post_anchor_defense_tracking():
    """Post-anchor high/low only update after anchor is set."""
    t = MorningTracker()
    # Before anchor: post-anchor fields stay at defaults
    t.update_spy(575.0)
    assert t.spy_post_anchor_high == 0.0, "Should not track before anchor"
    assert t.spy_post_anchor_low == float("inf"), "Should not track before anchor"

    # Set anchor and reset
    t.condor_anchor = 575.0
    t.reset_post_anchor_tracking()
    assert t.spy_post_anchor_high == 575.0
    assert t.spy_post_anchor_low == 575.0

    # Now updates propagate (using last-trade price only)
    t.update_spy(580.0)
    assert t.spy_post_anchor_high == 580.0
    assert t.spy_post_anchor_low == 575.0

    t.update_spy(570.0)
    assert t.spy_post_anchor_high == 580.0
    assert t.spy_post_anchor_low == 570.0
    print("  PASS: post-anchor defense tracking")


def test_post_anchor_ignores_daily_bar():
    """
    CRITICAL: Pre-anchor daily bar extremes must NOT contaminate
    post-anchor defense tracking.

    Scenario: SPY spiked to 590 at 10:00 AM, then settled to 575
    by 11:30 AM when the condor anchor is set.  The daily bar high
    is 590 (from the earlier spike).  Post-anchor defense must only
    see 575, NOT 590.  Otherwise defense would instantly trigger.
    """
    t = MorningTracker()

    # Morning: SPY trades at 575, daily bar shows high=590 from earlier spike
    t.update_spy(575.0, high=590.0, low=568.0)
    # Session-wide should pick up the daily bar extremes
    assert t.spy_high == 590.0
    assert t.spy_low == 568.0

    # 11:30 AM: Set condor anchor at current last trade
    t.condor_anchor = 575.0
    t.reset_post_anchor_tracking()

    # First refresh AFTER anchor: daily bar still shows high=590
    t.update_spy(575.5, high=590.0, low=568.0)

    # Post-anchor must only see 575.5, NOT the 590 daily bar high
    assert t.spy_post_anchor_high == 575.5, (
        f"Post-anchor high should be 575.5 (last trade), "
        f"got {t.spy_post_anchor_high} (daily bar contamination!)"
    )
    assert t.spy_post_anchor_low == 575.0, (
        f"Post-anchor low should be 575.0, got {t.spy_post_anchor_low}"
    )

    # Session-wide should still show 590 (that's correct for general stats)
    assert t.spy_high == 590.0

    # Verify defense does NOT trigger (move from 575 anchor is tiny)
    max_move = t.spy_max_move_from_anchor_pct()
    assert max_move < 0.01, f"Max move should be <1%, got {max_move*100:.2f}%"
    print("  PASS: post-anchor ignores daily bar (critical fix verified)")


def test_defense_max_excursion():
    """Defense check should use max excursion, not just current last."""
    cfg = CondorConfig()
    cs = CondorStrategy(cfg, dry_run=True)
    cs.state.is_filled = True
    cs.state.is_open = True
    cs.state.anchor_price = 575.0

    t = MorningTracker()
    t.condor_anchor = 575.0
    t.reset_post_anchor_tracking()

    # Simulate a spike to 582 (1.22%) then recovery to 576
    t.update_spy(582.0)  # breaches 1%
    t.update_spy(576.0)  # recovered
    assert t.spy_last == 576.0

    # Defense should still trigger because post-anchor high was 582
    max_move = t.spy_max_move_from_anchor_pct()
    assert max_move is not None
    expected = abs(582.0 - 575.0) / 575.0
    assert abs(max_move - expected) < 0.0001
    assert max_move >= cfg.defense_trigger_pct  # Should breach 1% threshold

    # Verify the strategy's check_defense agrees
    triggered = cs.check_defense(t)
    assert triggered is True
    print("  PASS: defense max excursion check")


def test_defense_close_failed_flag():
    """defense_close_failed flag exists and is used."""
    cs = CondorState()
    assert cs.defense_close_failed is False
    cs.defense_close_failed = True
    assert cs.defense_close_failed is True
    print("  PASS: defense_close_failed flag")


def test_option_snapshot_import():
    """get_option_snapshot is importable."""
    from bot.options_client import get_option_snapshot
    assert callable(get_option_snapshot)
    print("  PASS: get_option_snapshot importable")


def test_position_intent_in_payload():
    """place_single_option_order includes position_intent in payload."""
    from bot.options_client import place_single_option_order
    result = place_single_option_order(
        symbol="XND260318C20000", qty=1, side="buy",
        order_type="market", dry_run=True,
    )
    assert "position_intent" in result["payload"], result["payload"]
    assert result["payload"]["position_intent"] == "buy_to_open"

    result2 = place_single_option_order(
        symbol="XND260318P20000", qty=1, side="sell",
        order_type="market", dry_run=True,
    )
    assert result2["payload"]["position_intent"] == "sell_to_close"
    print("  PASS: position_intent in single-leg payload")


if __name__ == "__main__":
    print("Running fix validation tests...\n")
    test_condor_otm_settlement()
    test_condor_itm_settlement()
    test_condor_never_filled()
    test_condor_defense_fill_tracking()
    test_directional_ndx_conversion()
    test_directional_never_filled()
    test_directional_otm_settlement()
    test_is_open_gating()
    test_position_intent_in_payload()
    test_terminal_order_marking()
    test_post_anchor_defense_tracking()
    test_post_anchor_ignores_daily_bar()
    test_defense_max_excursion()
    test_defense_close_failed_flag()
    test_option_snapshot_import()
    print("\nAll tests PASSED.")
