"""
Main Bot Orchestrator — Runs daily near market close.
Checks all strategy signals, executes trades, manages state.

Usage:
    python -m bot.main              # Run once (for scheduler/cron)
    python -m bot.main --dry-run    # Show signals without trading
    python -m bot.main --status     # Show current state and positions
"""
import sys
import logging
import os
from datetime import datetime

from bot import config
from bot.state_manager import load_state, save_state, log_trade
from bot import alpaca_client as broker
from bot import data
from bot import strategies

# ═══════════════════════════════════════════════════
# Logging setup
# ═══════════════════════════════════════════════════
os.makedirs(config.LOG_DIR, exist_ok=True)

logger = logging.getLogger("bot")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def show_status():
    """Display current bot state and Alpaca positions."""
    state = load_state()
    print("\n" + "=" * 60)
    print("  BOT STATUS")
    print("=" * 60)
    print(f"  Last run:      {state.get('last_run', 'never')}")
    print(f"  MA holding:    {state.get('ma_holding', 'None')}")
    print(f"  MA counters:   QQQ(+{state.get('ma_qa',0)}/-{state.get('ma_qb',0)}) "
          f"TLT(+{state.get('ma_ta',0)}/-{state.get('ma_tb',0)})")
    print(f"  Dip active:    {state.get('dip_active', False)}")
    if state.get("dip_active"):
        print(f"    Source:      {state.get('dip_source')}")
        print(f"    Entry:       ${state.get('dip_entry_price', 0):.2f}")
        print(f"    Buy date:    {state.get('dip_buy_date')}")
        print(f"    Days held:   {state.get('dip_days_held', 0)}")
        print(f"    Exit mode:   {state.get('dip_exit_mode')}")

    try:
        account = broker.get_account()
        print(f"\n  ALPACA ACCOUNT")
        print(f"  Equity:        ${float(account.equity):,.2f}")
        print(f"  Cash:          ${float(account.cash):,.2f}")
        print(f"  Buying power:  ${float(account.buying_power):,.2f}")

        positions = broker.get_all_positions()
        if positions:
            print(f"\n  POSITIONS")
            for pos in positions:
                pnl = float(pos.unrealized_pl)
                pnl_pct = float(pos.unrealized_plpc) * 100
                print(f"    {pos.symbol:<6} {float(pos.qty):>10.4f} shares  "
                      f"${float(pos.market_value):>10,.2f}  "
                      f"P&L: {pnl:>+8,.2f} ({pnl_pct:>+5.1f}%)")
        else:
            print(f"\n  No open positions")
    except Exception as e:
        print(f"\n  Could not fetch Alpaca data: {e}")

    # Recent trades
    history = state.get("trade_history", [])
    if history:
        print(f"\n  RECENT TRADES (last 10)")
        for t in history[-10:]:
            print(f"    {t['timestamp']}  {t['action']:<6} {t['ticker']:<6} "
                  f"qty={t.get('qty','?')} @ ${t.get('price',0):.2f}  {t.get('reason','')}")

    print("=" * 60)


def bootstrap_ma_counters(state, qqq_closes, tlt_closes):
    """On first run, compute MA confirmation counters from historical data
    so the bot can immediately enter positions instead of waiting days."""
    if state.get("ma_bootstrapped"):
        return  # Already bootstrapped

    logger.info("First run — bootstrapping MA counters from history...")
    period = config.MA_PERIOD
    buf = config.MA_BUFFER_PCT
    qa, qb, ta, tb = 0, 0, 0, 0

    for i in range(period, len(qqq_closes)):
        qqq_sma = sum(qqq_closes[i - period + 1:i + 1]) / period
        if qqq_closes[i] > qqq_sma * (1 + buf):
            qa += 1; qb = 0
        elif qqq_closes[i] < qqq_sma * (1 - buf):
            qb += 1; qa = 0

    for i in range(period, len(tlt_closes)):
        tlt_sma = sum(tlt_closes[i - period + 1:i + 1]) / period
        if tlt_closes[i] > tlt_sma * (1 + buf):
            ta += 1; tb = 0
        elif tlt_closes[i] < tlt_sma * (1 - buf):
            tb += 1; ta = 0

    state["ma_qa"] = qa
    state["ma_qb"] = qb
    state["ma_ta"] = ta
    state["ma_tb"] = tb
    state["ma_bootstrapped"] = True
    logger.info(f"Bootstrapped: qa={qa} qb={qb} ta={ta} tb={tb}")


def _fetch_common_data():
    """Fetch account + market data used by multiple modes. Returns None on failure."""
    try:
        equity = broker.get_equity()
        cash = broker.get_cash()
        logger.info(f"Account: equity=${equity:,.2f} cash=${cash:,.2f}")
    except Exception as e:
        logger.error(f"Could not fetch account: {e}")
        return None

    try:
        logger.info("Fetching market data...")
        all_bars = data.fetch_daily_bars(config.ALL_TICKERS, lookback_days=150)

        ctx = {
            "equity": equity, "cash": cash,
            "spy_dates":  all_bars.get("SPY", {}).get("dates", []),
            "spy_closes": all_bars.get("SPY", {}).get("closes", []),
            "spy_opens":  all_bars.get("SPY", {}).get("opens", []),
            "qqq_closes": all_bars.get("QQQ", {}).get("closes", []),
            "tlt_closes": all_bars.get("TLT", {}).get("closes", []),
            "upro_closes": all_bars.get("UPRO", {}).get("closes", []),
        }
        ctx["upro_price"] = ctx["upro_closes"][-1] if ctx["upro_closes"] else 0

        if not ctx["spy_closes"] or not ctx["qqq_closes"] or not ctx["tlt_closes"]:
            logger.error("Missing critical price data -- aborting")
            for t in ["SPY", "QQQ", "TLT", "UPRO"]:
                n = len(all_bars.get(t, {}).get("closes", []))
                logger.error(f"  {t}: {n} bars")
            return None

        logger.info(
            f"Daily bars: SPY={ctx['spy_closes'][-1]:.2f} QQQ={ctx['qqq_closes'][-1]:.2f} "
            f"TLT={ctx['tlt_closes'][-1]:.2f} UPRO={ctx['upro_price']:.2f} "
            f"({len(ctx['spy_closes'])} bars, may be stale at 3:30)"
        )

        # Fetch live prices — daily bars at 3:30 PM are yesterday's close
        try:
            live_tickers = ["SPY", "UPRO", config.MA_TRADE_GROWTH,
                            config.MA_TRADE_SAFE, config.MA_TRADE_ALT]
            live = data.fetch_live_prices(list(set(live_tickers)))
            ctx["spy_live"] = float(live["SPY"]) if live.get("SPY") else None
            ctx["upro_live"] = float(live["UPRO"]) if live.get("UPRO") else None
            ctx["live_prices"] = {k: float(v) for k, v in live.items() if v}
            if ctx["spy_live"]:
                logger.info(f"Live: SPY=${ctx['spy_live']:.2f} UPRO=${ctx['upro_live']:.2f}")
            else:
                logger.warning("Could not get live prices -- using daily bar closes")
        except Exception as e:
            logger.warning(f"Live price fetch failed: {e} -- using daily bar closes")
            ctx["spy_live"] = None
            ctx["upro_live"] = None
            ctx["live_prices"] = {}

        return ctx
    except Exception as e:
        logger.error(f"Could not fetch market data: {e}", exc_info=True)
        return None


def _sync_ma_holding_from_broker(state):
    """Ensure state['ma_holding'] matches actual Alpaca positions."""
    ma_tickers = {
        config.MA_TRADE_GROWTH,
        config.MA_TRADE_SAFE,
        config.MA_TRADE_ALT,
    }

    try:
        positions = broker.get_all_positions()
    except Exception as e:
        logger.error(f"Could not fetch positions to sync MA holding: {e}")
        return state.get("ma_holding"), 0.0

    active = []
    for pos in positions:
        symbol = getattr(pos, "symbol", None)
        if symbol in ma_tickers:
            qty = float(getattr(pos, "qty", 0) or 0)
            if abs(qty) > 0:
                market_value = float(getattr(pos, "market_value", 0) or 0)
                active.append((symbol, market_value))

    active.sort(key=lambda item: abs(item[1]), reverse=True)
    actual_symbol = active[0][0] if active else None
    actual_value = active[0][1] if active else 0.0

    if state.get("ma_holding") != actual_symbol:
        logger.warning(
            f"MA HOLDING SYNC: state={state.get('ma_holding')} -> broker={actual_symbol}"
        )
        state["ma_holding"] = actual_symbol

    state["ma_position_value"] = actual_value
    return actual_symbol, actual_value


def _check_market_open(dry_run=False):
    """Return True if market is open (or dry_run)."""
    try:
        clock = broker.get_clock()
        if not clock.is_open and not dry_run:
            logger.info("Market is closed. Exiting.")
            return False
    except Exception as e:
        logger.error(f"Could not check market clock: {e}")
        if not dry_run:
            return False
    return True


def _handle_dip_exit(state, spy_closes, upro_price, spy_live=None,
                     upro_is_live=False, dry_run=False):
    """Check and execute dip trade exit. Returns True if an exit occurred.
    For BB SMA exits, appends spy_live to spy_closes so the check uses current price.

    Hard rule: if dip_active and upro_is_live is False, we refuse to evaluate
    exits to avoid stop-loss/take-profit decisions on stale daily bar prices.
    Will retry live price fetch once before skipping."""
    if not state.get("dip_active"):
        return False

    # Hard rule: never evaluate dip exits on stale prices
    if not upro_is_live:
        logger.warning("UPRO price is stale (daily bar) — retrying live fetch...")
        try:
            retry = data.fetch_live_prices(["UPRO", "SPY"])
            upro_retry = float(retry["UPRO"]) if retry.get("UPRO") else None
            spy_retry = float(retry["SPY"]) if retry.get("SPY") else None
            if upro_retry:
                upro_price = upro_retry
                spy_live = spy_retry
                upro_is_live = True
                logger.info(f"Retry succeeded: UPRO=${upro_price:.2f}")
            else:
                logger.error("UPRO live price retry failed — SKIPPING dip exit evaluation "
                             "to avoid decisions on stale data")
                return False
        except Exception as e:
            logger.error(f"UPRO live price retry error: {e} — SKIPPING dip exit evaluation")
            return False

    # Only increment days_held once per trading day (not per intraday run)
    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("dip_last_hold_date") != today:
        state["dip_days_held"] = state.get("dip_days_held", 0) + 1
        state["dip_last_hold_date"] = today

    # Build spy_closes with live price for accurate BB SMA exit check
    closes_for_exit = list(spy_closes)
    if spy_live is not None and state.get("dip_source") == "BB":
        closes_for_exit = closes_for_exit + [float(spy_live)]

    should_exit, reason = strategies.check_dip_exit(state, closes_for_exit, upro_price)

    if should_exit:
        logger.info(f"DIP EXIT: {reason}")
        if not dry_run:
            broker.close_position(config.DIP_TRADE_TICKER)
            log_trade(state, "SELL", config.DIP_TRADE_TICKER, "all", upro_price, reason)
        else:
            logger.info(f"  [DRY RUN] Would close {config.DIP_TRADE_TICKER}")

        state["dip_active"] = False
        state["dip_source"] = None
        state["dip_entry_price"] = 0
        state["dip_buy_date"] = None
        state["dip_days_held"] = 0
        state["dip_exit_mode"] = "hold"
        state["dip_last_hold_date"] = None
        return True
    else:
        logger.info(f"DIP HOLD: {reason}")
        return False


# ═══════════════════════════════════════════════════
# MODE: rebalance  (Monday 10:30 AM)
#   - Adjust MA position to target allocation (50% of equity)
#   - Does NOT touch dip positions
# ═══════════════════════════════════════════════════

def run_rebalance(dry_run=False):
    """Rebalance MA bucket to target allocation.
    Compares current MA position market value to target (MA_ALLOC_PCT * equity).
    Buys or trims to close the gap. Threshold of 2% avoids micro-trades."""
    state = load_state()
    logger.info("=" * 60)
    logger.info("REBALANCE RUN" + (" [DRY RUN]" if dry_run else ""))

    def finish(message):
        state["last_rebalance"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        save_state(state)
        logger.info(message)
        logger.info("=" * 60)

    if not _check_market_open(dry_run):
        return finish("REBALANCE SKIPPED (market closed)")

    try:
        equity = broker.get_equity()
        cash = broker.get_cash()
        logger.info(f"Account: equity=${equity:,.2f} cash=${cash:,.2f}")
    except Exception as e:
        logger.error(f"Could not fetch account data: {e}")
        return finish("REBALANCE SKIPPED (account fetch failed)")

    ma_ticker, current_value = _sync_ma_holding_from_broker(state)
    if not ma_ticker:
        logger.info("REBALANCE: no MA position held — state synced, nothing to do")
        return finish("REBALANCE COMPLETE (no MA position)")

    base_capital = max(current_value, 0) + cash
    state["ma_capital_base"] = base_capital
    if base_capital <= 0:
        logger.info("REBALANCE: no cash or MA capital available — skipping")
        return finish("REBALANCE SKIPPED (no capital)")

    target_value = base_capital * config.MA_ALLOC_PCT
    diff = target_value - current_value
    drift_pct = abs(diff) / base_capital if base_capital > 0 else 0

    logger.info(f"REBALANCE: {ma_ticker} current=${current_value:,.2f} "
                f"target=${target_value:,.2f} diff=${diff:+,.2f} "
                f"(drift={drift_pct:.1%}, base=${base_capital:,.2f})")

    REBALANCE_THRESHOLD = 0.02
    if drift_pct < REBALANCE_THRESHOLD:
        logger.info(f"REBALANCE: drift {drift_pct:.1%} < {REBALANCE_THRESHOLD:.0%} threshold — no action")
        return finish("REBALANCE COMPLETE (within threshold)")

    if diff > 0:
        buy_amount = min(diff, cash)
        if buy_amount < 1.0:
            logger.info(f"REBALANCE: need ${diff:,.2f} more but only ${cash:,.2f} cash — skipping")
            return finish("REBALANCE SKIPPED (insufficient cash)")

        logger.info(f"REBALANCE BUY: {ma_ticker} +${buy_amount:,.2f}")
        if not dry_run:
            broker.buy_notional(ma_ticker, buy_amount)
            log_trade(state, "BUY", ma_ticker, f"${buy_amount:.0f}", 0,
                      f"rebalance +${buy_amount:.0f}")
        else:
            logger.info(f"  [DRY RUN] Would buy ${buy_amount:,.2f} of {ma_ticker}")
        return finish("REBALANCE COMPLETE (buy)")

    # diff <= 0 → trim
    trim_amount = abs(diff)
    pos = broker.get_position(ma_ticker)
    if not pos or float(getattr(pos, "market_value", 0) or 0) <= 0:
        logger.info("REBALANCE: no live position to trim — skipping")
        return finish("REBALANCE SKIPPED (no position)")

    trim_pct = trim_amount / float(pos.market_value)
    trim_qty = float(pos.qty) * trim_pct
    if trim_qty < 0.001:
        logger.info(f"REBALANCE: trim too small ({trim_qty:.4f} shares) — skipping")
        return finish("REBALANCE SKIPPED (trim too small)")

    logger.info(f"REBALANCE TRIM: {ma_ticker} -{trim_qty:.4f} shares (~${trim_amount:,.2f})")
    if not dry_run:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        client = broker.get_trading_client()
        order = client.submit_order(MarketOrderRequest(
            symbol=ma_ticker,
            qty=round(trim_qty, 4),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        ))
        logger.info(f"TRIM order_id={order.id}")
        log_trade(state, "SELL", ma_ticker, f"{trim_qty:.4f}", 0,
                  f"rebalance -${trim_amount:.0f}")
    else:
        logger.info(f"  [DRY RUN] Would trim {trim_qty:.4f} shares of {ma_ticker}")

    return finish("REBALANCE COMPLETE (trim)")


# ═══════════════════════════════════════════════════
# MODE: tuesday_recovery  (Monday 3:30 PM only)
#   - Check dip trade exit (in case one is active)
#   - Monday dip signal -> buy UPRO for Tuesday bounce
#   - Does NOT touch MA crossover or BB
# ═══════════════════════════════════════════════════

def run_tuesday_recovery(dry_run=False):
    """Monday 3:30 PM: check for Monday dip signal, buy UPRO for Tuesday bounce."""
    state = load_state()
    logger.info("=" * 60)
    logger.info("TUESDAY RECOVERY RUN" + (" [DRY RUN]" if dry_run else ""))

    if not _check_market_open(dry_run):
        return

    ctx = _fetch_common_data()
    if not ctx:
        return

    # Use live prices
    upro_now = ctx.get("upro_live") or ctx["upro_price"]
    spy_live = ctx.get("spy_live")
    has_live = ctx.get("upro_live") is not None

    # Exit any active dip trade first
    _handle_dip_exit(state, ctx["spy_closes"], upro_now, spy_live=spy_live,
                     upro_is_live=has_live, dry_run=dry_run)

    # Monday dip check (live SPY vs prev close)
    is_week_start = data.is_first_trading_day_of_week(ctx["spy_dates"])
    md_buy, md_alloc, md_reason = strategies.check_monday_dip(
        state, ctx["spy_closes"], spy_live, is_week_start
    )

    if md_buy:
        cash = broker.get_cash()
        invest_amount = ctx["equity"] * md_alloc
        invest_amount = min(invest_amount, cash)
        logger.info(f"MONDAY DIP BUY: {config.DIP_TRADE_TICKER} ${invest_amount:,.2f} "
                     f"(cash avail=${cash:,.2f}) ({md_reason})")

        if not dry_run:
            broker.buy_notional(config.DIP_TRADE_TICKER, invest_amount)
            log_trade(state, "BUY", config.DIP_TRADE_TICKER, f"${invest_amount:.0f}",
                      upro_now, md_reason)

        state["dip_active"] = True
        state["dip_source"] = "MD"
        state["dip_entry_price"] = upro_now
        state["dip_buy_date"] = datetime.now().strftime("%Y-%m-%d")
        state["dip_days_held"] = 0
        state["dip_last_hold_date"] = None
        state["dip_exit_mode"] = "hold"
    else:
        logger.info(f"MONDAY DIP: no signal ({md_reason})")

    state["last_recovery_run"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_state(state)
    logger.info("TUESDAY RECOVERY RUN COMPLETE")
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════
# MODE: monday_close  (Monday 3:30 PM only)
#   - Tuesday recovery (Monday dip buy) FIRST
#   - Then MA crossover + BB (same as daily_close)
#   - Single Monday entry point, avoids double-fire
# ═══════════════════════════════════════════════════

def run_monday_close(dry_run=False):
    """Monday 3:30 PM: tuesday recovery + MA rotation + BB entry."""
    state = load_state()
    logger.info("=" * 60)
    logger.info("MONDAY CLOSE RUN" + (" [DRY RUN]" if dry_run else ""))

    if not _check_market_open(dry_run):
        return

    ctx = _fetch_common_data()
    if not ctx:
        return

    # Bootstrap MA counters on first run
    bootstrap_ma_counters(state, ctx["qqq_closes"], ctx["tlt_closes"])

    # Use live prices (daily bars at 3:30 PM are yesterday's close)
    upro_now = ctx.get("upro_live") or ctx["upro_price"]
    spy_live = ctx.get("spy_live")
    has_live = ctx.get("upro_live") is not None

    # 1. Check dip trade exit (live prices for accurate SL/TP/SMA)
    _handle_dip_exit(state, ctx["spy_closes"], upro_now, spy_live=spy_live,
                     upro_is_live=has_live, dry_run=dry_run)

    # 2. Tuesday recovery: Monday dip check (live SPY vs Friday close)
    is_week_start = data.is_first_trading_day_of_week(ctx["spy_dates"])
    md_buy, md_alloc, md_reason = strategies.check_monday_dip(
        state, ctx["spy_closes"], spy_live, is_week_start
    )

    if md_buy:
        cash = broker.get_cash()
        invest_amount = ctx["equity"] * md_alloc
        invest_amount = min(invest_amount, cash)
        logger.info(f"MONDAY DIP BUY: {config.DIP_TRADE_TICKER} ${invest_amount:,.2f} "
                     f"(cash avail=${cash:,.2f}) ({md_reason})")

        if not dry_run:
            broker.buy_notional(config.DIP_TRADE_TICKER, invest_amount)
            log_trade(state, "BUY", config.DIP_TRADE_TICKER, f"${invest_amount:.0f}",
                      upro_now, md_reason)

        state["dip_active"] = True
        state["dip_source"] = "MD"
        state["dip_entry_price"] = upro_now
        state["dip_buy_date"] = datetime.now().strftime("%Y-%m-%d")
        state["dip_days_held"] = 0
        state["dip_last_hold_date"] = None
        state["dip_exit_mode"] = "hold"
    else:
        logger.info(f"MONDAY DIP: no signal ({md_reason})")

    # 3. MA crossover check
    ma_target = strategies.check_ma_crossover(state, ctx["qqq_closes"], ctx["tlt_closes"])
    current_ma = state.get("ma_holding")

    if ma_target != current_ma:
        logger.info(f"MA ROTATION: {current_ma} -> {ma_target}")

        if current_ma and not dry_run:
            sell_price = ctx.get("live_prices", {}).get(current_ma, 0)
            broker.close_position(current_ma)
            log_trade(state, "SELL", current_ma, "all", sell_price, f"MA rotation -> {ma_target}")

        if ma_target:
            cash = broker.get_cash()
            invest_amount = ctx["equity"] * config.MA_ALLOC_PCT
            invest_amount = min(invest_amount, cash)
            buy_price = ctx.get("live_prices", {}).get(ma_target, 0)
            logger.info(f"MA BUY: {ma_target} notional=${invest_amount:,.2f} (cash avail=${cash:,.2f})")
            if not dry_run:
                broker.buy_notional(ma_target, invest_amount)
                log_trade(state, "BUY", ma_target, f"${invest_amount:.0f}", buy_price, "MA crossover")
            else:
                logger.info(f"  [DRY RUN] Would buy ${invest_amount:,.2f} of {ma_target}")

        state["ma_holding"] = ma_target
    else:
        logger.info(f"MA: holding {current_ma or 'nothing'} (no change)")

    # 4. BB reversion check (live SPY for current Bollinger position)
    if not state.get("dip_active"):
        bb_buy, bb_alloc, bb_reason = strategies.check_bb_entry(
            state, ctx["spy_closes"], spy_live=spy_live
        )

        if bb_buy:
            cash = broker.get_cash()
            invest_amount = ctx["equity"] * bb_alloc
            invest_amount = min(invest_amount, cash)
            logger.info(f"BB REVERSION BUY: {config.BB_TRADE_TICKER} ${invest_amount:,.2f} "
                         f"(cash avail=${cash:,.2f}) ({bb_reason})")

            if not dry_run:
                broker.buy_notional(config.BB_TRADE_TICKER, invest_amount)
                log_trade(state, "BUY", config.BB_TRADE_TICKER, f"${invest_amount:.0f}",
                          upro_now, bb_reason)

            state["dip_active"] = True
            state["dip_source"] = "BB"
            state["dip_entry_price"] = upro_now
            state["dip_buy_date"] = datetime.now().strftime("%Y-%m-%d")
            state["dip_days_held"] = 0
            state["dip_last_hold_date"] = None
            state["dip_exit_mode"] = "sma"

    state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_state(state)
    logger.info("MONDAY CLOSE RUN COMPLETE")
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════
# MODE: daily_close  (Tue-Fri 3:30 PM)
#   - Check dip trade exit
#   - MA crossover rotation
#   - BB reversion entry (only if no dip active)
#   - Does NOT check Monday dip (that's monday_close)
# ═══════════════════════════════════════════════════

def run_daily_close(dry_run=False):
    """Tue-Fri 3:30 PM: MA rotation + BB entry + dip exits."""
    state = load_state()
    logger.info("=" * 60)
    logger.info("DAILY CLOSE RUN" + (" [DRY RUN]" if dry_run else ""))

    if not _check_market_open(dry_run):
        return

    ctx = _fetch_common_data()
    if not ctx:
        return

    # Bootstrap MA counters on first run
    bootstrap_ma_counters(state, ctx["qqq_closes"], ctx["tlt_closes"])

    # Use live prices (daily bars at 3:30 PM are yesterday's close)
    upro_now = ctx.get("upro_live") or ctx["upro_price"]
    spy_live = ctx.get("spy_live")
    has_live = ctx.get("upro_live") is not None

    # 1. Check dip trade exit (live prices for accurate SL/TP/SMA)
    _handle_dip_exit(state, ctx["spy_closes"], upro_now, spy_live=spy_live,
                     upro_is_live=has_live, dry_run=dry_run)

    # 2. MA crossover check
    ma_target = strategies.check_ma_crossover(state, ctx["qqq_closes"], ctx["tlt_closes"])
    current_ma = state.get("ma_holding")

    if ma_target != current_ma:
        logger.info(f"MA ROTATION: {current_ma} -> {ma_target}")

        if current_ma and not dry_run:
            sell_price = ctx.get("live_prices", {}).get(current_ma, 0)
            broker.close_position(current_ma)
            log_trade(state, "SELL", current_ma, "all", sell_price, f"MA rotation -> {ma_target}")

        if ma_target:
            cash = broker.get_cash()
            invest_amount = ctx["equity"] * config.MA_ALLOC_PCT
            invest_amount = min(invest_amount, cash)
            buy_price = ctx.get("live_prices", {}).get(ma_target, 0)
            logger.info(f"MA BUY: {ma_target} notional=${invest_amount:,.2f} (cash avail=${cash:,.2f})")
            if not dry_run:
                broker.buy_notional(ma_target, invest_amount)
                log_trade(state, "BUY", ma_target, f"${invest_amount:.0f}", buy_price, "MA crossover")
            else:
                logger.info(f"  [DRY RUN] Would buy ${invest_amount:,.2f} of {ma_target}")

        state["ma_holding"] = ma_target
    else:
        logger.info(f"MA: holding {current_ma or 'nothing'} (no change)")

    # 3. BB reversion check (live SPY for current Bollinger position)
    if not state.get("dip_active"):
        bb_buy, bb_alloc, bb_reason = strategies.check_bb_entry(
            state, ctx["spy_closes"], spy_live=spy_live
        )

        if bb_buy:
            cash = broker.get_cash()
            invest_amount = ctx["equity"] * bb_alloc
            invest_amount = min(invest_amount, cash)
            logger.info(f"BB REVERSION BUY: {config.BB_TRADE_TICKER} ${invest_amount:,.2f} "
                         f"(cash avail=${cash:,.2f}) ({bb_reason})")

            if not dry_run:
                broker.buy_notional(config.BB_TRADE_TICKER, invest_amount)
                log_trade(state, "BUY", config.BB_TRADE_TICKER, f"${invest_amount:.0f}",
                          upro_now, bb_reason)

            state["dip_active"] = True
            state["dip_source"] = "BB"
            state["dip_entry_price"] = upro_now
            state["dip_buy_date"] = datetime.now().strftime("%Y-%m-%d")
            state["dip_days_held"] = 0
            state["dip_last_hold_date"] = None
            state["dip_exit_mode"] = "sma"

    state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_state(state)
    logger.info("DAILY CLOSE RUN COMPLETE")
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════
# MODE: combined  (--force fallback, runs everything)
# ═══════════════════════════════════════════════════

def run_bot(dry_run=False):
    """Combined run — all strategies. Used by --force for manual overrides."""
    state = load_state()
    logger.info("=" * 60)
    logger.info("FULL BOT RUN" + (" [DRY RUN]" if dry_run else ""))

    if not _check_market_open(dry_run):
        return

    ctx = _fetch_common_data()
    if not ctx:
        return

    bootstrap_ma_counters(state, ctx["qqq_closes"], ctx["tlt_closes"])

    # Use live prices (daily bars at 3:30 PM are yesterday's close)
    upro_now = ctx.get("upro_live") or ctx["upro_price"]
    spy_live = ctx.get("spy_live")
    has_live = ctx.get("upro_live") is not None

    # 1. Dip exit
    _handle_dip_exit(state, ctx["spy_closes"], upro_now, spy_live=spy_live,
                     upro_is_live=has_live, dry_run=dry_run)

    # 2. MA crossover
    ma_target = strategies.check_ma_crossover(state, ctx["qqq_closes"], ctx["tlt_closes"])
    current_ma = state.get("ma_holding")

    if ma_target != current_ma:
        logger.info(f"MA ROTATION: {current_ma} -> {ma_target}")
        if current_ma and not dry_run:
            sell_price = ctx.get("live_prices", {}).get(current_ma, 0)
            broker.close_position(current_ma)
            log_trade(state, "SELL", current_ma, "all", sell_price, f"MA rotation -> {ma_target}")
        if ma_target:
            cash = broker.get_cash()
            invest_amount = ctx["equity"] * config.MA_ALLOC_PCT
            invest_amount = min(invest_amount, cash)
            buy_price = ctx.get("live_prices", {}).get(ma_target, 0)
            logger.info(f"MA BUY: {ma_target} notional=${invest_amount:,.2f} (cash avail=${cash:,.2f})")
            if not dry_run:
                broker.buy_notional(ma_target, invest_amount)
                log_trade(state, "BUY", ma_target, f"${invest_amount:.0f}", buy_price, "MA crossover")
            else:
                logger.info(f"  [DRY RUN] Would buy ${invest_amount:,.2f} of {ma_target}")
        state["ma_holding"] = ma_target
    else:
        logger.info(f"MA: holding {current_ma or 'nothing'} (no change)")

    # 3. Monday dip (live SPY vs prev close)
    is_week_start = data.is_first_trading_day_of_week(ctx["spy_dates"])
    md_buy, md_alloc, md_reason = strategies.check_monday_dip(
        state, ctx["spy_closes"], spy_live, is_week_start
    )
    if md_buy:
        cash = broker.get_cash()
        invest_amount = ctx["equity"] * md_alloc
        invest_amount = min(invest_amount, cash)
        logger.info(f"MONDAY DIP BUY: {config.DIP_TRADE_TICKER} ${invest_amount:,.2f} ({md_reason})")
        if not dry_run:
            broker.buy_notional(config.DIP_TRADE_TICKER, invest_amount)
            log_trade(state, "BUY", config.DIP_TRADE_TICKER, f"${invest_amount:.0f}",
                      upro_now, md_reason)
        state["dip_active"] = True
        state["dip_source"] = "MD"
        state["dip_entry_price"] = upro_now
        state["dip_buy_date"] = datetime.now().strftime("%Y-%m-%d")
        state["dip_days_held"] = 0
        state["dip_last_hold_date"] = None
        state["dip_exit_mode"] = "hold"

    # 4. BB reversion (live SPY for current Bollinger)
    if not state.get("dip_active") and not md_buy:
        bb_buy, bb_alloc, bb_reason = strategies.check_bb_entry(
            state, ctx["spy_closes"], spy_live=spy_live
        )
        if bb_buy:
            cash = broker.get_cash()
            invest_amount = ctx["equity"] * bb_alloc
            invest_amount = min(invest_amount, cash)
            logger.info(f"BB REVERSION BUY: {config.BB_TRADE_TICKER} ${invest_amount:,.2f} ({bb_reason})")
            if not dry_run:
                broker.buy_notional(config.BB_TRADE_TICKER, invest_amount)
                log_trade(state, "BUY", config.BB_TRADE_TICKER, f"${invest_amount:.0f}",
                          upro_now, bb_reason)
            state["dip_active"] = True
            state["dip_source"] = "BB"
            state["dip_entry_price"] = upro_now
            state["dip_buy_date"] = datetime.now().strftime("%Y-%m-%d")
            state["dip_days_held"] = 0
            state["dip_last_hold_date"] = None
            state["dip_exit_mode"] = "sma"

    state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_state(state)
    logger.info("FULL BOT RUN COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    if "--status" in sys.argv:
        show_status()
    elif "--dry-run" in sys.argv:
        run_bot(dry_run=True)
    else:
        run_bot(dry_run=False)
