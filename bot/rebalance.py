"""
Rebalance Script - Manually rebalance portfolio to 50% 3 ETF rotation, 50% cash
This script is called manually to set the initial allocation
"""
import sys
import logging
import os
import argparse
from datetime import datetime

from bot import config
from bot.state_manager import load_state, save_state, log_trade
from bot import alpaca_client as broker
from bot import data
from bot import strategies
from bot.trade_reporter import log_trade_with_reporting

# Import Morning Momentum state store for safety check
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "Alpaca_Morning_Momentum"))
from bot.state_manager import StateStore as MMStateStore

# ═══════════════════════════════════════════════════
# Logging setup
# ═══════════════════════════════════════════════════
os.makedirs(config.LOG_DIR, exist_ok=True)

logger = logging.getLogger("rebalance")
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


def bootstrap_ma_counters(state, qqq_closes, tlt_closes):
    """Bootstrap MA counters on first run"""
    if state.get("ma_bootstrapped"):
        return

    logger.info("Bootstrapping MA counters from history...")
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


def sync_ma_holding_from_broker(state):
    """Ensure state['ma_holding'] matches actual Alpaca positions"""
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


def check_market_open(dry_run=False):
    """Return True if market is open (or dry_run)"""
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


def fetch_common_data():
    """Fetch account + market data"""
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
            return None

        # Fetch live prices
        try:
            live_tickers = ["SPY", "UPRO", config.MA_TRADE_GROWTH,
                            config.MA_TRADE_SAFE, config.MA_TRADE_ALT]
            live = data.fetch_live_prices(list(set(live_tickers)))
            ctx["spy_live"] = float(live["SPY"]) if live.get("SPY") else None
            ctx["upro_live"] = float(live["UPRO"]) if live.get("UPRO") else None
            ctx["live_prices"] = {k: float(v) for k, v in live.items() if v}
        except Exception as e:
            logger.warning(f"Live price fetch failed: {e}")
            ctx["spy_live"] = None
            ctx["upro_live"] = None
            ctx["live_prices"] = {}

        return ctx
    except Exception as e:
        logger.error(f"Could not fetch market data: {e}", exc_info=True)
        return None


def run_rebalance(dry_run=False, force=False):
    """Rebalance portfolio to 50% 3 ETF rotation, 50% cash"""
    state = load_state()
    logger.info("MMStateStore imported from %s", MMStateStore.__module__)
    logger.info("=" * 60)
    logger.info("REBALANCE RUN" + (" [DRY RUN]" if dry_run else ""))
    logger.info("Target: 50% 3 ETF rotation, 50% cash")

    def finish(message):
        state["last_rebalance"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        save_state(state)
        logger.info(message)
        logger.info("=" * 60)

    if not check_market_open(dry_run):
        return finish("REBALANCE SKIPPED (market closed)")

    # ⚠️ SAFETY CHECK: Ensure no Morning Momentum positions are open
    mm_state_store = MMStateStore("state/mm_positions.json")
    mm_positions = mm_state_store.load_positions()
    
    if mm_positions and not dry_run:
        logger.warning("⚠️  SAFETY WARNING: Morning Momentum positions detected:")
        for symbol, pos in mm_positions.items():
            logger.warning(f"  - {symbol}: {pos.get('qty', 0)} shares @ ${pos.get('entry_price', 0):.2f}")
        logger.error("❌ REBALANCE ABORTED: Close all Morning Momentum positions before rebalancing")
        logger.error("   Morning momentum trades should be closed by 10:30 AM automatically")
        logger.error("   If positions remain open, check the morning momentum bot status")
        return finish("REBALANCE ABORTED (Morning Momentum positions open)")
    
    if mm_positions and dry_run:
        logger.warning("⚠️  DRY RUN: Morning Momentum positions detected (would abort in live mode):")
        for symbol, pos in mm_positions.items():
            logger.warning(f"  - {symbol}: {pos.get('qty', 0)} shares @ ${pos.get('entry_price', 0):.2f}")
        logger.info("💡 In live mode, rebalancing would be aborted for safety")

    ctx = fetch_common_data()
    if not ctx:
        return finish("REBALANCE SKIPPED (data fetch failed)")

    # Bootstrap MA counters
    bootstrap_ma_counters(state, ctx["qqq_closes"], ctx["tlt_closes"])

    # Get current MA position
    ma_ticker, current_value = sync_ma_holding_from_broker(state)
    
    # Calculate target allocation (50% of equity)
    target_value = ctx["equity"] * 0.50
    
    logger.info(f"Current allocation: {ma_ticker or 'None'} = ${current_value:,.2f}")
    logger.info(f"Target allocation: 50% = ${target_value:,.2f}")
    logger.info(f"Available cash: ${ctx['cash']:,.2f}")

    # Determine if we need to adjust
    if ma_ticker:
        diff = target_value - current_value
        drift_pct = abs(diff) / ctx["equity"] if ctx["equity"] > 0 else 0
        
        logger.info(f"Difference: ${diff:+,.2f} ({drift_pct:.1%})")
        
        REBALANCE_THRESHOLD = 0.02  # 2% threshold
        if drift_pct < REBALANCE_THRESHOLD and not force:
            logger.info(f"Drift {drift_pct:.1%} < {REBALANCE_THRESHOLD:.0%} threshold — no action")
            return finish("REBALANCE COMPLETE (within threshold)")
        
        if diff > 0:
            # Need to buy more
            buy_amount = min(diff, ctx["cash"])
            if buy_amount < 1.0:
                logger.info(f"Need ${diff:,.2f} more but only ${ctx['cash']:,.2f} cash — skipping")
                return finish("REBALANCE SKIPPED (insufficient cash)")

            logger.info(f"REBALANCE BUY: {ma_ticker} +${buy_amount:,.2f}")
            if not dry_run:
                broker.buy_notional(ma_ticker, buy_amount)
                log_trade(state, "BUY", ma_ticker, f"${buy_amount:.0f}", 0,
                          f"rebalance 50% allocation")
                try:
                    log_trade_with_reporting(
                        ma_ticker,
                        "BUY",
                        buy_amount / ctx.get("live_prices", {}).get(ma_ticker, 1),
                        ctx.get("live_prices", {}).get(ma_ticker, 0),
                        "etf_rotation",
                        notes="rebalance 50% allocation",
                    )
                except Exception:
                    logger.exception("Rebalance reporting failed for %s BUY; continuing", ma_ticker)
            else:
                logger.info(f"  [DRY RUN] Would buy ${buy_amount:,.2f} of {ma_ticker}")
            return finish("REBALANCE COMPLETE (buy)")
        
        else:
            # Need to sell some
            trim_amount = abs(diff)
            pos = broker.get_position(ma_ticker)
            if not pos or float(getattr(pos, "market_value", 0) or 0) <= 0:
                logger.info("No live position to trim — skipping")
                return finish("REBALANCE SKIPPED (no position)")

            trim_pct = trim_amount / float(pos.market_value)
            trim_qty = float(pos.qty) * trim_pct
            if trim_qty < 0.001:
                logger.info(f"Trim too small ({trim_qty:.4f} shares) — skipping")
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
                          f"rebalance 50% allocation")
                try:
                    log_trade_with_reporting(
                        ma_ticker,
                        "SELL",
                        trim_qty,
                        ctx.get("live_prices", {}).get(ma_ticker, 0),
                        "etf_rotation",
                        notes="rebalance 50% allocation",
                    )
                except Exception:
                    logger.exception("Rebalance reporting failed for %s SELL; continuing", ma_ticker)
            else:
                logger.info(f"  [DRY RUN] Would trim {trim_qty:.4f} shares of {ma_ticker}")
            return finish("REBALANCE COMPLETE (trim)")
    
    else:
        # No MA position - determine best entry
        ma_target = strategies.check_ma_crossover(state, ctx["qqq_closes"], ctx["tlt_closes"])
        
        if not ma_target:
            logger.info("No MA signal - staying in cash")
            return finish("REBALANCE COMPLETE (no signal)")
        
        # Enter new position
        buy_amount = min(target_value, ctx["cash"])
        if buy_amount < 1.0:
            logger.info(f"Target ${target_value:,.2f} but only ${ctx['cash']:,.2f} cash — skipping")
            return finish("REBALANCE SKIPPED (insufficient cash)")

        logger.info(f"REBALANCE ENTER: {ma_target} ${buy_amount:,.2f}")
        if not dry_run:
            broker.buy_notional(ma_target, buy_amount)
            log_trade(state, "BUY", ma_target, f"${buy_amount:.0f}", 0,
                      f"rebalance 50% allocation")
            state["ma_holding"] = ma_target
            try:
                log_trade_with_reporting(
                    ma_target,
                    "BUY",
                    buy_amount / ctx.get("live_prices", {}).get(ma_target, 1),
                    ctx.get("live_prices", {}).get(ma_target, 0),
                    "etf_rotation",
                    notes="rebalance 50% allocation",
                )
            except Exception:
                logger.exception("Rebalance reporting failed for %s BUY; continuing", ma_target)
        else:
            logger.info(f"  [DRY RUN] Would buy ${buy_amount:,.2f} of {ma_target}")
        
        return finish("REBALANCE COMPLETE (enter)")


def check_morning_momentum_positions():
    """Check if any Morning Momentum positions are open"""
    mm_state_store = MMStateStore("state/mm_positions.json")
    mm_positions = mm_state_store.load_positions()
    
    if not mm_positions:
        print("✅ No Morning Momentum positions detected")
        return False
    
    print("⚠️  Morning Momentum positions currently open:")
    for symbol, pos in mm_positions.items():
        entry_price = pos.get('entry_price', 0)
        qty = pos.get('qty', 0)
        entry_time = pos.get('entry_time', 'Unknown')
        print(f"  - {symbol}: {qty} shares @ ${entry_price:.2f} (entered: {entry_time})")
    
    print(f"\n💡 These should be closed automatically by 10:30 AM")
    print(f"   If they remain open, check the morning momentum bot status")
    return True


def main():
    parser = argparse.ArgumentParser(description="Rebalance portfolio to 50/50 allocation")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without trading")
    parser.add_argument("--force", action="store_true", help="Force rebalance even within threshold")
    parser.add_argument("--check-mm", action="store_true", help="Check for open Morning Momentum positions only")
    args = parser.parse_args()
    
    if args.check_mm:
        check_morning_momentum_positions()
        return
    
    run_rebalance(dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
