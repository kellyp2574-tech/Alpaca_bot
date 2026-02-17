"""
Bot Scheduler — Dispatches to the correct strategy mode based on --mode argument.
Cron handles all timing; the scheduler just runs the requested mode and exits.

Modes:
  intraday       Live-price stop loss / take profit checks (no new entries)
  monday_close   Monday dip buy + MA crossover + BB reversion + dip exits (Mon only)
  daily_close    MA crossover + BB reversion + dip exits (Tue-Fri)

Cron setup (all times ET, triggers at :30 to avoid open/close bells):
  # Intraday exits — Mon-Fri, 10:30 AM to 2:30 PM
  30 10-14 * * 1-5 cd /path/to/Alpaca_bot && python -m bot.scheduler --mode intraday

  # Monday close — Monday 3:30 PM (tuesday recovery + MA + BB)
  30 15 * * 1 cd /path/to/Alpaca_bot && python -m bot.scheduler --mode monday_close

  # Daily close — Tue-Fri 3:30 PM (MA rotation + BB entries + dip exits)
  30 15 * * 2-5 cd /path/to/Alpaca_bot && python -m bot.scheduler --mode daily_close

Manual:
  python -m bot.scheduler --force              # Run all strategies (combined)
  python -m bot.scheduler --force --dry-run    # Dry run, no trades
  python -m bot.scheduler --status             # Show current state
"""
import sys
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from bot import config
from bot.main import run_bot, run_daily_close, run_monday_close, show_status
from bot.state_manager import load_state, save_state
from bot import alpaca_client as broker
from bot import data
from bot import strategies

ET = ZoneInfo("America/New_York")

# ═══════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════
os.makedirs(config.LOG_DIR, exist_ok=True)

logger = logging.getLogger("bot")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(config.LOG_FILE)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)


def run_intraday_exits(dry_run=False):
    """Intraday check: evaluate stop loss / take profit on active dip trades
    using LIVE prices (latest trade), not daily bars.
    Does NOT open new positions or check MA crossover.
    Manages both MD and BB positions (both tracked via dip_active)."""
    state = load_state()

    if not state.get("dip_active"):
        logger.info("INTRADAY CHECK: no active dip trade, nothing to do")
        return

    logger.info("INTRADAY CHECK: evaluating active dip trade "
                f"(source={state.get('dip_source')})...")

    spy_closes = []
    try:
        # Live prices for current intraday values
        live = data.fetch_live_prices(["UPRO", "SPY"])
        upro_price = float(live["UPRO"]) if live.get("UPRO") else None
        spy_price = float(live["SPY"]) if live.get("SPY") else None

        if not upro_price:
            logger.warning("Could not get live UPRO price -- skipping intraday check")
            return

        if spy_price:
            logger.info(f"Live prices: UPRO=${upro_price:.2f} SPY=${spy_price:.2f}")
        else:
            logger.info(f"Live prices: UPRO=${upro_price:.2f}")

        # Daily SPY closes for BB SMA exit context (only needed for BB trades)
        if state.get("dip_source") == "BB":
            bars = data.fetch_daily_bars(["SPY"], lookback_days=30)
            spy_closes = bars.get("SPY", {}).get("closes", [])
            # Append live SPY price so SMA check uses current value
            if spy_price and spy_closes:
                spy_closes = spy_closes + [spy_price]

    except Exception as e:
        logger.error(f"Could not fetch intraday data: {e}")
        return

    should_exit, reason = strategies.check_dip_exit(state, spy_closes, upro_price)

    if should_exit:
        logger.info(f"INTRADAY EXIT: {reason}")
        if not dry_run:
            broker.close_position(config.DIP_TRADE_TICKER)
            from bot.state_manager import log_trade
            log_trade(state, "SELL", config.DIP_TRADE_TICKER, "all", upro_price, f"intraday: {reason}")

            state["dip_active"] = False
            state["dip_source"] = None
            state["dip_entry_price"] = 0
            state["dip_buy_date"] = None
            state["dip_days_held"] = 0
            state["dip_exit_mode"] = "hold"
            save_state(state)
        else:
            logger.info(f"  [DRY RUN] Would close {config.DIP_TRADE_TICKER}")
    else:
        logger.info(f"INTRADAY HOLD: {reason}")


def _market_is_open():
    """Return True if market is open. Returns False on weekends, holidays, errors."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
        logger.info("Weekend -- market closed")
        return False
    try:
        clock = broker.get_clock()
        if not clock.is_open:
            logger.info("Market closed (holiday or outside hours)")
            return False
        return True
    except Exception as e:
        logger.error(f"Could not check market clock: {e} -- assuming closed")
        return False


def dispatch(mode, dry_run=False):
    """Dispatch to the correct mode. Guards against closed-market execution."""
    now = datetime.now(ET)
    logger.info(f"SCHEDULER: mode={mode} at {now.strftime('%Y-%m-%d %H:%M ET')} ({now.strftime('%A')})"
                + (" [DRY RUN]" if dry_run else ""))
    logger.info(f"  python={sys.executable}  cwd={os.getcwd()}")

    # ── Market guard (skip unless --force) ──
    if mode != "force" and not _market_is_open():
        logger.info(f"Skipping mode={mode} -- market not open (use --force to override)")
        return

    if mode == "intraday":
        run_intraday_exits(dry_run=dry_run)

    elif mode == "monday_close":
        run_monday_close(dry_run=dry_run)

    elif mode == "daily_close":
        run_daily_close(dry_run=dry_run)

    elif mode == "force":
        run_bot(dry_run=dry_run)

    else:
        logger.error(f"Unknown mode: {mode}")
        print(f"Usage: python -m bot.scheduler --mode [intraday|monday_close|daily_close]")
        print(f"       python -m bot.scheduler --force")
        sys.exit(1)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv

    if "--status" in sys.argv:
        show_status()
        sys.exit(0)

    if "--force" in sys.argv:
        dispatch("force", dry_run=dry_run)
    elif "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            dispatch(sys.argv[idx + 1], dry_run=dry_run)
        else:
            print("Error: --mode requires an argument")
            sys.exit(1)
    else:
        print("Usage: python -m bot.scheduler --mode [intraday|monday_close|daily_close]")
        print("       python -m bot.scheduler --force [--dry-run]")
        print("       python -m bot.scheduler --status")
        sys.exit(1)
