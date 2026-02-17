"""
Strategy Logic — Pure signal generation for all three strategies.
No side effects — just takes data in, returns decisions out.
"""
import logging
from bot import config

logger = logging.getLogger("bot.strategies")


# ═══════════════════════════════════════════════════
# Strategy A: MA Crossover
# ═══════════════════════════════════════════════════

def check_ma_crossover(state, closes_qqq, closes_tlt):
    """
    Check MA crossover signals and return the target holding.
    Updates confirmation counters in state.
    Returns: target ticker string ("QLD", "UBT", "DBMF") or None

    Note: Counters are intentionally hysteresis-heavy — they persist when price
    is inside the buffer band (no decay). This prevents whipsaws in sideways
    markets but can delay switching when price drifts back to neutral.
    Entry requires MA_CONFIRM_ENTRY days (2); exit requires MA_CONFIRM_EXIT
    days (5). The asymmetry is intentional: slow to exit reduces churn.
    """
    from bot.data import compute_sma

    if len(closes_qqq) < config.MA_PERIOD or len(closes_tlt) < config.MA_PERIOD:
        logger.warning("Not enough data for MA crossover calculation")
        return state.get("ma_holding")

    qqq_sma = compute_sma(closes_qqq, config.MA_PERIOD)
    tlt_sma = compute_sma(closes_tlt, config.MA_PERIOD)
    qqq_price = closes_qqq[-1]
    tlt_price = closes_tlt[-1]

    qa = state.get("ma_qa", 0)
    qb = state.get("ma_qb", 0)
    ta = state.get("ma_ta", 0)
    tb = state.get("ma_tb", 0)

    # Update QQQ counters
    if qqq_price > qqq_sma * (1 + config.MA_BUFFER_PCT):
        qa += 1; qb = 0
    elif qqq_price < qqq_sma * (1 - config.MA_BUFFER_PCT):
        qb += 1; qa = 0

    # Update TLT counters
    if tlt_price > tlt_sma * (1 + config.MA_BUFFER_PCT):
        ta += 1; tb = 0
    elif tlt_price < tlt_sma * (1 - config.MA_BUFFER_PCT):
        tb += 1; ta = 0

    # Save counters
    state["ma_qa"] = qa
    state["ma_qb"] = qb
    state["ma_ta"] = ta
    state["ma_tb"] = tb

    # Confirmation checks
    ce = max(config.MA_CONFIRM_ENTRY, 1)
    cx = max(config.MA_CONFIRM_EXIT, 1)
    q_above = qa >= ce
    q_below = qb >= cx
    t_above = ta >= ce
    t_below = tb >= cx

    current = state.get("ma_holding")
    target = current

    if current == config.MA_TRADE_GROWTH:
        if q_below:
            target = config.MA_TRADE_SAFE if t_above else config.MA_TRADE_ALT
    elif current == config.MA_TRADE_SAFE:
        if q_above:
            target = config.MA_TRADE_GROWTH
        elif t_below:
            target = config.MA_TRADE_ALT
    elif current == config.MA_TRADE_ALT:
        if q_above:
            target = config.MA_TRADE_GROWTH
        elif t_above:
            target = config.MA_TRADE_SAFE
    else:
        # No position yet — always enter; MA strategy should never be all-cash
        if q_above:
            target = config.MA_TRADE_GROWTH
        elif t_above:
            target = config.MA_TRADE_SAFE
        else:
            target = config.MA_TRADE_ALT

    if target != current:
        logger.info(
            f"MA SIGNAL: {current} -> {target} | "
            f"QQQ={qqq_price:.2f} SMA={qqq_sma:.2f} (qa={qa} qb={qb}) | "
            f"TLT={tlt_price:.2f} SMA={tlt_sma:.2f} (ta={ta} tb={tb})"
        )

    return target


# ═══════════════════════════════════════════════════
# Strategy B: Monday Dip
# ═══════════════════════════════════════════════════

def check_monday_dip(state, spy_closes, spy_live, is_first_day_of_week):
    """
    Check if Monday dip conditions are met.
    Returns: (should_buy: bool, alloc_amount: float, reason: str)
    alloc_amount is a fraction (e.g. 0.30 for 30% of equity).

    Args:
        spy_closes: daily bar closes (last element = Friday/prev close at 3:30 PM)
        spy_live:   current live SPY price (snapshot at trigger time)
    """
    if not is_first_day_of_week:
        return False, 0.0, "Not first trading day of week"

    if state.get("dip_active"):
        return False, 0.0, "Dip trade already active"

    if not spy_closes or spy_live is None:
        return False, 0.0, "Not enough SPY data"

    prev_close = spy_closes[-1]   # Friday close (from daily bars)
    spy_now = float(spy_live)     # live price at 3:30 PM Monday

    if spy_now >= prev_close:
        return False, 0.0, f"SPY not down (now={spy_now:.2f} prev={prev_close:.2f})"

    dip_pct = (prev_close - spy_now) / prev_close
    if dip_pct < config.DIP_MIN_PCT:
        return False, 0.0, f"Dip too small ({dip_pct*100:.2f}% < {config.DIP_MIN_PCT*100:.1f}%)"

    logger.info(
        f"MONDAY DIP SIGNAL: SPY {prev_close:.2f} → {spy_now:.2f} "
        f"(dip={dip_pct*100:.2f}%)"
    )
    return True, config.DIP_ALLOC_CAP, f"Monday dip {dip_pct*100:.2f}%"


# ═══════════════════════════════════════════════════
# Strategy C: BB Reversion
# ═══════════════════════════════════════════════════

def check_bb_entry(state, spy_closes, spy_live=None):
    """
    Check if BB reversion entry conditions are met.
    Returns: (should_buy: bool, alloc_amount: float, reason: str)

    Args:
        spy_closes: daily bar closes (may not include today at 3:30 PM)
        spy_live:   current live SPY price — if provided, appended to closes
                    so the Bollinger check uses today's actual price
    """
    from bot.data import compute_sma, compute_bollinger_lower

    if state.get("dip_active"):
        return False, 0.0, "Dip trade already active"

    # Append live price so BB calc includes current conditions
    closes = list(spy_closes)
    if spy_live is not None:
        closes = closes + [float(spy_live)]

    if len(closes) < config.BB_PERIOD:
        return False, 0.0, "Not enough data for BB calculation"

    spy_price = closes[-1]
    lower_band = compute_bollinger_lower(closes, config.BB_PERIOD, config.BB_SIGMA)
    sma = compute_sma(closes, config.BB_PERIOD)

    if spy_price > lower_band:
        return False, 0.0, f"SPY {spy_price:.2f} above BB lower {lower_band:.2f}"

    logger.info(
        f"BB REVERSION SIGNAL: SPY={spy_price:.2f} below BB lower={lower_band:.2f} "
        f"(SMA={sma:.2f})"
    )
    return True, config.BB_ALLOC_CAP, f"BB entry SPY={spy_price:.2f} < BB={lower_band:.2f}"


def check_dip_exit(state, spy_closes, upro_price):
    """
    Check if the active dip trade should be exited.
    Returns: (should_exit: bool, reason: str)

    Callers must ensure:
      - upro_price is the LIVE current UPRO price
      - spy_closes[-1] is the LIVE SPY price (append before calling)
        for accurate BB SMA exit checks
      - days_held is incremented once per trading day before calling
    """
    from bot.data import compute_sma

    if not state.get("dip_active"):
        return False, "No active dip trade"

    entry_price = state.get("dip_entry_price", 0)
    days_held = state.get("dip_days_held", 0)
    source = state.get("dip_source", "")
    exit_mode = state.get("dip_exit_mode", "hold")

    if entry_price <= 0:
        return True, "Invalid entry price, closing"

    current_pnl = (upro_price - entry_price) / entry_price

    # Stop loss (BB trades only — 10%)
    if source == "BB" and current_pnl <= -config.BB_STOP_LOSS:
        return True, f"STOP LOSS hit: {current_pnl*100:+.1f}% (limit: -{config.BB_STOP_LOSS*100:.0f}%)"

    # Take profit
    if source == "MD" and current_pnl >= config.DIP_TAKE_PROFIT:
        return True, f"TAKE PROFIT (MD): {current_pnl*100:+.1f}% (target: +{config.DIP_TAKE_PROFIT*100:.1f}%)"
    if source == "BB" and current_pnl >= config.BB_TAKE_PROFIT:
        return True, f"TAKE PROFIT (BB): {current_pnl*100:+.1f}% (target: +{config.BB_TAKE_PROFIT*100:.1f}%)"

    # Normal exit
    if exit_mode == "hold":
        # Monday Dip: exit after hold_days
        if days_held >= config.DIP_HOLD_DAYS:
            return True, f"MD hold complete ({days_held}d): P&L {current_pnl*100:+.1f}%"
    elif exit_mode == "sma":
        # BB Reversion: exit when SPY returns to 20d SMA
        if len(spy_closes) >= config.BB_PERIOD:
            spy_sma = compute_sma(spy_closes, config.BB_PERIOD)
            spy_price = spy_closes[-1]
            if spy_price >= spy_sma:
                return True, f"BB SMA exit: SPY={spy_price:.2f} >= SMA={spy_sma:.2f}, P&L {current_pnl*100:+.1f}%"

    return False, f"Holding ({source} day {days_held}): P&L {current_pnl*100:+.1f}%"
