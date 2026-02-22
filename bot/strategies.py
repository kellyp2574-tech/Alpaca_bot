"""
Strategy Logic — Pure signal generation for 3 ETF rotation strategy.
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
# Helper Functions
# ═══════════════════════════════════════════════════
