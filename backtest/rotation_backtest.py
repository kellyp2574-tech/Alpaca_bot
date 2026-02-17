"""
Forex Momentum Rotation Backtest

Rotates between forex pairs based on momentum. Key forex differences vs crypto:
  - Pairs are relative (no inherent appreciation -- must trade to profit)
  - Spread costs instead of commission fees
  - Leverage multiplies returns (and losses)
  - 20+ years of data for robust validation
  - Daily bars (forex is 24/5)

For simplicity, we normalize all pairs to "long base vs USD" direction.
Pairs like USDJPY are inverted so momentum is comparable across all pairs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from Alpaca_bot.backtest.forex_data import (
    PAIR_MAP, ForexPair, PairData, load_pair_cached
)


# ── Helpers ────────────────────────────────────────────────────────

# Pairs where USD is the base (USDXXX) need to be inverted so that
# "price going up" means "the non-USD currency is strengthening"
# This makes momentum comparable: positive = currency strengthening vs USD
USD_BASE_PAIRS = {"USDJPY", "USDCHF", "USDCAD", "USDSEK", "USDNOK", "USDMXN", "USDZAR", "USDSGD"}


def normalize_closes(pair_data: PairData) -> list[float]:
    """Return closes normalized so up = base currency strengthening.
    For XXXUSD pairs, closes are already correct.
    For USDXXX pairs, invert (1/price) so up = foreign currency strengthening.
    """
    if pair_data.symbol in USD_BASE_PAIRS:
        return [1.0 / c for c in pair_data.closes]
    return list(pair_data.closes)


def momentum(closes: list[float], lookback: int) -> float:
    """Simple return over lookback period."""
    if len(closes) < lookback + 1:
        return 0.0
    return (closes[-1] - closes[-lookback - 1]) / closes[-lookback - 1]


# ── Main backtest ──────────────────────────────────────────────────

def run_rotation_backtest(
    pairs: list[PairData],
    rebalance_days: int = 20,       # rebalance every N trading days (~1 month)
    lookback_days: int = 60,        # momentum lookback (~3 months)
    spread_cost: float = 0.0003,    # ~3 pips on majors as fraction (~0.03%)
    leverage: float = 1.0,          # 1x = no leverage, 10x = 10:1
    starting_cash: float = 10000.0,
    min_advantage: float = 0.005,   # min momentum edge to rotate (0.5%)
):
    """
    Rotation backtest across forex pairs.

    Always holds exactly ONE pair (long) at a time.
    Every rebalance_days, compute momentum for all pairs,
    switch to the strongest one if it has meaningful edge.

    Leverage amplifies both the return on each hold period
    and the spread cost per rotation.
    """
    # Align dates: find common trading days
    all_date_sets = [set(p.dates) for p in pairs]
    common_dates = sorted(all_date_sets[0].intersection(*all_date_sets[1:]))

    if not common_dates:
        print("ERROR: no common dates across pairs")
        return

    # Build aligned normalized price arrays
    price_map = {}
    raw_price_map = {}
    for pair in pairs:
        date_to_close = dict(zip(pair.dates, pair.closes))
        raw_closes = [date_to_close[d] for d in common_dates]
        raw_price_map[pair.symbol] = raw_closes

        norm_closes = normalize_closes(PairData(
            symbol=pair.symbol, dates=common_dates,
            timestamps=[], opens=[], highs=[], lows=[],
            closes=raw_closes, volumes=[],
        ))
        price_map[pair.symbol] = norm_closes

    n_bars = len(common_dates)
    symbols = [p.symbol for p in pairs]
    print(f"Aligned {n_bars} trading days across {len(symbols)} pairs: {symbols}")
    print(f"Period: {common_dates[0]} to {common_dates[-1]}")
    years = n_bars / 252
    print(f"Duration: {years:.1f} years ({n_bars} trading days)")
    print(f"Rebalance every {rebalance_days} days, momentum lookback {lookback_days} days")
    print(f"Spread cost: {spread_cost*100:.2f}% per trade, Leverage: {leverage:.0f}x")
    print(f"Min advantage to rotate: {min_advantage*100:.1f}%")
    print()

    # Start: hold first pair
    equity = starting_cash
    current_holding = symbols[0]
    entry_price = price_map[current_holding][0]
    # Pay spread on entry
    equity -= equity * spread_cost
    
    n_trades = 1
    trade_log = []
    trade_log.append({
        'bar': 0, 'date': common_dates[0],
        'action': 'BUY', 'from': '-', 'to': current_holding,
        'equity': equity, 'hold_pnl_pct': None, 'hold_days': None,
        'mom_scores': {s: 0.0 for s in symbols},
    })
    last_entry_bar = 0
    last_entry_price = entry_price
    bars_since_rebalance = 0

    # Track equity curve for stats
    equity_curve = [equity]

    for i in range(1, n_bars):
        bars_since_rebalance += 1

        # Mark-to-market with leverage
        current_price = price_map[current_holding][i]
        prev_price = price_map[current_holding][i - 1]
        daily_return = (current_price - prev_price) / prev_price
        equity *= (1 + daily_return * leverage)
        equity_curve.append(equity)

        # Rebalance check
        if bars_since_rebalance >= rebalance_days and i >= lookback_days:
            bars_since_rebalance = 0

            # Compute momentum for all pairs
            mom = {}
            for sym in symbols:
                closes_so_far = price_map[sym][:i+1]
                mom[sym] = momentum(closes_so_far, lookback_days)

            best = max(symbols, key=lambda s: mom[s])

            # Only rotate if the best pair has a meaningful edge
            current_mom = mom[current_holding]
            best_mom = mom[best]
            if best != current_holding and (best_mom - current_mom) >= min_advantage:
                # Pay spread cost (both exit and entry)
                total_spread = spread_cost * 2 * leverage
                hold_return = (current_price - last_entry_price) / last_entry_price
                hold_days = i - last_entry_bar

                equity *= (1 - total_spread)

                trade_log.append({
                    'bar': i, 'date': common_dates[i],
                    'action': 'ROTATE', 'from': current_holding, 'to': best,
                    'equity': equity,
                    'hold_pnl_pct': hold_return * leverage,
                    'hold_days': hold_days,
                    'mom_scores': dict(mom),
                })

                n_trades += 1
                last_entry_bar = i
                last_entry_price = price_map[best][i]
                current_holding = best

    # Final stats
    final_equity = equity
    total_return = (final_equity / starting_cash - 1) * 100
    annual_return = ((final_equity / starting_cash) ** (1 / years) - 1) * 100

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    # Win rate
    rotations = [t for t in trade_log if t['action'] == 'ROTATE']
    wins = sum(1 for t in rotations if t['hold_pnl_pct'] and t['hold_pnl_pct'] > 0)
    win_rate = wins / len(rotations) * 100 if rotations else 0

    # B&H each pair
    print("=" * 100)
    print("  RESULTS")
    print("=" * 100)
    print(f"  {'Strategy':<25} {'End Value':>12} {'Total Ret':>10} {'Annual':>8} {'MaxDD':>8} {'Trades':>8}")
    print(f"  {'-'*71}")
    print(f"  {'ROTATION':<25} ${final_equity:>11.2f} {total_return:>+9.1f}% {annual_return:>+7.1f}% {-max_dd*100:>7.1f}% {n_trades:>8}")

    for sym in symbols:
        # Compute B&H return for each pair
        start_p = price_map[sym][0]
        end_p = price_map[sym][-1]
        bh_ret = (end_p / start_p - 1) * 100
        bh_annual = ((end_p / start_p) ** (1 / years) - 1) * 100
        bh_equity = starting_cash * (1 + bh_ret / 100)
        # B&H max drawdown
        bh_peak = price_map[sym][0]
        bh_max_dd = 0
        for p in price_map[sym]:
            if p > bh_peak:
                bh_peak = p
            dd = (bh_peak - p) / bh_peak
            if dd > bh_max_dd:
                bh_max_dd = dd
        print(f"  {'B&H ' + sym:<25} ${bh_equity:>11.2f} {bh_ret:>+9.1f}% {bh_annual:>+7.1f}% {-bh_max_dd*100:>7.1f}% {'1':>8}")

    print()
    print(f"  Win rate: {wins}/{len(rotations)} ({win_rate:.0f}%)")
    print(f"  Avg hold: {sum(t['hold_days'] for t in rotations if t['hold_days']) / max(len(rotations), 1):.0f} days")
    print(f"  Total spread costs: ~${starting_cash * spread_cost * 2 * leverage * len(rotations):.2f}")

    # ── Detailed Trade Log ─────────────────────────────────────────
    print()
    print("=" * 100)
    print("  DETAILED TRADE LOG (first 30 and last 10)")
    print("=" * 100)
    print(f"  {'#':<4} {'Date':<12} {'Action':<8} {'From':<8} {'To':<8} "
          f"{'Hold':>6} {'Lev P&L':>10} {'Equity':>12} {'Top momentum scores'}")
    print(f"  {'-'*98}")

    show_trades = trade_log[:31] + ([{'separator': True}] if len(trade_log) > 41 else []) + trade_log[-10:] if len(trade_log) > 41 else trade_log

    for idx, t in enumerate(show_trades):
        if 'separator' in t:
            print(f"  {'...':^98}")
            continue

        if t['action'] == 'BUY':
            print(f"  {0:<4} {t['date']:<12} {'ENTRY':<8} {'-':<8} {t['to']:<8} "
                  f"{'':>6} {'':>10} ${t['equity']:>11.2f}")
        else:
            hold_str = f"{t['hold_days']}d"
            pnl_str = f"{t['hold_pnl_pct']:+.2%}"
            sorted_mom = sorted(t['mom_scores'].items(), key=lambda x: x[1], reverse=True)
            mom_str = "  ".join(f"{s}:{m:+.1%}" for s, m in sorted_mom[:4])
            actual_idx = trade_log.index(t)
            print(f"  {actual_idx:<4} {t['date']:<12} {'ROTATE':<8} {t['from']:<8} {t['to']:<8} "
                  f"{hold_str:>6} {pnl_str:>10} ${t['equity']:>11.2f} {mom_str}")

    # Final hold
    final_hold_days = n_bars - 1 - last_entry_bar
    final_hold_ret = (price_map[current_holding][-1] - last_entry_price) / last_entry_price * leverage
    print(f"  {'>':<4} {common_dates[-1]:<12} {'HOLD':<8} {'':<8} {current_holding:<8} "
          f"{final_hold_days}d{'':>4} {final_hold_ret:>+10.2%} ${final_equity:>11.2f} (still holding)")

    # ── Yearly Breakdown ───────────────────────────────────────────
    print()
    print("=" * 100)
    print("  YEARLY BREAKDOWN")
    print("=" * 100)
    print(f"  {'Year':<8} {'Rotation':>10} {'Trades':>8} {'Held most':<12}")
    print(f"  {'-'*38}")

    # Compute yearly returns from equity curve
    year_start_equity = equity_curve[0]
    year_start_idx = 0
    current_year = common_dates[0][:4]
    hold_counts = {}

    for i in range(1, n_bars):
        yr = common_dates[i][:4]
        if yr != current_year or i == n_bars - 1:
            yr_return = (equity_curve[i-1 if i < n_bars-1 else i] / year_start_equity - 1) * 100
            yr_trades = sum(1 for t in rotations if t['date'][:4] == current_year)
            # Find most held pair in that year
            print(f"  {current_year:<8} {yr_return:>+9.1f}% {yr_trades:>8}")
            year_start_equity = equity_curve[i-1 if i < n_bars-1 else i]
            current_year = yr

    print()
    return final_equity


# ── CLI ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Forex Momentum Rotation Backtest")
    p.add_argument("--pairs", default="EURUSD,GBPUSD,USDJPY,AUDUSD,NZDUSD,USDCHF,USDCAD",
                   help="Comma-separated pair symbols")
    p.add_argument("--start", default="2003-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--rebalance-days", type=int, default=20, help="Trading days between rebalances")
    p.add_argument("--lookback-days", type=int, default=60, help="Momentum lookback in trading days")
    p.add_argument("--min-advantage", type=float, default=0.005, help="Min momentum edge to rotate")
    p.add_argument("--spread-cost", type=float, default=0.0003, help="Spread as fraction per trade")
    p.add_argument("--leverage", type=float, default=1.0, help="Leverage multiplier (e.g. 10 for 10:1)")
    p.add_argument("--starting-cash", type=float, default=10000.0)
    p.add_argument("--cache-dir", default="state/forex_cache")
    args = p.parse_args()

    pair_symbols = [s.strip().upper() for s in args.pairs.split(",")]

    print(f"Loading data for: {pair_symbols}")
    pairs_data = []
    for sym in pair_symbols:
        pair = PAIR_MAP.get(sym)
        if not pair:
            print(f"  Unknown pair: {sym}, skipping")
            continue
        data = load_pair_cached(pair, cache_dir=args.cache_dir, start=args.start, end=args.end)
        print(f"  {sym}: {len(data.closes)} daily bars ({data.dates[0]} to {data.dates[-1]})")
        pairs_data.append(data)

    if len(pairs_data) < 2:
        print("ERROR: need at least 2 pairs for rotation")
        return

    print()
    run_rotation_backtest(
        pairs=pairs_data,
        rebalance_days=args.rebalance_days,
        lookback_days=args.lookback_days,
        spread_cost=args.spread_cost,
        leverage=args.leverage,
        starting_cash=args.starting_cash,
        min_advantage=args.min_advantage,
    )


if __name__ == "__main__":
    main()
