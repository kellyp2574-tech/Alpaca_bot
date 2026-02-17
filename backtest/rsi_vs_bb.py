"""
RSI vs Bollinger Band Reversion — Head-to-Head Comparison
==========================================================
Tests whether RSI(14) < 30 is a better entry signal than Bollinger Band
for our UPRO dip-buying strategy, and whether a VIX Rank filter improves either.

Variants tested:
  A. BB Reversion (current):  SPY < lower BB(20,2) -> buy UPRO, exit at SMA / SL / TP
  B. RSI Reversion:           RSI(14) < 30 on SPY -> buy UPRO, same exits
  C. BB + VIX Rank filter:    A but only when VIX Rank < threshold
  D. RSI + VIX Rank filter:   B but only when VIX Rank < threshold
  E. RSI article-style:       RSI(14) < 30, 3% TP, 10-day max hold, no SL

All use: 0.2% slippage, 20% allocation, SPY signal -> UPRO trade.
"""
import argparse
import json
import math
import os
from datetime import datetime

import yfinance as yf


# ── Data Loading (cached) ─────────────────────────────────────

def load_etf_cached(ticker, cache_dir="state/etf_cache", start="2009-06-25", end="2025-01-01"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_daily.json")

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            data = json.load(f)
        dates = [d for d in data['dates'] if start <= d <= end]
        closes = [data['closes'][data['dates'].index(d)] for d in dates]
        return dates, closes

    print(f"  Fetching {ticker} from Yahoo Finance...")
    df = yf.download(ticker, start="1999-01-01", end="2025-01-01", progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.get_level_values(0)

    all_dates = [d.strftime("%Y-%m-%d") for d in df.index]
    all_closes = df['Close'].tolist()

    with open(cache_file, 'w') as f:
        json.dump({'dates': all_dates, 'closes': all_closes}, f)
    print(f"    {len(all_dates)} daily bars ({all_dates[0]} to {all_dates[-1]})")

    dates = [d for d, c in zip(all_dates, all_closes) if start <= d <= end]
    closes = [c for d, c in zip(all_dates, all_closes) if start <= d <= end]
    return dates, closes


def load_vix_cached(cache_dir="state/etf_cache", start="2009-06-25", end="2025-01-01"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "VIX_daily.json")

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            data = json.load(f)
        dates = [d for d in data['dates'] if start <= d <= end]
        closes = [data['closes'][data['dates'].index(d)] for d in dates]
        return dates, closes

    print(f"  Fetching ^VIX from Yahoo Finance...")
    df = yf.download("^VIX", start="1999-01-01", end="2025-01-01", progress=False)
    if df.empty:
        raise ValueError("No VIX data")
    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.get_level_values(0)

    all_dates = [d.strftime("%Y-%m-%d") for d in df.index]
    all_closes = df['Close'].tolist()

    with open(cache_file, 'w') as f:
        json.dump({'dates': all_dates, 'closes': all_closes}, f)
    print(f"    {len(all_dates)} daily bars ({all_dates[0]} to {all_dates[-1]})")

    dates = [d for d in all_dates if start <= d <= end]
    closes = [c for d, c in zip(all_dates, all_closes) if start <= d <= end]
    return dates, closes


# ── Technical Indicators ──────────────────────────────────────

def compute_sma(closes, i, period):
    """SMA ending at index i."""
    if i < period - 1:
        return closes[i] if closes else 0
    return sum(closes[i - period + 1:i + 1]) / period


def compute_std(closes, i, period):
    """Population std dev ending at index i."""
    if i < period - 1:
        return 0
    window = closes[i - period + 1:i + 1]
    mean = sum(window) / period
    return math.sqrt(sum((x - mean) ** 2 for x in window) / period)


def compute_rsi(closes, i, period=14):
    """RSI ending at index i using Wilder smoothing."""
    if i < period:
        return 50  # not enough data
    gains = []
    losses = []
    for j in range(i - period + 1, i + 1):
        change = closes[j] - closes[j - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_vix_rank(vix_closes, i, lookback=252):
    """VIX percentile rank over the last `lookback` days (0-100)."""
    if i < lookback:
        window = vix_closes[:i + 1]
    else:
        window = vix_closes[i - lookback + 1:i + 1]
    if len(window) < 10:
        return 50  # not enough data
    current = vix_closes[i]
    below = sum(1 for v in window if v < current)
    return (below / len(window)) * 100


# ── Backtest Engine ───────────────────────────────────────────

def run_backtest(
    entry_mode: str = "bb",        # "bb" or "rsi"
    exit_mode: str = "sma",        # "sma" (BB-style) or "time" (article-style)
    trade_ticker: str = "UPRO",
    signal_ticker: str = "SPY",
    bb_period: int = 20,
    bb_sigma: float = 2.0,
    rsi_period: int = 14,
    rsi_threshold: float = 30.0,
    sma_exit_period: int = 20,     # exit when SPY returns to this SMA
    stop_loss: float = 0.10,       # 10% SL (0=off)
    take_profit: float = 0.05,     # 5% TP for BB-style, 3% for article-style
    max_hold_days: int = 0,        # 0=no limit (BB-style), 10 for article-style
    vix_rank_max: float = 100.0,   # 100=no filter, 70=article filter
    vix_rank_lookback: int = 252,
    slippage: float = 0.002,       # 0.2%
    alloc_pct: float = 0.20,       # 20% of equity per trade
    starting_cash: float = 10000.0,
    cash_apy: float = 0.0335,
    cache_dir: str = "state/etf_cache",
    start: str = "2009-06-25",
    end: str = "2025-01-01",
    verbose: bool = True,
):
    daily_cash_rate = (1 + cash_apy) ** (1 / 252) - 1

    # Load data
    tickers = list(set([trade_ticker, signal_ticker, 'SPY']))
    price_data = {}
    for t in tickers:
        dates, closes = load_etf_cached(t, cache_dir=cache_dir, start=start, end=end)
        price_data[t] = {'dates': dates, 'closes': dict(zip(dates, closes))}

    vix_dates, vix_closes_raw = load_vix_cached(cache_dir=cache_dir, start=start, end=end)
    vix_data = dict(zip(vix_dates, vix_closes_raw))

    # Common dates
    all_date_sets = [set(price_data[t]['dates']) for t in tickers] + [set(vix_dates)]
    common_dates = sorted(set.intersection(*all_date_sets))
    n = len(common_dates)
    years = n / 252

    # Build aligned arrays
    spy = [price_data['SPY']['closes'][d] for d in common_dates]
    trade = [price_data[trade_ticker]['closes'][d] for d in common_dates]
    vix = [vix_data[d] for d in common_dates]

    if verbose:
        print(f"  Period: {common_dates[0]} to {common_dates[-1]} ({years:.1f} yr, {n} bars)")

    # State
    cash = starting_cash
    invested = 0.0
    holding = False
    buy_bar = -1
    entry_price_trade = 0.0
    entry_price_spy = 0.0
    trade_log = []
    equity_curve = [cash]
    peak = cash
    max_dd = 0.0
    invested_days = 0

    for i in range(1, n):
        # Cash earns interest
        if not holding:
            cash += cash * daily_cash_rate

        # Mark to market
        if holding:
            prev_p = trade[i - 1]
            cur_p = trade[i]
            if prev_p > 0:
                invested *= (1 + (cur_p - prev_p) / prev_p)
            invested_days += 1

        equity = invested + cash
        equity_curve.append(equity)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd

        # ── Exit checks ──
        if holding:
            cur_pnl_trade = (trade[i] - entry_price_trade) / entry_price_trade

            # Stop loss
            if stop_loss > 0 and cur_pnl_trade <= -stop_loss:
                invested *= (1 - slippage)
                cash += invested
                trade_log.append({
                    'date': common_dates[i], 'action': 'STOP',
                    'pnl': cur_pnl_trade, 'days': i - buy_bar,
                })
                invested = 0; holding = False
                continue

            # Take profit
            if take_profit > 0 and cur_pnl_trade >= take_profit:
                invested *= (1 - slippage)
                cash += invested
                trade_log.append({
                    'date': common_dates[i], 'action': 'TP',
                    'pnl': cur_pnl_trade, 'days': i - buy_bar,
                })
                invested = 0; holding = False
                continue

            # SMA exit (BB-style): SPY returns to SMA
            if exit_mode == "sma":
                spy_sma = compute_sma(spy, i, sma_exit_period)
                if spy[i] >= spy_sma:
                    pnl = cur_pnl_trade
                    invested *= (1 - slippage)
                    cash += invested
                    trade_log.append({
                        'date': common_dates[i], 'action': 'SMA_EXIT',
                        'pnl': pnl, 'days': i - buy_bar,
                    })
                    invested = 0; holding = False
                    continue

            # Max hold exit (article-style)
            if max_hold_days > 0 and (i - buy_bar) >= max_hold_days:
                pnl = cur_pnl_trade
                invested *= (1 - slippage)
                cash += invested
                trade_log.append({
                    'date': common_dates[i], 'action': 'MAX_HOLD',
                    'pnl': pnl, 'days': i - buy_bar,
                })
                invested = 0; holding = False
                continue

        # ── Entry checks ──
        if not holding and i >= max(bb_period, rsi_period, 20):
            signal = False

            if entry_mode == "bb":
                spy_sma = compute_sma(spy, i, bb_period)
                spy_std = compute_std(spy, i, bb_period)
                lower_bb = spy_sma - bb_sigma * spy_std
                if spy[i] < lower_bb:
                    signal = True

            elif entry_mode == "rsi":
                rsi_val = compute_rsi(spy, i, rsi_period)
                if rsi_val < rsi_threshold:
                    signal = True

            # VIX Rank filter
            if signal and vix_rank_max < 100:
                vr = compute_vix_rank(vix, i, vix_rank_lookback)
                if vr > vix_rank_max:
                    signal = False

            if signal:
                invest_amt = (invested + cash) * alloc_pct
                invest_amt = min(invest_amt, cash)
                if invest_amt < 10:
                    continue
                cash -= invest_amt
                invested = invest_amt * (1 - slippage)
                entry_price_trade = trade[i]
                entry_price_spy = spy[i]
                buy_bar = i
                holding = True
                trade_log.append({
                    'date': common_dates[i], 'action': 'BUY',
                    'pnl': None, 'days': None,
                })

    # Final equity
    equity = invested + cash
    total_ret = (equity / starting_cash - 1) * 100
    ann_ret = ((equity / starting_cash) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Trade stats
    exits = [t for t in trade_log if t['action'] in ('STOP', 'TP', 'SMA_EXIT', 'MAX_HOLD')]
    n_trades = len(exits)
    wins = sum(1 for t in exits if (t.get('pnl') or 0) > 0)
    stops = sum(1 for t in exits if t['action'] == 'STOP')
    tps = sum(1 for t in exits if t['action'] == 'TP')
    sma_exits = sum(1 for t in exits if t['action'] == 'SMA_EXIT')
    max_holds = sum(1 for t in exits if t['action'] == 'MAX_HOLD')
    avg_pnl = sum((t.get('pnl') or 0) for t in exits) / max(n_trades, 1) * 100
    avg_hold = sum((t.get('days') or 0) for t in exits) / max(n_trades, 1)
    pct_invested = invested_days / n * 100

    # Gross wins / gross losses for profit factor
    gross_wins = sum((t.get('pnl') or 0) for t in exits if (t.get('pnl') or 0) > 0)
    gross_losses = abs(sum((t.get('pnl') or 0) for t in exits if (t.get('pnl') or 0) < 0))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # SPY B&H
    spy_ret = (spy[-1] / spy[0] - 1) * 100
    spy_ann = ((spy[-1] / spy[0]) ** (1 / years) - 1) * 100

    # Yearly breakdown
    yearly = {}
    for t in exits:
        yr = t['date'][:4]
        if yr not in yearly:
            yearly[yr] = {'wins': 0, 'losses': 0, 'pnls': []}
        if (t.get('pnl') or 0) > 0:
            yearly[yr]['wins'] += 1
        else:
            yearly[yr]['losses'] += 1
        yearly[yr]['pnls'].append((t.get('pnl') or 0) * 100)

    if verbose:
        label = f"{entry_mode.upper()} entry | {exit_mode.upper()} exit"
        vix_str = f" | VIX Rank < {vix_rank_max:.0f}" if vix_rank_max < 100 else ""
        print(f"  Config: {label}{vix_str}")
        print(f"  Trade: {trade_ticker} | Alloc: {alloc_pct*100:.0f}% | Slippage: {slippage*100:.1f}%")
        if entry_mode == "bb":
            print(f"  Entry: SPY < BB({bb_period}, {bb_sigma}sig)")
        else:
            print(f"  Entry: SPY RSI({rsi_period}) < {rsi_threshold:.0f}")
        if exit_mode == "sma":
            print(f"  Exit: SPY >= SMA({sma_exit_period}) | SL: {stop_loss*100:.0f}% | TP: {take_profit*100:.0f}%")
        else:
            print(f"  Exit: {max_hold_days}d max hold | SL: {stop_loss*100:.0f}% | TP: {take_profit*100:.0f}%")

        print(f"\n  {'Metric':<24} {'Value':>12}")
        print(f"  {'-'*38}")
        print(f"  {'CAGR':<24} {ann_ret:>+11.2f}%")
        print(f"  {'Total Return':<24} {total_ret:>+11.1f}%")
        print(f"  {'Max Drawdown':<24} {-max_dd*100:>11.1f}%")
        print(f"  {'$10k ->':<24} ${equity:>10,.0f}")
        print(f"  {'Trades':<24} {n_trades:>12}")
        print(f"  {'Win Rate':<24} {wins}/{n_trades} ({wins*100//max(n_trades,1)}%)")
        print(f"  {'Avg P&L':<24} {avg_pnl:>+11.2f}%")
        print(f"  {'Avg Hold':<24} {avg_hold:>11.1f}d")
        print(f"  {'Profit Factor':<24} {profit_factor:>11.2f}")
        print(f"  {'Time Invested':<24} {pct_invested:>11.1f}%")
        print(f"  {'Stops':<24} {stops:>12}")
        print(f"  {'Take Profits':<24} {tps:>12}")
        if sma_exits:
            print(f"  {'SMA Exits':<24} {sma_exits:>12}")
        if max_holds:
            print(f"  {'Max Hold Exits':<24} {max_holds:>12}")

        print(f"\n  SPY B&H: {spy_ann:+.2f}% CAGR | {spy_ret:+.1f}% total")

        # Yearly
        print(f"\n  {'Year':<6} {'Trades':>8} {'Wins':>6} {'Win%':>6} {'Avg P&L':>10} {'Sum P&L':>10}")
        print(f"  {'-'*48}")
        for yr in sorted(yearly.keys()):
            y = yearly[yr]
            total = y['wins'] + y['losses']
            win_pct = y['wins'] * 100 // max(total, 1)
            avg = sum(y['pnls']) / max(total, 1)
            total_pnl = sum(y['pnls'])
            print(f"  {yr:<6} {total:>8} {y['wins']:>6} {win_pct:>5}% {avg:>+9.2f}% {total_pnl:>+9.1f}%")

    return {
        'entry_mode': entry_mode, 'exit_mode': exit_mode,
        'vix_rank_max': vix_rank_max,
        'ann_ret': ann_ret, 'total_ret': total_ret,
        'max_dd': max_dd, 'equity': equity,
        'trades': n_trades, 'win_rate': wins / max(n_trades, 1),
        'avg_pnl': avg_pnl, 'avg_hold': avg_hold,
        'profit_factor': profit_factor, 'stops': stops, 'tps': tps,
        'pct_invested': pct_invested,
    }


# ── Head-to-Head Comparison ──────────────────────────────────

def run_comparison(start="2009-06-25", end="2025-01-01", cache_dir="state/etf_cache"):
    print("=" * 90)
    print("  RSI vs BOLLINGER BAND REVERSION -- HEAD-TO-HEAD COMPARISON")
    print("  Signal: SPY | Trade: UPRO | Alloc: 20% | Slippage: 0.2%")
    print("=" * 90)

    configs = [
        # A. Our current BB strategy
        {"label": "A. BB (current)",
         "entry_mode": "bb", "exit_mode": "sma",
         "stop_loss": 0.10, "take_profit": 0.05, "max_hold_days": 0,
         "vix_rank_max": 100},

        # B. RSI with same exits as BB
        {"label": "B. RSI (BB-style exits)",
         "entry_mode": "rsi", "exit_mode": "sma",
         "stop_loss": 0.10, "take_profit": 0.05, "max_hold_days": 0,
         "vix_rank_max": 100},

        # C. BB + VIX Rank filter
        {"label": "C. BB + VIX<70",
         "entry_mode": "bb", "exit_mode": "sma",
         "stop_loss": 0.10, "take_profit": 0.05, "max_hold_days": 0,
         "vix_rank_max": 70},

        # D. RSI + VIX Rank filter
        {"label": "D. RSI + VIX<70",
         "entry_mode": "rsi", "exit_mode": "sma",
         "stop_loss": 0.10, "take_profit": 0.05, "max_hold_days": 0,
         "vix_rank_max": 70},

        # E. RSI article-style (3% TP, 10d max hold, no SL)
        {"label": "E. RSI article-style",
         "entry_mode": "rsi", "exit_mode": "time",
         "stop_loss": 0.0, "take_profit": 0.03, "max_hold_days": 10,
         "vix_rank_max": 100},

        # F. RSI article-style + VIX filter
        {"label": "F. RSI article + VIX<70",
         "entry_mode": "rsi", "exit_mode": "time",
         "stop_loss": 0.0, "take_profit": 0.03, "max_hold_days": 10,
         "vix_rank_max": 70},
    ]

    results = []
    for cfg in configs:
        label = cfg.pop("label")
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
        r = run_backtest(
            **cfg, trade_ticker="UPRO", signal_ticker="SPY",
            slippage=0.002, alloc_pct=0.20,
            start=start, end=end, cache_dir=cache_dir,
            verbose=True,
        )
        r['label'] = label
        results.append(r)

    # Summary table
    print(f"\n\n{'='*110}")
    print(f"  SUMMARY COMPARISON")
    print(f"{'='*110}")
    print(f"  {'Config':<28} {'CAGR':>8} {'MaxDD':>8} {'$10k->':>10} {'Trades':>8} {'Win%':>6} "
          f"{'AvgP&L':>8} {'PF':>6} {'Stops':>6} {'AvgHold':>8}")
    print(f"  {'-'*104}")

    for r in results:
        wr = r['win_rate'] * 100
        print(f"  {r['label']:<28} {r['ann_ret']:>+7.2f}% {-r['max_dd']*100:>7.1f}% "
              f"${r['equity']:>9,.0f} {r['trades']:>8} {wr:>5.0f}% "
              f"{r['avg_pnl']:>+7.2f}% {r['profit_factor']:>5.1f} {r['stops']:>6} "
              f"{r['avg_hold']:>7.1f}d")

    return results


# ── RSI Threshold Sweep ───────────────────────────────────────

def run_rsi_sweep(start="2009-06-25", end="2025-01-01", cache_dir="state/etf_cache"):
    """Sweep RSI thresholds to find optimal entry level."""
    print("=" * 100)
    print("  RSI THRESHOLD SWEEP -- SPY signal, UPRO trade, SMA exit, 20% alloc")
    print("=" * 100)

    print(f"\n  {'RSI<':>6} {'VIX<':>6} {'CAGR':>8} {'MaxDD':>8} {'$10k->':>10} {'Trades':>8} "
          f"{'Win%':>6} {'AvgP&L':>8} {'PF':>6} {'Stops':>6}")
    print(f"  {'-'*80}")

    for rsi_thresh in [20, 25, 30, 35, 40]:
        for vix_max in [100, 70, 50]:
            r = run_backtest(
                entry_mode="rsi", exit_mode="sma",
                rsi_threshold=rsi_thresh,
                stop_loss=0.10, take_profit=0.05, max_hold_days=0,
                vix_rank_max=vix_max,
                trade_ticker="UPRO", signal_ticker="SPY",
                slippage=0.002, alloc_pct=0.20,
                start=start, end=end, cache_dir=cache_dir,
                verbose=False,
            )
            vix_str = f"{vix_max:.0f}" if vix_max < 100 else "off"
            wr = r['win_rate'] * 100
            print(f"  {rsi_thresh:>5.0f} {vix_str:>6} {r['ann_ret']:>+7.2f}% {-r['max_dd']*100:>7.1f}% "
                  f"${r['equity']:>9,.0f} {r['trades']:>8} {wr:>5.0f}% "
                  f"{r['avg_pnl']:>+7.2f}% {r['profit_factor']:>5.1f} {r['stops']:>6}")


# ── CLI ───────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="RSI vs BB Reversion Comparison")
    p.add_argument("--start", default="2009-06-25", help="Start date (UPRO inception)")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--cache-dir", default="state/etf_cache")
    p.add_argument("--sweep", action="store_true", help="Run RSI threshold sweep")
    p.add_argument("--compare", action="store_true", help="Run head-to-head comparison (default)")
    p.add_argument("--all", action="store_true", help="Run both comparison and sweep")
    args = p.parse_args()

    if args.all or (not args.sweep and not args.compare):
        # Default: run both
        run_comparison(start=args.start, end=args.end, cache_dir=args.cache_dir)
        print("\n\n")
        run_rsi_sweep(start=args.start, end=args.end, cache_dir=args.cache_dir)
    elif args.sweep:
        run_rsi_sweep(start=args.start, end=args.end, cache_dir=args.cache_dir)
    elif args.compare:
        run_comparison(start=args.start, end=args.end, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
