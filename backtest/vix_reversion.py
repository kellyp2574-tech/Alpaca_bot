"""
VIX-Filtered Mean Reversion Strategy
======================================
Only buy when volatility is elevated AND SPY has a significant dip:
- VIX > threshold (e.g. 25)
- SPY down ≥ X% on the day (close < open)
- Buy at close, sell 1-2 days later
- Test on specific days (Wednesday focus) and any day
"""
import argparse
import json
import os
from datetime import datetime

import yfinance as yf


# ── Data Loading ───────────────────────────────────────────────

def load_etf_ohlc_cached(ticker, cache_dir="state/etf_cache", start="1999-01-01", end="2025-01-01"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_ohlc.json")

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            data = json.load(f)
        mask = [start <= d <= end for d in data['dates']]
        return ([d for d,m in zip(data['dates'],mask) if m],
                [o for o,m in zip(data['opens'],mask) if m],
                [c for c,m in zip(data['closes'],mask) if m])

    print(f"  Fetching {ticker} OHLC from Yahoo Finance...")
    df = yf.download(ticker, start="1999-01-01", end="2025-01-01", progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.get_level_values(0)

    all_dates = [d.strftime("%Y-%m-%d") for d in df.index]
    all_opens = df['Open'].tolist()
    all_closes = df['Close'].tolist()

    with open(cache_file, 'w') as f:
        json.dump({'dates': all_dates, 'opens': all_opens, 'closes': all_closes}, f)
    print(f"    {len(all_dates)} daily bars ({all_dates[0]} to {all_dates[-1]})")

    mask = [start <= d <= end for d in all_dates]
    return ([d for d,m in zip(all_dates,mask) if m],
            [o for o,m in zip(all_opens,mask) if m],
            [c for c,m in zip(all_closes,mask) if m])


def load_vix_cached(cache_dir="state/etf_cache", start="1999-01-01", end="2025-01-01"):
    """Load VIX close data (^VIX)."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "VIX_daily.json")

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            data = json.load(f)
        mask = [start <= d <= end for d in data['dates']]
        return ([d for d,m in zip(data['dates'],mask) if m],
                [c for c,m in zip(data['closes'],mask) if m])

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

    mask = [start <= d <= end for d in all_dates]
    return ([d for d,m in zip(all_dates,mask) if m],
            [c for c,m in zip(all_closes,mask) if m])


# ── VIX Mean Reversion Strategy ────────────────────────────────

DAY_NAMES = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}

def run_vix_reversion(
    trade_ticker: str = 'SPY',
    signal_ticker: str = 'SPY',
    vix_min: float = 25.0,       # min VIX level to trigger
    dip_min: float = 0.01,       # min intraday dip (close < open)
    hold_days: int = 2,          # sell N trading days later
    day_filter: int = -1,        # -1=any day, 0=Mon, 2=Wed, etc.
    slippage: float = 0.0,       # IBKR $0 commission
    starting_cash: float = 10000.0,
    alloc_pct: float = 1.0,
    take_profit: float = 0.0,    # TP threshold (0=off)
    cache_dir: str = "state/etf_cache",
    start: str = "1999-01-01",
    end: str = "2025-01-01",
    verbose: bool = True,
):
    trade_cost = slippage

    # Load data
    tickers = list(set([trade_ticker, signal_ticker]))
    price_data = {}
    for t in tickers:
        dates, opens, closes = load_etf_ohlc_cached(t, cache_dir=cache_dir, start=start, end=end)
        price_data[t] = {
            'dates': dates,
            'opens': dict(zip(dates, opens)),
            'closes': dict(zip(dates, closes)),
        }

    vix_dates, vix_closes = load_vix_cached(cache_dir=cache_dir, start=start, end=end)
    vix_data = dict(zip(vix_dates, vix_closes))

    # Common dates (need VIX + all tickers)
    date_sets = [set(price_data[t]['dates']) for t in tickers] + [set(vix_dates)]
    common_dates = sorted(set.intersection(*date_sets))
    n = len(common_dates)
    years = n / 252

    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in common_dates]
    weekdays = [d.weekday() for d in date_objs]

    # Build close price arrays for mark-to-market
    close_arr = {t: [price_data[t]['closes'][d] for d in common_dates] for t in tickers}

    # Run backtest
    cash = starting_cash
    holding = False
    buy_bar = -1
    entry_price = 0
    trade_log = []
    peak = cash
    max_dd = 0
    invested = 0.0

    for i in range(1, n):
        # Mark to market
        if holding:
            prev_p = close_arr[trade_ticker][i-1]
            cur_p = close_arr[trade_ticker][i]
            invested *= (1 + (cur_p - prev_p) / prev_p)

        equity = invested + cash

        # Take profit check
        if holding and take_profit > 0 and (i - buy_bar) < hold_days:
            cur_pnl = (close_arr[trade_ticker][i] - entry_price) / entry_price
            if cur_pnl >= take_profit:
                invested *= (1 - trade_cost)
                cash += invested
                trade_log.append({'date': common_dates[i], 'action': 'TP',
                                  'pnl': cur_pnl, 'days': i - buy_bar, 'vix': 0})
                invested = 0; holding = False

        # Sell after hold_days
        if holding and (i - buy_bar) >= hold_days:
            pnl = (close_arr[trade_ticker][i] - entry_price) / entry_price
            invested *= (1 - trade_cost)
            cash += invested
            trade_log.append({'date': common_dates[i], 'action': 'SELL',
                              'pnl': pnl, 'days': i - buy_bar, 'vix': 0})
            invested = 0; holding = False

        # Check for buy signal
        if not holding:
            d = common_dates[i]
            # Day filter
            if day_filter >= 0 and weekdays[i] != day_filter:
                continue

            # VIX check
            vix_val = vix_data.get(d, 0)
            if vix_val < vix_min:
                continue

            # Intraday dip check: close < open by at least dip_min
            sig_open = price_data[signal_ticker]['opens'][d]
            sig_close = price_data[signal_ticker]['closes'][d]
            intraday_move = (sig_close - sig_open) / sig_open

            if intraday_move >= 0:  # not a down day
                continue
            if abs(intraday_move) < dip_min:
                continue

            # BUY at close
            invest_amt = cash * alloc_pct
            cash -= invest_amt
            invested = invest_amt * (1 - trade_cost)
            entry_price = close_arr[trade_ticker][i]
            buy_bar = i
            holding = True
            trade_log.append({'date': d, 'action': 'BUY',
                              'pnl': None, 'days': None, 'vix': vix_val,
                              'dip': intraday_move})

        equity = invested + cash
        if equity > peak: peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd: max_dd = dd

    # Final
    equity = invested + cash
    total_ret = (equity / starting_cash - 1) * 100
    ann_ret = ((equity / starting_cash) ** (1/years) - 1) * 100

    exits = [t for t in trade_log if t['action'] in ('SELL', 'TP')]
    wins = sum(1 for t in exits if t.get('pnl', 0) > 0)
    n_trades = len(exits)
    avg_pnl = sum(t.get('pnl', 0) for t in exits) / max(n_trades, 1) * 100
    tps = sum(1 for t in exits if t['action'] == 'TP')
    avg_hold = sum(t.get('days', 0) for t in exits) / max(n_trades, 1)

    # Yearly
    yr_eq = {}; running = starting_cash
    for t in trade_log:
        if t['action'] in ('SELL', 'TP'):
            yr = t['date'][:4]
            # crude: track equity at each exit
    # Better: just compute from equity curve
    buy_dates = [t for t in trade_log if t['action'] == 'BUY']
    yearly_pnl = {}
    for t in exits:
        yr = t['date'][:4]
        if yr not in yearly_pnl: yearly_pnl[yr] = []
        yearly_pnl[yr].append(t.get('pnl', 0))

    # SPY B&H
    spy_start = close_arr[signal_ticker][0]
    spy_end = close_arr[signal_ticker][-1]
    spy_ret = (spy_end / spy_start - 1) * 100

    day_str = DAY_NAMES.get(day_filter, "Any day") if day_filter >= 0 else "Any day"

    if verbose:
        print(f"\n{'='*70}")
        print(f"  VIX MEAN REVERSION — {trade_ticker} | {day_str}")
        print(f"{'='*70}")
        print(f"  Period: {common_dates[0]} to {common_dates[-1]} ({years:.1f} yr)")
        print(f"  VIX ≥ {vix_min} | SPY dip ≥ {dip_min*100:.1f}% | Hold: {hold_days}d")
        tp_str = f" | TP: {take_profit*100:.1f}%" if take_profit > 0 else ""
        print(f"  Slippage: {slippage*100:.2f}% | Alloc: {alloc_pct*100:.0f}%{tp_str}")

        print(f"\n  CAGR: {ann_ret:+.2f}% | Max DD: {-max_dd*100:.1f}% | $10k → ${equity:,.0f}")
        print(f"  Trades: {n_trades} | Wins: {wins} ({wins*100//max(n_trades,1)}%) | Avg P&L: {avg_pnl:+.3f}%")
        if tps: print(f"  Take profits: {tps}")
        print(f"  Avg hold: {avg_hold:.1f} days")

        # Show some trade examples
        buys = [t for t in trade_log if t['action'] == 'BUY']
        if buys:
            print(f"\n  Sample trades (first 10):")
            print(f"  {'Date':<12} {'VIX':>6} {'Dip':>8}")
            for t in buys[:10]:
                print(f"  {t['date']:<12} {t['vix']:>6.1f} {t.get('dip',0)*100:>+7.2f}%")

    return {
        'ann_ret': ann_ret, 'max_dd': max_dd, 'trades': n_trades,
        'win_rate': wins / max(n_trades, 1), 'avg_pnl': avg_pnl,
        'equity': equity, 'tps': tps, 'avg_hold': avg_hold,
        'yearly_pnl': yearly_pnl,
    }


# ── Sweep ──────────────────────────────────────────────────────

def run_sweep():
    print("=" * 110)
    print("  VIX MEAN REVERSION — FULL SWEEP")
    print("  VIX ≥ threshold + SPY intraday dip → buy close, sell N days later")
    print("  IBKR $0 commission")
    print("=" * 110)

    # Test matrix
    tickers = ['SPY', 'SSO', 'UPRO']
    vix_levels = [20, 25, 30]
    dip_mins = [0.005, 0.01, 0.015, 0.02]
    holds = [1, 2, 3]
    days = [(-1, 'Any'), (2, 'Wed')]

    for day_val, day_name in days:
        for ticker in tickers:
            print(f'\n  --- {ticker} | {day_name} only | 0% slippage ---')
            print(f'  {"Config":<50} {"CAGR":>8} {"MaxDD":>8} {"Trades":>8} {"Win%":>6} {"AvgP&L":>8} {"$10k→":>12}')
            print(f'  {"-"*100}')

            for vix in vix_levels:
                for dip in dip_mins:
                    for hold in holds:
                        r = run_vix_reversion(
                            trade_ticker=ticker, vix_min=vix, dip_min=dip,
                            hold_days=hold, day_filter=day_val,
                            slippage=0.0, verbose=False)
                        if r['trades'] >= 5:  # skip configs with too few trades
                            dip_str = f"{dip*100:.1f}%"
                            label = f"VIX≥{vix} dip≥{dip_str} hold={hold}d"
                            print(f'  {label:<50} {r["ann_ret"]:>+7.2f}% {-r["max_dd"]*100:>+7.1f}% '
                                  f'{r["trades"]:>8} {r["win_rate"]*100:>5.0f}% {r["avg_pnl"]:>+7.3f}% '
                                  f'${r["equity"]:>10,.0f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VIX Mean Reversion Strategy")
    parser.add_argument('--trade-ticker', default='SPY')
    parser.add_argument('--vix-min', type=float, default=25.0)
    parser.add_argument('--dip-min', type=float, default=0.01)
    parser.add_argument('--hold', type=int, default=2)
    parser.add_argument('--day', type=int, default=-1, help='-1=any, 0=Mon, 2=Wed')
    parser.add_argument('--slippage', type=float, default=0.0)
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()

    if args.sweep:
        run_sweep()
    else:
        run_vix_reversion(
            trade_ticker=args.trade_ticker, vix_min=args.vix_min,
            dip_min=args.dip_min, hold_days=args.hold,
            day_filter=args.day, slippage=args.slippage,
        )
