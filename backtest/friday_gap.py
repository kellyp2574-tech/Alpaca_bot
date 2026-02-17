"""
Friday Gap Continuation Strategy
==================================
Compare Thursday close with Friday open:
- If Friday opens UP from Thursday close → buy LONG at Friday open, sell Friday close
- If Friday opens DOWN from Thursday close → buy SHORT/inverse at Friday open, sell Friday close

Essentially a momentum bet that the overnight gap direction continues through Friday.
"""
import argparse
import json
import os
from datetime import datetime

import yfinance as yf


# ── Data Loading (OHLC) ───────────────────────────────────────

def load_etf_ohlc_cached(ticker, cache_dir="state/etf_cache", start="1999-01-01", end="2025-01-01"):
    """Load daily OHLC data with caching."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_ohlc.json")

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            data = json.load(f)
        mask = [start <= d <= end for d in data['dates']]
        dates = [d for d, m in zip(data['dates'], mask) if m]
        opens = [o for o, m in zip(data['opens'], mask) if m]
        closes = [c for c, m in zip(data['closes'], mask) if m]
        return dates, opens, closes

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
    dates = [d for d, m in zip(all_dates, mask) if m]
    opens = [o for o, m in zip(all_opens, mask) if m]
    closes = [c for c, m in zip(all_closes, mask) if m]
    return dates, opens, closes


# ── Friday Gap Strategy ────────────────────────────────────────

def run_friday_gap(
    long_ticker: str = 'SPY',     # ETF to buy when gap is UP
    short_ticker: str = 'SH',     # inverse ETF to buy when gap is DOWN
    signal_ticker: str = 'SPY',   # ticker to measure gap
    slippage: float = 0.0,        # IBKR $0 commission
    starting_cash: float = 10000.0,
    alloc_pct: float = 1.0,
    min_gap: float = 0.0,         # min overnight gap to trigger (0=any)
    long_only: bool = False,      # only trade gap-up (skip gap-down)
    cache_dir: str = "state/etf_cache",
    start: str = "1999-01-01",
    end: str = "2025-01-01",
    verbose: bool = True,
):
    trade_cost = slippage

    # Load OHLC for all tickers
    tickers = list(set([long_ticker, short_ticker, signal_ticker]))
    price_data = {}
    for t in tickers:
        dates, opens, closes = load_etf_ohlc_cached(t, cache_dir=cache_dir, start=start, end=end)
        price_data[t] = {'dates': dates, 'opens': dict(zip(dates, opens)), 'closes': dict(zip(dates, closes))}

    # Common dates
    date_sets = [set(price_data[t]['dates']) for t in tickers]
    common_dates = sorted(set.intersection(*date_sets))
    n = len(common_dates)
    years = n / 252

    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in common_dates]
    weekdays = [d.weekday() for d in date_objs]

    # Find Thu-Fri pairs in same week
    weeks = {}
    for i, d in enumerate(date_objs):
        key = (d.isocalendar()[0], d.isocalendar()[1])
        if key not in weeks:
            weeks[key] = {}
        weeks[key][weekdays[i]] = i

    # Run backtest
    cash = starting_cash
    trade_log = []
    peak = cash
    max_dd = 0
    n_trades = 0; n_wins = 0; n_long = 0; n_short = 0
    total_pnl = 0.0

    for key in sorted(weeks.keys()):
        wk = weeks[key]
        if 3 not in wk or 4 not in wk:
            continue

        thu_idx = wk[3]
        fri_idx = wk[4]

        sig = signal_ticker
        thu_close = price_data[sig]['closes'][common_dates[thu_idx]]
        fri_open = price_data[sig]['opens'][common_dates[fri_idx]]
        gap = (fri_open - thu_close) / thu_close

        # Apply min gap filter
        if min_gap > 0 and abs(gap) < min_gap:
            continue

        # Determine direction
        if gap > 0:
            # Gap UP → buy long, bet it continues up through Friday
            trade_ticker = long_ticker
            direction = 'LONG'
            n_long += 1
        else:
            if long_only:
                continue
            # Gap DOWN → buy inverse/short ETF, bet it continues down
            trade_ticker = short_ticker
            direction = 'SHORT'
            n_short += 1

        # Entry: Friday open | Exit: Friday close
        entry_price = price_data[trade_ticker]['opens'][common_dates[fri_idx]]
        exit_price = price_data[trade_ticker]['closes'][common_dates[fri_idx]]

        # Execute trade
        invest = cash * alloc_pct
        shares_value = invest * (1 - trade_cost)
        pnl = (exit_price - entry_price) / entry_price
        shares_value *= (1 + pnl)
        shares_value *= (1 - trade_cost)

        cash = cash - invest + shares_value
        n_trades += 1
        total_pnl += pnl
        if pnl > 0:
            n_wins += 1

        trade_log.append({
            'date': common_dates[fri_idx],
            'direction': direction,
            'ticker': trade_ticker,
            'gap': gap,
            'pnl': pnl,
            'equity': cash,
        })

        if cash > peak:
            peak = cash
        dd = (peak - cash) / peak
        if dd > max_dd:
            max_dd = dd

    # Results
    total_ret = (cash / starting_cash - 1) * 100
    ann_ret = ((cash / starting_cash) ** (1 / years) - 1) * 100
    avg_pnl = total_pnl / max(n_trades, 1) * 100

    long_trades = [t for t in trade_log if t['direction'] == 'LONG']
    short_trades = [t for t in trade_log if t['direction'] == 'SHORT']
    long_wins = sum(1 for t in long_trades if t['pnl'] > 0)
    short_wins = sum(1 for t in short_trades if t['pnl'] > 0)
    long_avg = sum(t['pnl'] for t in long_trades) / max(len(long_trades), 1) * 100
    short_avg = sum(t['pnl'] for t in short_trades) / max(len(short_trades), 1) * 100

    # Yearly
    yr_data = {}
    for t in trade_log:
        yr = t['date'][:4]
        if yr not in yr_data:
            yr_data[yr] = {'trades': 0, 'wins': 0, 'long': 0, 'short': 0}
        yr_data[yr]['trades'] += 1
        if t['pnl'] > 0: yr_data[yr]['wins'] += 1
        if t['direction'] == 'LONG': yr_data[yr]['long'] += 1
        else: yr_data[yr]['short'] += 1

    yearly_returns = {}
    running_eq = starting_cash
    for yr in sorted(yr_data.keys()):
        yr_start_eq = running_eq
        for t in trade_log:
            if t['date'][:4] == yr:
                running_eq = t['equity']
        yearly_returns[yr] = (running_eq / yr_start_eq - 1) * 100

    worst_yr = min(yearly_returns, key=yearly_returns.get) if yearly_returns else 'n/a'

    if verbose:
        lo = " (LONG-ONLY)" if long_only else ""
        mg = f" | min gap {min_gap*100:.2f}%" if min_gap > 0 else ""
        print(f"\n{'='*70}")
        print(f"  FRIDAY GAP{lo} — L:{long_ticker} S:{short_ticker}{mg}")
        print(f"{'='*70}")
        print(f"  Period: {common_dates[0]} to {common_dates[-1]} ({years:.1f} years)")
        print(f"  Signal: {signal_ticker} Thu close vs Fri open")
        print(f"  Trade: Fri open → Fri close (intraday)")
        print(f"  Slippage: {slippage*100:.2f}% | Alloc: {alloc_pct*100:.0f}%")

        print(f"\n  {'Metric':<20} {'Value':>15}")
        print(f"  {'-'*36}")
        print(f"  {'CAGR':<20} {ann_ret:>+14.2f}%")
        print(f"  {'Max DD':<20} {-max_dd*100:>+14.1f}%")
        print(f"  {'Trades':<20} {n_trades:>15}")
        print(f"  {'Win Rate':<20} {n_wins*100//max(n_trades,1):>14}%")
        print(f"  {'Avg P&L':<20} {avg_pnl:>+14.3f}%")
        print(f"  {'$10k →':<20} ${cash:>13,.0f}")

        print(f"\n  LONG:  {len(long_trades)} trades ({long_wins} wins, {long_wins*100//max(len(long_trades),1)}%) | Avg: {long_avg:+.3f}%")
        print(f"  SHORT: {len(short_trades)} trades ({short_wins} wins, {short_wins*100//max(len(short_trades),1)}%) | Avg: {short_avg:+.3f}%")

        print(f"\n  {'Year':<6} {'Return':>10} {'Trades':>8} {'Win%':>8} {'L/S':>8}")
        print(f"  {'-'*42}")
        for yr in sorted(yearly_returns):
            yd = yr_data[yr]
            wr = yd['wins'] * 100 // max(yd['trades'], 1)
            m = " <-- worst" if yr == worst_yr else ""
            print(f"  {yr:<6} {yearly_returns[yr]:>+9.1f}% {yd['trades']:>8} {wr:>7}% {yd['long']:>3}L/{yd['short']:>2}S{m}")

    return {
        'ann_ret': ann_ret, 'max_dd': max_dd, 'trades': n_trades,
        'win_rate': n_wins / max(n_trades, 1), 'avg_pnl': avg_pnl,
        'long_avg': long_avg, 'short_avg': short_avg,
        'n_long': len(long_trades), 'n_short': len(short_trades),
        'equity': cash, 'worst_yr': worst_yr,
        'worst_yr_ret': yearly_returns.get(worst_yr, 0), 'yearly': yearly_returns,
    }


# ── Sweep ──────────────────────────────────────────────────────

def run_sweep():
    print("=" * 110)
    print("  FRIDAY GAP CONTINUATION — Thu close vs Fri open → trade Fri intraday")
    print("  IBKR $0 commission")
    print("=" * 110)

    pairs = [
        ('SPY', 'SH',   'SPY/SH 1x',   False),
        ('SSO', 'SDS',  'SSO/SDS 2x',   False),
        ('UPRO','SPXU', 'UPRO/SPXU 3x', False),
        ('SPY', 'SH',   'SPY long-only', True),
        ('SSO', 'SDS',  'SSO long-only', True),
        ('UPRO','SPXU', 'UPRO long-only',True),
    ]

    print(f'\n  {"Config":<45} {"CAGR":>8} {"MaxDD":>8} {"Trades":>8} {"Win%":>6} {"AvgP&L":>8} {"L_avg":>8} {"S_avg":>8} {"$10k→":>12}')
    print(f'  {"-"*112}')

    for long_t, short_t, desc, lo in pairs:
        for mg in [0.0, 0.001, 0.002, 0.005]:
            for slip in [0.0, 0.0005]:
                mv = "any" if mg == 0 else f">{mg*100:.2f}%"
                sl = f"{slip*100:.2f}%"
                label = f"{desc} gap={mv} sl={sl}"
                r = run_friday_gap(
                    long_ticker=long_t, short_ticker=short_t,
                    slippage=slip, min_gap=mg, long_only=lo, verbose=False)
                if r['trades'] > 0:
                    print(f'  {label:<45} {r["ann_ret"]:>+7.2f}% {-r["max_dd"]*100:>+7.1f}% '
                          f'{r["trades"]:>8} {r["win_rate"]*100:>5.0f}% {r["avg_pnl"]:>+7.3f}% '
                          f'{r["long_avg"]:>+7.3f}% {r["short_avg"]:>+7.3f}% '
                          f'${r["equity"]:>10,.0f}')

    # Best config detailed
    print(f'\n\n  --- Best config detailed ---')
    run_friday_gap(long_ticker='UPRO', short_ticker='SPXU', slippage=0.0, min_gap=0.0, long_only=False, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Friday Gap Continuation Strategy")
    parser.add_argument('--long-ticker', default='SPY')
    parser.add_argument('--short-ticker', default='SH')
    parser.add_argument('--slippage', type=float, default=0.0)
    parser.add_argument('--min-gap', type=float, default=0.0)
    parser.add_argument('--long-only', action='store_true')
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()

    if args.sweep:
        run_sweep()
    else:
        run_friday_gap(
            long_ticker=args.long_ticker, short_ticker=args.short_ticker,
            slippage=args.slippage, min_gap=args.min_gap,
            long_only=args.long_only,
        )
