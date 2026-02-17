"""
Mid-Week Reversal Strategy
============================
Look at the Mon→Wed trend on SPY:
- If SPY rose Mon-Wed: buy a SHORT/inverse ETF Thursday open, sell Friday close
- If SPY fell Mon-Wed: buy a LONG ETF Thursday open, sell Friday close

Essentially a mean-reversion bet that the Thu-Fri move reverses the Mon-Wed trend.
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


# ── Mid-Week Reversal Strategy ─────────────────────────────────

def run_midweek_reversal(
    long_ticker: str = 'SPY',    # ETF to buy when Mon-Wed is down
    short_ticker: str = 'SH',    # inverse ETF to buy when Mon-Wed is up
    signal_ticker: str = 'SPY',  # ticker to measure Mon-Wed trend
    slippage: float = 0.002,
    starting_cash: float = 10000.0,
    alloc_pct: float = 1.0,
    long_only: bool = False,     # only trade when Mon-Wed is down (skip short)
    min_move: float = 0.0,       # min Mon-Wed move to trigger (0=any move)
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

    # Build arrays
    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in common_dates]
    weekdays = [d.weekday() for d in date_objs]

    # Find valid weeks: need Mon, Wed, Thu, Fri in same week
    # Group by ISO week
    weeks = {}
    for i, d in enumerate(date_objs):
        iso_yr, iso_wk, _ = d.isocalendar()
        key = (iso_yr, iso_wk)
        if key not in weeks:
            weeks[key] = {}
        weeks[key][weekdays[i]] = i  # weekday -> index in common_dates

    # Run backtest
    cash = starting_cash
    trade_log = []
    equity_curve = [cash]
    peak = cash
    max_dd = 0
    n_trades = 0
    n_wins = 0
    n_long = 0
    n_short = 0
    total_pnl = 0.0

    for key in sorted(weeks.keys()):
        wk = weeks[key]
        # Need at least Monday (0), Wednesday (2), Thursday (3), Friday (4)
        if 0 not in wk or 2 not in wk or 3 not in wk or 4 not in wk:
            continue

        mon_idx = wk[0]
        wed_idx = wk[2]
        thu_idx = wk[3]
        fri_idx = wk[4]

        # Mon-Wed trend: compare Monday open to Wednesday close
        sig = signal_ticker
        mon_open = price_data[sig]['opens'][common_dates[mon_idx]]
        wed_close = price_data[sig]['closes'][common_dates[wed_idx]]
        mon_wed_move = (wed_close - mon_open) / mon_open

        # Apply min move filter
        if min_move > 0 and abs(mon_wed_move) < min_move:
            continue

        # Determine trade direction
        if mon_wed_move > 0:
            if long_only:
                continue  # skip up-weeks in long-only mode
            # SPY rose Mon-Wed → buy SHORT ETF (bet on reversal down)
            trade_ticker = short_ticker
            direction = 'SHORT'
            n_short += 1
        else:
            # SPY fell Mon-Wed → buy LONG ETF (bet on reversal up)
            trade_ticker = long_ticker
            direction = 'LONG'
            n_long += 1

        # Entry: Thursday open
        entry_price = price_data[trade_ticker]['opens'][common_dates[thu_idx]]
        # Exit: Friday close
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
            'date_buy': common_dates[thu_idx],
            'date_sell': common_dates[fri_idx],
            'direction': direction,
            'ticker': trade_ticker,
            'mon_wed': mon_wed_move,
            'pnl': pnl,
            'equity': cash,
        })

        equity_curve.append(cash)
        if cash > peak:
            peak = cash
        dd = (peak - cash) / peak
        if dd > max_dd:
            max_dd = dd

    # Results
    total_ret = (cash / starting_cash - 1) * 100
    ann_ret = ((cash / starting_cash) ** (1 / years) - 1) * 100

    avg_pnl = total_pnl / max(n_trades, 1) * 100

    # Split stats by direction
    long_trades = [t for t in trade_log if t['direction'] == 'LONG']
    short_trades = [t for t in trade_log if t['direction'] == 'SHORT']
    long_wins = sum(1 for t in long_trades if t['pnl'] > 0)
    short_wins = sum(1 for t in short_trades if t['pnl'] > 0)
    long_avg = sum(t['pnl'] for t in long_trades) / max(len(long_trades), 1) * 100
    short_avg = sum(t['pnl'] for t in short_trades) / max(len(short_trades), 1) * 100

    # Yearly breakdown
    yr_data = {}
    for t in trade_log:
        yr = t['date_buy'][:4]
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
            if t['date_buy'][:4] == yr:
                running_eq = t['equity']
        yearly_returns[yr] = (running_eq / yr_start_eq - 1) * 100

    worst_yr = min(yearly_returns, key=yearly_returns.get) if yearly_returns else 'n/a'

    # SPY B&H benchmark
    spy_start = price_data[signal_ticker]['closes'][common_dates[0]]
    spy_end = price_data[signal_ticker]['closes'][common_dates[-1]]
    spy_ret = (spy_end / spy_start - 1) * 100
    spy_ann = ((spy_end / spy_start) ** (1/years) - 1) * 100

    min_str = f" | min move {min_move*100:.1f}%" if min_move > 0 else ""

    if verbose:
        print(f"\n{'='*70}")
        print(f"  MID-WEEK REVERSAL — Long:{long_ticker} Short:{short_ticker}{min_str}")
        print(f"{'='*70}")
        print(f"  Period: {common_dates[0]} to {common_dates[-1]} ({years:.1f} years)")
        print(f"  Signal: {signal_ticker} Mon open → Wed close")
        print(f"  Trade: Buy Thu open → Sell Fri close")
        print(f"  Slippage: {slippage*100:.1f}% | Alloc: {alloc_pct*100:.0f}%")

        print(f"\n  {'Strategy':<20} {'End Value':>12} {'Total':>10} {'Annual':>8} {'MaxDD':>8} {'Trades':>8}")
        print(f"  {'-'*68}")
        print(f"  {'REVERSAL':<20} ${cash:>11,.2f} {total_ret:>+9.1f}% {ann_ret:>+7.1f}% {-max_dd*100:>7.1f}% {n_trades:>8}")
        print(f"  {'SPY B&H':<20} ${starting_cash*(1+spy_ret/100):>11,.2f} {spy_ret:>+9.1f}% {spy_ann:>+7.1f}%")

        print(f"\n  Overall: {n_wins}/{n_trades} wins ({n_wins*100//max(n_trades,1)}%) | Avg P&L: {avg_pnl:+.3f}%")
        print(f"  LONG trades:  {len(long_trades)} ({long_wins} wins, {long_wins*100//max(len(long_trades),1)}%) | Avg: {long_avg:+.3f}%")
        print(f"  SHORT trades: {len(short_trades)} ({short_wins} wins, {short_wins*100//max(len(short_trades),1)}%) | Avg: {short_avg:+.3f}%")

        print(f"\n  {'Year':<6} {'Return':>10} {'Trades':>8} {'Win%':>8} {'L/S':>8}")
        print(f"  {'-'*42}")
        for yr in sorted(yearly_returns):
            yd = yr_data[yr]
            wr = yd['wins'] * 100 // max(yd['trades'], 1)
            m = " <-- worst" if yr == worst_yr else ""
            print(f"  {yr:<6} {yearly_returns[yr]:>+9.1f}% {yd['trades']:>8} {wr:>7}% {yd['long']:>3}L/{yd['short']:>2}S{m}")

    return {
        'ann_ret': ann_ret,
        'max_dd': max_dd,
        'total_ret': total_ret,
        'trades': n_trades,
        'win_rate': n_wins / max(n_trades, 1),
        'avg_pnl': avg_pnl,
        'long_avg': long_avg,
        'short_avg': short_avg,
        'n_long': len(long_trades),
        'n_short': len(short_trades),
        'equity': cash,
        'worst_yr': worst_yr,
        'worst_yr_ret': yearly_returns.get(worst_yr, 0),
        'yearly': yearly_returns,
    }


# ── Sweep ──────────────────────────────────────────────────────

def run_sweep():
    print("=" * 80)
    print("  MID-WEEK REVERSAL — FULL SWEEP")
    print("  Mon open → Wed close trend → buy reverse Thu open → sell Fri close")
    print("=" * 80)

    # ETF pairs: (long_ticker, short_ticker, description)
    pairs = [
        ('SPY',  'SH',   'SPY 1x / SH -1x'),
        ('SSO',  'SDS',  'SSO 2x / SDS -2x'),
        ('UPRO', 'SPXU', 'UPRO 3x / SPXU -3x'),
        ('SPY',  'SPY',  'SPY long-only (ignore short signal)'),
    ]

    slip = 0.002

    for long_t, short_t, desc in pairs:
        print(f'\n  --- {desc} (0.2% slippage) ---')
        print(f'  {"Config":<40} {"CAGR":>8} {"MaxDD":>8} {"Trades":>8} {"Win%":>6} {"AvgP&L":>8} {"L_avg":>8} {"S_avg":>8} {"$10k→":>12}')
        print(f'  {"-"*108}')

        for alloc in [1.0, 0.5]:
            for min_mv in [0.0, 0.005, 0.01, 0.02]:
                mv_str = "any" if min_mv == 0 else f">{min_mv*100:.1f}%"
                label = f"alloc={alloc*100:.0f}% min={mv_str}"
                r = run_midweek_reversal(
                    long_ticker=long_t, short_ticker=short_t,
                    slippage=slip, alloc_pct=alloc, min_move=min_mv,
                    verbose=False)
                print(f'  {label:<40} {r["ann_ret"]:>+7.2f}% {-r["max_dd"]*100:>+7.1f}% '
                      f'{r["trades"]:>8} {r["win_rate"]*100:>5.0f}% {r["avg_pnl"]:>+7.3f}% '
                      f'{r["long_avg"]:>+7.3f}% {r["short_avg"]:>+7.3f}% '
                      f'${r["equity"]:>10,.0f}')


# ── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mid-Week Reversal Strategy")
    parser.add_argument('--long-ticker', default='SPY')
    parser.add_argument('--short-ticker', default='SH')
    parser.add_argument('--slippage', type=float, default=0.002)
    parser.add_argument('--alloc', type=float, default=1.0)
    parser.add_argument('--min-move', type=float, default=0.0)
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()

    if args.sweep:
        run_sweep()
    else:
        run_midweek_reversal(
            long_ticker=args.long_ticker,
            short_ticker=args.short_ticker,
            slippage=args.slippage,
            alloc_pct=args.alloc,
            min_move=args.min_move,
        )
