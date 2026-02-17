"""
Thursday Gold Timing Strategy
==============================
Buy a gold ETF on Thursday, sell at Friday close.
Tests: GLD (1x), UGL (2x) — buy at open vs close on Thursday.

Uses OHLC data to test both entry timings.
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


# ── Thursday Gold Strategy ─────────────────────────────────────

DAY_NAMES = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}


def run_gold_weekly(
    trade_ticker: str = 'GLD',
    buy_day: int = 4,            # 0=Mon..4=Fri — day to buy
    buy_at: str = 'close',       # 'open' or 'close' on buy day
    sell_day: int = 0,           # 0=Mon..4=Fri — day to sell
    sell_at: str = 'close',      # 'open' or 'close' on sell day
    slippage: float = 0.002,
    starting_cash: float = 10000.0,
    alloc_pct: float = 1.0,
    cache_dir: str = "state/etf_cache",
    start: str = "1999-01-01",
    end: str = "2025-01-01",
    verbose: bool = True,
):
    trade_cost = slippage

    # Load OHLC data
    dates, opens, closes = load_etf_ohlc_cached(trade_ticker, cache_dir=cache_dir, start=start, end=end)
    n = len(dates)
    years = n / 252

    # Also load GLD for benchmark if trading leveraged
    if trade_ticker != 'GLD':
        gld_dates, gld_opens, gld_closes = load_etf_ohlc_cached('GLD', cache_dir=cache_dir, start=start, end=end)
        gld_map_close = dict(zip(gld_dates, gld_closes))
    else:
        gld_map_close = dict(zip(dates, closes))

    # Day-of-week detection
    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    weekdays = [d.weekday() for d in date_objs]

    # Find valid buy→sell pairs
    # If sell_day > buy_day: same week (e.g., buy Mon sell Fri)
    # If sell_day <= buy_day: next week (e.g., buy Fri sell Mon = weekend hold)
    pairs = []
    for i in range(n):
        if weekdays[i] == buy_day:
            # Find next occurrence of sell_day after i
            for j in range(i + 1, min(i + 7, n)):
                if weekdays[j] == sell_day:
                    pairs.append((i, j))
                    break

    # Run backtest
    cash = starting_cash
    trade_log = []
    equity_curve = [cash]
    peak = cash
    max_dd = 0
    n_trades = 0
    n_wins = 0
    total_pnl = 0.0

    for buy_idx, sell_idx in pairs:
        # Entry price
        entry_price = opens[buy_idx] if buy_at == 'open' else closes[buy_idx]
        # Exit price
        exit_price = opens[sell_idx] if sell_at == 'open' else closes[sell_idx]

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
            'date_buy': dates[buy_idx],
            'date_sell': dates[sell_idx],
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

    # GLD B&H benchmark
    gld_start = gld_map_close.get(dates[0], closes[0])
    gld_end = gld_map_close.get(dates[-1], closes[-1])
    gld_ret = (gld_end / gld_start - 1) * 100
    gld_ann = ((gld_end / gld_start) ** (1/years) - 1) * 100

    avg_pnl = total_pnl / max(n_trades, 1) * 100

    # Yearly breakdown
    yr_data = {}
    for t in trade_log:
        yr = t['date_buy'][:4]
        if yr not in yr_data:
            yr_data[yr] = {'trades': 0, 'wins': 0}
        yr_data[yr]['trades'] += 1
        if t['pnl'] > 0:
            yr_data[yr]['wins'] += 1

    yearly_returns = {}
    running_eq = starting_cash
    for yr in sorted(yr_data.keys()):
        yr_start_eq = running_eq
        for t in trade_log:
            if t['date_buy'][:4] == yr:
                running_eq = t['equity']
        yearly_returns[yr] = (running_eq / yr_start_eq - 1) * 100

    worst_yr = min(yearly_returns, key=yearly_returns.get) if yearly_returns else 'n/a'

    buy_str = f"{DAY_NAMES[buy_day]} {buy_at}"
    sell_str = f"{DAY_NAMES[sell_day]} {sell_at}"

    if verbose:
        print(f"\n{'='*70}")
        print(f"  GOLD WEEKLY — {trade_ticker} | Buy {buy_str} → Sell {sell_str}")
        print(f"{'='*70}")
        print(f"  Period: {dates[0]} to {dates[-1]} ({years:.1f} years)")
        print(f"  Slippage: {slippage*100:.1f}% | Alloc: {alloc_pct*100:.0f}%")
        print(f"  Valid pairs found: {len(pairs)}")

        print(f"\n  {'Strategy':<20} {'End Value':>12} {'Total':>10} {'Annual':>8} {'MaxDD':>8} {'Trades':>8}")
        print(f"  {'-'*68}")
        print(f"  {'GOLD WEEKLY':<20} ${cash:>11,.2f} {total_ret:>+9.1f}% {ann_ret:>+7.1f}% {-max_dd*100:>7.1f}% {n_trades:>8}")
        print(f"  {'GLD B&H':<20} ${starting_cash*(1+gld_ret/100):>11,.2f} {gld_ret:>+9.1f}% {gld_ann:>+7.1f}%")

        print(f"\n  Win rate: {n_wins}/{n_trades} ({n_wins*100//max(n_trades,1)}%)")
        print(f"  Avg trade P&L: {avg_pnl:+.3f}%")

        print(f"\n  {'Year':<6} {'Return':>10} {'Trades':>8} {'Win%':>8}")
        print(f"  {'-'*34}")
        for yr in sorted(yearly_returns):
            yd = yr_data[yr]
            wr = yd['wins'] * 100 // max(yd['trades'], 1)
            m = " <-- worst" if yr == worst_yr else ""
            print(f"  {yr:<6} {yearly_returns[yr]:>+9.1f}% {yd['trades']:>8} {wr:>7}%{m}")

    return {
        'ann_ret': ann_ret,
        'max_dd': max_dd,
        'total_ret': total_ret,
        'trades': n_trades,
        'win_rate': n_wins / max(n_trades, 1),
        'avg_pnl': avg_pnl,
        'equity': cash,
        'worst_yr': worst_yr,
        'worst_yr_ret': yearly_returns.get(worst_yr, 0),
        'yearly': yearly_returns,
    }


# ── Sweep ──────────────────────────────────────────────────────

def run_sweep():
    print("=" * 80)
    print("  GOLD WEEKLY TIMING — FULL SWEEP")
    print("=" * 80)

    # All day combos to test
    combos = [
        # (buy_day, buy_at, sell_day, sell_at, description)
        (3, 'close', 4, 'close', 'Thu close → Fri close'),
        (3, 'open',  4, 'close', 'Thu open  → Fri close'),
        (4, 'close', 0, 'close', 'Fri close → Mon close (weekend)'),
        (4, 'open',  0, 'close', 'Fri open  → Mon close (weekend)'),
        (4, 'close', 0, 'open',  'Fri close → Mon open  (weekend)'),
        (3, 'close', 0, 'close', 'Thu close → Mon close (Thu+wknd)'),
        (3, 'open',  0, 'close', 'Thu open  → Mon close (Thu+wknd)'),
        (3, 'close', 0, 'open',  'Thu close → Mon open  (Thu+wknd)'),
    ]

    slip = 0.002  # realistic IBKR slippage

    for ticker in ['GLD', 'UGL', 'SHNY']:
        print(f'\n  --- {ticker} (0.2% slippage, 100% alloc) ---')
        print(f'  {"Pattern":<35} {"CAGR":>8} {"MaxDD":>8} {"Trades":>8} {"Win%":>6} {"AvgP&L":>8} {"Yrs":>5} {"$10k→":>12}')
        print(f'  {"-"*98}')

        for bd, ba, sd, sa, desc in combos:
            try:
                r = run_gold_weekly(
                    trade_ticker=ticker, buy_day=bd, buy_at=ba,
                    sell_day=sd, sell_at=sa, slippage=slip,
                    alloc_pct=1.0, verbose=False)
                yrs = len(r['yearly'])
                print(f'  {desc:<35} {r["ann_ret"]:>+7.2f}% {-r["max_dd"]*100:>+7.1f}% '
                      f'{r["trades"]:>8} {r["win_rate"]*100:>5.0f}% {r["avg_pnl"]:>+7.3f}% '
                      f'{yrs:>5} ${r["equity"]:>10,.0f}')
            except Exception as e:
                print(f'  {desc:<35} ERROR: {e}')


# ── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thursday Gold Timing Strategy")
    parser.add_argument('--trade-ticker', default='GLD')
    parser.add_argument('--buy-at', choices=['open', 'close'], default='close')
    parser.add_argument('--slippage', type=float, default=0.002)
    parser.add_argument('--alloc', type=float, default=1.0)
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()

    if args.sweep:
        run_sweep()
    else:
        run_gold_thursday(
            trade_ticker=args.trade_ticker,
            buy_at=args.buy_at,
            slippage=args.slippage,
            alloc_pct=args.alloc,
        )
