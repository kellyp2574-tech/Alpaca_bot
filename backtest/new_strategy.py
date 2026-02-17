"""
Monday Dip Timing Strategy
===========================
Buy SPY (or leveraged ETF) when Monday's close < prior Friday's close.
Sell 2 trading days later (Wednesday close).

Uses daily close as proxy for "1hr before close" (very close for SPY).
"""
import argparse
import json
import os
from datetime import datetime

import yfinance as yf


# ── Data Loading ────────────────────────────────────────────────

def load_etf_cached(ticker, cache_dir="state/etf_cache", start="1999-01-01", end="2025-01-01"):
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


# ── Monday Dip Strategy ─────────────────────────────────────────

def sma(closes, period):
    if len(closes) < period:
        return closes[-1] if closes else 0
    return sum(closes[-period:]) / period


def run_monday_dip(
    signal_ticker: str = 'SPY',   # ticker to check for dip signal
    trade_ticker: str = 'SPY',    # ticker to actually trade (SPY, SSO, UPRO)
    hold_days: int = 2,           # sell N trading days after buy
    commission: float = 0.001,
    starting_cash: float = 10000.0,
    cash_apy: float = 0.0335,
    alloc_pct: float = 1.0,       # fraction to invest per trade
    sma_filter: int = 0,          # only buy when SPY > N-day SMA (0=off)
    stop_loss: float = 0.0,       # sell early if position drops X% (0=off)
    take_profit: float = 0.0,     # sell early if position gains X% before hold_days (0=off)
    max_dip: float = 0.0,         # skip if Monday dip > X% (e.g. 0.03 = skip >3% drops, 0=off)
    min_dip: float = 0.0,         # skip if Monday dip < X% (e.g. 0.002 = need >0.2% dip, 0=off)
    cache_dir: str = "state/etf_cache",
    start: str = "1999-01-01",
    end: str = "2025-01-01",
    verbose: bool = True,
):
    daily_cash_rate = (1 + cash_apy) ** (1/252) - 1

    # Load data
    tickers = list(set([signal_ticker, trade_ticker, 'SPY']))
    price_data = {}
    date_data = {}
    for t in tickers:
        dates, closes = load_etf_cached(t, cache_dir=cache_dir, start=start, end=end)
        price_data[t] = dict(zip(dates, closes))
        date_data[t] = set(dates)

    common_dates = sorted(set.intersection(*(date_data[t] for t in tickers)))
    prices = {t: [price_data[t][d] for d in common_dates] for t in tickers}
    n = len(common_dates)
    years = n / 252

    # Parse dates to get day-of-week
    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in common_dates]
    weekdays = [d.weekday() for d in date_objs]  # 0=Mon, 4=Fri

    # Identify first trading day of each week
    # A new week starts when the weekday is <= the previous day's weekday
    # (e.g., Mon after Fri, or Tue after Fri if Mon was holiday)
    week_starts = []
    for i in range(1, n):
        if weekdays[i] <= weekdays[i-1]:
            week_starts.append(i)

    if verbose:
        lev_str = f" | Signal: {signal_ticker}" if signal_ticker != trade_ticker else ""
        print(f"  Period: {common_dates[0]} to {common_dates[-1]} ({years:.1f} years)")
        print(f"  Trade: {trade_ticker}{lev_str} | Hold: {hold_days} days | Alloc: {alloc_pct*100:.0f}%")
        print(f"  Commission: {commission*100:.2f}% | Cash APY: {cash_apy*100:.2f}%")
        print(f"  Weekly check days: {len(week_starts)}")

    # State
    equity = starting_cash
    invested = 0.0
    cash = starting_cash
    holding = False
    buy_bar = -1
    entry_price = 0
    n_trades = 0
    trade_log = []
    equity_curve = [equity]
    peak = equity
    max_dd = 0
    invested_days = 0

    for i in range(1, n):
        # Cash earns interest when not invested
        if not holding:
            interest = cash * daily_cash_rate
            cash += interest

        # Mark to market if holding
        if holding:
            prev = prices[trade_ticker][i-1]
            cur = prices[trade_ticker][i]
            invested *= (1 + (cur - prev) / prev)
            invested_days += 1

        equity = invested + cash
        equity_curve.append(equity)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd

        # Check stop-loss first (before hold_days check)
        if holding and stop_loss > 0:
            cur_pnl = (prices[trade_ticker][i] - entry_price) / entry_price
            if cur_pnl <= -stop_loss:
                invested *= (1 - commission)
                cash += invested
                invested = 0
                holding = False
                trade_log.append({
                    'date': common_dates[i],
                    'action': 'STOP',
                    'ticker': trade_ticker,
                    'pnl': cur_pnl,
                    'days': i - buy_bar,
                    'equity': cash,
                })
                n_trades += 1

        # Check take-profit (before hold_days, after stop-loss)
        if holding and take_profit > 0 and (i - buy_bar) < hold_days:
            cur_pnl = (prices[trade_ticker][i] - entry_price) / entry_price
            if cur_pnl >= take_profit:
                invested *= (1 - commission)
                cash += invested
                invested = 0
                holding = False
                trade_log.append({
                    'date': common_dates[i],
                    'action': 'TP',
                    'ticker': trade_ticker,
                    'pnl': cur_pnl,
                    'days': i - buy_bar,
                    'equity': cash,
                })
                n_trades += 1

        # Check if we need to sell (held for hold_days)
        if holding and (i - buy_bar) >= hold_days:
            sell_pnl = (prices[trade_ticker][i] - entry_price) / entry_price
            invested *= (1 - commission)
            cash += invested
            invested = 0
            holding = False
            trade_log.append({
                'date': common_dates[i],
                'action': 'SELL',
                'ticker': trade_ticker,
                'pnl': sell_pnl,
                'days': i - buy_bar,
                'equity': cash,
            })
            n_trades += 1

        # Check if this is a week-start day and we're not holding
        if not holding and i in week_starts:
            # Compare: this day's close vs previous trading day's close
            signal_today = prices[signal_ticker][i]
            signal_prev = prices[signal_ticker][i-1]
            dip_pct = (signal_prev - signal_today) / signal_prev  # positive = dip

            if signal_today < signal_prev:
                # Apply filters
                skip = False

                # SMA filter: only buy in uptrend
                if sma_filter > 0 and i >= sma_filter:
                    spy_sma = sma(prices['SPY'][:i+1], sma_filter)
                    if prices['SPY'][i] < spy_sma:
                        skip = True

                # Dip magnitude filters
                if max_dip > 0 and dip_pct > max_dip:
                    skip = True  # dip too large (panic selling)
                if min_dip > 0 and dip_pct < min_dip:
                    skip = True  # dip too small

                if not skip:
                    # Monday dip detected — BUY
                    invest_amount = cash * alloc_pct
                    cash -= invest_amount
                    invested = invest_amount * (1 - commission)
                    entry_price = prices[trade_ticker][i]
                    buy_bar = i
                    holding = True
                    trade_log.append({
                        'date': common_dates[i],
                        'action': 'BUY',
                        'ticker': trade_ticker,
                        'pnl': None,
                        'days': None,
                        'equity': invested + cash,
                        'signal': f"{signal_ticker} dip {dip_pct*100:.2f}%",
                    })
                    n_trades += 1

    # Final equity
    equity = invested + cash
    total_ret = (equity / starting_cash - 1) * 100
    ann_ret = ((equity / starting_cash) ** (1/years) - 1) * 100

    # SPY B&H benchmark
    spy_dates, spy_closes = load_etf_cached('SPY', cache_dir=cache_dir, start=start, end=end)
    spy_map = dict(zip(spy_dates, spy_closes))
    spy_start = spy_map.get(common_dates[0], spy_closes[0])
    spy_end = spy_map.get(common_dates[-1], spy_closes[-1])
    spy_ret = (spy_end / spy_start - 1) * 100
    spy_ann = ((spy_end / spy_start) ** (1/years) - 1) * 100

    # Trade stats (include stops and TPs as exits)
    exits = [t for t in trade_log if t['action'] in ('SELL', 'STOP', 'TP')]
    wins = sum(1 for t in exits if t['pnl'] and t['pnl'] > 0)
    total_sells = len(exits)
    stops = sum(1 for t in exits if t['action'] == 'STOP')
    tps = sum(1 for t in exits if t['action'] == 'TP')
    avg_pnl = sum(t['pnl'] for t in exits) * 100 / max(total_sells, 1)
    avg_hold = sum(t['days'] for t in exits) / max(total_sells, 1)
    pct_invested = invested_days / n * 100

    if verbose:
        print(f"\n  {'Strategy':<20} {'End Value':>12} {'Total':>10} {'Annual':>8} {'MaxDD':>8} {'Trades':>8}")
        print(f"  {'-'*66}")
        print(f"  {'MON DIP':<20} ${equity:>11,.2f} {total_ret:>+9.1f}% {ann_ret:>+7.1f}% {-max_dd*100:>7.1f}% {total_sells:>8}")
        print(f"  {'SPY B&H':<20} ${starting_cash*(1+spy_ret/100):>11,.2f} {spy_ret:>+9.1f}% {spy_ann:>+7.1f}%")
        print(f"\n  Win rate: {wins}/{total_sells} ({wins*100//max(total_sells,1)}%)")
        print(f"  Avg trade P&L: {avg_pnl:+.2f}%")
        print(f"  Avg hold: {avg_hold:.1f} days")
        print(f"  Time invested: {pct_invested:.1f}%")
        print(f"  Weeks checked: {len(week_starts)} | Trades taken: {total_sells} ({total_sells*100//len(week_starts)}% of weeks)")

        # Yearly breakdown
        print(f"\n  {'Year':<6} {'Strategy':>10} {'SPY B&H':>10} {'Trades':>8} {'Win%':>8}")
        print(f"  {'-'*44}")

        yr_start_eq = equity_curve[0]
        yr_start_spy = prices['SPY'][0] if 'SPY' in prices else spy_start
        current_year = common_dates[0][:4]
        yr_trades = []

        for j in range(1, n):
            yr = common_dates[j][:4]
            # Collect trades for this year
            for t in sells:
                if t['date'][:4] == current_year and t not in yr_trades:
                    yr_trades.append(t)

            if yr != current_year or j == n - 1:
                idx = j - 1 if yr != current_year else j
                yr_ret = (equity_curve[idx] / yr_start_eq - 1) * 100

                spy_idx_price = spy_map.get(common_dates[idx], spy_end)
                spy_yr = (spy_idx_price / yr_start_spy - 1) * 100

                yr_wins = sum(1 for t in yr_trades if t['pnl'] > 0)
                yr_total = len(yr_trades)
                win_pct = f"{yr_wins*100//yr_total}%" if yr_total > 0 else "n/a"

                print(f"  {current_year:<6} {yr_ret:>+9.1f}% {spy_yr:>+9.1f}% {yr_total:>8} {win_pct:>8}")

                yr_start_eq = equity_curve[idx]
                yr_start_spy = spy_idx_price
                current_year = yr
                yr_trades = []

        # Trade log (last 30 trades)
        if verbose and len(trade_log) > 0:
            show_n = min(60, len(trade_log))
            print(f"\n  Last {show_n//2} round trips:")
            print(f"  {'Date':<12} {'Action':<6} {'Ticker':<6} {'P&L':>8} {'Hold':>6} {'Equity':>12}")
            print(f"  {'-'*52}")
            for t in trade_log[-show_n:]:
                pnl_str = f"{t['pnl']*100:+.2f}%" if t['pnl'] is not None else ""
                days_str = f"{t['days']}d" if t['days'] is not None else ""
                print(f"  {t['date']:<12} {t['action']:<6} {t['ticker']:<6} {pnl_str:>8} {days_str:>6} ${t['equity']:>11,.2f}")

    return {
        'equity': equity, 'total_ret': total_ret, 'ann_ret': ann_ret,
        'max_dd': max_dd, 'trades': total_sells, 'years': years,
        'win_rate': wins / max(total_sells, 1), 'avg_pnl': avg_pnl,
        'pct_invested': pct_invested,
    }


# ── CLI ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Monday Dip Timing Strategy")
    p.add_argument("--signal-ticker", default="SPY", help="Ticker to check for Monday dip")
    p.add_argument("--trade-ticker", default="SPY", help="Ticker to trade (SPY, SSO, UPRO)")
    p.add_argument("--hold-days", type=int, default=2, help="Days to hold after buy")
    p.add_argument("--commission", type=float, default=0.001)
    p.add_argument("--starting-cash", type=float, default=10000.0)
    p.add_argument("--cash-apy", type=float, default=0.0335)
    p.add_argument("--alloc-pct", type=float, default=1.0)
    p.add_argument("--cache-dir", default="state/etf_cache")
    p.add_argument("--start", default="1999-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--sweep", action="store_true", help="Sweep hold days and allocation")
    args = p.parse_args()

    if args.sweep:
        print("=" * 90)
        print("  MONDAY DIP TIMING — SWEEP")
        print("=" * 90)

        results = []
        for ticker in ['SPY', 'SSO', 'UPRO']:
            for hold in [1, 2, 3, 5]:
                for alloc in [0.5, 1.0]:
                    r = run_monday_dip(
                        signal_ticker='SPY', trade_ticker=ticker,
                        hold_days=hold, alloc_pct=alloc,
                        commission=args.commission, starting_cash=args.starting_cash,
                        cash_apy=args.cash_apy, cache_dir=args.cache_dir,
                        start=args.start, end=args.end, verbose=False,
                    )
                    label = f"{ticker} hold={hold}d alloc={alloc*100:.0f}%"
                    results.append((label, r))

        print(f"\n  {'Config':<32} {'Return':>10} {'Annual':>8} {'MaxDD':>8} {'Trades':>8} {'Win%':>6} {'Invested':>9}")
        print(f"  {'-'*85}")
        for label, r in sorted(results, key=lambda x: -x[1]['ann_ret']):
            print(f"  {label:<32} {r['total_ret']:>+9.1f}% {r['ann_ret']:>+7.1f}% {-r['max_dd']*100:>+7.1f}% "
                  f"{r['trades']:>8} {r['win_rate']*100:>5.0f}% {r['pct_invested']:>8.1f}%")
    else:
        print("=" * 80)
        print(f"  MONDAY DIP TIMING — {args.trade_ticker}")
        print("=" * 80)
        run_monday_dip(
            signal_ticker=args.signal_ticker, trade_ticker=args.trade_ticker,
            hold_days=args.hold_days, commission=args.commission,
            starting_cash=args.starting_cash, cash_apy=args.cash_apy,
            alloc_pct=args.alloc_pct, cache_dir=args.cache_dir,
            start=args.start, end=args.end,
        )


if __name__ == "__main__":
    main()
