"""
Leverage Comparison: 1x vs 2x vs 3x MA Crossover
==================================================
Tests the 5-day exit confirmation (2-day entry) with 100d SMA, 3% buffer
across 1x (QQQ/TLT), 2x (QLD/UBT), and 3x (TQQQ/TMF).

Includes: CAGR, Max DD, worst 3-year, rolling 5-year min, slippage sensitivity.
"""
import json
import math
import os
from datetime import datetime


# ── Data Loading ──────────────────────────────────────────────

def load_etf_cached(ticker, cache_dir="state/etf_cache", start="2010-02-12", end="2025-01-01"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_daily.json")

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            data = json.load(f)
        dates = [d for d in data['dates'] if start <= d <= end]
        closes = [data['closes'][data['dates'].index(d)] for d in dates]
        return dates, closes

    import yfinance as yf
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


def sma(closes, period):
    if len(closes) < period:
        return closes[-1] if closes else 0
    return sum(closes[-period:]) / period


# ── MA Rotation Backtest ──────────────────────────────────────

def run_ma(
    trade_growth='QQQ', trade_safety='TLT',
    alt_ticker='',
    ma_period=100, buffer_pct=0.03,
    confirm_entry=2, confirm_exit=5,
    slippage=0.002,
    starting_cash=10000.0, cash_apy=0.0335,
    cache_dir="state/etf_cache",
    start="2010-02-12", end="2025-01-01",
):
    daily_cash_rate = (1 + cash_apy) ** (1 / 252) - 1

    sig_growth, sig_safety = 'QQQ', 'TLT'

    all_tickers = list(set([
        sig_growth, sig_safety, trade_growth, trade_safety, 'SPY'
    ] + ([alt_ticker] if alt_ticker else [])))

    price_data = {}; date_data = {}
    for t in all_tickers:
        dates, closes = load_etf_cached(t, cache_dir=cache_dir, start=start, end=end)
        price_data[t] = dict(zip(dates, closes))
        date_data[t] = set(dates)

    common_dates = sorted(set.intersection(*(date_data[t] for t in all_tickers)))
    prices = {t: [price_data[t][d] for d in common_dates] for t in all_tickers}
    n = len(common_dates)
    years = n / 252

    # State
    cash = starting_cash
    invested = 0.0
    holding = None; direction = None
    entry_price = 0.0; entry_bar = 0
    qa, qb, ta, tb = 0, 0, 0, 0
    n_trades = 0; trade_log = []

    equity_curve = [starting_cash]
    peak_eq = starting_cash; max_dd = 0.0
    growth_days = 0; safety_days = 0; alt_days = 0; cash_days = 0

    for i in range(1, n):
        cash += cash * daily_cash_rate

        if holding is not None:
            prev_p = prices[holding][i - 1]
            cur_p = prices[holding][i]
            if prev_p > 0:
                invested *= (1 + (cur_p - prev_p) / prev_p)

        if direction == 'growth': growth_days += 1
        elif direction == 'safety': safety_days += 1
        elif direction == 'alt': alt_days += 1
        else: cash_days += 1

        total_eq = invested + cash
        equity_curve.append(total_eq)
        if total_eq > peak_eq: peak_eq = total_eq
        dd = (peak_eq - total_eq) / peak_eq
        if dd > max_dd: max_dd = dd

        if i < ma_period:
            continue

        qqq_sma = sma(prices[sig_growth][:i + 1], ma_period)
        tlt_sma = sma(prices[sig_safety][:i + 1], ma_period)
        qqq_price = prices[sig_growth][i]
        tlt_price = prices[sig_safety][i]

        if qqq_price > qqq_sma * (1 + buffer_pct): qa += 1; qb = 0
        elif qqq_price < qqq_sma * (1 - buffer_pct): qb += 1; qa = 0

        if tlt_price > tlt_sma * (1 + buffer_pct): ta += 1; tb = 0
        elif tlt_price < tlt_sma * (1 - buffer_pct): tb += 1; ta = 0

        qqq_above = qa >= confirm_entry
        qqq_below = qb >= confirm_exit
        tlt_above = ta >= confirm_entry
        tlt_below = tb >= confirm_exit

        fallback = alt_ticker if alt_ticker else None
        target = holding; target_dir = direction

        if direction == 'growth':
            if qqq_below:
                if tlt_above: target = trade_safety; target_dir = 'safety'
                else: target = fallback; target_dir = 'alt' if fallback else None
        elif direction == 'safety':
            if qqq_above: target = trade_growth; target_dir = 'growth'
            elif tlt_below: target = fallback; target_dir = 'alt' if fallback else None
        elif direction == 'alt':
            if qqq_above: target = trade_growth; target_dir = 'growth'
            elif tlt_above: target = trade_safety; target_dir = 'safety'
        else:
            if qqq_above: target = trade_growth; target_dir = 'growth'
            elif tlt_above: target = trade_safety; target_dir = 'safety'

        if target != holding:
            if holding is not None:
                hold_pnl = (prices[holding][i] - entry_price) / entry_price
                invested *= (1 - slippage); cash += invested; invested = 0
                trade_log.append({'date': common_dates[i], 'action': 'SELL',
                    'ticker': holding, 'pnl': hold_pnl, 'days': i - entry_bar})
                n_trades += 1

            if target is not None:
                invest_amt = cash + invested
                cash = 0; invested = invest_amt * (1 - slippage)
                entry_price = prices[target][i]; entry_bar = i
                trade_log.append({'date': common_dates[i], 'action': 'BUY',
                    'ticker': target, 'pnl': None, 'days': None})
                n_trades += 1
            else:
                cash = cash + invested; invested = 0

            holding = target; direction = target_dir

    total_eq = invested + cash
    total_ret = (total_eq / starting_cash - 1) * 100
    ann_ret = ((total_eq / starting_cash) ** (1 / years) - 1) * 100

    # SPY B&H
    spy_ret = (prices['SPY'][-1] / prices['SPY'][0] - 1) * 100
    spy_ann = ((1 + spy_ret / 100) ** (1 / years) - 1) * 100

    # Trade ticker B&H
    g_bh_ret = (prices[trade_growth][-1] / prices[trade_growth][0] - 1) * 100
    g_bh_ann = ((1 + g_bh_ret / 100) ** (1 / years) - 1) * 100
    g_bh_peak = prices[trade_growth][0]; g_bh_dd = 0
    for p in prices[trade_growth]:
        if p > g_bh_peak: g_bh_peak = p
        d = (g_bh_peak - p) / g_bh_peak
        if d > g_bh_dd: g_bh_dd = d

    # Rolling windows
    bars_3yr = 252 * 3; bars_5yr = 252 * 5
    worst_3yr = float('inf'); worst_5yr = float('inf')
    neg_5yr = 0; total_5yr_windows = 0

    for j in range(bars_3yr, n):
        ret = (equity_curve[j] / equity_curve[j - bars_3yr]) ** (1 / 3) - 1
        if ret < worst_3yr: worst_3yr = ret

    for j in range(bars_5yr, n):
        ret = (equity_curve[j] / equity_curve[j - bars_5yr]) ** (1 / 5) - 1
        if ret < worst_5yr: worst_5yr = ret
        total_5yr_windows += 1
        if ret < 0: neg_5yr += 1

    # Longest underwater
    uw_start = 0; longest_uw = 0; uw_peak = equity_curve[0]
    for j in range(1, n):
        if equity_curve[j] >= uw_peak:
            uw_peak = equity_curve[j]; uw_start = j
        else:
            uw_len = j - uw_start
            if uw_len > longest_uw: longest_uw = uw_len

    # Yearly
    yearly = {}
    yr_start = equity_curve[0]
    current_year = common_dates[0][:4]
    for j in range(1, n):
        yr = common_dates[j][:4]
        if yr != current_year or j == n - 1:
            idx = j - 1 if yr != current_year else j
            yearly[current_year] = (equity_curve[idx] / yr_start - 1) * 100
            yr_start = equity_curve[idx]
            current_year = yr

    sells = [t for t in trade_log if t['action'] == 'SELL']
    wins = sum(1 for t in sells if (t.get('pnl') or 0) > 0)
    avg_hold = sum((t.get('days') or 0) for t in sells) / max(len(sells), 1)

    total_days = growth_days + safety_days + alt_days + cash_days
    pct_g = growth_days / total_days * 100 if total_days else 0
    pct_s = safety_days / total_days * 100 if total_days else 0

    return {
        'ann_ret': ann_ret, 'total_ret': total_ret, 'max_dd': max_dd * 100,
        'equity': total_eq, 'trades': len(sells), 'years': years,
        'worst_3yr': worst_3yr * 100, 'worst_5yr': worst_5yr * 100,
        'neg_5yr': neg_5yr, 'longest_uw': longest_uw / 21,  # approx months
        'win_rate': wins / max(len(sells), 1) * 100,
        'avg_hold': avg_hold,
        'pct_growth': pct_g, 'pct_safety': pct_s,
        'yearly': yearly, 'g_bh_ann': g_bh_ann, 'g_bh_dd': g_bh_dd * 100,
        'spy_ann': spy_ann,
    }


# ── Main Comparison ───────────────────────────────────────────

def main():
    print("=" * 120)
    print("  LEVERAGE COMPARISON: 1x vs 2x vs 3x MA CROSSOVER")
    print("  100-day SMA | 3% buffer | 2-day entry / 5-day exit confirm | Cash fallback")
    print("=" * 120)

    configs = [
        # (label, trade_growth, trade_safety, alt_ticker)
        ("1x QQQ/TLT", "QQQ", "TLT", ""),
        ("1x + DBMF", "QQQ", "TLT", "DBMF"),
        ("2x QLD/UBT", "QLD", "UBT", ""),
        ("2x + DBMF", "QLD", "UBT", "DBMF"),
        ("3x TQQQ/TMF", "TQQQ", "TMF", ""),
        ("3x + DBMF", "TQQQ", "TMF", "DBMF"),
    ]

    # ── 0.2% slippage ──
    print(f"\n{'='*120}")
    print(f"  SLIPPAGE: 0.2%")
    print(f"{'='*120}")

    results_02 = []
    for label, tg, ts, alt in configs:
        r = run_ma(trade_growth=tg, trade_safety=ts, alt_ticker=alt, slippage=0.002)
        r['label'] = label
        results_02.append(r)

    print(f"\n  {'Config':<18} {'CAGR':>8} {'MaxDD':>8} {'$10k->':>10} {'W3yr':>8} {'R5yr':>8} "
          f"{'N5yr':>6} {'UW(mo)':>8} {'Trades':>7} {'Win%':>6} {'BH CAGR':>8} {'BH DD':>8}")
    print(f"  {'-'*110}")
    for r in results_02:
        print(f"  {r['label']:<18} {r['ann_ret']:>+7.2f}% {-r['max_dd']:>7.1f}% "
              f"${r['equity']:>9,.0f} {r['worst_3yr']:>+7.2f}% {r['worst_5yr']:>+7.2f}% "
              f"{r['neg_5yr']:>6} {r['longest_uw']:>7.1f} {r['trades']:>7} "
              f"{r['win_rate']:>5.0f}% {r['g_bh_ann']:>+7.1f}% {-r['g_bh_dd']:>7.1f}%")

    # ── 0.5% slippage ──
    print(f"\n{'='*120}")
    print(f"  SLIPPAGE: 0.5%")
    print(f"{'='*120}")

    results_05 = []
    for label, tg, ts, alt in configs:
        r = run_ma(trade_growth=tg, trade_safety=ts, alt_ticker=alt, slippage=0.005)
        r['label'] = label
        results_05.append(r)

    print(f"\n  {'Config':<18} {'CAGR':>8} {'MaxDD':>8} {'$10k->':>10} {'W3yr':>8} {'R5yr':>8} "
          f"{'N5yr':>6} {'UW(mo)':>8} {'Trades':>7} {'Win%':>6}")
    print(f"  {'-'*90}")
    for r in results_05:
        print(f"  {r['label']:<18} {r['ann_ret']:>+7.2f}% {-r['max_dd']:>7.1f}% "
              f"${r['equity']:>9,.0f} {r['worst_3yr']:>+7.2f}% {r['worst_5yr']:>+7.2f}% "
              f"{r['neg_5yr']:>6} {r['longest_uw']:>7.1f} {r['trades']:>7} "
              f"{r['win_rate']:>5.0f}%")

    # ── Slippage cost ──
    print(f"\n{'='*90}")
    print(f"  SLIPPAGE COST (0.5% - 0.2%)")
    print(f"{'='*90}")
    print(f"  {'Config':<18} {'CAGR hit':>10} {'DD hit':>10}")
    print(f"  {'-'*40}")
    for r02, r05 in zip(results_02, results_05):
        print(f"  {r02['label']:<18} {r05['ann_ret']-r02['ann_ret']:>+9.2f}% {r05['max_dd']-r02['max_dd']:>+9.1f}%")

    # ── Yearly breakdown for key configs ──
    key_labels = ["2x QLD/UBT", "2x + DBMF", "3x TQQQ/TMF", "3x + DBMF"]
    key_results = [r for r in results_02 if r['label'] in key_labels]

    print(f"\n{'='*90}")
    print(f"  YEARLY RETURNS (0.2% slippage)")
    print(f"{'='*90}")

    all_years = sorted(key_results[0]['yearly'].keys())
    header = f"  {'Year':<6}"
    for r in key_results:
        header += f" {r['label']:>14}"
    print(header)
    print(f"  {'-'*(6 + 15 * len(key_results))}")

    for yr in all_years:
        row = f"  {yr:<6}"
        for r in key_results:
            row += f" {r['yearly'].get(yr, 0):>+13.1f}%"
        print(row)

    # ── Final recommendation ──
    print(f"\n{'='*90}")
    print(f"  2x vs 3x DIRECT COMPARISON (DBMF alt, 0.2% slippage)")
    print(f"{'='*90}")

    r2 = [r for r in results_02 if r['label'] == '2x + DBMF'][0]
    r3 = [r for r in results_02 if r['label'] == '3x + DBMF'][0]

    print(f"\n  {'Metric':<32} {'2x QLD/UBT':>14} {'3x TQQQ/TMF':>14} {'Delta':>12}")
    print(f"  {'-'*74}")
    print(f"  {'CAGR':<32} {r2['ann_ret']:>+13.2f}% {r3['ann_ret']:>+13.2f}% {r3['ann_ret']-r2['ann_ret']:>+11.2f}%")
    print(f"  {'Max Drawdown':<32} {-r2['max_dd']:>13.1f}% {-r3['max_dd']:>13.1f}% {r2['max_dd']-r3['max_dd']:>+11.1f}%")
    print(f"  {'$10k ->':<32} ${r2['equity']:>12,.0f} ${r3['equity']:>12,.0f}")
    print(f"  {'Worst 3-Year (ann)':<32} {r2['worst_3yr']:>+13.2f}% {r3['worst_3yr']:>+13.2f}% {r3['worst_3yr']-r2['worst_3yr']:>+11.2f}%")
    print(f"  {'Rolling 5-Year Min (ann)':<32} {r2['worst_5yr']:>+13.2f}% {r3['worst_5yr']:>+13.2f}% {r3['worst_5yr']-r2['worst_5yr']:>+11.2f}%")
    print(f"  {'Negative 5-Year Windows':<32} {r2['neg_5yr']:>13} {r3['neg_5yr']:>13}")
    print(f"  {'Longest Underwater (mo)':<32} {r2['longest_uw']:>13.1f} {r3['longest_uw']:>13.1f}")
    print(f"  {'Trades':<32} {r2['trades']:>13} {r3['trades']:>13}")
    print(f"  {'Win Rate':<32} {r2['win_rate']:>12.0f}% {r3['win_rate']:>12.0f}%")

    # Also show 2x no DBMF vs 3x no DBMF
    r2n = [r for r in results_02 if r['label'] == '2x QLD/UBT'][0]
    r3n = [r for r in results_02 if r['label'] == '3x TQQQ/TMF'][0]

    print(f"\n  {'--- Without DBMF ---'}")
    print(f"  {'Metric':<32} {'2x QLD/UBT':>14} {'3x TQQQ/TMF':>14} {'Delta':>12}")
    print(f"  {'-'*74}")
    print(f"  {'CAGR':<32} {r2n['ann_ret']:>+13.2f}% {r3n['ann_ret']:>+13.2f}% {r3n['ann_ret']-r2n['ann_ret']:>+11.2f}%")
    print(f"  {'Max Drawdown':<32} {-r2n['max_dd']:>13.1f}% {-r3n['max_dd']:>13.1f}% {r2n['max_dd']-r3n['max_dd']:>+11.1f}%")
    print(f"  {'$10k ->':<32} ${r2n['equity']:>12,.0f} ${r3n['equity']:>12,.0f}")
    print(f"  {'Worst 3-Year (ann)':<32} {r2n['worst_3yr']:>+13.2f}% {r3n['worst_3yr']:>+13.2f}% {r3n['worst_3yr']-r2n['worst_3yr']:>+11.2f}%")
    print(f"  {'Rolling 5-Year Min (ann)':<32} {r2n['worst_5yr']:>+13.2f}% {r3n['worst_5yr']:>+13.2f}% {r3n['worst_5yr']-r2n['worst_5yr']:>+11.2f}%")
    print(f"  {'Negative 5-Year Windows':<32} {r2n['neg_5yr']:>13} {r3n['neg_5yr']:>13}")


if __name__ == "__main__":
    main()
