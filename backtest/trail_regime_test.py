"""
Trail Stop Regime Analysis
===========================
Tests the 8% trail + 20d cooldown across specific historical market regimes
using QQQ/TLT 1x (longest available history, since 2003).

Regimes tested:
  1. 2004-2007: Chop / slow recovery after dot-com
  2. 2011:      Euro sovereign debt crisis
  3. 2015:      Flat / choppy regime
  4. 2018:      Volatility spike (Feb + Q4)
  5. 2020:      COVID crash
  6. 2022:      Rate hike bear market (the big one)
  7. Full period: 2003-2024 (21+ years)
"""
import json
import math
import os
from datetime import datetime

import yfinance as yf


# ── Data Loading ──────────────────────────────────────────────

def load_etf_cached(ticker, cache_dir="state/etf_cache", start="2002-01-01", end="2025-01-01"):
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


def sma(closes, period):
    if len(closes) < period:
        return closes[-1] if closes else 0
    return sum(closes[-period:]) / period


# ── MA Rotation Backtest (1x, cash fallback) ──────────────────

def run_ma_rotation(
    trail_pct=0.0,
    trail_cooldown=0,
    ma_period=100,
    buffer_pct=0.03,
    confirm_entry=2,
    confirm_exit=5,
    slippage=0.002,
    starting_cash=10000.0,
    cash_apy=0.0335,
    cache_dir="state/etf_cache",
    start="2003-01-01",
    end="2025-01-01",
):
    daily_cash_rate = (1 + cash_apy) ** (1 / 252) - 1

    sig_growth, sig_safety = 'QQQ', 'TLT'
    trade_growth, trade_safety = 'QQQ', 'TLT'  # 1x for long history

    all_tickers = ['QQQ', 'TLT', 'SPY']
    price_data = {}
    date_data = {}
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
    holding = None
    direction = None
    entry_price = 0.0
    entry_bar = 0
    signal_peak = 0.0
    trail_triggers = 0
    last_trail_bar = -999
    last_trail_dir = None
    qa, qb, ta, tb = 0, 0, 0, 0
    n_trades = 0

    equity_curve = [starting_cash]
    peak_eq = starting_cash
    max_dd = 0.0
    trade_log = []

    for i in range(1, n):
        cash += cash * daily_cash_rate

        if holding is not None:
            prev_p = prices[holding][i - 1]
            cur_p = prices[holding][i]
            if prev_p > 0:
                invested *= (1 + (cur_p - prev_p) / prev_p)

        total_eq = invested + cash
        equity_curve.append(total_eq)
        if total_eq > peak_eq:
            peak_eq = total_eq
        dd = (peak_eq - total_eq) / peak_eq
        if dd > max_dd:
            max_dd = dd

        if i < ma_period:
            continue

        qqq_sma = sma(prices['QQQ'][:i + 1], ma_period)
        tlt_sma = sma(prices['TLT'][:i + 1], ma_period)
        qqq_price = prices['QQQ'][i]
        tlt_price = prices['TLT'][i]

        if qqq_price > qqq_sma * (1 + buffer_pct):
            qa += 1; qb = 0
        elif qqq_price < qqq_sma * (1 - buffer_pct):
            qb += 1; qa = 0

        if tlt_price > tlt_sma * (1 + buffer_pct):
            ta += 1; tb = 0
        elif tlt_price < tlt_sma * (1 - buffer_pct):
            tb += 1; ta = 0

        qqq_above = qa >= confirm_entry
        qqq_below = qb >= confirm_exit
        tlt_above = ta >= confirm_entry
        tlt_below = tb >= confirm_exit

        # ── Trailing stop ──
        trail_fired = False
        if trail_pct > 0 and holding is not None and direction in ('growth', 'safety'):
            sig_price = qqq_price if direction == 'growth' else tlt_price
            if sig_price > signal_peak:
                signal_peak = sig_price
            dd_from_peak = (signal_peak - sig_price) / signal_peak

            if dd_from_peak >= trail_pct:
                trail_fired = True
                trail_triggers += 1
                last_trail_bar = i
                last_trail_dir = direction

                hold_pnl = (prices[holding][i] - entry_price) / entry_price
                invested *= (1 - slippage)
                cash += invested; invested = 0
                trade_log.append({
                    'date': common_dates[i], 'action': 'TRAIL',
                    'pnl': hold_pnl, 'days': i - entry_bar,
                })
                n_trades += 1

                # Rotate
                new_target = None; new_dir = None
                if direction == 'growth' and tlt_price > tlt_sma:
                    new_target = 'TLT'; new_dir = 'safety'
                elif direction == 'safety' and qqq_price > qqq_sma:
                    new_target = 'QQQ'; new_dir = 'growth'

                if new_target:
                    invest_amt = (cash + invested)
                    cash = 0
                    invested = invest_amt * (1 - slippage)
                    entry_price = prices[new_target][i]
                    entry_bar = i
                    signal_peak = qqq_price if new_dir == 'growth' else tlt_price
                    holding = new_target; direction = new_dir
                    n_trades += 1
                else:
                    holding = None; direction = None; signal_peak = 0

        # ── Normal MA rotation ──
        if not trail_fired:
            target = holding; target_dir = direction

            if direction == 'growth':
                if qqq_below:
                    if tlt_above:
                        target = 'TLT'; target_dir = 'safety'
                    else:
                        target = None; target_dir = None
            elif direction == 'safety':
                if qqq_above:
                    target = 'QQQ'; target_dir = 'growth'
                elif tlt_below:
                    target = None; target_dir = None
            else:
                if qqq_above:
                    target = 'QQQ'; target_dir = 'growth'
                elif tlt_above:
                    target = 'TLT'; target_dir = 'safety'

            # Cooldown
            if trail_cooldown > 0 and target_dir == last_trail_dir:
                if (i - last_trail_bar) < trail_cooldown:
                    target = holding; target_dir = direction

            if target != holding:
                if holding is not None:
                    hold_pnl = (prices[holding][i] - entry_price) / entry_price
                    invested *= (1 - slippage)
                    cash += invested; invested = 0
                    trade_log.append({
                        'date': common_dates[i], 'action': 'SELL',
                        'pnl': hold_pnl, 'days': i - entry_bar,
                    })
                    n_trades += 1

                if target is not None:
                    invest_amt = cash + invested
                    cash = 0
                    invested = invest_amt * (1 - slippage)
                    entry_price = prices[target][i]
                    entry_bar = i
                    if target_dir == 'growth':
                        signal_peak = qqq_price
                    elif target_dir == 'safety':
                        signal_peak = tlt_price
                    else:
                        signal_peak = 0
                    n_trades += 1
                else:
                    cash = cash + invested; invested = 0; signal_peak = 0

                holding = target; direction = target_dir

    total_eq = invested + cash
    total_ret = (total_eq / starting_cash - 1) * 100
    ann_ret = ((total_eq / starting_cash) ** (1 / years) - 1) * 100

    # Yearly returns
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

    # Period max drawdowns
    def period_dd(eq_curve, dates, p_start, p_end):
        peak = 0; mdd = 0; p_ret = None; first_eq = None
        for j, d in enumerate(dates):
            if p_start <= d <= p_end:
                if first_eq is None:
                    first_eq = eq_curve[j]
                    peak = eq_curve[j]
                if eq_curve[j] > peak:
                    peak = eq_curve[j]
                dd = (peak - eq_curve[j]) / peak
                if dd > mdd:
                    mdd = dd
                last_eq = eq_curve[j]
        if first_eq and first_eq > 0:
            p_ret = (last_eq / first_eq - 1) * 100
        return p_ret, mdd * 100

    # Period trail counts
    def period_trails(tlog, p_start, p_end):
        return sum(1 for t in tlog if t['action'] == 'TRAIL' and p_start <= t['date'] <= p_end)

    # Rolling windows
    bars_3yr = 252 * 3
    bars_5yr = 252 * 5
    worst_3yr = float('inf')
    worst_5yr = float('inf')
    neg_5yr = 0; total_5yr = 0

    for j in range(bars_3yr, n):
        ret = (equity_curve[j] / equity_curve[j - bars_3yr]) ** (1 / 3) - 1
        if ret < worst_3yr: worst_3yr = ret

    for j in range(bars_5yr, n):
        ret = (equity_curve[j] / equity_curve[j - bars_5yr]) ** (1 / 5) - 1
        if ret < worst_5yr: worst_5yr = ret
        total_5yr += 1
        if ret < 0: neg_5yr += 1

    return {
        'ann_ret': ann_ret, 'total_ret': total_ret, 'max_dd': max_dd * 100,
        'equity': total_eq, 'trades': n_trades, 'trail_triggers': trail_triggers,
        'yearly': yearly,
        'worst_3yr': worst_3yr * 100, 'worst_5yr': worst_5yr * 100,
        'neg_5yr': neg_5yr,
        'equity_curve': equity_curve, 'dates': common_dates,
        'period_dd': period_dd, 'trade_log': trade_log,
        'period_trails': period_trails,
    }


# ── Regime Analysis ───────────────────────────────────────────

def main():
    print("=" * 110)
    print("  TRAILING STOP REGIME ROBUSTNESS TEST")
    print("  MA Crossover: QQQ/TLT 1x | 100d SMA | 3% buffer | 2d/5d confirm | Cash fallback")
    print("  Trail: 8% on 1x signal ETF + 20-day cooldown")
    print("  Full history: 2003-2024 (21+ years)")
    print("=" * 110)

    # Run both configs over full history
    common_kw = dict(
        ma_period=100, buffer_pct=0.03, confirm_entry=2, confirm_exit=5,
        slippage=0.002, start="2003-01-01", end="2025-01-01",
    )

    print("\n  Running no-trail baseline...")
    no_trail = run_ma_rotation(trail_pct=0.0, trail_cooldown=0, **common_kw)
    print("  Running 8% trail + 20d cooldown...")
    with_trail = run_ma_rotation(trail_pct=0.08, trail_cooldown=20, **common_kw)

    # ── Full period summary ──
    print(f"\n{'='*90}")
    print(f"  FULL PERIOD: {no_trail['dates'][0]} to {no_trail['dates'][-1]}")
    print(f"{'='*90}")
    print(f"  {'Metric':<32} {'No Trail':>14} {'8%+20d Trail':>14} {'Delta':>12}")
    print(f"  {'-'*74}")
    print(f"  {'CAGR':<32} {no_trail['ann_ret']:>+13.2f}% {with_trail['ann_ret']:>+13.2f}% {with_trail['ann_ret']-no_trail['ann_ret']:>+11.2f}%")
    print(f"  {'Total Return':<32} {no_trail['total_ret']:>+13.1f}% {with_trail['total_ret']:>+13.1f}%")
    print(f"  {'Max Drawdown':<32} {-no_trail['max_dd']:>13.1f}% {-with_trail['max_dd']:>13.1f}% {no_trail['max_dd']-with_trail['max_dd']:>+11.1f}%")
    print(f"  {'$10k ->':<32} ${no_trail['equity']:>12,.0f} ${with_trail['equity']:>12,.0f}")
    print(f"  {'Worst 3-Year (ann)':<32} {no_trail['worst_3yr']:>+13.2f}% {with_trail['worst_3yr']:>+13.2f}% {with_trail['worst_3yr']-no_trail['worst_3yr']:>+11.2f}%")
    print(f"  {'Rolling 5-Year Min (ann)':<32} {no_trail['worst_5yr']:>+13.2f}% {with_trail['worst_5yr']:>+13.2f}% {with_trail['worst_5yr']-no_trail['worst_5yr']:>+11.2f}%")
    print(f"  {'Negative 5-Year Windows':<32} {no_trail['neg_5yr']:>13} {with_trail['neg_5yr']:>13}")
    print(f"  {'Total Trades':<32} {no_trail['trades']:>13} {with_trail['trades']:>13}")
    print(f"  {'Trail Stops':<32} {'0':>13} {with_trail['trail_triggers']:>13}")

    # ── Yearly comparison ──
    print(f"\n{'='*90}")
    print(f"  YEARLY RETURNS")
    print(f"{'='*90}")
    print(f"  {'Year':<6} {'No Trail':>10} {'Trail':>10} {'Delta':>10} {'Trails':>8} {'Winner':>10}")
    print(f"  {'-'*56}")

    all_years = sorted(set(list(no_trail['yearly'].keys()) + list(with_trail['yearly'].keys())))
    for yr in all_years:
        nt = no_trail['yearly'].get(yr, 0)
        wt = with_trail['yearly'].get(yr, 0)
        delta = wt - nt
        trails = with_trail['period_trails'](with_trail['trade_log'], f"{yr}-01-01", f"{yr}-12-31")
        winner = "TRAIL" if delta > 0.5 else ("BASE" if delta < -0.5 else "~same")
        marker = " <--" if abs(delta) > 3 else ""
        print(f"  {yr:<6} {nt:>+9.1f}% {wt:>+9.1f}% {delta:>+9.1f}% {trails:>8} {winner:>10}{marker}")

    # ── Regime-specific deep dive ──
    regimes = [
        ("2004-2007 Chop",      "2004-01-01", "2007-12-31"),
        ("2008 GFC",            "2008-01-01", "2008-12-31"),
        ("2011 Euro Crisis",    "2011-01-01", "2011-12-31"),
        ("2013-2014 Bull",      "2013-01-01", "2014-12-31"),
        ("2015 Flat",           "2015-01-01", "2015-12-31"),
        ("2018 Vol Spike",      "2018-01-01", "2018-12-31"),
        ("2020 COVID",          "2020-01-01", "2020-12-31"),
        ("2021 Bull",           "2021-01-01", "2021-12-31"),
        ("2022 Rate Hike Bear", "2022-01-01", "2022-12-31"),
        ("2023-2024 Recovery",  "2023-01-01", "2024-12-31"),
    ]

    print(f"\n{'='*110}")
    print(f"  REGIME-SPECIFIC ANALYSIS")
    print(f"{'='*110}")
    print(f"  {'Regime':<24} {'NT Return':>10} {'TR Return':>10} {'Delta':>10} "
          f"{'NT MaxDD':>10} {'TR MaxDD':>10} {'DD Improv':>10} {'Trails':>8}")
    print(f"  {'-'*94}")

    for name, p_start, p_end in regimes:
        nt_ret, nt_dd = no_trail['period_dd'](no_trail['equity_curve'], no_trail['dates'], p_start, p_end)
        tr_ret, tr_dd = with_trail['period_dd'](with_trail['equity_curve'], with_trail['dates'], p_start, p_end)
        trails = with_trail['period_trails'](with_trail['trade_log'], p_start, p_end)

        if nt_ret is not None and tr_ret is not None:
            delta = tr_ret - nt_ret
            dd_improv = nt_dd - tr_dd
            print(f"  {name:<24} {nt_ret:>+9.1f}% {tr_ret:>+9.1f}% {delta:>+9.1f}% "
                  f"{-nt_dd:>9.1f}% {-tr_dd:>9.1f}% {dd_improv:>+9.1f}% {trails:>8}")

    # ── Specific crisis events deep dive ──
    print(f"\n{'='*90}")
    print(f"  TRAIL STOPS FIRED (all {with_trail['trail_triggers']} events)")
    print(f"{'='*90}")
    print(f"  {'Date':<12} {'Held':>6} {'P&L':>8} {'Days':>6}")
    print(f"  {'-'*34}")
    for t in with_trail['trade_log']:
        if t['action'] == 'TRAIL':
            print(f"  {t['date']:<12} {'':>6} {t['pnl']*100:>+7.1f}% {t['days']:>5}d")

    # ── Verdict ──
    print(f"\n{'='*90}")
    print(f"  VERDICT")
    print(f"{'='*90}")

    # Count regimes where trail helped vs hurt
    helped = 0; hurt = 0; neutral = 0
    for name, p_start, p_end in regimes:
        nt_ret, _ = no_trail['period_dd'](no_trail['equity_curve'], no_trail['dates'], p_start, p_end)
        tr_ret, _ = with_trail['period_dd'](with_trail['equity_curve'], with_trail['dates'], p_start, p_end)
        if nt_ret is not None and tr_ret is not None:
            delta = tr_ret - nt_ret
            if delta > 1.0: helped += 1
            elif delta < -1.0: hurt += 1
            else: neutral += 1

    print(f"  Regimes where trail helped (>1%):   {helped}")
    print(f"  Regimes where trail hurt (<-1%):     {hurt}")
    print(f"  Regimes neutral (+/-1%):             {neutral}")
    print(f"  Trail stops fired:                   {with_trail['trail_triggers']}")
    print(f"  Full-period CAGR delta:              {with_trail['ann_ret']-no_trail['ann_ret']:+.2f}%")
    print(f"  Full-period DD improvement:          {no_trail['max_dd']-with_trail['max_dd']:+.1f}%")


if __name__ == "__main__":
    main()
