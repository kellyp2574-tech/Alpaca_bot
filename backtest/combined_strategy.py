"""
Combined Strategy Backtest: MA Crossover + Monday Dip + BB Reversion
=====================================================================
Runs all three strategies together with shared capital to measure
the real-world impact of adding a trailing stop to the MA rotation.

Capital allocation:
  - MA Crossover: 50% of equity (QLD/UBT/DBMF with 100d SMA)
  - Monday Dip:   up to 30% of equity (SPY signal -> UPRO, 2-day hold)
  - BB Reversion: up to 20% of equity (SPY below BB -> UPRO, SMA exit)
  - Dip strategies use available cash (what's left after MA allocation)

Comparisons:
  A. Current strategy (no trailing stop)
  B. With 8% trail + 20-day cooldown on 1x signal ETF

Metrics: CAGR, Max DD, worst 3-year, rolling 5-year min, slippage sensitivity.
"""
import argparse
import json
import math
import os
from datetime import datetime


# ── Data Loading ──────────────────────────────────────────────

def load_etf_cached(ticker, cache_dir="state/etf_cache", start="2009-06-25", end="2025-01-01"):
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


# ── Technical Indicators ──────────────────────────────────────

def compute_sma(closes, i, period):
    if i < period - 1:
        return closes[i] if closes else 0
    return sum(closes[i - period + 1:i + 1]) / period


def compute_std(closes, i, period):
    if i < period - 1:
        return 0
    window = closes[i - period + 1:i + 1]
    mean = sum(window) / period
    return math.sqrt(sum((x - mean) ** 2 for x in window) / period)


# ── Combined Backtest ─────────────────────────────────────────

def run_combined(
    # MA Crossover params
    ma_period: int = 100,
    buffer_pct: float = 0.03,
    confirm_entry: int = 2,
    confirm_exit: int = 5,
    ma_alloc: float = 0.50,
    trail_pct: float = 0.0,
    trail_cooldown: int = 0,
    # Monday Dip params
    md_enabled: bool = True,
    md_alloc_cap: float = 0.30,
    md_min_dip: float = 0.005,
    md_hold_days: int = 2,
    md_take_profit: float = 0.025,
    # BB Reversion params
    bb_enabled: bool = True,
    bb_alloc_cap: float = 0.20,
    bb_period: int = 20,
    bb_sigma: float = 2.0,
    bb_stop_loss: float = 0.10,
    bb_take_profit: float = 0.05,
    # General
    slippage: float = 0.002,
    starting_cash: float = 10000.0,
    cash_apy: float = 0.0335,
    cache_dir: str = "state/etf_cache",
    start: str = "2009-06-25",
    end: str = "2025-01-01",
    verbose: bool = True,
    label: str = "",
):
    daily_cash_rate = (1 + cash_apy) ** (1 / 252) - 1

    # Tickers
    sig_growth, sig_safety = 'QQQ', 'TLT'
    trade_growth, trade_safety = 'QLD', 'UBT'
    alt_ticker = 'DBMF'
    dip_ticker = 'UPRO'

    all_tickers = list(set([
        sig_growth, sig_safety, trade_growth, trade_safety,
        alt_ticker, dip_ticker, 'SPY',
    ]))

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

    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in common_dates]
    weekdays = [d.weekday() for d in date_objs]

    # Identify first trading day of each week
    week_starts = set()
    for i in range(1, n):
        if weekdays[i] <= weekdays[i - 1]:
            week_starts.add(i)

    if verbose:
        trail_str = f" | Trail: {trail_pct*100:.0f}%+{trail_cooldown}d cool" if trail_pct > 0 else " | No trail"
        print(f"  Period: {common_dates[0]} to {common_dates[-1]} ({years:.1f} yr, {n} bars)")
        print(f"  MA: {ma_period}d, {buffer_pct*100:.0f}% buf, {confirm_entry}d/{confirm_exit}d confirm{trail_str}")
        print(f"  Monday Dip: {md_alloc_cap*100:.0f}% cap, {md_min_dip*100:.1f}% min, {md_hold_days}d hold, {md_take_profit*100:.1f}% TP")
        print(f"  BB Reversion: {bb_alloc_cap*100:.0f}% cap, BB({bb_period},{bb_sigma}), {bb_stop_loss*100:.0f}% SL, {bb_take_profit*100:.0f}% TP")
        print(f"  Slippage: {slippage*100:.1f}% | Cash APY: {cash_apy*100:.2f}%")

    # ── State ──
    # Cash pool (uninvested)
    cash = starting_cash

    # MA rotation state
    ma_invested = 0.0
    ma_holding = None  # trade_growth, trade_safety, alt_ticker, or None
    ma_direction = None  # 'growth', 'safety', 'alt'
    ma_entry_price = 0.0
    ma_entry_bar = 0
    ma_signal_peak = 0.0
    ma_trail_triggers = 0
    ma_last_trail_bar = -999
    ma_last_trail_dir = None

    # MA confirmation counters
    qa, qb, ta, tb = 0, 0, 0, 0

    # Dip trade state (shared between Monday Dip and BB)
    dip_invested = 0.0
    dip_active = False
    dip_source = None  # 'MD' or 'BB'
    dip_entry_price = 0.0  # entry price of UPRO
    dip_entry_spy = 0.0    # SPY price at entry (for BB SMA exit)
    dip_buy_bar = 0
    dip_days_held = 0

    # Tracking
    equity_curve = [starting_cash]
    peak_eq = starting_cash
    max_dd = 0.0
    trade_log = []
    n_trades = 0
    md_trades = 0
    md_wins = 0
    bb_trades = 0
    bb_wins = 0
    bb_stops = 0
    ma_trades = 0

    for i in range(1, n):
        # Cash earns interest
        cash += cash * daily_cash_rate

        # Mark MA position to market
        if ma_holding is not None:
            prev_p = prices[ma_holding][i - 1]
            cur_p = prices[ma_holding][i]
            if prev_p > 0:
                ma_invested *= (1 + (cur_p - prev_p) / prev_p)

        # Mark dip position to market
        if dip_active:
            prev_p = prices[dip_ticker][i - 1]
            cur_p = prices[dip_ticker][i]
            if prev_p > 0:
                dip_invested *= (1 + (cur_p - prev_p) / prev_p)

        total_equity = cash + ma_invested + dip_invested
        equity_curve.append(total_equity)
        if total_equity > peak_eq:
            peak_eq = total_equity
        dd = (peak_eq - total_equity) / peak_eq
        if dd > max_dd:
            max_dd = dd

        if i < ma_period:
            continue

        # ════════════════════════════════════════════
        # 1. DIP TRADE EXIT CHECK (before new entries)
        # ════════════════════════════════════════════
        if dip_active:
            dip_days_held += 1
            upro_price = prices[dip_ticker][i]
            dip_pnl = (upro_price - dip_entry_price) / dip_entry_price

            should_exit = False
            exit_reason = ""

            if dip_source == 'MD':
                # Monday Dip: exit after hold_days or TP
                if dip_pnl >= md_take_profit:
                    should_exit = True; exit_reason = f"MD TP {dip_pnl*100:+.1f}%"
                elif dip_days_held >= md_hold_days:
                    should_exit = True; exit_reason = f"MD hold done {dip_pnl*100:+.1f}%"

            elif dip_source == 'BB':
                # BB: exit at SMA, SL, or TP
                spy_sma = compute_sma(prices['SPY'], i, bb_period)
                if dip_pnl <= -bb_stop_loss:
                    should_exit = True; exit_reason = f"BB SL {dip_pnl*100:+.1f}%"
                    bb_stops += 1
                elif dip_pnl >= bb_take_profit:
                    should_exit = True; exit_reason = f"BB TP {dip_pnl*100:+.1f}%"
                elif prices['SPY'][i] >= spy_sma:
                    should_exit = True; exit_reason = f"BB SMA exit {dip_pnl*100:+.1f}%"

            if should_exit:
                dip_invested *= (1 - slippage)
                cash += dip_invested
                trade_log.append({
                    'date': common_dates[i], 'action': 'DIP_SELL',
                    'ticker': dip_ticker, 'pnl': dip_pnl,
                    'days': dip_days_held, 'detail': exit_reason,
                })
                if dip_source == 'MD':
                    md_trades += 1
                    if dip_pnl > 0: md_wins += 1
                elif dip_source == 'BB':
                    bb_trades += 1
                    if dip_pnl > 0: bb_wins += 1
                n_trades += 1
                dip_invested = 0; dip_active = False; dip_source = None

        # ════════════════════════════════════════════
        # 2. MA CROSSOVER LOGIC (with optional trail)
        # ════════════════════════════════════════════
        qqq_sma = compute_sma(prices[sig_growth], i, ma_period)
        tlt_sma = compute_sma(prices[sig_safety], i, ma_period)
        qqq_price = prices[sig_growth][i]
        tlt_price = prices[sig_safety][i]

        # Update confirmation counters
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

        # ── Trailing stop check ──
        trail_fired = False
        if trail_pct > 0 and ma_holding is not None and ma_direction in ('growth', 'safety'):
            sig_ticker = sig_growth if ma_direction == 'growth' else sig_safety
            sig_price = prices[sig_ticker][i]
            if sig_price > ma_signal_peak:
                ma_signal_peak = sig_price
            dd_from_peak = (ma_signal_peak - sig_price) / ma_signal_peak

            if dd_from_peak >= trail_pct:
                trail_fired = True
                ma_trail_triggers += 1
                ma_last_trail_bar = i
                ma_last_trail_dir = ma_direction

                # Force exit MA position
                hold_pnl = (prices[ma_holding][i] - ma_entry_price) / ma_entry_price
                ma_invested *= (1 - slippage)
                cash += ma_invested
                trade_log.append({
                    'date': common_dates[i], 'action': 'TRAIL',
                    'ticker': ma_holding, 'pnl': hold_pnl,
                    'days': i - ma_entry_bar,
                    'detail': f"{sig_ticker} -{dd_from_peak*100:.1f}% from peak",
                })
                n_trades += 1; ma_trades += 1
                ma_invested = 0

                # Check for immediate rotation
                new_target = None; new_dir = None
                if ma_direction == 'growth':
                    if tlt_price > tlt_sma:
                        new_target = trade_safety; new_dir = 'safety'
                    else:
                        new_target = alt_ticker; new_dir = 'alt'
                elif ma_direction == 'safety':
                    if qqq_price > qqq_sma:
                        new_target = trade_growth; new_dir = 'growth'
                    else:
                        new_target = alt_ticker; new_dir = 'alt'

                if new_target:
                    total_eq = cash + ma_invested + dip_invested
                    invest_amt = total_eq * ma_alloc
                    invest_amt = min(invest_amt, cash)
                    cash -= invest_amt
                    ma_invested = invest_amt * (1 - slippage)
                    ma_entry_price = prices[new_target][i]
                    ma_entry_bar = i
                    ma_signal_peak = prices[sig_growth if new_dir == 'growth' else sig_safety][i]
                    ma_holding = new_target; ma_direction = new_dir
                    trade_log.append({
                        'date': common_dates[i], 'action': 'MA_BUY',
                        'ticker': new_target, 'pnl': None, 'days': None,
                        'detail': 'trail rotation',
                    })
                    n_trades += 1; ma_trades += 1
                else:
                    ma_holding = None; ma_direction = None; ma_signal_peak = 0

        # ── Normal MA rotation (skip if trail just fired) ──
        if not trail_fired:
            fallback = alt_ticker
            target = ma_holding; target_dir = ma_direction

            if ma_direction == 'growth':
                if qqq_below:
                    target = trade_safety if tlt_above else fallback
                    target_dir = 'safety' if tlt_above else 'alt'
            elif ma_direction == 'safety':
                if qqq_above:
                    target = trade_growth; target_dir = 'growth'
                elif tlt_below:
                    target = fallback; target_dir = 'alt'
            elif ma_direction == 'alt':
                if qqq_above:
                    target = trade_growth; target_dir = 'growth'
                elif tlt_above:
                    target = trade_safety; target_dir = 'safety'
            else:  # None / cash
                if qqq_above:
                    target = trade_growth; target_dir = 'growth'
                elif tlt_above:
                    target = trade_safety; target_dir = 'safety'

            # Cooldown check
            if trail_cooldown > 0 and target_dir == ma_last_trail_dir:
                if (i - ma_last_trail_bar) < trail_cooldown:
                    target = ma_holding; target_dir = ma_direction

            if target != ma_holding:
                # Sell current MA
                if ma_holding is not None:
                    hold_pnl = (prices[ma_holding][i] - ma_entry_price) / ma_entry_price
                    ma_invested *= (1 - slippage)
                    cash += ma_invested; ma_invested = 0
                    trade_log.append({
                        'date': common_dates[i], 'action': 'MA_SELL',
                        'ticker': ma_holding, 'pnl': hold_pnl,
                        'days': i - ma_entry_bar, 'detail': 'MA signal',
                    })
                    n_trades += 1; ma_trades += 1

                # Buy new MA
                if target is not None:
                    total_eq = cash + ma_invested + dip_invested
                    invest_amt = total_eq * ma_alloc
                    invest_amt = min(invest_amt, cash)
                    cash -= invest_amt
                    ma_invested = invest_amt * (1 - slippage)
                    ma_entry_price = prices[target][i]
                    ma_entry_bar = i
                    if target_dir == 'growth':
                        ma_signal_peak = prices[sig_growth][i]
                    elif target_dir == 'safety':
                        ma_signal_peak = prices[sig_safety][i]
                    else:
                        ma_signal_peak = 0
                    trade_log.append({
                        'date': common_dates[i], 'action': 'MA_BUY',
                        'ticker': target, 'pnl': None, 'days': None,
                        'detail': 'MA signal',
                    })
                    n_trades += 1; ma_trades += 1

                ma_holding = target; ma_direction = target_dir

        # ════════════════════════════════════════════
        # 3. MONDAY DIP CHECK (priority over BB)
        # ════════════════════════════════════════════
        if md_enabled and not dip_active and i in week_starts:
            spy_today = prices['SPY'][i]
            spy_prev = prices['SPY'][i - 1]
            if spy_today < spy_prev:
                dip_pct = (spy_prev - spy_today) / spy_prev
                if dip_pct >= md_min_dip:
                    total_eq = cash + ma_invested + dip_invested
                    invest_amt = total_eq * md_alloc_cap
                    invest_amt = min(invest_amt, cash * 0.95)
                    if invest_amt >= 10:
                        cash -= invest_amt
                        dip_invested = invest_amt * (1 - slippage)
                        dip_entry_price = prices[dip_ticker][i]
                        dip_entry_spy = prices['SPY'][i]
                        dip_buy_bar = i
                        dip_days_held = 0
                        dip_active = True
                        dip_source = 'MD'
                        trade_log.append({
                            'date': common_dates[i], 'action': 'DIP_BUY',
                            'ticker': dip_ticker, 'pnl': None, 'days': None,
                            'detail': f'MD dip {dip_pct*100:.2f}%',
                        })
                        n_trades += 1

        # ════════════════════════════════════════════
        # 4. BB REVERSION CHECK (only if no dip active)
        # ════════════════════════════════════════════
        if bb_enabled and not dip_active and i >= bb_period:
            spy_sma = compute_sma(prices['SPY'], i, bb_period)
            spy_std = compute_std(prices['SPY'], i, bb_period)
            lower_bb = spy_sma - bb_sigma * spy_std

            if prices['SPY'][i] < lower_bb:
                total_eq = cash + ma_invested + dip_invested
                invest_amt = total_eq * bb_alloc_cap
                invest_amt = min(invest_amt, cash * 0.95)
                if invest_amt >= 10:
                    cash -= invest_amt
                    dip_invested = invest_amt * (1 - slippage)
                    dip_entry_price = prices[dip_ticker][i]
                    dip_entry_spy = prices['SPY'][i]
                    dip_buy_bar = i
                    dip_days_held = 0
                    dip_active = True
                    dip_source = 'BB'
                    trade_log.append({
                        'date': common_dates[i], 'action': 'DIP_BUY',
                        'ticker': dip_ticker, 'pnl': None, 'days': None,
                        'detail': f'BB SPY={prices["SPY"][i]:.2f} < BB={lower_bb:.2f}',
                    })
                    n_trades += 1

    # ── Final equity ──
    total_equity = cash + ma_invested + dip_invested
    total_ret = (total_equity / starting_cash - 1) * 100
    ann_ret = ((total_equity / starting_cash) ** (1 / years) - 1) * 100

    # SPY B&H
    spy_ret = (prices['SPY'][-1] / prices['SPY'][0] - 1) * 100
    spy_ann = ((1 + spy_ret / 100) ** (1 / years) - 1) * 100
    spy_peak_p = prices['SPY'][0]
    spy_max_dd = 0
    for p in prices['SPY']:
        if p > spy_peak_p: spy_peak_p = p
        d = (spy_peak_p - p) / spy_peak_p
        if d > spy_max_dd: spy_max_dd = d

    # ── Rolling window analysis ──
    # Annualized returns for every 3-year and 5-year window
    worst_3yr = float('inf')
    worst_5yr = float('inf')
    bars_3yr = 252 * 3
    bars_5yr = 252 * 5
    neg_5yr_windows = 0
    total_5yr_windows = 0

    for j in range(bars_3yr, n):
        ret_3 = (equity_curve[j] / equity_curve[j - bars_3yr]) ** (1 / 3) - 1
        if ret_3 < worst_3yr:
            worst_3yr = ret_3

    for j in range(bars_5yr, n):
        ret_5 = (equity_curve[j] / equity_curve[j - bars_5yr]) ** (1 / 5) - 1
        if ret_5 < worst_5yr:
            worst_5yr = ret_5
        total_5yr_windows += 1
        if ret_5 < 0:
            neg_5yr_windows += 1

    # Longest underwater period
    uw_start = 0
    longest_uw = 0
    uw_peak = equity_curve[0]
    for j in range(1, n):
        if equity_curve[j] >= uw_peak:
            uw_peak = equity_curve[j]
            uw_start = j
        else:
            uw_len = j - uw_start
            if uw_len > longest_uw:
                longest_uw = uw_len

    # Yearly returns
    yearly = {}
    yr_start_eq = equity_curve[0]
    yr_start_spy = prices['SPY'][0]
    current_year = common_dates[0][:4]
    for j in range(1, n):
        yr = common_dates[j][:4]
        if yr != current_year or j == n - 1:
            idx = j - 1 if yr != current_year else j
            yr_ret = (equity_curve[idx] / yr_start_eq - 1) * 100
            spy_yr = (prices['SPY'][idx] / yr_start_spy - 1) * 100
            yr_trails = sum(1 for t in trade_log if t['date'][:4] == current_year and t['action'] == 'TRAIL')
            yr_md = sum(1 for t in trade_log if t['date'][:4] == current_year
                       and t['action'] == 'DIP_SELL' and 'MD' in t.get('detail', ''))
            yr_bb = sum(1 for t in trade_log if t['date'][:4] == current_year
                       and t['action'] == 'DIP_SELL' and 'BB' in t.get('detail', ''))
            yearly[current_year] = {
                'ret': yr_ret, 'spy': spy_yr,
                'trails': yr_trails, 'md': yr_md, 'bb': yr_bb,
            }
            yr_start_eq = equity_curve[idx]
            yr_start_spy = prices['SPY'][idx]
            current_year = yr

    if verbose:
        print(f"\n  {'='*60}")
        lbl = label if label else ("With Trail" if trail_pct > 0 else "No Trail")
        print(f"  RESULTS: {lbl}")
        print(f"  {'='*60}")

        print(f"\n  {'Metric':<32} {'Value':>14}")
        print(f"  {'-'*48}")
        print(f"  {'CAGR':<32} {ann_ret:>+13.2f}%")
        print(f"  {'Total Return':<32} {total_ret:>+13.1f}%")
        print(f"  {'Max Drawdown':<32} {-max_dd*100:>13.1f}%")
        print(f"  {'$10k ->':<32} ${total_equity:>12,.0f}")
        print(f"  {'Worst 3-Year (ann)':<32} {worst_3yr*100:>+13.2f}%")
        print(f"  {'Rolling 5-Year Min (ann)':<32} {worst_5yr*100:>+13.2f}%")
        print(f"  {'Negative 5-Year Windows':<32} {neg_5yr_windows:>13}")
        print(f"  {'Longest Underwater':<32} {longest_uw/252:>12.1f} mo")
        print(f"  {'Total Trades':<32} {n_trades:>13}")
        print(f"  {'MA Rotations':<32} {ma_trades:>13}")
        if trail_pct > 0:
            print(f"  {'Trail Stops':<32} {ma_trail_triggers:>13}")
        print(f"  {'Monday Dip Trades':<32} {md_trades:>13} ({md_wins} wins, {md_wins*100//max(md_trades,1)}%)")
        print(f"  {'BB Trades':<32} {bb_trades:>13} ({bb_wins} wins, {bb_wins*100//max(bb_trades,1)}%)")
        if bb_stops:
            print(f"  {'BB Stops':<32} {bb_stops:>13}")

        print(f"\n  SPY B&H: {spy_ann:+.2f}% CAGR | {spy_ret:+.1f}% total | {-spy_max_dd*100:.1f}% max DD")

        # Yearly
        print(f"\n  {'Year':<6} {'Strategy':>10} {'SPY B&H':>10} {'Trails':>8} {'MD':>6} {'BB':>6}")
        print(f"  {'-'*50}")
        for yr in sorted(yearly.keys()):
            y = yearly[yr]
            print(f"  {yr:<6} {y['ret']:>+9.1f}% {y['spy']:>+9.1f}% {y['trails']:>8} {y['md']:>6} {y['bb']:>6}")

    return {
        'label': label,
        'ann_ret': ann_ret, 'total_ret': total_ret,
        'max_dd': max_dd, 'equity': total_equity,
        'worst_3yr': worst_3yr * 100, 'worst_5yr': worst_5yr * 100,
        'neg_5yr_windows': neg_5yr_windows,
        'longest_uw': longest_uw / 252,
        'n_trades': n_trades, 'ma_trades': ma_trades,
        'trail_triggers': ma_trail_triggers,
        'md_trades': md_trades, 'md_wins': md_wins,
        'bb_trades': bb_trades, 'bb_wins': bb_wins, 'bb_stops': bb_stops,
        'yearly': yearly,
        'equity_curve': equity_curve,
    }


# ── Head-to-Head Comparison ──────────────────────────────────

def run_comparison(start="2009-06-25", end="2025-01-01", cache_dir="state/etf_cache"):
    print("=" * 90)
    print("  COMBINED STRATEGY: NO TRAIL vs 8% TRAIL+20d COOLDOWN")
    print("  MA Crossover (50%) + Monday Dip (30%) + BB Reversion (20%)")
    print("  QLD/UBT 2x | DBMF alt | UPRO dip trade | 0.2% slippage")
    print("=" * 90)

    configs = [
        {"label": "A. Current (no trail)", "trail_pct": 0.0, "trail_cooldown": 0, "slippage": 0.002},
        {"label": "B. 8% trail + 20d cool", "trail_pct": 0.08, "trail_cooldown": 20, "slippage": 0.002},
    ]

    results = []
    for cfg in configs:
        r = run_combined(
            **cfg, start=start, end=end, cache_dir=cache_dir, verbose=True,
        )
        results.append(r)

    # ── Slippage sensitivity ──
    print(f"\n\n{'='*90}")
    print(f"  SLIPPAGE SENSITIVITY (0.2% vs 0.5%)")
    print(f"{'='*90}")

    slip_configs = [
        {"label": "A. No trail @ 0.2%", "trail_pct": 0.0, "trail_cooldown": 0, "slippage": 0.002},
        {"label": "A. No trail @ 0.5%", "trail_pct": 0.0, "trail_cooldown": 0, "slippage": 0.005},
        {"label": "B. Trail @ 0.2%", "trail_pct": 0.08, "trail_cooldown": 20, "slippage": 0.002},
        {"label": "B. Trail @ 0.5%", "trail_pct": 0.08, "trail_cooldown": 20, "slippage": 0.005},
    ]

    slip_results = []
    for cfg in slip_configs:
        r = run_combined(
            **cfg, start=start, end=end, cache_dir=cache_dir, verbose=False,
        )
        slip_results.append(r)

    # ── Summary Tables ──
    all_results = results + slip_results[1::2]  # add the 0.5% versions

    print(f"\n\n{'='*110}")
    print(f"  FULL COMPARISON SUMMARY")
    print(f"{'='*110}")
    print(f"  {'Config':<28} {'CAGR':>8} {'MaxDD':>8} {'$10k->':>10} {'W3yr':>8} {'R5yr':>8} "
          f"{'Neg5yr':>7} {'UW(mo)':>8} {'Trades':>8} {'Trails':>8}")
    print(f"  {'-'*106}")

    for r in slip_results:
        print(f"  {r['label']:<28} {r['ann_ret']:>+7.2f}% {-r['max_dd']*100:>7.1f}% "
              f"${r['equity']:>9,.0f} {r['worst_3yr']:>+7.2f}% {r['worst_5yr']:>+7.2f}% "
              f"{r['neg_5yr_windows']:>7} {r['longest_uw']:>7.1f} "
              f"{r['n_trades']:>8} {r['trail_triggers']:>8}")

    # Delta analysis
    a02 = slip_results[0]  # no trail 0.2%
    b02 = slip_results[2]  # trail 0.2%
    a05 = slip_results[1]  # no trail 0.5%
    b05 = slip_results[3]  # trail 0.5%

    print(f"\n  {'DELTA (Trail vs No Trail)':<28}")
    print(f"  {'-'*80}")
    print(f"  {'@ 0.2% slippage':<28} {b02['ann_ret']-a02['ann_ret']:>+7.2f}% "
          f"{(-b02['max_dd']+a02['max_dd'])*100:>+7.1f}% "
          f"{'':>10} {b02['worst_3yr']-a02['worst_3yr']:>+7.2f}% "
          f"{b02['worst_5yr']-a02['worst_5yr']:>+7.2f}%")
    print(f"  {'@ 0.5% slippage':<28} {b05['ann_ret']-a05['ann_ret']:>+7.2f}% "
          f"{(-b05['max_dd']+a05['max_dd'])*100:>+7.1f}% "
          f"{'':>10} {b05['worst_3yr']-a05['worst_3yr']:>+7.2f}% "
          f"{b05['worst_5yr']-a05['worst_5yr']:>+7.2f}%")

    # Slippage cost
    print(f"\n  {'SLIPPAGE COST (0.5% vs 0.2%)':<28}")
    print(f"  {'-'*80}")
    print(f"  {'No trail':<28} {a05['ann_ret']-a02['ann_ret']:>+7.2f}% CAGR")
    print(f"  {'With trail':<28} {b05['ann_ret']-b02['ann_ret']:>+7.2f}% CAGR")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Combined Strategy Backtest")
    p.add_argument("--start", default="2009-06-25")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--cache-dir", default="state/etf_cache")
    args = p.parse_args()

    run_comparison(start=args.start, end=args.end, cache_dir=args.cache_dir)
