"""
Moving Average Crossover: QQQ / TLT / Cash

Three-state strategy checked DAILY:
  1. QQQ > its N-day SMA  →  100% QQQ
  2. QQQ < SMA, TLT > its N-day SMA  →  100% TLT
  3. Both below SMA  →  100% Cash (earns brokerage yield)

Tests with 30, 50, and 63-day moving averages.
"""
from __future__ import annotations

import argparse
import json
import os


# ── Data Loading ──────────────────────────────────────────────────

def load_etf_cached(ticker: str, cache_dir: str = "state/etf_cache",
                    start: str = "1999-01-01", end: str = "2025-01-01"
                    ) -> tuple[list[str], list[float]]:
    """Load daily adjusted close from cache or yfinance."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_daily.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            data = json.load(f)
        return data["dates"], data["closes"]

    import yfinance as yf
    print(f"  Fetching {ticker} from Yahoo Finance...")
    df = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]

    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    closes = df["Close"].tolist()
    print(f"    {len(closes)} daily bars ({dates[0]} to {dates[-1]})")

    with open(cache_file, "w") as f:
        json.dump({"dates": dates, "closes": closes}, f)

    return dates, closes


# ── Helpers ───────────────────────────────────────────────────────

def sma(closes: list[float], period: int) -> float:
    """Simple moving average of last N values."""
    if len(closes) < period:
        return closes[-1] if closes else 0
    return sum(closes[-period:]) / period


# ── Main Backtest ─────────────────────────────────────────────────

def run_backtest(
    ma_period: int = 50,
    buffer_pct: float = 0.0,    # hysteresis band: must be X% above MA to enter, X% below to exit
    confirm_days: int = 0,      # must stay above/below MA for N consecutive days to trigger
    confirm_entry: int = 0,     # override confirm_days for entries only (0 = use confirm_days)
    confirm_exit: int = 0,      # override confirm_days for exits only (0 = use confirm_days)
    commission: float = 0.001,
    starting_cash: float = 10000.0,
    cash_apy: float = 0.0,
    use_leveraged = False,       # False=1x, True/'3x'=TQQQ/TMF, '2x'=QLD/UBT
    alloc_pct: float = 1.0,     # fraction of equity to invest (rest stays cash buffer)
    alt_ticker: str = '',        # rotation alt when both signals fail (e.g. 'DBMF'). ''=cash
    buffer_ticker: str = '',     # ETF for uninvested buffer (e.g. 'BIL'). ''=cash at APY
    cache_dir: str = "state/etf_cache",
    start: str = "1999-01-01",
    end: str = "2025-01-01",
    verbose: bool = True,
):
    daily_cash_rate = (1 + cash_apy) ** (1/252) - 1

    # Signal tickers (always 1x for clean signals)
    sig_growth = 'QQQ'
    sig_safety = 'TLT'
    # Trade tickers (what we actually hold)
    if use_leveraged == '2x':
        trade_growth, trade_safety, lev_label = 'QLD', 'UBT', '2x'
    elif use_leveraged:
        trade_growth, trade_safety, lev_label = 'TQQQ', 'TMF', '3x'
    else:
        trade_growth, trade_safety, lev_label = 'QQQ', 'TLT', '1x'

    # Load data
    all_tickers = list(set([sig_growth, sig_safety, trade_growth, trade_safety, 'SPY']))
    if alt_ticker:
        all_tickers.append(alt_ticker)
    if buffer_ticker:
        all_tickers.append(buffer_ticker)
    all_tickers = list(set(all_tickers))
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

    if verbose:
        print(f"  Period: {common_dates[0]} to {common_dates[-1]} ({years:.1f} years)")
        buf_str = f" | Buffer: {buffer_pct*100:.1f}%" if buffer_pct > 0 else ""
        conf_str = f" | Confirm: {confirm_days}d" if confirm_days > 0 else ""
        alloc_str = f" | Alloc: {alloc_pct*100:.0f}%" if alloc_pct < 1.0 else ""
        alt_str = f" | Alt: {alt_ticker}" if alt_ticker else ""
        buf_tick_str = f" | Buffer: {buffer_ticker}" if buffer_ticker else ""
        print(f"  {lev_label} ({trade_growth}/{trade_safety}) | Signals: {sig_growth}/{sig_safety}{alt_str}{buf_tick_str}")
        print(f"  MA period: {ma_period}-day{buf_str}{conf_str}{alloc_str} | Commission: {commission*100:.2f}% | Cash APY: {cash_apy*100:.2f}%")

    # State: tracking invested vs cash buffer for partial allocation
    total_equity = starting_cash
    invested_equity = 0.0
    if buffer_ticker:
        buffer_equity = starting_cash
        cash_buffer = 0.0
    else:
        buffer_equity = 0.0
        cash_buffer = starting_cash
    holding = None  # trade_growth, trade_safety, or None (cash)
    n_trades = 0
    trade_log = []
    entry_price = 0
    entry_bar = 0
    cash_interest_total = 0.0
    cash_days = 0
    growth_days = 0
    safety_days = 0
    alt_days = 0

    # Tracking
    equity_curve = [total_equity]
    peak_eq = total_equity
    max_dd = 0

    # Confirmation counters: track consecutive days a signal has been active
    qqq_above_count = 0
    tlt_above_count = 0
    qqq_below_count = 0
    tlt_below_count = 0

    for i in range(1, n):
        # Buffer portion: track ETF or earn flat interest
        if buffer_ticker and buffer_equity > 0:
            prev_buf = prices[buffer_ticker][i-1]
            cur_buf = prices[buffer_ticker][i]
            buffer_equity *= (1 + (cur_buf - prev_buf) / prev_buf)
        else:
            cb_interest = cash_buffer * daily_cash_rate
            cash_buffer += cb_interest
            cash_interest_total += cb_interest

        if holding is None:
            cash_days += 1
        elif holding == alt_ticker:
            prev = prices[holding][i-1]
            cur = prices[holding][i]
            invested_equity *= (1 + (cur - prev) / prev)
            alt_days += 1
        else:
            # Mark to market the invested portion
            prev = prices[holding][i-1]
            cur = prices[holding][i]
            invested_equity *= (1 + (cur - prev) / prev)
            if holding == trade_growth:
                growth_days += 1
            else:
                safety_days += 1

        total_equity = invested_equity + cash_buffer + buffer_equity
        equity_curve.append(total_equity)

        if total_equity > peak_eq:
            peak_eq = total_equity
        dd = (peak_eq - total_equity) / peak_eq
        if dd > max_dd:
            max_dd = dd

        # Need enough bars for MA
        if i < ma_period:
            continue

        # Compute MAs using 1x SIGNAL tickers (not trade tickers)
        qqq_sma = sma(prices[sig_growth][:i+1], ma_period)
        tlt_sma = sma(prices[sig_safety][:i+1], ma_period)

        qqq_price = prices[sig_growth][i]
        tlt_price = prices[sig_safety][i]

        # Update confirmation counters
        if qqq_price > qqq_sma * (1 + buffer_pct):
            qqq_above_count += 1
            qqq_below_count = 0
        elif qqq_price < qqq_sma * (1 - buffer_pct):
            qqq_below_count += 1
            qqq_above_count = 0
        # else: in the buffer band, don't reset either counter

        if tlt_price > tlt_sma * (1 + buffer_pct):
            tlt_above_count += 1
            tlt_below_count = 0
        elif tlt_price < tlt_sma * (1 - buffer_pct):
            tlt_below_count += 1
            tlt_above_count = 0

        # Confirmed signals — asymmetric entry/exit thresholds
        ce = max(confirm_entry if confirm_entry > 0 else confirm_days, 1)
        cx = max(confirm_exit if confirm_exit > 0 else confirm_days, 1)
        qqq_confirmed_above = qqq_above_count >= ce   # entry signal
        qqq_confirmed_below = qqq_below_count >= cx   # exit signal
        tlt_confirmed_above = tlt_above_count >= ce
        tlt_confirmed_below = tlt_below_count >= cx

        # Determine target state using hysteresis:
        # - To ENTER a position, need confirmed above signal
        # - To EXIT a position, need confirmed below signal
        # - If no confirmed signal either way, hold current position
        target = holding  # default: stay put

        fallback = alt_ticker if alt_ticker else None  # DBMF or cash

        if holding == trade_growth:
            if qqq_confirmed_below:
                target = trade_safety if tlt_confirmed_above else fallback
        elif holding == trade_safety:
            if qqq_confirmed_above:
                target = trade_growth
            elif tlt_confirmed_below:
                target = fallback
        elif holding == alt_ticker:
            # In alt — check if growth or safety recovered
            if qqq_confirmed_above:
                target = trade_growth
            elif tlt_confirmed_above:
                target = trade_safety
        else:  # holding is None (cash)
            if qqq_confirmed_above:
                target = trade_growth
            elif tlt_confirmed_above:
                target = trade_safety

        # Switch if needed
        if target != holding:
            # Sell current
            if holding is not None:
                hold_pnl = (prices[holding][i] - entry_price) / entry_price
                invested_equity *= (1 - commission)
                cash_buffer += invested_equity
                invested_equity = 0
                total_equity = cash_buffer + buffer_equity
                trade_log.append({
                    'date': common_dates[i],
                    'action': 'SELL',
                    'ticker': holding,
                    'pnl': hold_pnl,
                    'days': i - entry_bar,
                    'equity': total_equity,
                })
                n_trades += 1

            # Buy new
            if target is not None:
                total_equity = invested_equity + cash_buffer + buffer_equity
                invest_amount = total_equity * alloc_pct
                uninvested = total_equity - invest_amount
                if buffer_ticker:
                    buffer_equity = uninvested
                    cash_buffer = 0
                else:
                    cash_buffer = uninvested
                    buffer_equity = 0
                invested_equity = invest_amount * (1 - commission)
                entry_price = prices[target][i]
                entry_bar = i
                trade_log.append({
                    'date': common_dates[i],
                    'action': 'BUY',
                    'ticker': target,
                    'pnl': None,
                    'days': None,
                    'equity': invested_equity + cash_buffer + buffer_equity,
                })
                n_trades += 1
            else:
                # Going to cash — move everything to buffer or cash
                total_equity = cash_buffer + buffer_equity
                if buffer_ticker:
                    buffer_equity = total_equity
                    cash_buffer = 0
                else:
                    cash_buffer = total_equity
                    buffer_equity = 0

            holding = target

    # Results
    total_equity = invested_equity + cash_buffer + buffer_equity
    equity = total_equity  # for compatibility
    total_ret = (total_equity / starting_cash - 1) * 100
    ann_ret = ((total_equity / starting_cash) ** (1/years) - 1) * 100

    # SPY benchmark
    spy_ret = (prices['SPY'][-1] / prices['SPY'][0] - 1) * 100
    spy_ann = ((1 + spy_ret/100) ** (1/years) - 1) * 100
    spy_peak = prices['SPY'][0]
    spy_max_dd = 0
    for p in prices['SPY']:
        if p > spy_peak: spy_peak = p
        dd = (spy_peak - p) / spy_peak
        if dd > spy_max_dd: spy_max_dd = dd

    # Growth B&H (QQQ or TQQQ)
    growth_bh_ret = (prices[trade_growth][-1] / prices[trade_growth][0] - 1) * 100
    growth_bh_ann = ((1 + growth_bh_ret/100) ** (1/years) - 1) * 100
    growth_bh_peak = prices[trade_growth][0]
    growth_bh_max_dd = 0
    for p in prices[trade_growth]:
        if p > growth_bh_peak: growth_bh_peak = p
        dd = (growth_bh_peak - p) / growth_bh_peak
        if dd > growth_bh_max_dd: growth_bh_max_dd = dd

    total_days = growth_days + safety_days + cash_days
    pct_qqq = growth_days / total_days * 100 if total_days else 0
    pct_tlt = safety_days / total_days * 100 if total_days else 0
    pct_cash = cash_days / total_days * 100 if total_days else 0

    if verbose:
        print(f"\n  {'Strategy':<20} {'End Value':>12} {'Total':>10} {'Annual':>8} {'MaxDD':>8} {'Trades':>8}")
        print(f"  {'-'*66}")
        print(f"  {'MA CROSSOVER':<20} ${total_equity:>11,.2f} {total_ret:>+9.1f}% {ann_ret:>+7.1f}% {-max_dd*100:>7.1f}% {n_trades:>8}")
        print(f"  {trade_growth+' B&H':<20} ${starting_cash*(1+growth_bh_ret/100):>11,.2f} {growth_bh_ret:>+9.1f}% {growth_bh_ann:>+7.1f}% {-growth_bh_max_dd*100:>7.1f}%        1")
        print(f"  {'SPY B&H':<20} ${starting_cash*(1+spy_ret/100):>11,.2f} {spy_ret:>+9.1f}% {spy_ann:>+7.1f}% {-spy_max_dd*100:>7.1f}%        1")
        print(f"\n  Time allocation: {trade_growth} {pct_qqq:.0f}% | {trade_safety} {pct_tlt:.0f}% | Cash {pct_cash:.0f}%")
        if cash_apy > 0:
            print(f"  Cash interest earned: ${cash_interest_total:,.2f}")

        # Win rate
        sells = [t for t in trade_log if t['action'] == 'SELL']
        wins = sum(1 for t in sells if t['pnl'] and t['pnl'] > 0)
        print(f"  Win rate: {wins}/{len(sells)} ({wins/max(len(sells),1)*100:.0f}%)")
        avg_hold = sum(t['days'] for t in sells if t['days']) / max(len(sells), 1)
        print(f"  Avg hold: {avg_hold:.0f} days")

        # Yearly breakdown
        print(f"\n  {'Year':<6} {'Strategy':>10} {'QQQ B&H':>10} {'SPY B&H':>10} {'Held':<20}")
        print(f"  {'-'*60}")

        yr_start_eq = equity_curve[0]
        yr_start_qqq = prices['QQQ'][0]
        yr_start_spy = prices['SPY'][0]
        current_year = common_dates[0][:4]
        yr_holding_days = {'QQQ': 0, 'TLT': 0, 'CASH': 0}
        track_holding = None

        for j in range(1, n):
            yr = common_dates[j][:4]
            if yr != current_year or j == n - 1:
                idx = j - 1 if yr != current_year else j
                yr_ret = (equity_curve[idx] / yr_start_eq - 1) * 100
                qqq_yr = (prices['QQQ'][idx] / yr_start_qqq - 1) * 100
                spy_yr = (prices['SPY'][idx] / yr_start_spy - 1) * 100

                print(f"  {current_year:<6} {yr_ret:>+9.1f}% {qqq_yr:>+9.1f}% {spy_yr:>+9.1f}%")

                yr_start_eq = equity_curve[idx]
                yr_start_qqq = prices['QQQ'][idx]
                yr_start_spy = prices['SPY'][idx]
                current_year = yr

        # Trade log
        if verbose:
            print(f"\n  {'Date':<12} {'Action':<6} {'Ticker':<6} {'P&L':>8} {'Hold':>6} {'Equity':>12}")
            print(f"  {'-'*52}")
            for t in trade_log:
                pnl_str = f"{t['pnl']*100:+.1f}%" if t['pnl'] is not None else ""
                days_str = f"{t['days']}d" if t['days'] is not None else ""
                print(f"  {t['date']:<12} {t['action']:<6} {t['ticker']:<6} {pnl_str:>8} {days_str:>6} ${t['equity']:>11,.2f}")

    return {
        'equity': equity, 'total_ret': total_ret, 'ann_ret': ann_ret,
        'max_dd': max_dd, 'trades': n_trades, 'years': years,
        'pct_qqq': pct_qqq, 'pct_tlt': pct_tlt, 'pct_cash': pct_cash,
        'cash_interest': cash_interest_total,
    }


# ── Hybrid: 1x on weak signals, 3x on strong signals ─────────────

def run_hybrid(
    ma_period: int = 100,
    buffer_pct: float = 0.03,
    confirm_days: int = 3,
    lev_threshold: float = 0.05,  # how far above MA to upgrade to 3x (e.g. 0.05 = 5%)
    commission: float = 0.001,
    starting_cash: float = 10000.0,
    cash_apy: float = 0.0335,
    cache_dir: str = "state/etf_cache",
    start: str = "1999-01-01",
    end: str = "2025-01-01",
    verbose: bool = True,
):
    daily_cash_rate = (1 + cash_apy) ** (1/252) - 1

    # All tickers needed
    all_tickers = ['QQQ', 'TLT', 'TQQQ', 'TMF', 'SPY']
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

    if verbose:
        print(f"  Period: {common_dates[0]} to {common_dates[-1]} ({years:.1f} years)")
        print(f"  HYBRID: 1x (QQQ/TLT) on moderate signals, 3x (TQQQ/TMF) on strong signals")
        print(f"  MA: {ma_period}-day | Buffer: {buffer_pct*100:.1f}% | Confirm: {confirm_days}d | 3x threshold: {lev_threshold*100:.0f}% above MA")
        print(f"  Commission: {commission*100:.2f}% | Cash APY: {cash_apy*100:.2f}%")

    # Map: direction → (1x ticker, 3x ticker)
    GROWTH = {'1x': 'QQQ', '3x': 'TQQQ'}
    SAFETY = {'1x': 'TLT', '3x': 'TMF'}

    # State
    equity = starting_cash
    holding = None  # actual ticker held (QQQ, TQQQ, TLT, TMF, or None)
    n_trades = 0
    trade_log = []
    entry_price = 0
    entry_bar = 0
    cash_interest_total = 0.0
    days_in = {'QQQ': 0, 'TQQQ': 0, 'TLT': 0, 'TMF': 0, 'CASH': 0}

    equity_curve = [equity]
    peak_eq = equity
    max_dd = 0

    qqq_above_count = 0
    tlt_above_count = 0
    qqq_below_count = 0
    tlt_below_count = 0

    for i in range(1, n):
        # Cash earns interest
        if holding is None:
            interest = equity * daily_cash_rate
            equity += interest
            cash_interest_total += interest
            days_in['CASH'] += 1
        else:
            prev = prices[holding][i-1]
            cur = prices[holding][i]
            equity *= (1 + (cur - prev) / prev)
            days_in[holding] += 1

        equity_curve.append(equity)
        if equity > peak_eq:
            peak_eq = equity
        dd = (peak_eq - equity) / peak_eq
        if dd > max_dd:
            max_dd = dd

        if i < ma_period:
            continue

        # Signals from 1x tickers
        qqq_sma = sma(prices['QQQ'][:i+1], ma_period)
        tlt_sma = sma(prices['TLT'][:i+1], ma_period)
        qqq_price = prices['QQQ'][i]
        tlt_price = prices['TLT'][i]

        qqq_dist = (qqq_price - qqq_sma) / qqq_sma  # how far above/below MA
        tlt_dist = (tlt_price - tlt_sma) / tlt_sma

        # Update confirmation counters (same as before)
        if qqq_price > qqq_sma * (1 + buffer_pct):
            qqq_above_count += 1
            qqq_below_count = 0
        elif qqq_price < qqq_sma * (1 - buffer_pct):
            qqq_below_count += 1
            qqq_above_count = 0

        if tlt_price > tlt_sma * (1 + buffer_pct):
            tlt_above_count += 1
            tlt_below_count = 0
        elif tlt_price < tlt_sma * (1 - buffer_pct):
            tlt_below_count += 1
            tlt_above_count = 0

        min_confirm = max(confirm_days, 1)
        qqq_confirmed_above = qqq_above_count >= min_confirm
        qqq_confirmed_below = qqq_below_count >= min_confirm
        tlt_confirmed_above = tlt_above_count >= min_confirm
        tlt_confirmed_below = tlt_below_count >= min_confirm

        # Determine direction (same 3-state logic)
        growth_tickers = {GROWTH['1x'], GROWTH['3x']}
        safety_tickers = {SAFETY['1x'], SAFETY['3x']}

        direction = None  # 'growth', 'safety', or None (cash)
        if holding in growth_tickers:
            if qqq_confirmed_below:
                direction = 'safety' if tlt_confirmed_above else None
            else:
                direction = 'growth'
        elif holding in safety_tickers:
            if qqq_confirmed_above:
                direction = 'growth'
            elif tlt_confirmed_below:
                direction = None
            else:
                direction = 'safety'
        else:  # cash
            if qqq_confirmed_above:
                direction = 'growth'
            elif tlt_confirmed_above:
                direction = 'safety'

        # Determine leverage level based on signal strength
        if direction == 'growth':
            target = GROWTH['3x'] if qqq_dist >= lev_threshold else GROWTH['1x']
        elif direction == 'safety':
            target = SAFETY['3x'] if tlt_dist >= lev_threshold else SAFETY['1x']
        else:
            target = None

        # Switch if needed
        if target != holding:
            # Sell current
            if holding is not None:
                hold_pnl = (prices[holding][i] - entry_price) / entry_price
                equity *= (1 - commission)
                trade_log.append({
                    'date': common_dates[i], 'action': 'SELL',
                    'ticker': holding, 'pnl': hold_pnl,
                    'days': i - entry_bar, 'equity': equity,
                })
                n_trades += 1

            # Buy new
            if target is not None:
                equity *= (1 - commission)
                entry_price = prices[target][i]
                entry_bar = i
                trade_log.append({
                    'date': common_dates[i], 'action': 'BUY',
                    'ticker': target, 'pnl': None,
                    'days': None, 'equity': equity,
                })
                n_trades += 1

            holding = target

    # Results
    total_ret = (equity / starting_cash - 1) * 100
    ann_ret = ((equity / starting_cash) ** (1/years) - 1) * 100

    spy_ret = (prices['SPY'][-1] / prices['SPY'][0] - 1) * 100
    spy_ann = ((1 + spy_ret/100) ** (1/years) - 1) * 100
    spy_peak = prices['SPY'][0]
    spy_max_dd = 0
    for p in prices['SPY']:
        if p > spy_peak: spy_peak = p
        d = (spy_peak - p) / spy_peak
        if d > spy_max_dd: spy_max_dd = d

    total_days = sum(days_in.values())

    if verbose:
        print(f"\n  {'Strategy':<20} {'End Value':>12} {'Total':>10} {'Annual':>8} {'MaxDD':>8} {'Trades':>8}")
        print(f"  {'-'*66}")
        print(f"  {'HYBRID MA':<20} ${equity:>11,.2f} {total_ret:>+9.1f}% {ann_ret:>+7.1f}% {-max_dd*100:>7.1f}% {n_trades:>8}")
        print(f"  {'SPY B&H':<20} ${starting_cash*(1+spy_ret/100):>11,.2f} {spy_ret:>+9.1f}% {spy_ann:>+7.1f}% {-spy_max_dd*100:>7.1f}%        1")
        print(f"\n  Time: QQQ {days_in['QQQ']*100//total_days}% | TQQQ {days_in['TQQQ']*100//total_days}% | "
              f"TLT {days_in['TLT']*100//total_days}% | TMF {days_in['TMF']*100//total_days}% | Cash {days_in['CASH']*100//total_days}%")
        if cash_apy > 0:
            print(f"  Cash interest: ${cash_interest_total:,.2f}")

        sells = [t for t in trade_log if t['action'] == 'SELL']
        wins = sum(1 for t in sells if t['pnl'] and t['pnl'] > 0)
        print(f"  Win rate: {wins}/{len(sells)} ({wins/max(len(sells),1)*100:.0f}%)")

        # Yearly
        print(f"\n  {'Year':<6} {'Strategy':>10} {'QQQ B&H':>10} {'SPY B&H':>10}")
        print(f"  {'-'*40}")
        yr_start_eq = equity_curve[0]
        yr_start_qqq = prices['QQQ'][0]
        yr_start_spy = prices['SPY'][0]
        current_year = common_dates[0][:4]
        for j in range(1, n):
            yr = common_dates[j][:4]
            if yr != current_year or j == n - 1:
                idx = j - 1 if yr != current_year else j
                yr_ret = (equity_curve[idx] / yr_start_eq - 1) * 100
                qqq_yr = (prices['QQQ'][idx] / yr_start_qqq - 1) * 100
                spy_yr = (prices['SPY'][idx] / yr_start_spy - 1) * 100
                print(f"  {current_year:<6} {yr_ret:>+9.1f}% {qqq_yr:>+9.1f}% {spy_yr:>+9.1f}%")
                yr_start_eq = equity_curve[idx]
                yr_start_qqq = prices['QQQ'][idx]
                yr_start_spy = prices['SPY'][idx]
                current_year = yr

        # Trade log
        print(f"\n  {'Date':<12} {'Action':<6} {'Ticker':<6} {'P&L':>8} {'Hold':>6} {'Equity':>12}")
        print(f"  {'-'*52}")
        for t in trade_log:
            pnl_str = f"{t['pnl']*100:+.1f}%" if t['pnl'] is not None else ""
            days_str = f"{t['days']}d" if t['days'] is not None else ""
            print(f"  {t['date']:<12} {t['action']:<6} {t['ticker']:<6} {pnl_str:>8} {days_str:>6} ${t['equity']:>11,.2f}")

    return {
        'equity': equity, 'total_ret': total_ret, 'ann_ret': ann_ret,
        'max_dd': max_dd, 'trades': n_trades, 'years': years,
        'days_in': days_in, 'cash_interest': cash_interest_total,
    }


# ── CLI ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="QQQ/TLT/Cash MA Crossover Backtest")
    p.add_argument("--ma-period", type=int, default=50, help="Moving average period (e.g. 30, 50, 63)")
    p.add_argument("--buffer-pct", type=float, default=0.0, help="Hysteresis buffer as decimal (e.g. 0.02 for 2%%)")
    p.add_argument("--confirm-days", type=int, default=0, help="Consecutive days to confirm signal")
    p.add_argument("--commission", type=float, default=0.001, help="Commission per trade (0.001 = 0.1%%)")
    p.add_argument("--starting-cash", type=float, default=10000.0)
    p.add_argument("--cash-apy", type=float, default=0.0, help="Annual yield on cash (e.g. 0.0335 for 3.35%%)")
    p.add_argument("--leveraged", default="", help="Leverage level: '2x' for QLD/UBT, '3x' for TQQQ/TMF (signals still use QQQ/TLT)")
    p.add_argument("--alloc-pct", type=float, default=1.0, help="Fraction of equity to invest (e.g. 0.5 for 50%%)")
    p.add_argument("--cache-dir", default="state/etf_cache")
    p.add_argument("--start", default="1999-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--sweep", action="store_true", help="Run sweep across MA periods, buffers, and confirm days")
    p.add_argument("--hybrid", action="store_true", help="Hybrid mode: 1x on weak signals, 3x on strong signals")
    p.add_argument("--lev-threshold", type=float, default=0.05, help="Distance above MA to upgrade to 3x (e.g. 0.05 for 5%%)")
    args = p.parse_args()

    common_kw = dict(
        commission=args.commission, starting_cash=args.starting_cash,
        cache_dir=args.cache_dir, start=args.start, end=args.end,
    )

    if args.hybrid and args.sweep:
        print("=" * 90)
        print("  HYBRID 1x/3x — MA CROSSOVER SWEEP (with 3.35% cash yield)")
        print("  1x (QQQ/TLT) on moderate signals, 3x (TQQQ/TMF) on strong signals")
        print("=" * 90)

        results = []
        for ma in [50, 63, 100, 150, 200]:
            for buf in [0.0, 0.02, 0.03, 0.05]:
                for conf in [0, 3, 5]:
                    for lt in [0.03, 0.05, 0.07, 0.10]:
                        r = run_hybrid(
                            ma_period=ma, buffer_pct=buf, confirm_days=conf,
                            lev_threshold=lt, cash_apy=0.0335,
                            verbose=False, **common_kw,
                        )
                        label = f"MA{ma} b{buf*100:.0f}% c{conf}d t{lt*100:.0f}%"
                        results.append((label, r))

        print(f"\n{'='*90}")
        print(f"  TOP 20 BY ANNUAL RETURN (Max DD < 30%)")
        print(f"{'='*90}")
        print(f"  {'Config':<30} {'Return':>10} {'Annual':>8} {'Max DD':>8} {'Trades':>8}")
        print(f"  {'-'*64}")
        good = [(l, r) for l, r in results if r['max_dd'] < 0.30]
        for label, r in sorted(good, key=lambda x: -x[1]['ann_ret'])[:20]:
            print(f"  {label:<30} {r['total_ret']:>+9.1f}% {r['ann_ret']:>+7.1f}% {-r['max_dd']*100:>+7.1f}% {r['trades']:>8}")

        print(f"\n  TOP 20 BY ANNUAL RETURN (Max DD < 40%)")
        print(f"  {'Config':<30} {'Return':>10} {'Annual':>8} {'Max DD':>8} {'Trades':>8}")
        print(f"  {'-'*64}")
        good40 = [(l, r) for l, r in results if r['max_dd'] < 0.40]
        for label, r in sorted(good40, key=lambda x: -x[1]['ann_ret'])[:20]:
            print(f"  {label:<30} {r['total_ret']:>+9.1f}% {r['ann_ret']:>+7.1f}% {-r['max_dd']*100:>+7.1f}% {r['trades']:>8}")

        print(f"\n  ALL RESULTS (sorted by DD, top 30)")
        print(f"  {'Config':<30} {'Return':>10} {'Annual':>8} {'Max DD':>8} {'Trades':>8}")
        print(f"  {'-'*64}")
        for label, r in sorted(results, key=lambda x: x[1]['max_dd'])[:30]:
            print(f"  {label:<30} {r['total_ret']:>+9.1f}% {r['ann_ret']:>+7.1f}% {-r['max_dd']*100:>+7.1f}% {r['trades']:>8}")
    elif args.hybrid:
        print("=" * 80)
        print(f"  HYBRID 1x/3x — {args.ma_period}-DAY MA CROSSOVER")
        print("=" * 80)
        run_hybrid(
            ma_period=args.ma_period, buffer_pct=args.buffer_pct,
            confirm_days=args.confirm_days, lev_threshold=args.lev_threshold,
            cash_apy=args.cash_apy, verbose=True, **common_kw,
        )
    else:
        lev = args.leveraged if args.leveraged else False
        lev_labels = {'2x': 'QLD/UBT', '3x': 'TQQQ/TMF'}
        lev_label = lev_labels.get(lev, 'QQQ/TLT')
        print("=" * 80)
        print(f"  {lev_label} / CASH — {args.ma_period}-DAY MA CROSSOVER")
        print("=" * 80)
        run_backtest(
            ma_period=args.ma_period, buffer_pct=args.buffer_pct,
            confirm_days=args.confirm_days, cash_apy=args.cash_apy,
            use_leveraged=lev, alloc_pct=args.alloc_pct,
            verbose=True, **common_kw,
        )


if __name__ == "__main__":
    main()
