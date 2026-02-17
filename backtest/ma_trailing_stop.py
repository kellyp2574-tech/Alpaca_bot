"""
MA Crossover + Trailing Stop on 1x Signal ETF
===============================================
Adds a trailing stop to the MA crossover rotation strategy.

Current strategy (no trail):
  QQQ > 100-day SMA -> hold QLD (2x QQQ)
  TLT > 100-day SMA -> hold UBT (2x TLT)
  Both below         -> hold DBMF / cash
  3% buffer, 2-day entry confirm, 5-day exit confirm

New addition:
  Track peak of the 1x SIGNAL ETF (QQQ or TLT) while holding.
  If the 1x ETF drops X% from its peak, force a rotation:
    - Check which other option has its 1x ETF above its MA
    - Rotate there, or go to fallback (DBMF/cash)
  This catches sharp drawdowns before the slow MA signal reacts.

Test trails: 7%, 8%, 10%, 12%, 15%, 20% on the 1x ETF.
(A 10% drop in QQQ ~= 20% drop in QLD, so these are meaningful.)
"""
import argparse
import json
import os


# ── Data Loading ──────────────────────────────────────────────

def load_etf_cached(ticker, cache_dir="state/etf_cache", start="1999-01-01", end="2025-01-01"):
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


# ── Backtest ──────────────────────────────────────────────────

def run_backtest(
    ma_period: int = 100,
    buffer_pct: float = 0.03,
    confirm_entry: int = 2,
    confirm_exit: int = 5,
    trail_pct: float = 0.0,        # trailing stop on 1x signal ETF (0=off)
    trail_cooldown: int = 0,        # days after trail triggers before re-entering same direction
    use_leveraged: str = '2x',      # '2x' for QLD/UBT, '3x' for TQQQ/TMF, '' for 1x
    alt_ticker: str = 'DBMF',
    commission: float = 0.002,      # 0.2% slippage
    starting_cash: float = 10000.0,
    cash_apy: float = 0.0335,
    alloc_pct: float = 0.50,        # 50% of equity (MA strategy allocation)
    cache_dir: str = "state/etf_cache",
    start: str = "2009-06-25",
    end: str = "2025-01-01",
    verbose: bool = True,
):
    daily_cash_rate = (1 + cash_apy) ** (1 / 252) - 1

    # Signal tickers (always 1x)
    sig_growth = 'QQQ'
    sig_safety = 'TLT'

    # Trade tickers
    if use_leveraged == '2x':
        trade_growth, trade_safety, lev_label = 'QLD', 'UBT', '2x'
    elif use_leveraged == '3x':
        trade_growth, trade_safety, lev_label = 'TQQQ', 'TMF', '3x'
    else:
        trade_growth, trade_safety, lev_label = 'QQQ', 'TLT', '1x'

    # Load data
    all_tickers = list(set([
        sig_growth, sig_safety, trade_growth, trade_safety, 'SPY'
    ] + ([alt_ticker] if alt_ticker else [])))

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
        trail_str = f" | Trail: {trail_pct*100:.0f}% on 1x" if trail_pct > 0 else " | Trail: OFF"
        cool_str = f" ({trail_cooldown}d cooldown)" if trail_cooldown > 0 and trail_pct > 0 else ""
        alt_str = f" | Alt: {alt_ticker}" if alt_ticker else ""
        print(f"  Period: {common_dates[0]} to {common_dates[-1]} ({years:.1f} yr)")
        print(f"  {lev_label} ({trade_growth}/{trade_safety}){alt_str}")
        print(f"  MA: {ma_period}d | Buffer: {buffer_pct*100:.0f}% | Entry: {confirm_entry}d | Exit: {confirm_exit}d{trail_str}{cool_str}")
        print(f"  Alloc: {alloc_pct*100:.0f}% | Slippage: {commission*100:.1f}% | Cash APY: {cash_apy*100:.2f}%")

    # State
    invested = 0.0
    cash_buffer = starting_cash
    holding = None  # trade_growth, trade_safety, alt_ticker, or None
    direction = None  # 'growth', 'safety', 'alt', None
    entry_price = 0.0
    entry_bar = 0
    n_trades = 0
    trade_log = []

    # Trailing stop state
    signal_peak = 0.0           # peak of 1x signal ETF since entry
    trail_triggers = 0          # count of trail stop triggers
    last_trail_bar = -999       # bar index of last trail trigger
    last_trail_direction = None  # direction that was stopped out

    # Confirmation counters
    qa = 0  # QQQ above count
    qb = 0  # QQQ below count
    ta = 0  # TLT above count
    tb = 0  # TLT below count

    # Tracking
    equity_curve = [starting_cash]
    peak_eq = starting_cash
    max_dd = 0.0
    growth_days = 0
    safety_days = 0
    alt_days = 0
    cash_days = 0

    for i in range(1, n):
        # Cash earns interest
        cash_buffer += cash_buffer * daily_cash_rate

        # Mark to market
        if holding is not None:
            prev_p = prices[holding][i - 1]
            cur_p = prices[holding][i]
            if prev_p > 0:
                invested *= (1 + (cur_p - prev_p) / prev_p)

        if direction == 'growth':
            growth_days += 1
        elif direction == 'safety':
            safety_days += 1
        elif direction == 'alt':
            alt_days += 1
        else:
            cash_days += 1

        total_equity = invested + cash_buffer
        equity_curve.append(total_equity)
        if total_equity > peak_eq:
            peak_eq = total_equity
        dd = (peak_eq - total_equity) / peak_eq
        if dd > max_dd:
            max_dd = dd

        if i < ma_period:
            continue

        # ── Compute signals ──
        qqq_sma_val = sma(prices[sig_growth][:i + 1], ma_period)
        tlt_sma_val = sma(prices[sig_safety][:i + 1], ma_period)
        qqq_price = prices[sig_growth][i]
        tlt_price = prices[sig_safety][i]

        # Update confirmation counters
        if qqq_price > qqq_sma_val * (1 + buffer_pct):
            qa += 1; qb = 0
        elif qqq_price < qqq_sma_val * (1 - buffer_pct):
            qb += 1; qa = 0

        if tlt_price > tlt_sma_val * (1 + buffer_pct):
            ta += 1; tb = 0
        elif tlt_price < tlt_sma_val * (1 - buffer_pct):
            tb += 1; ta = 0

        qqq_above = qa >= confirm_entry
        qqq_below = qb >= confirm_exit
        tlt_above = ta >= confirm_entry
        tlt_below = tb >= confirm_exit

        # ── Trailing stop check ──
        trail_fired = False
        if trail_pct > 0 and holding is not None and direction in ('growth', 'safety'):
            # Track peak of the 1x SIGNAL ETF
            sig_ticker = sig_growth if direction == 'growth' else sig_safety
            sig_price = prices[sig_ticker][i]

            if sig_price > signal_peak:
                signal_peak = sig_price

            drawdown_from_peak = (signal_peak - sig_price) / signal_peak

            if drawdown_from_peak >= trail_pct:
                trail_fired = True
                trail_triggers += 1
                last_trail_bar = i
                last_trail_direction = direction

                # Force exit
                hold_pnl = (prices[holding][i] - entry_price) / entry_price
                invested *= (1 - commission)
                cash_buffer += invested
                invested = 0

                trade_log.append({
                    'date': common_dates[i], 'action': 'TRAIL',
                    'ticker': holding, 'pnl': hold_pnl,
                    'days': i - entry_bar,
                    'equity': cash_buffer,
                    'detail': f"{sig_ticker} -{drawdown_from_peak*100:.1f}% from peak",
                })
                n_trades += 1

                # Immediately check what else is above its MA
                new_target = None
                new_direction = None

                if direction == 'growth':
                    # Was in QLD, got stopped out — check TLT
                    if tlt_price > tlt_sma_val:
                        new_target = trade_safety
                        new_direction = 'safety'
                    elif alt_ticker:
                        new_target = alt_ticker
                        new_direction = 'alt'
                elif direction == 'safety':
                    # Was in UBT, got stopped out — check QQQ
                    if qqq_price > qqq_sma_val:
                        new_target = trade_growth
                        new_direction = 'growth'
                    elif alt_ticker:
                        new_target = alt_ticker
                        new_direction = 'alt'

                if new_target is not None:
                    total_eq = cash_buffer
                    invest_amt = total_eq * alloc_pct
                    cash_buffer = total_eq - invest_amt
                    invested = invest_amt * (1 - commission)
                    entry_price = prices[new_target][i]
                    entry_bar = i
                    signal_peak = prices[sig_growth if new_direction == 'growth' else sig_safety][i]
                    holding = new_target
                    direction = new_direction

                    trade_log.append({
                        'date': common_dates[i], 'action': 'BUY',
                        'ticker': new_target, 'pnl': None, 'days': None,
                        'equity': invested + cash_buffer,
                        'detail': 'trail rotation',
                    })
                    n_trades += 1
                else:
                    holding = None
                    direction = None
                    signal_peak = 0

                continue  # skip normal rotation logic this bar

        # ── Normal MA rotation logic ──
        fallback_ticker = alt_ticker if alt_ticker else None
        target = holding
        target_dir = direction

        if direction == 'growth':
            if qqq_below:
                if tlt_above:
                    target = trade_safety; target_dir = 'safety'
                else:
                    target = fallback_ticker; target_dir = 'alt' if fallback_ticker else None
        elif direction == 'safety':
            if qqq_above:
                target = trade_growth; target_dir = 'growth'
            elif tlt_below:
                target = fallback_ticker; target_dir = 'alt' if fallback_ticker else None
        elif direction == 'alt':
            if qqq_above:
                target = trade_growth; target_dir = 'growth'
            elif tlt_above:
                target = trade_safety; target_dir = 'safety'
        else:  # cash
            if qqq_above:
                target = trade_growth; target_dir = 'growth'
            elif tlt_above:
                target = trade_safety; target_dir = 'safety'

        # Cooldown: after trail stop, don't re-enter same direction for N bars
        if trail_cooldown > 0 and target_dir == last_trail_direction:
            if (i - last_trail_bar) < trail_cooldown:
                target = holding
                target_dir = direction

        # Execute rotation
        if target != holding:
            # Sell current
            if holding is not None:
                hold_pnl = (prices[holding][i] - entry_price) / entry_price
                invested *= (1 - commission)
                cash_buffer += invested
                invested = 0
                trade_log.append({
                    'date': common_dates[i], 'action': 'SELL',
                    'ticker': holding, 'pnl': hold_pnl,
                    'days': i - entry_bar,
                    'equity': cash_buffer,
                    'detail': 'MA signal',
                })
                n_trades += 1

            # Buy new
            if target is not None:
                total_eq = invested + cash_buffer
                invest_amt = total_eq * alloc_pct
                cash_buffer = total_eq - invest_amt
                invested = invest_amt * (1 - commission)
                entry_price = prices[target][i]
                entry_bar = i
                # Reset trailing stop peak
                if target_dir == 'growth':
                    signal_peak = prices[sig_growth][i]
                elif target_dir == 'safety':
                    signal_peak = prices[sig_safety][i]
                else:
                    signal_peak = 0

                trade_log.append({
                    'date': common_dates[i], 'action': 'BUY',
                    'ticker': target, 'pnl': None, 'days': None,
                    'equity': invested + cash_buffer,
                    'detail': 'MA signal',
                })
                n_trades += 1
            else:
                # Going to cash
                cash_buffer = invested + cash_buffer
                invested = 0
                signal_peak = 0

            holding = target
            direction = target_dir

    # ── Results ──
    total_equity = invested + cash_buffer
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

    # Trade ticker B&H
    growth_bh_ret = (prices[trade_growth][-1] / prices[trade_growth][0] - 1) * 100
    growth_bh_ann = ((1 + growth_bh_ret / 100) ** (1 / years) - 1) * 100

    sells = [t for t in trade_log if t['action'] in ('SELL', 'TRAIL')]
    wins = sum(1 for t in sells if (t.get('pnl') or 0) > 0)
    trail_sells = sum(1 for t in sells if t['action'] == 'TRAIL')
    avg_hold = sum((t.get('days') or 0) for t in sells) / max(len(sells), 1)

    total_days = growth_days + safety_days + alt_days + cash_days
    pct_growth = growth_days / total_days * 100 if total_days else 0
    pct_safety = safety_days / total_days * 100 if total_days else 0
    pct_alt = alt_days / total_days * 100 if total_days else 0
    pct_cash = cash_days / total_days * 100 if total_days else 0

    if verbose:
        print(f"\n  {'Strategy':<20} {'End Value':>12} {'Total':>10} {'Annual':>8} {'MaxDD':>8} {'Trades':>8}")
        print(f"  {'-'*66}")
        trail_label = f"MA+Trail{trail_pct*100:.0f}%" if trail_pct > 0 else "MA (no trail)"
        print(f"  {trail_label:<20} ${total_equity:>11,.2f} {total_ret:>+9.1f}% {ann_ret:>+7.1f}% {-max_dd*100:>7.1f}% {n_trades:>8}")
        print(f"  {trade_growth+' B&H':<20} ${starting_cash*(1+growth_bh_ret/100):>11,.2f} {growth_bh_ret:>+9.1f}% {growth_bh_ann:>+7.1f}%")
        print(f"  {'SPY B&H':<20} ${starting_cash*(1+spy_ret/100):>11,.2f} {spy_ret:>+9.1f}% {spy_ann:>+7.1f}% {-spy_max_dd*100:>7.1f}%")

        print(f"\n  Time: {trade_growth} {pct_growth:.0f}% | {trade_safety} {pct_safety:.0f}%", end="")
        if alt_ticker:
            print(f" | {alt_ticker} {pct_alt:.0f}%", end="")
        print(f" | Cash {pct_cash:.0f}%")

        print(f"  Win rate: {wins}/{len(sells)} ({wins*100//max(len(sells),1)}%)")
        print(f"  Avg hold: {avg_hold:.0f} days")
        if trail_pct > 0:
            print(f"  Trail stops triggered: {trail_triggers}")

        # Yearly breakdown
        print(f"\n  {'Year':<6} {'Strategy':>10} {trade_growth+' B&H':>10} {'SPY B&H':>10} {'Trades':>8} {'Trails':>8}")
        print(f"  {'-'*56}")

        yr_start_eq = equity_curve[0]
        yr_start_g = prices[trade_growth][0]
        yr_start_spy = prices['SPY'][0]
        current_year = common_dates[0][:4]

        for j in range(1, n):
            yr = common_dates[j][:4]
            if yr != current_year or j == n - 1:
                idx = j - 1 if yr != current_year else j
                yr_ret = (equity_curve[idx] / yr_start_eq - 1) * 100
                g_yr = (prices[trade_growth][idx] / yr_start_g - 1) * 100
                spy_yr = (prices['SPY'][idx] / yr_start_spy - 1) * 100

                yr_trades = sum(1 for t in trade_log if t['date'][:4] == current_year and t['action'] in ('SELL', 'TRAIL'))
                yr_trails = sum(1 for t in trade_log if t['date'][:4] == current_year and t['action'] == 'TRAIL')

                print(f"  {current_year:<6} {yr_ret:>+9.1f}% {g_yr:>+9.1f}% {spy_yr:>+9.1f}% {yr_trades:>8} {yr_trails:>8}")

                yr_start_eq = equity_curve[idx]
                yr_start_g = prices[trade_growth][idx]
                yr_start_spy = prices['SPY'][idx]
                current_year = yr

        # Trade log (show trail stops highlighted)
        if verbose:
            print(f"\n  {'Date':<12} {'Action':<7} {'Ticker':<6} {'P&L':>8} {'Hold':>6} {'Equity':>12} {'Detail'}")
            print(f"  {'-'*70}")
            for t in trade_log:
                pnl_str = f"{t['pnl']*100:+.1f}%" if t['pnl'] is not None else ""
                days_str = f"{t['days']}d" if t['days'] is not None else ""
                detail = t.get('detail', '')
                action = t['action']
                if action == 'TRAIL':
                    action = 'TRAIL*'  # highlight
                print(f"  {t['date']:<12} {action:<7} {t['ticker']:<6} {pnl_str:>8} {days_str:>6} ${t['equity']:>11,.2f} {detail}")

    return {
        'ann_ret': ann_ret, 'total_ret': total_ret, 'max_dd': max_dd,
        'equity': total_equity, 'trades': n_trades, 'years': years,
        'trail_triggers': trail_triggers, 'win_rate': wins / max(len(sells), 1),
        'avg_hold': avg_hold, 'trail_pct': trail_pct,
        'pct_growth': pct_growth, 'pct_safety': pct_safety,
    }


# ── Comparison: No Trail vs Various Trail Stops ──────────────

def run_comparison(start="2009-06-25", end="2025-01-01", cache_dir="state/etf_cache"):
    print("=" * 100)
    print("  MA CROSSOVER + TRAILING STOP COMPARISON")
    print("  Signal: QQQ/TLT 1x | Trade: QLD/UBT 2x | Alt: DBMF | Alloc: 50%")
    print("  MA: 100d | Buffer: 3% | Entry: 2d | Exit: 5d | Slippage: 0.2%")
    print("=" * 100)

    trails = [0.0, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20]
    results = []

    for trail in trails:
        label = f"Trail {trail*100:.0f}%" if trail > 0 else "No trail"
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
        r = run_backtest(
            trail_pct=trail,
            start=start, end=end, cache_dir=cache_dir,
            verbose=True,
        )
        r['label'] = label
        results.append(r)

    # Summary
    print(f"\n\n{'='*110}")
    print(f"  SUMMARY: TRAILING STOP IMPACT")
    print(f"{'='*110}")
    print(f"  {'Config':<16} {'CAGR':>8} {'MaxDD':>8} {'$10k->':>10} {'Trades':>8} {'Win%':>6} "
          f"{'AvgHold':>8} {'Trails':>8} {'%Growth':>8} {'%Safety':>8}")
    print(f"  {'-'*96}")

    for r in results:
        wr = r['win_rate'] * 100
        print(f"  {r['label']:<16} {r['ann_ret']:>+7.2f}% {-r['max_dd']*100:>7.1f}% "
              f"${r['equity']:>9,.0f} {r['trades']:>8} {wr:>5.0f}% "
              f"{r['avg_hold']:>7.0f}d {r['trail_triggers']:>8} "
              f"{r['pct_growth']:>7.0f}% {r['pct_safety']:>7.0f}%")

    return results


# ── Trail + Cooldown Sweep ────────────────────────────────────

def run_sweep(start="2009-06-25", end="2025-01-01", cache_dir="state/etf_cache"):
    print("=" * 110)
    print("  TRAIL STOP SWEEP (with cooldown variations)")
    print("  MA100 | Buffer 3% | Entry 2d | Exit 5d | QLD/UBT 2x | DBMF alt | 50% alloc")
    print("=" * 110)

    print(f"\n  {'Trail':>6} {'Cool':>6} {'CAGR':>8} {'MaxDD':>8} {'$10k->':>10} {'Trades':>8} "
          f"{'Win%':>6} {'AvgHold':>8} {'Trails':>8}")
    print(f"  {'-'*72}")

    for trail in [0.0, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20]:
        cooldowns = [0] if trail == 0 else [0, 5, 10, 20]
        for cool in cooldowns:
            r = run_backtest(
                trail_pct=trail, trail_cooldown=cool,
                start=start, end=end, cache_dir=cache_dir,
                verbose=False,
            )
            trail_str = f"{trail*100:.0f}%" if trail > 0 else "off"
            cool_str = f"{cool}d" if cool > 0 else "-"
            wr = r['win_rate'] * 100
            print(f"  {trail_str:>6} {cool_str:>6} {r['ann_ret']:>+7.2f}% {-r['max_dd']*100:>7.1f}% "
                  f"${r['equity']:>9,.0f} {r['trades']:>8} {wr:>5.0f}% "
                  f"{r['avg_hold']:>7.0f}d {r['trail_triggers']:>8}")


# ── CLI ───────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="MA Crossover + Trailing Stop Backtest")
    p.add_argument("--start", default="2009-06-25")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--cache-dir", default="state/etf_cache")
    p.add_argument("--trail", type=float, default=0.0, help="Trail stop %% on 1x ETF (e.g. 0.10 for 10%%)")
    p.add_argument("--cooldown", type=int, default=0, help="Days after trail before re-entering same direction")
    p.add_argument("--compare", action="store_true", help="Run head-to-head comparison")
    p.add_argument("--sweep", action="store_true", help="Run full sweep with cooldown variations")
    p.add_argument("--all", action="store_true", help="Run compare + sweep")
    args = p.parse_args()

    if args.all:
        run_comparison(start=args.start, end=args.end, cache_dir=args.cache_dir)
        print("\n\n")
        run_sweep(start=args.start, end=args.end, cache_dir=args.cache_dir)
    elif args.compare:
        run_comparison(start=args.start, end=args.end, cache_dir=args.cache_dir)
    elif args.sweep:
        run_sweep(start=args.start, end=args.end, cache_dir=args.cache_dir)
    else:
        # Default: single run
        print("=" * 80)
        trail_str = f" + {args.trail*100:.0f}% Trail" if args.trail > 0 else ""
        print(f"  MA CROSSOVER{trail_str} — QLD/UBT 2x")
        print("=" * 80)
        run_backtest(
            trail_pct=args.trail, trail_cooldown=args.cooldown,
            start=args.start, end=args.end, cache_dir=args.cache_dir,
            verbose=True,
        )


if __name__ == "__main__":
    main()
