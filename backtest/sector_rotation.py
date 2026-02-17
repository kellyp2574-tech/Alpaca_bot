"""
Equity Sector Rotation Backtest

Rotates between US sector ETFs based on momentum, with optional
bond ETF (TLT/SHY) as a safe haven when all sectors are falling.

Key advantages over crypto/forex rotation:
  - Stocks appreciate long-term (underlying trend helps)
  - Sectors have REAL divergence (tech vs energy vs healthcare)
  - 20+ years of data for robust validation
  - Dividends included in adjusted close prices
  - Academic backing for sector momentum strategies

Uses yfinance for free historical data (adjusted for splits & dividends).
"""
from __future__ import annotations

import argparse
import yfinance as yf
import json
import os
from datetime import datetime


# ── Data Loading ──────────────────────────────────────────────────

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLY": "Consumer Discretionary",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
    # Benchmarks & safe havens
    "SPY": "S&P 500 (benchmark)",
    "TLT": "20+ Year Treasury Bonds",
    "SHY": "1-3 Year Treasury Bonds",
    "GLD": "Gold",
    "QQQ": "Nasdaq 100",
}


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


# ── Helpers ────────────────────────────────────────────────────────

def momentum(closes: list[float], lookback: int) -> float:
    """Simple return over lookback period."""
    if len(closes) < lookback + 1:
        return 0.0
    return (closes[-1] - closes[-lookback - 1]) / closes[-lookback - 1]


def sma(closes: list[float], period: int) -> float:
    """Simple moving average of last N values."""
    if len(closes) < period:
        return closes[-1] if closes else 0
    return sum(closes[-period:]) / period


# ── Main Backtest ─────────────────────────────────────────────────

def run_sector_rotation(
    tickers: list[str],
    safe_haven: str = "TLT",
    crash_haven: str = "",          # separate asset for crash exits (e.g. GLD); empty = use safe_haven
    benchmark: str = "SPY",
    rebalance_days: int = 21,       # ~1 month
    lookback_days: int = 63,        # ~3 months momentum
    min_advantage: float = 0.02,    # 2% edge to rotate
    commission: float = 0.001,      # 0.1% per trade (cheap broker)
    starting_cash: float = 10000.0,
    use_safe_haven: bool = True,    # rotate to bonds when everything falls
    sma_filter: int = 200,          # only hold sector if above 200-day SMA
    leverage: float = 1.0,          # 1.0 = no margin, 2.0 = 2:1 margin
    margin_rate: float = 0.0,       # annual margin interest rate (e.g. 0.05 = 5%)
    crash_exit_pct: float = 0.0,    # emergency exit if SPY drops this much from 20-day high (e.g. 0.10 = 10%)
    min_reentry_mom: float = 0.0,    # minimum absolute momentum for sector eligibility (e.g. 0.03 = 3%)
    safe_haven_lock: int = 0,        # SMA period for re-entry after crash (0=no lock, 50/100/200)
    cache_dir: str = "state/etf_cache",
    start: str = "1999-01-01",
    end: str = "2025-01-01",
):
    # Resolve crash haven
    dynamic_crash = False
    if not crash_haven:
        crash_haven = safe_haven
    if crash_haven.upper() == "BEST":
        # Dynamic: pick best of safe_haven and GLD at crash time
        dynamic_crash = True
        crash_haven = safe_haven  # default, will be chosen dynamically
    havens = list(set([safe_haven, crash_haven, "GLD"]) if dynamic_crash else set([safe_haven, crash_haven]))

    # Load all data
    all_tickers = list(set(tickers + havens + [benchmark]))
    print(f"Loading data for {len(all_tickers)} tickers...")
    
    price_data = {}
    date_data = {}
    for t in all_tickers:
        dates, closes = load_etf_cached(t, cache_dir=cache_dir, start=start, end=end)
        price_data[t] = dict(zip(dates, closes))
        date_data[t] = set(dates)

    # Find common dates across sector tickers + safe haven
    needed = set(tickers)
    if use_safe_haven:
        for h in havens:
            needed.add(h)
    needed.add(benchmark)
    
    common_dates = sorted(set.intersection(*(date_data[t] for t in needed)))
    n_bars = len(common_dates)
    years = n_bars / 252

    # Build aligned price arrays
    prices = {}
    for t in needed:
        prices[t] = [price_data[t][d] for d in common_dates]

    print(f"\nAligned {n_bars} trading days across {len(needed)} tickers")
    print(f"Period: {common_dates[0]} to {common_dates[-1]} ({years:.1f} years)")
    print(f"Sectors: {tickers}")
    print(f"Safe haven: {safe_haven} | Crash haven: {crash_haven} | Benchmark: {benchmark}")
    print(f"Rebalance every {rebalance_days} days, momentum lookback {lookback_days} days")
    print(f"Min advantage: {min_advantage*100:.0f}% | SMA filter: {sma_filter}-day")
    print(f"Commission: {commission*100:.2f}% per trade")
    if leverage > 1:
        print(f"Leverage: {leverage:.1f}x | Margin rate: {margin_rate*100:.1f}% annual")
        print(f"  (Borrowing ${starting_cash * (leverage - 1):,.0f} at {margin_rate*100:.1f}%)")
    if crash_exit_pct > 0:
        print(f"Emergency exit: SPY drops >{crash_exit_pct*100:.0f}% from 20-day high -> {crash_haven}")
    if min_reentry_mom > 0:
        print(f"Momentum floor: sectors must have >{min_reentry_mom*100:.0f}% absolute momentum to be eligible")
    if safe_haven_lock > 0:
        print(f"Safe haven lock: once in haven, stay until SPY > {safe_haven_lock}-day SMA")
    print()

    # ── Run backtest ───────────────────────────────────────────────
    equity = starting_cash
    current_holding = tickers[0]  # start in first sector
    entry_price = prices[current_holding][0]
    equity *= (1 - commission)  # entry commission
    n_trades = 1
    trade_log = []
    equity_curve = [equity]
    bars_since_rebalance = 0
    last_entry_bar = 0
    last_entry_price = entry_price

    trade_log.append({
        'bar': 0, 'date': common_dates[0],
        'action': 'BUY', 'from': '-', 'to': current_holding,
        'equity': equity, 'hold_pnl': None, 'hold_days': None,
        'mom_scores': {},
    })

    emergency_exits = 0
    locked_in_haven = False

    for i in range(1, n_bars):
        bars_since_rebalance += 1

        # Mark to market (with leverage)
        current_price = prices[current_holding][i]
        prev_price = prices[current_holding][i - 1]
        daily_ret = (current_price - prev_price) / prev_price
        equity *= (1 + daily_ret * leverage)
        # Deduct daily margin interest on borrowed amount
        if leverage > 1:
            daily_interest = margin_rate / 252
            equity -= equity * (leverage - 1) / leverage * daily_interest
        equity_curve.append(equity)

        # ── Emergency exit check (DAILY, not just at rebalance) ────
        if crash_exit_pct > 0 and current_holding not in havens and i >= 20:
            spy_20d_high = max(prices[benchmark][i-20:i+1])
            spy_now = prices[benchmark][i]
            spy_drawdown = (spy_20d_high - spy_now) / spy_20d_high
            if spy_drawdown >= crash_exit_pct:
                # EMERGENCY: market crashing, flee to safety
                # Dynamic: pick best of available havens based on 20-day return
                if dynamic_crash and i >= 20:
                    best_haven = max(havens, key=lambda h: (prices[h][i] - prices[h][i-20]) / prices[h][i-20])
                else:
                    best_haven = crash_haven
                sell_price = prices[current_holding][i]
                hold_bars = i - last_entry_bar
                hold_pnl = (sell_price - last_entry_price) / last_entry_price
                equity *= (1 - commission * 2)
                trade_log.append({
                    'bar': i, 'date': common_dates[i],
                    'action': 'ROTATE', 'from': current_holding, 'to': best_haven,
                    'equity': equity, 'hold_pnl': hold_pnl, 'hold_days': hold_bars,
                    'mom_scores': {'EMERGENCY': spy_drawdown},
                })
                n_trades += 1
                emergency_exits += 1
                last_entry_bar = i
                last_entry_price = prices[best_haven][i]
                current_holding = best_haven
                locked_in_haven = safe_haven_lock
                bars_since_rebalance = 0
                continue

        # Rebalance check
        if bars_since_rebalance >= rebalance_days and i >= max(lookback_days, sma_filter):
            bars_since_rebalance = 0

            # Compute momentum for all sectors
            mom = {}
            for t in tickers:
                closes_so_far = prices[t][:i + 1]
                mom[t] = momentum(closes_so_far, lookback_days)

            # SMA filter + absolute momentum floor: only consider sectors
            # above their SMA AND with strong enough absolute momentum
            eligible = []
            for t in tickers:
                closes_so_far = prices[t][:i + 1]
                sma_val = sma(closes_so_far, sma_filter)
                if prices[t][i] > sma_val:
                    if min_reentry_mom > 0 and mom[t] < min_reentry_mom:
                        continue  # momentum too weak, skip
                    eligible.append(t)

            # If nothing is above SMA, use safe haven
            if not eligible and use_safe_haven:
                best = safe_haven
                # Compute safe haven momentum for logging
                mom[safe_haven] = momentum(prices[safe_haven][:i + 1], lookback_days)
            elif not eligible:
                continue  # stay put
            else:
                best = max(eligible, key=lambda t: mom[t])

            # Safe haven lock: don't leave haven until conditions met
            if locked_in_haven and current_holding in havens:
                spy_sma_val = sma(prices[benchmark][:i + 1], locked_in_haven)
                sma_met = prices[benchmark][i] > spy_sma_val
                if sma_met:
                    locked_in_haven = 0  # unlock, allow normal rotation
                else:
                    continue  # stay in haven, skip rebalance

            # Should we rotate?
            current_mom = mom.get(current_holding, 0)
            best_mom = mom.get(best, 0)
            should_rotate = False

            if best != current_holding:
                if current_holding not in eligible and best in eligible:
                    # Current dropped below SMA, best is above — always rotate
                    should_rotate = True
                elif current_holding not in eligible and best == safe_haven:
                    # Everything below SMA — go to safe haven
                    should_rotate = True
                elif current_holding in havens and eligible:
                    # We're in a haven but sectors recovered — rotate out
                    should_rotate = True
                elif (best_mom - current_mom) >= min_advantage:
                    should_rotate = True

            if should_rotate:
                sell_price = prices[current_holding][i]
                hold_bars = i - last_entry_bar
                hold_pnl = (sell_price - last_entry_price) / last_entry_price

                # Pay commission both ways
                equity *= (1 - commission * 2)

                trade_log.append({
                    'bar': i, 'date': common_dates[i],
                    'action': 'ROTATE', 'from': current_holding, 'to': best,
                    'equity': equity,
                    'hold_pnl': hold_pnl,
                    'hold_days': hold_bars,
                    'mom_scores': dict(mom),
                })

                n_trades += 1
                last_entry_bar = i
                last_entry_price = prices[best][i]
                current_holding = best

    # ── Results ────────────────────────────────────────────────────
    final_equity = equity
    total_return = (final_equity / starting_cash - 1) * 100
    annual_return = ((final_equity / starting_cash) ** (1 / years) - 1) * 100

    # Benchmark (SPY B&H)
    spy_ret = (prices[benchmark][-1] / prices[benchmark][0] - 1) * 100
    spy_annual = ((prices[benchmark][-1] / prices[benchmark][0]) ** (1 / years) - 1) * 100
    spy_equity = starting_cash * (1 + spy_ret / 100)

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    # SPY max drawdown
    spy_peak = prices[benchmark][0]
    spy_max_dd = 0
    for p in prices[benchmark]:
        if p > spy_peak:
            spy_peak = p
        dd = (spy_peak - p) / spy_peak
        if dd > spy_max_dd:
            spy_max_dd = dd

    # Win rate
    rotations = [t for t in trade_log if t['action'] == 'ROTATE']
    wins = sum(1 for t in rotations if t['hold_pnl'] and t['hold_pnl'] > 0)
    win_rate = wins / len(rotations) * 100 if rotations else 0

    print("=" * 110)
    print("  RESULTS")
    print("=" * 110)
    print(f"  {'Strategy':<30} {'End Value':>12} {'Total':>10} {'Annual':>8} {'MaxDD':>8} {'Trades':>8}")
    print(f"  {'-'*76}")
    print(f"  {'SECTOR ROTATION':<30} ${final_equity:>11.2f} {total_return:>+9.1f}% {annual_return:>+7.1f}% {-max_dd*100:>7.1f}% {n_trades:>8}")
    print(f"  {'SPY B&H':<30} ${spy_equity:>11.2f} {spy_ret:>+9.1f}% {spy_annual:>+7.1f}% {-spy_max_dd*100:>7.1f}% {'1':>8}")

    # Individual sector B&H
    for t in tickers:
        t_ret = (prices[t][-1] / prices[t][0] - 1) * 100
        t_annual = ((prices[t][-1] / prices[t][0]) ** (1 / years) - 1) * 100
        t_equity = starting_cash * (1 + t_ret / 100)
        label = f"B&H {t} ({SECTOR_ETFS.get(t, '')})"
        print(f"  {label:<30} ${t_equity:>11.2f} {t_ret:>+9.1f}% {t_annual:>+7.1f}%")

    if use_safe_haven:
        sh_ret = (prices[safe_haven][-1] / prices[safe_haven][0] - 1) * 100
        sh_annual = ((prices[safe_haven][-1] / prices[safe_haven][0]) ** (1 / years) - 1) * 100
        print(f"  {'B&H ' + safe_haven:<30} ${starting_cash*(1+sh_ret/100):>11.2f} {sh_ret:>+9.1f}% {sh_annual:>+7.1f}%")

    print()
    print(f"  Win rate: {wins}/{len(rotations)} ({win_rate:.0f}%)")
    avg_hold = sum(t['hold_days'] for t in rotations if t['hold_days']) / max(len(rotations), 1)
    print(f"  Avg hold: {avg_hold:.0f} days")
    if emergency_exits > 0:
        print(f"  Emergency exits (SPY crash): {emergency_exits}")

    # ── Trade Log ──────────────────────────────────────────────────
    print()
    print("=" * 110)
    print("  DETAILED TRADE LOG")
    print("=" * 110)
    print(f"  {'#':<4} {'Date':<12} {'Action':<8} {'From':<6} {'To':<6} "
          f"{'Hold':>6} {'Hold P&L':>10} {'Equity':>12} {'Top momentum scores'}")
    print(f"  {'-'*100}")

    for idx, t in enumerate(trade_log):
        if t['action'] == 'BUY':
            print(f"  {idx:<4} {t['date']:<12} {'ENTRY':<8} {'-':<6} {t['to']:<6} "
                  f"{'':>6} {'':>10} ${t['equity']:>11.2f}")
        else:
            hold_str = f"{t['hold_days']}d"
            pnl_str = f"{t['hold_pnl']:+.1%}"
            sorted_mom = sorted(t['mom_scores'].items(), key=lambda x: x[1], reverse=True)
            mom_str = "  ".join(f"{s}:{m:+.1%}" for s, m in sorted_mom[:5])
            print(f"  {idx:<4} {t['date']:<12} {'ROTATE':<8} {t['from']:<6} {t['to']:<6} "
                  f"{hold_str:>6} {pnl_str:>10} ${t['equity']:>11.2f} {mom_str}")

    # Final hold
    final_hold_days = n_bars - 1 - last_entry_bar
    final_hold_pnl = (prices[current_holding][-1] - last_entry_price) / last_entry_price
    print(f"  {'>':<4} {common_dates[-1]:<12} {'HOLD':<8} {'':<6} {current_holding:<6} "
          f"{final_hold_days}d{'':>4} {final_hold_pnl:>+10.1%} ${final_equity:>11.2f} (still holding)")

    # ── Yearly Breakdown ───────────────────────────────────────────
    print()
    print("=" * 110)
    print("  YEARLY BREAKDOWN")
    print("=" * 110)
    print(f"  {'Year':<6} {'Rotation':>10} {'SPY B&H':>10} {'Diff':>8} {'Trades':>8} {'Held'}")
    print(f"  {'-'*60}")

    yr_start_eq = equity_curve[0]
    yr_start_spy = prices[benchmark][0]
    current_year = common_dates[0][:4]
    yr_holding = current_holding

    for i in range(1, n_bars):
        yr = common_dates[i][:4]
        if yr != current_year or i == n_bars - 1:
            idx = i - 1 if yr != current_year else i
            yr_ret = (equity_curve[idx] / yr_start_eq - 1) * 100
            spy_yr_ret = (prices[benchmark][idx] / yr_start_spy - 1) * 100
            diff = yr_ret - spy_yr_ret
            yr_trades = sum(1 for t in rotations if t['date'][:4] == current_year)

            # Find what we held most of the year
            print(f"  {current_year:<6} {yr_ret:>+9.1f}% {spy_yr_ret:>+9.1f}% {diff:>+7.1f}% {yr_trades:>8}")

            yr_start_eq = equity_curve[idx]
            yr_start_spy = prices[benchmark][idx]
            current_year = yr

    print()
    return final_equity


# ── CLI ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Equity Sector Rotation Backtest")
    p.add_argument("--sectors", default="XLK,XLE,XLF,XLV,XLY,XLI,XLP,XLU,XLB",
                   help="Comma-separated sector ETF tickers")
    p.add_argument("--safe-haven", default="TLT")
    p.add_argument("--crash-haven", default="", help="Asset for crash exits (e.g. GLD). Defaults to safe-haven.")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--rebalance-days", type=int, default=21)
    p.add_argument("--lookback-days", type=int, default=63)
    p.add_argument("--min-advantage", type=float, default=0.02)
    p.add_argument("--commission", type=float, default=0.001)
    p.add_argument("--starting-cash", type=float, default=10000.0)
    p.add_argument("--no-safe-haven", action="store_true")
    p.add_argument("--sma-filter", type=int, default=200)
    p.add_argument("--leverage", type=float, default=1.0, help="Margin leverage (e.g. 2.0 for 2:1)")
    p.add_argument("--margin-rate", type=float, default=0.0, help="Annual margin interest rate (e.g. 0.05 for 5%%)")
    p.add_argument("--crash-exit", type=float, default=0.0, help="Emergency exit if SPY drops this pct from 20d high (e.g. 0.10)")
    p.add_argument("--min-reentry-mom", type=float, default=0.0, help="Momentum floor: min absolute momentum for sector eligibility (e.g. 0.03)")
    p.add_argument("--safe-haven-lock", type=int, default=0, help="SMA period for re-entry after crash exit (e.g. 50, 100, 200). 0=no lock.")
    p.add_argument("--cache-dir", default="state/etf_cache")
    p.add_argument("--start", default="1999-01-01")
    p.add_argument("--end", default="2025-01-01")
    args = p.parse_args()

    sectors = [s.strip().upper() for s in args.sectors.split(",")]

    run_sector_rotation(
        tickers=sectors,
        safe_haven=args.safe_haven,
        crash_haven=args.crash_haven,
        benchmark=args.benchmark,
        rebalance_days=args.rebalance_days,
        lookback_days=args.lookback_days,
        min_advantage=args.min_advantage,
        commission=args.commission,
        starting_cash=args.starting_cash,
        use_safe_haven=not args.no_safe_haven,
        sma_filter=args.sma_filter,
        leverage=args.leverage,
        margin_rate=args.margin_rate,
        crash_exit_pct=args.crash_exit,
        min_reentry_mom=args.min_reentry_mom,
        safe_haven_lock=args.safe_haven_lock,
        cache_dir=args.cache_dir,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
