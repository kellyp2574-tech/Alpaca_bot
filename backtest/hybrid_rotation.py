"""
Hybrid 1x/3x Sector Rotation Backtest

Normally holds 1x sector ETFs (XLK, XLE, XLF, etc.) using momentum rotation.
Upgrades to 3x leveraged versions (TECL, ERX, FAS) only when ALL conditions align:

UPGRADE to 3x when:
  1. Sector price > 200-day SMA (long-term uptrend)
  2. 50-day SMA > 200-day SMA (golden cross)
  3. 63-day momentum > threshold (e.g. 15%)
  4. SPY > its 200-day SMA (broad market healthy)

DOWNGRADE to 1x when ANY of:
  - Sector price < 50-day SMA (short-term break)
  - SPY < its 200-day SMA (market turning)
  - Sector momentum < 0 (trend over)

This aims to capture 3x gains during strong trends while maintaining
1x-level drawdowns during uncertain/choppy periods.
"""
from __future__ import annotations

import argparse
import os
import json

import yfinance as yf


# ── Sector Mapping: 1x → 3x ──────────────────────────────────────

SECTOR_1X_TO_3X = {
    "XLK": "TECL",   # Technology
    "XLE": "ERX",    # Energy
    "XLF": "FAS",    # Financials
    "XLV": "CURE",   # Healthcare
    # These 3x ETFs started too late (2017+), so no 3x available for long backtest:
    # "XLY": "WANT",  # Consumer Discretionary
    # "XLI": "DUSL",  # Industrials
    # "XLU": "UTSL",  # Utilities
}

SECTOR_NAMES = {
    "XLK": "Technology", "XLE": "Energy", "XLF": "Financials",
    "XLV": "Healthcare", "XLY": "Consumer Disc", "XLI": "Industrials",
    "XLP": "Consumer Staples", "XLU": "Utilities", "XLB": "Materials",
}


# ── Data Loading ──────────────────────────────────────────────────

def load_etf_cached(ticker: str, cache_dir: str = "state/etf_cache",
                    start: str = "1999-01-01", end: str = "2025-01-01"
                    ) -> tuple[list[str], list[float]]:
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

def momentum(closes: list[float], i: int, lookback: int) -> float:
    if i < lookback:
        return 0.0
    return (closes[i] - closes[i - lookback]) / closes[i - lookback]


def sma(closes: list[float], i: int, period: int) -> float:
    if i < period - 1:
        return closes[i]
    return sum(closes[i - period + 1:i + 1]) / period


def should_upgrade_3x(
    sector_closes: list[float],
    spy_closes: list[float],
    i: int,
    mom_threshold: float = 0.15,
    lookback: int = 63,
) -> bool:
    """Check if all 3x upgrade conditions are met."""
    if i < 200:
        return False

    price = sector_closes[i]
    sma50 = sma(sector_closes, i, 50)
    sma200 = sma(sector_closes, i, 200)
    mom_val = momentum(sector_closes, i, lookback)
    spy_sma200 = sma(spy_closes, i, 200)

    return (
        price > sma200           # above 200-day SMA
        and sma50 > sma200       # golden cross
        and mom_val >= mom_threshold  # strong momentum
        and spy_closes[i] > spy_sma200  # broad market healthy
    )


def should_downgrade_1x(
    sector_closes: list[float],
    spy_closes: list[float],
    i: int,
    lookback: int = 63,
) -> bool:
    """Check if any downgrade condition triggers."""
    if i < 200:
        return True  # not enough data, stay 1x

    price = sector_closes[i]
    sma50 = sma(sector_closes, i, 50)
    spy_sma200 = sma(spy_closes, i, 200)
    mom_val = momentum(sector_closes, i, lookback)

    return (
        price < sma50               # short-term trend broken
        or spy_closes[i] < spy_sma200  # broad market turning
        or mom_val < 0               # momentum gone
    )


# ── Main Backtest ─────────────────────────────────────────────────

def run_hybrid_backtest(
    sectors_1x: list[str],
    safe_haven: str = "TLT",
    benchmark: str = "SPY",
    rebalance_days: int = 21,
    lookback_days: int = 63,
    min_advantage: float = 0.02,
    commission: float = 0.001,
    starting_cash: float = 10000.0,
    sma_filter: int = 200,
    mom_threshold: float = 0.15,
    cache_dir: str = "state/etf_cache",
    start: str = "1999-01-01",
    end: str = "2025-01-01",
):
    # Determine which sectors have 3x equivalents
    sectors_3x = {s: SECTOR_1X_TO_3X[s] for s in sectors_1x if s in SECTOR_1X_TO_3X}

    # Load all needed tickers
    all_tickers = set(sectors_1x) | set(sectors_3x.values()) | {safe_haven, benchmark}
    print(f"Loading data for {len(all_tickers)} tickers...")
    print(f"1x sectors: {sectors_1x}")
    print(f"3x upgrades available: {sectors_3x}")
    print()

    price_data = {}
    date_data = {}
    for t in sorted(all_tickers):
        dates, closes = load_etf_cached(t, cache_dir=cache_dir, start=start, end=end)
        price_data[t] = dict(zip(dates, closes))
        date_data[t] = set(dates)
        print(f"  {t}: {len(closes)} bars ({dates[0]} to {dates[-1]})")

    # Find common dates across ALL tickers (1x, 3x, safe haven, benchmark)
    common_dates = sorted(set.intersection(*(date_data[t] for t in all_tickers)))
    n_bars = len(common_dates)
    years = n_bars / 252

    # Build aligned price arrays
    prices = {}
    for t in all_tickers:
        prices[t] = [price_data[t][d] for d in common_dates]

    print(f"\nAligned {n_bars} trading days ({years:.1f} years)")
    print(f"Period: {common_dates[0]} to {common_dates[-1]}")
    print(f"Rebalance every {rebalance_days} days, momentum lookback {lookback_days} days")
    print(f"3x upgrade threshold: momentum > {mom_threshold:.0%} + golden cross + SPY > 200 SMA")
    print(f"Commission: {commission*100:.2f}% per trade")
    print()

    # ── Run backtest ───────────────────────────────────────────────
    equity = starting_cash
    current_sector = sectors_1x[0]  # which sector we're in (1x ticker)
    current_ticker = current_sector  # actual ticker held (could be 1x or 3x)
    is_3x = False
    equity *= (1 - commission)
    n_trades = 1
    trade_log = []
    equity_curve = [equity]
    bars_since_rebalance = 0
    last_entry_bar = 0
    last_entry_price = prices[current_ticker][0]

    trade_log.append({
        'bar': 0, 'date': common_dates[0],
        'action': 'BUY', 'from': '-', 'to': current_ticker,
        'is_3x': False, 'equity': equity,
        'hold_pnl': None, 'hold_days': None,
        'reason': 'initial entry',
    })

    for i in range(1, n_bars):
        bars_since_rebalance += 1

        # Mark to market using current held ticker
        current_price = prices[current_ticker][i]
        prev_price = prices[current_ticker][i - 1]
        daily_ret = (current_price - prev_price) / prev_price
        equity *= (1 + daily_ret)
        equity_curve.append(equity)

        # ── Leverage check (every bar, not just rebalance) ─────────
        # Check if we need to downgrade from 3x to 1x (fast exit)
        if is_3x and i >= 200:
            if should_downgrade_1x(prices[current_sector], prices[benchmark], i, lookback_days):
                # Downgrade: sell 3x, buy 1x of same sector
                old_ticker = current_ticker
                current_ticker = current_sector  # back to 1x
                is_3x = False

                hold_pnl = (prices[old_ticker][i] - last_entry_price) / last_entry_price
                equity *= (1 - commission * 2)

                trade_log.append({
                    'bar': i, 'date': common_dates[i],
                    'action': 'DOWNGRADE', 'from': old_ticker, 'to': current_ticker,
                    'is_3x': False, 'equity': equity,
                    'hold_pnl': hold_pnl, 'hold_days': i - last_entry_bar,
                    'reason': 'downgrade trigger',
                })
                n_trades += 1
                last_entry_bar = i
                last_entry_price = prices[current_ticker][i]

        # ── Rebalance check ────────────────────────────────────────
        if bars_since_rebalance >= rebalance_days and i >= max(lookback_days, sma_filter):
            bars_since_rebalance = 0

            # Compute momentum for all 1x sectors
            mom = {}
            for s in sectors_1x:
                mom[s] = momentum(prices[s], i, lookback_days)

            # SMA filter: only sectors above their 200-day SMA
            eligible = []
            for s in sectors_1x:
                sma_val = sma(prices[s], i, sma_filter)
                if prices[s][i] > sma_val:
                    eligible.append(s)

            # Determine best sector
            if not eligible:
                best_sector = None
                best_ticker = safe_haven
            else:
                best_sector = max(eligible, key=lambda s: mom[s])
                best_ticker = best_sector

            # Should we rotate to a different sector?
            should_rotate = False
            if best_sector is None and current_sector is not None:
                should_rotate = True  # go to safe haven
            elif best_sector is not None and current_sector is None:
                should_rotate = True  # leave safe haven
            elif best_sector is not None and best_sector != current_sector:
                current_mom = mom.get(current_sector, 0)
                best_mom = mom.get(best_sector, 0)
                if current_sector not in eligible:
                    should_rotate = True
                elif current_sector == safe_haven:
                    should_rotate = True
                elif (best_mom - current_mom) >= min_advantage:
                    should_rotate = True

            if should_rotate:
                old_ticker = current_ticker
                hold_pnl = (prices[old_ticker][i] - last_entry_price) / last_entry_price

                # New sector — check if we should go 3x immediately
                new_is_3x = False
                if best_sector and best_sector in sectors_3x:
                    if should_upgrade_3x(prices[best_sector], prices[benchmark], i,
                                        mom_threshold, lookback_days):
                        new_is_3x = True

                if new_is_3x:
                    new_ticker = sectors_3x[best_sector]
                else:
                    new_ticker = best_ticker

                equity *= (1 - commission * 2)

                reason_parts = []
                sorted_mom = sorted(mom.items(), key=lambda x: x[1], reverse=True)
                reason_parts.append(" ".join(f"{s}:{m:+.1%}" for s, m in sorted_mom[:4]))

                trade_log.append({
                    'bar': i, 'date': common_dates[i],
                    'action': 'ROTATE', 'from': old_ticker, 'to': new_ticker,
                    'is_3x': new_is_3x, 'equity': equity,
                    'hold_pnl': hold_pnl, 'hold_days': i - last_entry_bar,
                    'reason': " ".join(reason_parts),
                })
                n_trades += 1
                last_entry_bar = i
                last_entry_price = prices[new_ticker][i]
                current_sector = best_sector
                current_ticker = new_ticker
                is_3x = new_is_3x

            elif not is_3x and current_sector and current_sector in sectors_3x:
                # Same sector — check if we should upgrade to 3x
                if should_upgrade_3x(prices[current_sector], prices[benchmark], i,
                                    mom_threshold, lookback_days):
                    old_ticker = current_ticker
                    new_ticker = sectors_3x[current_sector]
                    hold_pnl = (prices[old_ticker][i] - last_entry_price) / last_entry_price

                    equity *= (1 - commission * 2)
                    trade_log.append({
                        'bar': i, 'date': common_dates[i],
                        'action': 'UPGRADE', 'from': old_ticker, 'to': new_ticker,
                        'is_3x': True, 'equity': equity,
                        'hold_pnl': hold_pnl, 'hold_days': i - last_entry_bar,
                        'reason': f'3x upgrade: mom={mom[current_sector]:+.1%} golden_cross SPY_healthy',
                    })
                    n_trades += 1
                    last_entry_bar = i
                    last_entry_price = prices[new_ticker][i]
                    current_ticker = new_ticker
                    is_3x = True

    # ── Results ────────────────────────────────────────────────────
    final_equity = equity
    total_return = (final_equity / starting_cash - 1) * 100
    annual_return = ((final_equity / starting_cash) ** (1 / years) - 1) * 100

    # Benchmarks
    spy_ret = (prices[benchmark][-1] / prices[benchmark][0] - 1) * 100
    spy_annual = ((prices[benchmark][-1] / prices[benchmark][0]) ** (1 / years) - 1) * 100
    spy_equity = starting_cash * (1 + spy_ret / 100)

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0
    max_dd_date = common_dates[0]
    for idx, eq in enumerate(equity_curve):
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd
            max_dd_date = common_dates[idx]

    # SPY max drawdown
    spy_peak = prices[benchmark][0]
    spy_max_dd = 0
    for p in prices[benchmark]:
        if p > spy_peak:
            spy_peak = p
        dd = (spy_peak - p) / spy_peak
        if dd > spy_max_dd:
            spy_max_dd = dd

    # Stats
    rotations = [t for t in trade_log if t['action'] in ('ROTATE', 'UPGRADE', 'DOWNGRADE')]
    upgrades = [t for t in trade_log if t['action'] == 'UPGRADE']
    downgrades = [t for t in trade_log if t['action'] == 'DOWNGRADE']
    wins = sum(1 for t in rotations if t['hold_pnl'] and t['hold_pnl'] > 0)
    win_rate = wins / len(rotations) * 100 if rotations else 0

    # Time in 3x
    bars_in_3x = 0
    in_3x = False
    for t in trade_log:
        if t['action'] in ('UPGRADE', 'ROTATE') and t['is_3x']:
            in_3x = True
        elif t['action'] in ('DOWNGRADE', 'ROTATE') and not t['is_3x']:
            if in_3x:
                bars_in_3x += t.get('hold_days', 0) or 0
            in_3x = False
    pct_3x = bars_in_3x / n_bars * 100

    print("=" * 110)
    print("  RESULTS")
    print("=" * 110)
    print(f"  {'Strategy':<30} {'End Value':>12} {'Total':>10} {'Annual':>8} {'MaxDD':>8} {'Trades':>8}")
    print(f"  {'-'*76}")
    print(f"  {'HYBRID 1x/3x ROTATION':<30} ${final_equity:>11.2f} {total_return:>+9.1f}% {annual_return:>+7.1f}% {-max_dd*100:>7.1f}% {n_trades:>8}")
    print(f"  {'SPY B&H':<30} ${spy_equity:>11.2f} {spy_ret:>+9.1f}% {spy_annual:>+7.1f}% {-spy_max_dd*100:>7.1f}% {'1':>8}")
    print()
    print(f"  Win rate: {wins}/{len(rotations)} ({win_rate:.0f}%)")
    print(f"  Upgrades to 3x: {len(upgrades)} | Downgrades to 1x: {len(downgrades)}")
    print(f"  Time in 3x: ~{pct_3x:.0f}% of trading days")
    print(f"  Max drawdown occurred: {max_dd_date}")

    # ── Trade Log ──────────────────────────────────────────────────
    print()
    print("=" * 110)
    print("  DETAILED TRADE LOG")
    print("=" * 110)
    print(f"  {'#':<4} {'Date':<12} {'Action':<10} {'From':<6} {'To':<6} {'3x?':<4} "
          f"{'Hold':>6} {'P&L':>10} {'Equity':>12} {'Reason'}")
    print(f"  {'-'*105}")

    for idx, t in enumerate(trade_log):
        lev = "3x" if t['is_3x'] else "1x"
        if t['action'] == 'BUY':
            print(f"  {idx:<4} {t['date']:<12} {'ENTRY':<10} {'-':<6} {t['to']:<6} {lev:<4} "
                  f"{'':>6} {'':>10} ${t['equity']:>11.2f}")
        else:
            hold_str = f"{t['hold_days']}d" if t['hold_days'] else ""
            pnl_str = f"{t['hold_pnl']:+.1%}" if t['hold_pnl'] is not None else ""
            reason = t.get('reason', '')[:50]
            print(f"  {idx:<4} {t['date']:<12} {t['action']:<10} {t['from']:<6} {t['to']:<6} {lev:<4} "
                  f"{hold_str:>6} {pnl_str:>10} ${t['equity']:>11.2f} {reason}")

    # Final hold
    final_hold_days = n_bars - 1 - last_entry_bar
    final_hold_pnl = (prices[current_ticker][-1] - last_entry_price) / last_entry_price
    lev = "3x" if is_3x else "1x"
    print(f"  {'>':<4} {common_dates[-1]:<12} {'HOLD':<10} {'':<6} {current_ticker:<6} {lev:<4} "
          f"{final_hold_days}d{'':>4} {final_hold_pnl:>+10.1%} ${final_equity:>11.2f} (still holding)")

    # ── Yearly Breakdown ───────────────────────────────────────────
    print()
    print("=" * 110)
    print("  YEARLY BREAKDOWN")
    print("=" * 110)
    print(f"  {'Year':<6} {'Hybrid':>10} {'SPY B&H':>10} {'Diff':>8} {'Trades':>8} {'3x trades':>10}")
    print(f"  {'-'*52}")

    yr_start_eq = equity_curve[0]
    yr_start_spy = prices[benchmark][0]
    current_year = common_dates[0][:4]

    for i in range(1, n_bars):
        yr = common_dates[i][:4]
        if yr != current_year or i == n_bars - 1:
            idx = i - 1 if yr != current_year else i
            yr_ret = (equity_curve[idx] / yr_start_eq - 1) * 100
            spy_yr_ret = (prices[benchmark][idx] / yr_start_spy - 1) * 100
            diff = yr_ret - spy_yr_ret
            yr_trades = sum(1 for t in rotations if t['date'][:4] == current_year)
            yr_3x = sum(1 for t in trade_log if t['date'][:4] == current_year and t.get('is_3x'))
            print(f"  {current_year:<6} {yr_ret:>+9.1f}% {spy_yr_ret:>+9.1f}% {diff:>+7.1f}% {yr_trades:>8} {yr_3x:>10}")
            yr_start_eq = equity_curve[idx]
            yr_start_spy = prices[benchmark][idx]
            current_year = yr

    print()
    return final_equity


# ── CLI ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Hybrid 1x/3x Sector Rotation Backtest")
    p.add_argument("--sectors", default="XLK,XLE,XLF,XLV,XLY,XLI,XLP,XLU,XLB",
                   help="1x sector ETFs (3x auto-mapped where available)")
    p.add_argument("--safe-haven", default="TLT")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--rebalance-days", type=int, default=21)
    p.add_argument("--lookback-days", type=int, default=63)
    p.add_argument("--min-advantage", type=float, default=0.02)
    p.add_argument("--commission", type=float, default=0.001)
    p.add_argument("--starting-cash", type=float, default=10000.0)
    p.add_argument("--sma-filter", type=int, default=200)
    p.add_argument("--mom-threshold", type=float, default=0.15,
                   help="Momentum threshold to upgrade to 3x (e.g. 0.15 = 15%%)")
    p.add_argument("--cache-dir", default="state/etf_cache")
    p.add_argument("--start", default="1999-01-01")
    p.add_argument("--end", default="2025-01-01")
    args = p.parse_args()

    sectors = [s.strip().upper() for s in args.sectors.split(",")]

    run_hybrid_backtest(
        sectors_1x=sectors,
        safe_haven=args.safe_haven,
        benchmark=args.benchmark,
        rebalance_days=args.rebalance_days,
        lookback_days=args.lookback_days,
        min_advantage=args.min_advantage,
        commission=args.commission,
        starting_cash=args.starting_cash,
        sma_filter=args.sma_filter,
        mom_threshold=args.mom_threshold,
        cache_dir=args.cache_dir,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
