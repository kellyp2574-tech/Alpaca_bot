"""
Forex Composite Strategy Backtest

Combines three proven forex edges into a single scoring system:
  1. CARRY   - buy high-yield currencies, sell low-yield (earn interest differential)
  2. TREND   - go with the direction of the moving average (ride macro trends)
  3. REVERT  - fade extreme deviations from the mean (buy oversold, sell overbought)

Each pair gets a composite score. The bot goes LONG the highest-scored pair
and can optionally SHORT the lowest-scored pair.

Key difference from crypto: we can go BOTH long and short.
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone

from Alpaca_bot.backtest.forex_data import (
    PAIR_MAP, ForexPair, PairData, load_pair_cached
)


# ── Historical Interest Rates (approximate annual averages) ──────
# These are central bank benchmark rates by year.
# Used to compute carry: long high-rate currency, short low-rate currency.
# Source: FRED, central bank websites. Approximated for simplicity.

RATES = {
    # year: {currency: approximate avg annual rate %}
    2003: {"USD": 1.00, "EUR": 2.00, "GBP": 3.75, "JPY": 0.00, "AUD": 4.75, "NZD": 5.00, "CHF": 0.25, "CAD": 2.75},
    2004: {"USD": 1.50, "EUR": 2.00, "GBP": 4.50, "JPY": 0.00, "AUD": 5.25, "NZD": 5.75, "CHF": 0.50, "CAD": 2.25},
    2005: {"USD": 3.25, "EUR": 2.25, "GBP": 4.50, "JPY": 0.00, "AUD": 5.50, "NZD": 6.75, "CHF": 0.75, "CAD": 2.75},
    2006: {"USD": 5.00, "EUR": 3.00, "GBP": 4.75, "JPY": 0.25, "AUD": 6.00, "NZD": 7.25, "CHF": 1.50, "CAD": 4.25},
    2007: {"USD": 5.00, "EUR": 4.00, "GBP": 5.50, "JPY": 0.50, "AUD": 6.50, "NZD": 8.00, "CHF": 2.50, "CAD": 4.25},
    2008: {"USD": 2.00, "EUR": 3.75, "GBP": 4.50, "JPY": 0.30, "AUD": 6.00, "NZD": 6.50, "CHF": 1.50, "CAD": 3.00},
    2009: {"USD": 0.25, "EUR": 1.25, "GBP": 0.50, "JPY": 0.10, "AUD": 3.75, "NZD": 2.50, "CHF": 0.25, "CAD": 0.25},
    2010: {"USD": 0.25, "EUR": 1.00, "GBP": 0.50, "JPY": 0.10, "AUD": 4.50, "NZD": 3.00, "CHF": 0.25, "CAD": 1.00},
    2011: {"USD": 0.25, "EUR": 1.25, "GBP": 0.50, "JPY": 0.10, "AUD": 4.50, "NZD": 2.50, "CHF": 0.00, "CAD": 1.00},
    2012: {"USD": 0.25, "EUR": 0.75, "GBP": 0.50, "JPY": 0.10, "AUD": 3.50, "NZD": 2.50, "CHF": 0.00, "CAD": 1.00},
    2013: {"USD": 0.25, "EUR": 0.50, "GBP": 0.50, "JPY": 0.10, "AUD": 2.50, "NZD": 2.50, "CHF": 0.00, "CAD": 1.00},
    2014: {"USD": 0.25, "EUR": 0.15, "GBP": 0.50, "JPY": 0.10, "AUD": 2.50, "NZD": 3.50, "CHF":-0.25, "CAD": 1.00},
    2015: {"USD": 0.25, "EUR": 0.05, "GBP": 0.50, "JPY": 0.10, "AUD": 2.00, "NZD": 2.75, "CHF":-0.75, "CAD": 0.50},
    2016: {"USD": 0.50, "EUR": 0.00, "GBP": 0.25, "JPY":-0.10, "AUD": 1.50, "NZD": 2.00, "CHF":-0.75, "CAD": 0.50},
    2017: {"USD": 1.25, "EUR": 0.00, "GBP": 0.25, "JPY":-0.10, "AUD": 1.50, "NZD": 1.75, "CHF":-0.75, "CAD": 1.00},
    2018: {"USD": 2.25, "EUR": 0.00, "GBP": 0.75, "JPY":-0.10, "AUD": 1.50, "NZD": 1.75, "CHF":-0.75, "CAD": 1.75},
    2019: {"USD": 1.75, "EUR": 0.00, "GBP": 0.75, "JPY":-0.10, "AUD": 0.75, "NZD": 1.00, "CHF":-0.75, "CAD": 1.75},
    2020: {"USD": 0.25, "EUR": 0.00, "GBP": 0.10, "JPY":-0.10, "AUD": 0.25, "NZD": 0.25, "CHF":-0.75, "CAD": 0.25},
    2021: {"USD": 0.25, "EUR": 0.00, "GBP": 0.10, "JPY":-0.10, "AUD": 0.10, "NZD": 0.25, "CHF":-0.75, "CAD": 0.25},
    2022: {"USD": 3.00, "EUR": 1.50, "GBP": 2.50, "JPY":-0.10, "AUD": 2.50, "NZD": 3.50, "CHF": 0.50, "CAD": 3.75},
    2023: {"USD": 5.25, "EUR": 4.00, "GBP": 5.00, "JPY":-0.10, "AUD": 4.10, "NZD": 5.50, "CHF": 1.50, "CAD": 5.00},
    2024: {"USD": 5.00, "EUR": 3.75, "GBP": 5.00, "JPY": 0.25, "AUD": 4.35, "NZD": 5.00, "CHF": 1.25, "CAD": 4.25},
}


# ── Signal Components ─────────────────────────────────────────────

def get_carry_score(symbol: str, year: int) -> float:
    """Carry score: interest rate differential for the pair.
    
    For XXXUSD pairs (e.g. EURUSD): carry = rate(XXX) - rate(USD)
      Positive = you earn interest going long this pair
    For USDXXX pairs (e.g. USDJPY): carry = rate(USD) - rate(XXX)
      We invert these so "long" always means "long the non-USD currency"
    """
    rates = RATES.get(year, RATES.get(max(RATES.keys())))
    
    # Parse currencies from symbol
    if symbol.startswith("USD"):
        base, quote = "USD", symbol[3:]
        # Inverted: "long" = short USD, long quote
        return (rates.get(quote, 0) - rates.get(base, 0)) / 100
    elif symbol.endswith("USD"):
        base = symbol[:3]
        return (rates.get(base, 0) - rates.get("USD", 0)) / 100
    else:
        # Cross pair (e.g. EURGBP)
        base, quote = symbol[:3], symbol[3:]
        return (rates.get(base, 0) - rates.get(quote, 0)) / 100


def get_trend_score(closes: list[float], i: int, fast_period: int = 50,
                    slow_period: int = 200) -> float:
    """Trend score based on moving average alignment.
    
    Returns value between -1 and +1:
      +1 = strong uptrend (price > fast MA > slow MA)
      -1 = strong downtrend (price < fast MA < slow MA)
       0 = mixed/no trend
    """
    if i < slow_period:
        return 0.0
    
    price = closes[i]
    fast_ma = sum(closes[i - fast_period + 1:i + 1]) / fast_period
    slow_ma = sum(closes[i - slow_period + 1:i + 1]) / slow_period
    
    # Score: how far price is from slow MA, normalized
    deviation = (price - slow_ma) / slow_ma
    
    # Clamp to [-1, 1] range (10% deviation = max score)
    score = max(-1.0, min(1.0, deviation / 0.10))
    
    # Bonus if MAs are aligned (fast > slow for uptrend)
    if fast_ma > slow_ma and price > fast_ma:
        score = max(score, 0.3)  # at least moderately bullish
    elif fast_ma < slow_ma and price < fast_ma:
        score = min(score, -0.3)  # at least moderately bearish
    
    return score


def get_reversion_score(closes: list[float], i: int,
                        lookback: int = 120) -> float:
    """Mean reversion score based on z-score from rolling mean.
    
    Returns value between -1 and +1:
      +1 = deeply oversold (expect bounce UP - go long)
      -1 = deeply overbought (expect pullback DOWN - go short)
       0 = near fair value
    
    Note: this is CONTRARIAN - opposite of momentum.
    """
    if i < lookback:
        return 0.0
    
    window = closes[i - lookback + 1:i + 1]
    mean = sum(window) / len(window)
    std = (sum((x - mean) ** 2 for x in window) / len(window)) ** 0.5
    
    if std < 1e-10:
        return 0.0
    
    z_score = (closes[i] - mean) / std
    
    # Invert: negative z-score (oversold) = positive signal (buy)
    # Clamp to [-1, 1], z-score of 2 = max signal
    return max(-1.0, min(1.0, -z_score / 2.0))


# ── Composite Score ───────────────────────────────────────────────

def composite_score(
    symbol: str,
    closes: list[float],
    i: int,
    year: int,
    w_carry: float = 0.33,
    w_trend: float = 0.34,
    w_revert: float = 0.33,
) -> tuple[float, dict]:
    """Weighted composite of carry + trend + reversion scores.
    
    Returns (score, breakdown_dict).
    Score ranges roughly -1 to +1.
    Positive = go long, Negative = go short.
    """
    carry = get_carry_score(symbol, year)
    trend = get_trend_score(closes, i)
    revert = get_reversion_score(closes, i)
    
    # Normalize carry to roughly [-1, 1] range
    # Max carry diff is ~8% (NZD vs JPY in 2007), so /0.08
    carry_norm = max(-1.0, min(1.0, carry / 0.08))
    
    score = w_carry * carry_norm + w_trend * trend + w_revert * revert
    
    return score, {
        "carry": carry_norm,
        "trend": trend,
        "revert": revert,
        "raw_carry_pct": carry * 100,
    }


# ── Main Backtest ─────────────────────────────────────────────────

def run_composite_backtest(
    pairs: list[PairData],
    rebalance_days: int = 20,
    spread_cost: float = 0.0003,
    leverage: float = 1.0,
    starting_cash: float = 10000.0,
    w_carry: float = 0.33,
    w_trend: float = 0.34,
    w_revert: float = 0.33,
    allow_short: bool = True,
    trend_fast: int = 50,
    trend_slow: int = 200,
    revert_lookback: int = 120,
):
    # Align dates
    all_date_sets = [set(p.dates) for p in pairs]
    common_dates = sorted(all_date_sets[0].intersection(*all_date_sets[1:]))

    if not common_dates:
        print("ERROR: no common dates across pairs")
        return

    # Build aligned price arrays (normalized: up = non-USD strengthening)
    from Alpaca_bot.backtest.rotation_backtest import normalize_closes
    
    price_map = {}
    for pair in pairs:
        date_to_close = dict(zip(pair.dates, pair.closes))
        raw = [date_to_close[d] for d in common_dates]
        norm = normalize_closes(PairData(
            symbol=pair.symbol, dates=common_dates,
            timestamps=[], opens=[], highs=[], lows=[],
            closes=raw, volumes=[],
        ))
        price_map[pair.symbol] = norm

    n_bars = len(common_dates)
    symbols = [p.symbol for p in pairs]
    years = n_bars / 252

    print(f"Aligned {n_bars} trading days across {len(symbols)} pairs: {symbols}")
    print(f"Period: {common_dates[0]} to {common_dates[-1]} ({years:.1f} years)")
    print(f"Rebalance every {rebalance_days} days | Leverage: {leverage:.0f}x")
    print(f"Weights: carry={w_carry:.0%} trend={w_trend:.0%} revert={w_revert:.0%}")
    print(f"Short selling: {'enabled' if allow_short else 'disabled'}")
    print()

    # Start in cash equivalent (hold first pair as default)
    equity = starting_cash
    long_pair = None
    short_pair = None
    long_entry_price = None
    short_entry_price = None
    bars_since_rebalance = 0
    n_trades = 0
    trade_log = []
    equity_curve = []
    
    # Need enough data for trend slow MA
    start_bar = max(trend_slow, revert_lookback)

    for i in range(n_bars):
        # Mark to market
        if long_pair:
            daily_ret = (price_map[long_pair][i] - price_map[long_pair][i-1]) / price_map[long_pair][i-1] if i > 0 else 0
            equity *= (1 + daily_ret * leverage)
        if short_pair:
            daily_ret = (price_map[short_pair][i] - price_map[short_pair][i-1]) / price_map[short_pair][i-1] if i > 0 else 0
            equity *= (1 - daily_ret * leverage)  # short: profit when price falls

        equity_curve.append(equity)

        if i < start_bar:
            bars_since_rebalance += 1
            continue

        bars_since_rebalance += 1

        if bars_since_rebalance >= rebalance_days:
            bars_since_rebalance = 0
            year = int(common_dates[i][:4])

            # Score all pairs
            scores = {}
            breakdowns = {}
            for sym in symbols:
                s, bd = composite_score(
                    sym, price_map[sym], i, year,
                    w_carry=w_carry, w_trend=w_trend, w_revert=w_revert,
                )
                scores[sym] = s
                breakdowns[sym] = bd

            # Best long candidate (highest score)
            best_long = max(symbols, key=lambda s: scores[s])
            # Best short candidate (lowest score)
            best_short = min(symbols, key=lambda s: scores[s]) if allow_short else None

            # Only take positions with meaningful scores
            new_long = best_long if scores[best_long] > 0.05 else None
            new_short = best_short if allow_short and scores[best_short] < -0.05 else None

            changed = False

            # Update long position
            if new_long != long_pair:
                if long_pair:
                    equity *= (1 - spread_cost * leverage)  # exit spread
                if new_long:
                    equity *= (1 - spread_cost * leverage)  # entry spread
                    long_entry_price = price_map[new_long][i]

                if long_pair or new_long:
                    hold_pnl = None
                    if long_pair and long_entry_price:
                        hold_pnl = (price_map[long_pair][i] - long_entry_price) / long_entry_price * leverage
                    
                    trade_log.append({
                        'bar': i, 'date': common_dates[i],
                        'side': 'LONG',
                        'old': long_pair or '-',
                        'new': new_long or 'FLAT',
                        'equity': equity,
                        'hold_pnl': hold_pnl,
                        'scores': {s: f"{scores[s]:+.3f}" for s in sorted(scores, key=lambda x: scores[x], reverse=True)[:4]},
                        'breakdown': breakdowns.get(new_long or best_long, {}),
                    })
                    n_trades += 1
                    changed = True

                long_pair = new_long

            # Update short position
            if allow_short and new_short != short_pair:
                if short_pair:
                    equity *= (1 - spread_cost * leverage)
                if new_short:
                    equity *= (1 - spread_cost * leverage)
                    short_entry_price = price_map[new_short][i]

                if short_pair or new_short:
                    hold_pnl = None
                    if short_pair and short_entry_price:
                        hold_pnl = -(price_map[short_pair][i] - short_entry_price) / short_entry_price * leverage
                    
                    trade_log.append({
                        'bar': i, 'date': common_dates[i],
                        'side': 'SHORT',
                        'old': short_pair or '-',
                        'new': new_short or 'FLAT',
                        'equity': equity,
                        'hold_pnl': hold_pnl,
                        'scores': {s: f"{scores[s]:+.3f}" for s in sorted(scores, key=lambda x: scores[x])[:4]},
                        'breakdown': breakdowns.get(new_short or best_short, {}),
                    })
                    n_trades += 1
                    changed = True

                short_pair = new_short

    # ── Results ────────────────────────────────────────────────────
    final_equity = equity
    total_return = (final_equity / starting_cash - 1) * 100
    annual_return = ((final_equity / starting_cash) ** (1 / years) - 1) * 100 if final_equity > 0 else -100

    peak = equity_curve[0] if equity_curve else starting_cash
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    print("=" * 100)
    print("  RESULTS")
    print("=" * 100)
    print(f"  {'Strategy':<30} {'End Value':>12} {'Total':>10} {'Annual':>8} {'MaxDD':>8} {'Trades':>8}")
    print(f"  {'-'*76}")
    print(f"  {'COMPOSITE (carry+trend+rev)':<30} ${final_equity:>11.2f} {total_return:>+9.1f}% {annual_return:>+7.1f}% {-max_dd*100:>7.1f}% {n_trades:>8}")

    # Individual strategy B&H benchmarks
    for sym in symbols:
        start_p = price_map[sym][0]
        end_p = price_map[sym][-1]
        bh_ret = (end_p / start_p - 1) * 100
        bh_annual = ((end_p / start_p) ** (1 / years) - 1) * 100
        print(f"  {'B&H ' + sym:<30} ${'':>11} {bh_ret:>+9.1f}% {bh_annual:>+7.1f}%")

    # ── Trade Log ──────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("  TRADE LOG (first 20 and last 10)")
    print("=" * 100)
    print(f"  {'#':<4} {'Date':<12} {'Side':<6} {'From':<8} {'To':<8} {'P&L':>8} {'Equity':>12} {'Scores (top 4)'}")
    print(f"  {'-'*90}")

    show = trade_log[:21] + ([{'sep': True}] if len(trade_log) > 31 else []) + trade_log[-10:] if len(trade_log) > 31 else trade_log
    for t in show:
        if 'sep' in t:
            print(f"  {'...':^90}")
            continue
        pnl_str = f"{t['hold_pnl']:+.2%}" if t['hold_pnl'] is not None else ""
        scores_str = "  ".join(f"{k}:{v}" for k, v in list(t['scores'].items())[:4])
        print(f"  {trade_log.index(t):<4} {t['date']:<12} {t['side']:<6} {t['old']:<8} {t['new']:<8} {pnl_str:>8} ${t['equity']:>11.2f} {scores_str}")

    # ── Yearly Breakdown ───────────────────────────────────────────
    print()
    print("=" * 100)
    print("  YEARLY BREAKDOWN")
    print("=" * 100)
    print(f"  {'Year':<8} {'Return':>10} {'Trades':>8} {'Long':>10} {'Short':>10}")
    print(f"  {'-'*46}")

    year_start_eq = equity_curve[start_bar] if start_bar < len(equity_curve) else starting_cash
    current_year = common_dates[start_bar][:4] if start_bar < len(common_dates) else "2006"

    for i in range(start_bar + 1, n_bars):
        yr = common_dates[i][:4]
        if yr != current_year or i == n_bars - 1:
            eq_now = equity_curve[i - 1 if yr != current_year else i]
            yr_ret = (eq_now / year_start_eq - 1) * 100 if year_start_eq > 0 else 0
            yr_trades = sum(1 for t in trade_log if t['date'][:4] == current_year)
            print(f"  {current_year:<8} {yr_ret:>+9.1f}% {yr_trades:>8}")
            year_start_eq = eq_now
            current_year = yr

    print()
    return final_equity


# ── CLI ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Forex Composite Strategy Backtest")
    p.add_argument("--pairs", default="EURUSD,GBPUSD,USDJPY,AUDUSD,NZDUSD,USDCHF,USDCAD")
    p.add_argument("--start", default="2003-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--rebalance-days", type=int, default=20)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--spread-cost", type=float, default=0.0003)
    p.add_argument("--starting-cash", type=float, default=10000.0)
    p.add_argument("--w-carry", type=float, default=0.33)
    p.add_argument("--w-trend", type=float, default=0.34)
    p.add_argument("--w-revert", type=float, default=0.33)
    p.add_argument("--no-short", action="store_true", help="Disable short selling")
    p.add_argument("--cache-dir", default="state/forex_cache")
    args = p.parse_args()

    pair_symbols = [s.strip().upper() for s in args.pairs.split(",")]

    print(f"Loading data for: {pair_symbols}")
    pairs_data = []
    for sym in pair_symbols:
        pair = PAIR_MAP.get(sym)
        if not pair:
            print(f"  Unknown pair: {sym}, skipping")
            continue
        data = load_pair_cached(pair, cache_dir=args.cache_dir, start=args.start, end=args.end)
        print(f"  {sym}: {len(data.closes)} bars ({data.dates[0]} to {data.dates[-1]})")
        pairs_data.append(data)

    if len(pairs_data) < 2:
        print("ERROR: need at least 2 pairs")
        return

    print()
    run_composite_backtest(
        pairs=pairs_data,
        rebalance_days=args.rebalance_days,
        spread_cost=args.spread_cost,
        leverage=args.leverage,
        starting_cash=args.starting_cash,
        w_carry=args.w_carry,
        w_trend=args.w_trend,
        w_revert=args.w_revert,
        allow_short=not args.no_short,
    )


if __name__ == "__main__":
    main()
