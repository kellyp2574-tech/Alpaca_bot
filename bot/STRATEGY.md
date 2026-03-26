# Gap Momentum Strategy - VIX Conditional (FINAL)

**Status**: LOCKED IN  
**Strategy**: VIX-Conditioned Exit System  
**Price Range**: $0.50 - $5.00  
**Date**: March 25, 2026

---

## Strategy Overview

This strategy uses VIX-based conditional exits to adapt to market volatility regimes. Tested across 20 years (2005-2025) with 7 non-overlapping 3-year periods.

**Performance vs Baseline:**
- **VIX-Conditioned**: 1986.6% avg CAGR, -18.8% max DD
- **Baseline (trail all)**: 1703.6% avg CAGR, -17.4% max DD
- **Improvement**: +16.6% CAGR with VIX conditioning

---

## Entry Filters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Price Range** | $0.50 - $5.00 | Micro-cap momentum focus |
| **Gap %** | 3% - 50% | Minimum overnight gap |
| **ADV** | >=$5.0M | Minimum average daily dollar volume |
| **Entry Time** | 9:30 AM | Market open |
| **Entry Price** | Open price | No lookahead bias |

---

## Exit Rules (VIX-Conditioned)

**Core Logic**: 3-regime system based on VIX at market open

| VIX Level | Exit Strategy | Rationale |
|-----------|---------------|-----------|
| **VIX < 12** | Hard exit at **2:30 PM** | Low vol = less follow-through, take profits early |
| **12 ≤ VIX ≤ 22** | Trailing stop (15% activation / 3% trail) | Sweet spot - capture momentum |
| **VIX > 22** | Hard exit at **3:30 PM** | High vol = extended moves, let winners run |

**VIX Thresholds:**
- `vix_low`: 12.0
- `vix_high`: 22.0
- `trail_activation`: 15%
- `trail_stop`: 3%

---

## Position Sizing

| Parameter | Value |
|-----------|-------|
| **Daily Deployment** | 100% of trading equity |
| **Sizing Method** | Proportional to size_multiplier |
| **Low Price Multiplier** | 3x for stocks <$1.00 |
| **Liquidity Cap** | 0.3% of ADV |
| **Low Price Max** | 12% of daily capital |

---

## Backtest Results (20 Years)

### 3-Year Rolling Performance

| Period | CAGR | Max DD | Win Rate | Trades |
|--------|------|--------|----------|--------|
| 2005-2007 | 1110.5% | -10.9% | 69.3% | 940 |
| 2008-2010 | 1792.1% | -15.8% | 61.1% | 5,468 |
| 2011-2013 | 1165.6% | -8.8% | 71.5% | 1,155 |
| 2014-2016 | 1532.7% | -15.4% | 68.9% | 1,845 |
| 2017-2019 | 1792.6% | -53.5% | 68.8% | 1,386 |
| 2020-2022 | 3217.0% | -19.3% | 62.9% | 7,201 |
| 2023-2025 | 3295.4% | -7.9% | 57.9% | 10,963 |

**Key Finding**: VIX-conditioning beats baseline in ALL 7 periods, with biggest edge during high-volatility periods (2008-2010, 2020-2022).

---

## Why This Strategy Wins

1. **VIX < 12 (Low Volatility)**: Markets move slowly - early exit captures gains before they evaporate
2. **12-30 (Normal Volatility)**: Trailing stop captures trending moves while protecting capital
3. **VIX > 30 (High Volatility)**: Extended holding into 3:30 PM captures bigger moves when volatility drives momentum

---

## Data Requirements

### Files
- `data/meta/gap_candidates_complete.parquet` - Trade candidates
- `data/meta/regime_indicators.parquet` - VIX data

### VIX Data Format
- Date-indexed daily VIX close
- Used for VIX level at market open

---

## Production Implementation

### Key Components
- VIX lookup at 9:30 AM
- Hard exits at 2:30 PM (low VIX) or 3:30 PM (high VIX)
- Trailing stop monitoring during normal VIX
- Equal capital deployment across qualifying trades

---

## Test Scripts

- `test_vix_exits_20yr.py` - Full 20-year VIX-conditioned backtest
- `test_yearly_vix_dd.py` - Yearly restart DD analysis

---

## Research Findings

### What Was Tested

1. **Extreme VIX Strategies** (VIX >= 30):
   - Quick 10am exit
   - Tight trailing (5%/1.5%)
   - Skip entirely
   - Half-size positions
   
   **Finding**: Baseline VIX conditioning (3:30 PM exit) beats all extreme variants. Extreme VIX days don't drive the max DD - normal days do.

2. **VIX vs No VIX**:
   - Baseline (always trail): 1703.6% CAGR
   - VIX-conditioned: 1986.6% CAGR
   
   **Winner**: VIX-conditioned by 16.6%

---

## TODO

- [x] VIX-conditioned exit optimization
- [x] 20-year backtest with period breakdown
- [x] Extreme VIX strategy testing
- [x] Clean system comparison (A/B/C)
- [ ] Implement VIX-conditioned logic in bot
- [ ] Live paper trading validation

---

*Last Updated: March 25, 2026 - VIX CONDITIONAL CONFIG LOCKED*
