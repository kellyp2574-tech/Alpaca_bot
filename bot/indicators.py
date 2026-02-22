"""Indicator helpers used by the morning momentum bot."""

from __future__ import annotations

from typing import Sequence


def true_range(high: float, low: float, prev_close: float) -> float:
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def atr_1m(bars: Sequence, n: int) -> float:
    """Average true range using the most recent *n* 1m bars."""

    if len(bars) < n + 1:
        return 0.0
    trs = []
    for i in range(-n, 0):
        bar = bars[i]
        prev = bars[i - 1]
        trs.append(true_range(bar.h, bar.l, prev.c))
    return sum(trs) / n if trs else 0.0


class VWAPState:
    def __init__(self) -> None:
        self.pv = 0.0
        self.v = 0.0

    def update(self, bar) -> None:
        tp = (bar.h + bar.l + bar.c) / 3.0
        self.pv += tp * bar.v
        self.v += bar.v

    @property
    def vwap(self) -> float:
        return self.pv / self.v if self.v > 0 else 0.0
