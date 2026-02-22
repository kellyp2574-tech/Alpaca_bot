"""Time utilities for managing the bot's trading windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Optional

from zoneinfo import ZoneInfo

from .config import Config

MARKET_TZ = ZoneInfo("America/New_York")


def market_now(tz: ZoneInfo = MARKET_TZ) -> datetime:
    """Return the current time in the market time zone."""

    return datetime.now(tz)


def parse_time_str(value: str) -> time:
    """Parse an HH:MM string into a :class:`datetime.time`."""

    hour, minute = (int(part) for part in value.split(":", maxsplit=1))
    return time(hour=hour, minute=minute)


def _coerce_date(reference: Optional[datetime | date], tz: ZoneInfo) -> date:
    if reference is None:
        return market_now(tz).date()
    if isinstance(reference, datetime):
        return reference.astimezone(tz).date()
    return reference


@dataclass(frozen=True)
class TimeWindow:
    start: datetime
    end: datetime

    def contains(self, moment: datetime, inclusive_end: bool = False) -> bool:
        moment = moment.astimezone(self.start.tzinfo)
        if inclusive_end:
            return self.start <= moment <= self.end
        return self.start <= moment < self.end

    def minutes_since_start(self, moment: datetime) -> float:
        moment = moment.astimezone(self.start.tzinfo)
        return (moment - self.start).total_seconds() / 60.0

    def minutes_until_end(self, moment: datetime) -> float:
        moment = moment.astimezone(self.end.tzinfo)
        return (self.end - moment).total_seconds() / 60.0


def window_from_strings(
    reference: Optional[datetime | date],
    start_str: str,
    end_str: str,
    tz: ZoneInfo = MARKET_TZ,
) -> TimeWindow:
    """Construct a :class:`TimeWindow` for the given date and HH:MM strings."""

    trading_date = _coerce_date(reference, tz)
    start_dt = datetime.combine(trading_date, parse_time_str(start_str), tz)
    end_dt = datetime.combine(trading_date, parse_time_str(end_str), tz)
    if end_dt <= start_dt:
        raise ValueError("End time must be later than start time within the same day")
    return TimeWindow(start=start_dt, end=end_dt)


def market_datetime(
    reference: Optional[datetime | date],
    time_str: str,
    tz: ZoneInfo = MARKET_TZ,
) -> datetime:
    """Return a timezone-aware datetime on *reference*'s date using *time_str*."""

    trading_date = _coerce_date(reference, tz)
    return datetime.combine(trading_date, parse_time_str(time_str), tz)


def config_window(
    cfg: Config,
    start_attr: str,
    end_attr: str,
    *,
    reference: Optional[datetime | date] = None,
    tz: ZoneInfo = MARKET_TZ,
) -> TimeWindow:
    """Generic helper to build a TimeWindow from Config attributes."""

    start_str = getattr(cfg, start_attr)
    end_str = getattr(cfg, end_attr)
    return window_from_strings(reference, start_str, end_str, tz)


def is_in_window(
    cfg: Config, start_attr: str, end_attr: str, moment: Optional[datetime] = None
) -> bool:
    """Return True if *moment* (default: now) is within the config window."""

    if moment is None:
        moment = market_now()
    window = config_window(cfg, start_attr, end_attr, reference=moment)
    return window.contains(moment)


def minutes_until(
    cfg: Config, target_attr: str, moment: Optional[datetime] = None
) -> float:
    """Minutes from *moment* (default: now) until the config time string."""

    if moment is None:
        moment = market_now()
    target_time = datetime.combine(
        moment.astimezone(MARKET_TZ).date(),
        parse_time_str(getattr(cfg, target_attr)),
        MARKET_TZ,
    )
    return (target_time - moment.astimezone(MARKET_TZ)).total_seconds() / 60.0


def minutes_since(
    cfg: Config, start_attr: str, moment: Optional[datetime] = None
) -> float:
    """Minutes elapsed since the config time string on the same day."""

    if moment is None:
        moment = market_now()
    start_time = datetime.combine(
        moment.astimezone(MARKET_TZ).date(),
        parse_time_str(getattr(cfg, start_attr)),
        MARKET_TZ,
    )
    return (moment.astimezone(MARKET_TZ) - start_time).total_seconds() / 60.0
