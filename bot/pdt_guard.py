"""
PDT (Pattern Day Trader) Guard — Persistent day-trade ledger and
rolling 5-business-day tracker.

Tracks same-day round trips (open + close on the same trading day)
across all strategy sleeves.  Gates *discretionary* early exits when:

  - account equity <= PDT_EQUITY_BUFFER  (default $30,000), OR
  - rolling 5-business-day day-trade count >= MAX_DAY_TRADES (default 3)

Mandatory exits (defense, emergency, risk-reduction) are NEVER blocked.

Persistence:
  state/day_trade_log.json  — rolling ledger of confirmed day trades
  (survives process restarts)
"""
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from typing import Optional

from bot import config as cfg

logger = logging.getLogger("bot.pdt_guard")

# ─── Constants ────────────────────────────────────────────────────────────────

PDT_EQUITY_BUFFER = 30_000.0   # Block discretionary exits below this equity
MAX_DAY_TRADES = 3             # Max same-day round trips in rolling 5 biz days
ROLLING_WINDOW_BIZ_DAYS = 5

DAY_TRADE_LOG_PATH = os.path.join(cfg.STATE_DIR, "day_trade_log.json")
OPEN_POSITIONS_PATH = os.path.join(cfg.STATE_DIR, "open_positions.json")

# ─── Exit reasons ─────────────────────────────────────────────────────────────

EXIT_REASON_SCHEDULED = "scheduled_close"
EXIT_REASON_DISCRETIONARY = "discretionary_early_exit"
EXIT_REASON_DEFENSE = "defense"
EXIT_REASON_EMERGENCY = "emergency"

MANDATORY_EXIT_REASONS = {EXIT_REASON_DEFENSE, EXIT_REASON_EMERGENCY}


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class OpenPosition:
    """Tracks an intraday position opened by the bot."""
    symbol: str
    strategy: str              # "directional" or "condor"
    open_date: str             # YYYY-MM-DD (trading day)
    open_order_id: str
    open_timestamp: Optional[str] = None
    opened_today: bool = True
    counted_as_day_trade: bool = False


@dataclass
class DayTradeRecord:
    """A confirmed same-day round trip recorded in the persistent ledger."""
    trade_date: str            # YYYY-MM-DD
    strategy: str              # "directional" or "condor"
    symbol: str
    open_timestamp: str
    close_timestamp: str
    exit_reason: str
    counted_as_day_trade: bool = True


# ─── Persistent ledger ────────────────────────────────────────────────────────

def _load_ledger() -> list[dict]:
    """Load the day-trade ledger from disk."""
    if not os.path.exists(DAY_TRADE_LOG_PATH):
        return []
    try:
        with open(DAY_TRADE_LOG_PATH, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning("Failed to load day-trade ledger: %s", e)
        return []


def _save_ledger(records: list[dict]) -> None:
    """Write the day-trade ledger to disk."""
    os.makedirs(os.path.dirname(DAY_TRADE_LOG_PATH), exist_ok=True)
    try:
        with open(DAY_TRADE_LOG_PATH, "w") as f:
            json.dump(records, f, indent=2, default=str)
    except Exception as e:
        logger.error("Failed to save day-trade ledger: %s", e)


def _load_open_positions() -> dict[str, "OpenPosition"]:
    """Load open intraday positions from disk (survives midday restart)."""
    if not os.path.exists(OPEN_POSITIONS_PATH):
        return {}
    try:
        with open(OPEN_POSITIONS_PATH, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        positions = {}
        for sym, d in data.items():
            positions[sym] = OpenPosition(
                symbol=d.get("symbol", sym),
                strategy=d.get("strategy", ""),
                open_date=d.get("open_date", ""),
                open_order_id=d.get("open_order_id", ""),
                open_timestamp=d.get("open_timestamp"),
                opened_today=(d.get("open_date", "") == date.today().isoformat()),
                counted_as_day_trade=d.get("counted_as_day_trade", False),
            )
        logger.info("PDT: Loaded %d open position(s) from disk", len(positions))
        return positions
    except Exception as e:
        logger.warning("Failed to load open positions: %s", e)
        return {}


def _save_open_positions(positions: dict[str, "OpenPosition"]) -> None:
    """Write open intraday positions to disk."""
    os.makedirs(os.path.dirname(OPEN_POSITIONS_PATH), exist_ok=True)
    try:
        data = {sym: asdict(pos) for sym, pos in positions.items()}
        with open(OPEN_POSITIONS_PATH, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        logger.error("Failed to save open positions: %s", e)


def _business_days_ago(n: int, ref_date: Optional[date] = None) -> date:
    """Return the date N business days before ref_date (default: today)."""
    d = ref_date or date.today()
    count = 0
    while count < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:  # Mon–Fri
            count += 1
    return d


# ─── PDTGuard class ──────────────────────────────────────────────────────────

class PDTGuard:
    """
    Tracks intraday positions and same-day round trips.

    Usage:
        guard = PDTGuard()

        # When an entry fill is confirmed:
        guard.record_entry_fill("XND260318C20000", "directional",
                                order_id, fill_timestamp)

        # Before a discretionary early exit:
        if guard.can_take_discretionary_day_trade(current_equity):
            ... proceed with exit ...

        # When an exit fill is confirmed:
        guard.record_exit_fill("XND260318C20000", "directional",
                               close_timestamp, exit_reason)
    """

    def __init__(self):
        self._open_positions: dict[str, OpenPosition] = _load_open_positions()
        self._ledger: list[dict] = _load_ledger()
        self._prune_old_records()

    # ── Entry tracking ────────────────────────────────────────────────────

    def record_entry_fill(
        self,
        symbol: str,
        strategy: str,
        order_id: str,
        fill_timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a confirmed entry fill.  Call only after the fill is confirmed."""
        ts = fill_timestamp or datetime.now()
        today_str = ts.strftime("%Y-%m-%d")

        pos = OpenPosition(
            symbol=symbol,
            strategy=strategy,
            open_date=today_str,
            open_order_id=order_id,
            open_timestamp=ts.isoformat(),
            opened_today=True,
        )
        self._open_positions[symbol] = pos
        _save_open_positions(self._open_positions)
        logger.info(
            "PDT: Recorded entry fill — %s %s on %s (order=%s)",
            strategy, symbol, today_str, order_id,
        )

    # ── Exit tracking ─────────────────────────────────────────────────────

    def record_exit_fill(
        self,
        symbol: str,
        strategy: str,
        close_timestamp: Optional[datetime] = None,
        exit_reason: str = EXIT_REASON_SCHEDULED,
    ) -> bool:
        """
        Record a confirmed exit fill.  If the open and close are on the
        same trading day, increments the day-trade count.

        Returns True if this was counted as a day trade.
        """
        ts = close_timestamp or datetime.now()
        close_date_str = ts.strftime("%Y-%m-%d")

        pos = self._open_positions.pop(symbol, None)
        _save_open_positions(self._open_positions)

        is_day_trade = False
        if pos and pos.open_date == close_date_str and not pos.counted_as_day_trade:
            is_day_trade = True
            pos.counted_as_day_trade = True

            record = asdict(DayTradeRecord(
                trade_date=close_date_str,
                strategy=strategy,
                symbol=symbol,
                open_timestamp=pos.open_timestamp or "",
                close_timestamp=ts.isoformat(),
                exit_reason=exit_reason,
                counted_as_day_trade=True,
            ))
            self._ledger.append(record)
            _save_ledger(self._ledger)

            logger.warning(
                "PDT: Same-day round trip detected — %s %s on %s (reason=%s). "
                "Rolling count: %d",
                strategy, symbol, close_date_str, exit_reason,
                self.rolling_day_trade_count(),
            )
        else:
            logger.info(
                "PDT: Exit fill recorded — %s %s (not a same-day round trip)",
                strategy, symbol,
            )

        return is_day_trade

    # ── Query helpers ─────────────────────────────────────────────────────

    def rolling_day_trade_count(self, ref_date: Optional[date] = None) -> int:
        """Count day trades in the rolling 5-business-day window."""
        self._prune_old_records()
        cutoff = _business_days_ago(ROLLING_WINDOW_BIZ_DAYS, ref_date)
        cutoff_str = cutoff.isoformat()

        count = 0
        for rec in self._ledger:
            if rec.get("trade_date", "") >= cutoff_str and rec.get("counted_as_day_trade"):
                count += 1
        return count

    def can_take_discretionary_day_trade(self, current_equity: float) -> bool:
        """
        Check whether a discretionary early exit is allowed.

        Returns True only if:
          1. current_equity > PDT_EQUITY_BUFFER ($30,000)
          2. rolling 5-business-day day-trade count < MAX_DAY_TRADES (3)

        Mandatory exits (defense, emergency) should NOT use this check.
        """
        if current_equity <= PDT_EQUITY_BUFFER:
            logger.warning(
                "PDT GUARD: Discretionary exit BLOCKED — equity $%.2f <= $%.2f buffer",
                current_equity, PDT_EQUITY_BUFFER,
            )
            return False

        count = self.rolling_day_trade_count()
        if count >= MAX_DAY_TRADES:
            logger.warning(
                "PDT GUARD: Discretionary exit BLOCKED — %d day trades in rolling %d "
                "business days (max=%d)",
                count, ROLLING_WINDOW_BIZ_DAYS, MAX_DAY_TRADES,
            )
            return False

        logger.info(
            "PDT GUARD: Discretionary exit ALLOWED — equity=$%.2f, "
            "rolling day trades=%d/%d",
            current_equity, count, MAX_DAY_TRADES,
        )
        return True

    def get_status(self) -> dict:
        """Return a summary dict for reporting."""
        return {
            "rolling_day_trade_count": self.rolling_day_trade_count(),
            "max_day_trades": MAX_DAY_TRADES,
            "pdt_equity_buffer": PDT_EQUITY_BUFFER,
            "open_intraday_positions": len(self._open_positions),
            "ledger_size": len(self._ledger),
        }

    # ── Internal ──────────────────────────────────────────────────────────

    def _prune_old_records(self) -> None:
        """Remove ledger entries older than the rolling window."""
        cutoff = _business_days_ago(ROLLING_WINDOW_BIZ_DAYS + 2)  # small buffer
        cutoff_str = cutoff.isoformat()

        before = len(self._ledger)
        self._ledger = [
            r for r in self._ledger
            if r.get("trade_date", "") >= cutoff_str
        ]
        pruned = before - len(self._ledger)
        if pruned > 0:
            _save_ledger(self._ledger)
            logger.debug("PDT: Pruned %d old ledger records", pruned)
