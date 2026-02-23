"""Risk guardrails for entries and daily exposure."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from .morning_config import Config
from .clock import market_now


class RiskManager:
    def __init__(self, cfg: Config, *, state_store: Optional["StateStore"] = None) -> None:
        self.cfg = cfg
        self.state_store = state_store
        self.trades_taken = 0
        self.realized_r_total = 0.0
        self.day: Optional[date] = None
        self.daily_deploy_used = 0.0  # Track total deployed today

    def load_state(self, payload: Dict[str, Any]) -> None:
        if not payload:
            return
        self.trades_taken = int(payload.get("trades_taken", 0))
        self.realized_r_total = float(payload.get("realized_r_total", 0.0))
        self.daily_deploy_used = float(payload.get("daily_deploy_used", 0.0))
        day_str = payload.get("day")
        self.day = date.fromisoformat(day_str) if day_str else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trades_taken": self.trades_taken,
            "realized_r_total": self.realized_r_total,
            "daily_deploy_used": self.daily_deploy_used,
            "day": self.day.isoformat() if self.day else None,
        }

    def maybe_reset(self, today: Optional[date] = None) -> None:
        if today is None:
            today = market_now().date()
        if self.day is None or today != self.day:
            self.day = today
            self.trades_taken = 0
            self.realized_r_total = 0.0
            self.daily_deploy_used = 0.0  # Reset daily deploy
            self._persist()

    def can_enter(self, open_positions: int) -> Tuple[bool, str]:
        # daily_kill_r must be negative (e.g., -3R) - if 0 or positive, no kill
        if self.cfg.daily_kill_r < 0 and self.realized_r_total <= self.cfg.daily_kill_r:
            return False, "daily_kill"
        if self.trades_taken >= self.cfg.max_trades_per_day:
            return False, "max_trades"
        if open_positions >= self.cfg.max_concurrent:
            return False, "max_concurrent"
        return True, "ok"

    def can_deploy_amount(self, requested_amount: float) -> Tuple[bool, float]:
        """Check if we can deploy the requested amount under daily cap."""
        if self.cfg.max_daily_deploy < 0:
            return True, requested_amount  # Negative values disable the cap
        if self.cfg.max_daily_deploy == 0:
            return False, 0.0

        remaining = self.cfg.max_daily_deploy - self.daily_deploy_used
        allowed_amount = min(requested_amount, remaining)
        can_deploy = allowed_amount > 0
        
        return can_deploy, allowed_amount

    def on_deploy(self, amount: float) -> None:
        """Record deployment amount."""
        self.daily_deploy_used += amount
        self._persist()

    def on_new_trade(self) -> None:
        self.trades_taken += 1
        self._persist()

    def on_trade_closed(self, realized_r: float) -> None:
        self.realized_r_total += realized_r
        self._persist()

    def _persist(self) -> None:
        if self.state_store:
            self.state_store.save_risk_state(self.to_dict())


if TYPE_CHECKING:  # pragma: no cover
    from .state_manager import StateStore
