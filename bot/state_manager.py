"""
Persistent State Manager — Saves/loads bot state to JSON.
Tracks MA crossover counters, current holdings, dip trade state.
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Any, Optional

from zoneinfo import ZoneInfo

from bot.config import STATE_FILE, STATE_DIR


MARKET_TZ = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


DEFAULT_STATE = {
    "last_run": None,

    # MA Crossover state
    "ma_holding": None,          # "QLD", "UBT", "DBMF", or None
    "ma_qa": 0,                  # days QQQ above SMA+buffer
    "ma_qb": 0,                  # days QQQ below SMA-buffer
    "ma_ta": 0,                  # days TLT above SMA+buffer
    "ma_tb": 0,                  # days TLT below SMA-buffer

    # Dip trade state (shared by Monday Dip and BB Reversion)
    "dip_active": False,
    "dip_source": None,          # "MD" or "BB"
    "dip_entry_price": 0.0,      # UPRO entry price
    "dip_buy_date": None,        # date string "YYYY-MM-DD"
    "dip_days_held": 0,          # trading days held
    "dip_exit_mode": "hold",     # "hold" for MD, "sma" for BB

    # Trade history (recent)
    "trade_history": [],
}


def _read_json_strict(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        timestamp = datetime.now(MARKET_TZ).strftime("%Y%m%d%H%M%S")
        backup = path.with_name(f"{path.name}.corrupt-{timestamp}")
        try:
            os.replace(path, backup)
        except OSError:
            # If replace fails, surface original error with context
            raise RuntimeError(
                f"State file {path} is corrupted and could not be backed up"
            ) from exc
        raise RuntimeError(
            f"State file {path} is corrupted; moved to {backup}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to read state file {path}") from exc


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with NamedTemporaryFile(
            mode="w", dir=str(path.parent), delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(payload, tmp, indent=2, default=str)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
    except Exception as exc:
        # Ensure temp file is cleaned up on failure
        try:
            if 'tmp_path' in locals() and tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise RuntimeError(f"Failed to persist state file {path}") from exc


def load_state():
    os.makedirs(STATE_DIR, exist_ok=True)
    state_path = Path(STATE_FILE)
    if state_path.exists():
        saved = _read_json_strict(state_path)
        # Merge with defaults to handle new fields
        state = {**DEFAULT_STATE, **saved}
        return state
    return dict(DEFAULT_STATE)


def save_state(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    state["last_run"] = datetime.now(MARKET_TZ).strftime("%Y-%m-%d %H:%M:%S")
    _atomic_write_json(Path(STATE_FILE), state)


def log_trade(state, action, ticker, qty, price, reason=""):
    entry = {
        "timestamp": datetime.now(MARKET_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "ticker": ticker,
        "qty": qty,
        "price": price,
        "reason": reason,
    }
    state.setdefault("trade_history", [])
    state["trade_history"].append(entry)
    # Keep last 200 trades
    if len(state["trade_history"]) > 200:
        state["trade_history"] = state["trade_history"][-200:]
    return entry


class StateStore:
    """State store for morning momentum with separate file path."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_data(self) -> Dict[str, Any]:
        """Load data from file."""
        if self.file_path.exists():
            return _read_json_strict(self.file_path)
        return {}
    
    def _save_data(self, data: Dict[str, Any]) -> None:
        """Save data to file."""
        _atomic_write_json(self.file_path, data)
    
    def save_positions(self, positions: Dict[str, Any]) -> None:
        """Save positions to state store."""
        data = self._load_data()
        from .storage import position_state_to_dict
        data["positions"] = {
            symbol: position_state_to_dict(state) 
            for symbol, state in positions.items()
        }
        self._save_data(data)
    
    def load_positions(self) -> Dict[str, Any]:
        """Load positions from state store."""
        data = self._load_data()
        positions_data = data.get("positions", {})
        if not positions_data:
            return {}
        
        from .storage import position_state_from_dict
        positions: Dict[str, Any] = {}
        for symbol, state_data in positions_data.items():
            state = position_state_from_dict(state_data)
            if state is not None:
                positions[symbol] = state
            else:
                logger.error("Skipping corrupted position state for %s", symbol)
        return positions
    
    def save_pending_entry(self, pending_entry) -> None:
        """Save a pending entry to state store."""
        data = self._load_data()
        from .storage import pending_entry_to_dict
        data.setdefault("pending_entries", {})
        data["pending_entries"][pending_entry.client_order_id] = pending_entry_to_dict(pending_entry)
        self._save_data(data)
    
    def load_pending_entries(self) -> Dict[str, Any]:
        """Load pending entries from state store."""
        data = self._load_data()
        pending_data = data.get("pending_entries", {})
        if not pending_data:
            return {}
        
        from .storage import pending_entry_from_dict
        return {
            client_id: pending_entry_from_dict(entry_data)
            for client_id, entry_data in pending_data.items()
        }
    
    def clear_pending_entry(self, client_order_id: str) -> None:
        """Clear a pending entry from state store."""
        data = self._load_data()
        data.setdefault("pending_entries", {})
        data["pending_entries"].pop(client_order_id, None)
        self._save_data(data)
    
    def clear_pending_entries(self) -> None:
        """Clear all pending entries from state store."""
        data = self._load_data()
        data["pending_entries"] = {}
        self._save_data(data)
    
    def save_risk_state(self, risk_data: Dict[str, Any]) -> None:
        """Save risk manager state."""
        data = self._load_data()
        data["risk_state"] = risk_data
        self._save_data(data)
    
    def load_risk_state(self) -> Dict[str, Any]:
        """Load risk manager state."""
        data = self._load_data()
        return data.get("risk_state", {})
