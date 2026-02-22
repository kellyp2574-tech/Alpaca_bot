"""
Persistent State Manager — Saves/loads bot state to JSON.
Tracks MA crossover counters, current holdings, dip trade state.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from bot.config import STATE_FILE, STATE_DIR


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


def load_state():
    os.makedirs(STATE_DIR, exist_ok=True)
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            saved = json.load(f)
        # Merge with defaults to handle new fields
        state = {**DEFAULT_STATE, **saved}
        return state
    return dict(DEFAULT_STATE)


def save_state(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def log_trade(state, action, ticker, qty, price, reason=""):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
            try:
                with open(self.file_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading state from {self.file_path}: {e}")
        return {}
    
    def _save_data(self, data: Dict[str, Any]) -> None:
        """Save data to file."""
        try:
            with open(self.file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except IOError as e:
            print(f"Error saving state to {self.file_path}: {e}")
    
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
        return {
            symbol: position_state_from_dict(state_data)
            for symbol, state_data in positions_data.items()
        }
    
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
