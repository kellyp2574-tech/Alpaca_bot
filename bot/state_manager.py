"""State management for gap momentum bot"""
import json
import logging
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from bot import config

logger = logging.getLogger(__name__)


@dataclass
class PositionState:
    """Serializable position state"""
    symbol: str
    entry_price: float
    quantity: int
    entry_time: str
    entry_gap_pct: float
    adv_estimate: float
    peak_price: float = 0.0
    order_id: Optional[str] = None


class StateManager:
    """Manages bot state persistence"""

    def __init__(self):
        self.state_dir = config.STATE_DIR
        self.positions_file = config.POSITIONS_FILE
        self.daily_log_file = config.DAILY_LOG_FILE

        # Ensure directories exist
        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.positions_file), exist_ok=True)

    def save_positions(self, positions: Dict):
        """Save current positions to file"""
        state = {}
        for symbol, pos in positions.items():
            if isinstance(pos, dict):
                state[symbol] = pos
            else:
                state[symbol] = {
                    "symbol": pos.symbol,
                    "entry_price": pos.entry_price,
                    "quantity": pos.quantity,
                    "entry_time": pos.entry_time.isoformat() if hasattr(pos.entry_time, 'isoformat') else str(pos.entry_time),
                    "entry_gap_pct": pos.entry_gap_pct,
                    "adv_estimate": pos.adv_estimate,
                    "peak_price": pos.peak_price,
                    "order_id": getattr(pos, 'order_id', None),
                }

        try:
            with open(self.positions_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving positions: {e}")

    def load_positions(self) -> Dict[str, PositionState]:
        """Load positions from file"""
        if not os.path.exists(self.positions_file):
            return {}

        try:
            with open(self.positions_file, 'r') as f:
                data = json.load(f)

            positions = {}
            for symbol, pos_data in data.items():
                # Handle missing order_id for backward compatibility
                if 'order_id' not in pos_data:
                    pos_data['order_id'] = None
                positions[symbol] = PositionState(**pos_data)

            return positions
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
            return {}

    def clear_positions(self):
        """Clear positions file"""
        if os.path.exists(self.positions_file):
            os.remove(self.positions_file)

    def log_daily_summary(self, date: str, summary: dict):
        """Log daily trading summary"""
        log_entry = {
            "date": date,
            "timestamp": datetime.now().isoformat(),
            **summary
        }

        # Append to daily log
        logs = []
        if os.path.exists(self.daily_log_file):
            try:
                with open(self.daily_log_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []

        logs.append(log_entry)

        try:
            with open(self.daily_log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving daily log: {e}")
