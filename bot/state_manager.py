"""State management for gap momentum bot"""
import json
import logging
import os
from typing import Dict, Optional
from bot import config

logger = logging.getLogger(__name__)


class StateManager:
    """Manages bot state persistence"""

    def __init__(self):
        self.state_dir = config.STATE_DIR
        self.positions_file = config.POSITIONS_FILE

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
                    "adv_estimate": pos.adv_estimate,
                    "current_price": getattr(pos, 'current_price', pos.entry_price),
                }

        try:
            with open(self.positions_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving positions: {e}")

    def load_positions(self) -> Dict[str, dict]:
        """Load positions from file as raw dicts (consumed by PositionManager.load_positions)"""
        if not os.path.exists(self.positions_file):
            return {}

        try:
            with open(self.positions_file, 'r') as f:
                data = json.load(f)

            positions = {}
            for symbol, pos_data in data.items():
                if 'current_price' not in pos_data:
                    pos_data['current_price'] = pos_data.get('entry_price', 0)
                positions[symbol] = pos_data

            return positions
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
            return {}

    def clear_positions(self):
        """Clear positions file"""
        if os.path.exists(self.positions_file):
            os.remove(self.positions_file)

    def clear_bot_state(self):
        """Clear bot state file"""
        bot_state_file = os.path.join(self.state_dir, "bot_state.json")
        if os.path.exists(bot_state_file):
            os.remove(bot_state_file)

    def save_bot_state(self, state: dict):
        """Save bot state (VIX, stages) to file"""
        bot_state_file = os.path.join(self.state_dir, "bot_state.json")
        try:
            with open(bot_state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving bot state: {e}")

    def load_bot_state(self) -> Optional[dict]:
        """Load bot state from file"""
        bot_state_file = os.path.join(self.state_dir, "bot_state.json")
        if not os.path.exists(bot_state_file):
            return None
        try:
            with open(bot_state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading bot state: {e}")
            return None

