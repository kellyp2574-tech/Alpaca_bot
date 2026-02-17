"""
Persistent State Manager — Saves/loads bot state to JSON.
Tracks MA crossover counters, current holdings, dip trade state.
"""
import json
import os
from datetime import datetime
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
