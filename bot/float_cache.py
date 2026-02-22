"""SQLite-backed float cache with TTL semantics."""

from __future__ import annotations

import sqlite3
import time
import os
from typing import Optional


class FloatCache:
    def __init__(self, db_path: str = "floats.sqlite", ttl_hours: int = 24 * 7):
        # Make path absolute to avoid working directory issues
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), db_path)
        self.db_path = db_path
        self.ttl = ttl_hours * 3600
        print(f"FLOAT_CACHE path={self.db_path} ttl_hours={ttl_hours}")
        self._init()

    def _norm(self, symbol: str) -> str:
        """Normalize symbol to uppercase with no whitespace."""
        return symbol.strip().upper()

    def _init(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS floats (
                    symbol TEXT PRIMARY KEY,
                    float_shares REAL NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """)

    def get(self, symbol: str) -> Optional[float]:
        symbol = self._norm(symbol)
        now = int(time.time())
        with sqlite3.connect(self.db_path) as con:
            row = con.execute(
                "SELECT float_shares, updated_at FROM floats WHERE symbol=?",
                (symbol,),
            ).fetchone()
        if not row:
            return None
        float_shares, updated_at = float(row[0]), int(row[1])
        if now - updated_at > self.ttl:
            return None
        return float_shares

    def set(self, symbol: str, float_shares: float) -> None:
        symbol = self._norm(symbol)
        now = int(time.time())
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                INSERT OR REPLACE INTO floats(symbol, float_shares, updated_at)
                VALUES (?, ?, ?)
                """,
                (symbol, float_shares, now),
            )
