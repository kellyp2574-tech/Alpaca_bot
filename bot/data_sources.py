"""Factory helpers that wire up market-data clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .data_alpaca import AlpacaDataAdapter


@dataclass
class DataStack:
    """Container for the core data dependencies."""

    alpaca: AlpacaDataAdapter


def init_data_stack(
    alpaca_api_key: Optional[str] = None,
    alpaca_secret_key: Optional[str] = None,
    alpaca_feed: Optional[str] = None,
) -> DataStack:
    """Initialize and return the data stack."""
    alpaca = AlpacaDataAdapter(
        api_key=alpaca_api_key,
        secret_key=alpaca_secret_key,
        feed=alpaca_feed,
    )
    return DataStack(alpaca=alpaca)
