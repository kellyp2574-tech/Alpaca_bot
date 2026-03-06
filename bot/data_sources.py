"""Factory helpers that wire up market-data clients."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from dotenv import load_dotenv

if TYPE_CHECKING:
    from massive import RESTClient

from .data_alpaca import AlpacaDataAdapter

load_dotenv()  # Ensure .env is loaded before reading MASSIVE_API_KEY


@dataclass
class DataStack:
    """Container for the core data dependencies."""

    alpaca: AlpacaDataAdapter
    massive: Optional["RESTClient"] = None

    def unsubscribe_all(self) -> None:
        try:
            self.alpaca.close_stream()
        except Exception:
            pass


def init_data_stack(
    alpaca_api_key: Optional[str] = None,
    alpaca_secret_key: Optional[str] = None,
    alpaca_feed: Optional[str] = None,
    massive_api_key: Optional[str] = None,
) -> DataStack:
    """Initialize and return the data stack."""
    alpaca = AlpacaDataAdapter(
        api_key=alpaca_api_key,
        secret_key=alpaca_secret_key,
        feed=alpaca_feed,
    )
    
    massive = None
    massive_api_key = massive_api_key or os.getenv("MASSIVE_API_KEY")
    if massive_api_key:
        try:
            from massive import RESTClient
            massive = RESTClient(massive_api_key)
        except ImportError:
            logging.getLogger(__name__).warning("massive package not installed; Massive client unavailable")
    
    return DataStack(alpaca=alpaca, massive=massive)
