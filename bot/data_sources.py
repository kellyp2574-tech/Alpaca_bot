"""Factory helpers that wire up market-data clients and caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .data_alpaca import AlpacaDataAdapter
from .data_fmp import FMPClient
from .float_cache import FloatCache


@dataclass
class DataStack:
    """Container for the core data dependencies."""

    alpaca: AlpacaDataAdapter
    fmp: FMPClient
    float_cache: FloatCache


def init_data_stack(
    *,
    alpaca_api_key: Optional[str] = None,
    alpaca_secret_key: Optional[str] = None,
    alpaca_feed: Optional[str] = None,
    fmp_api_key: Optional[str] = None,
    float_db_path: str = "floats.sqlite",
    float_ttl_hours: int = 24 * 7,
) -> DataStack:
    """Instantiate adapters + cache, defaulting to environment credentials."""

    alpaca = AlpacaDataAdapter(
        api_key=alpaca_api_key,
        secret_key=alpaca_secret_key,
        feed=alpaca_feed,
    )
    fmp = FMPClient(api_key=fmp_api_key)
    cache = FloatCache(db_path=float_db_path, ttl_hours=float_ttl_hours)
    return DataStack(alpaca=alpaca, fmp=fmp, float_cache=cache)
