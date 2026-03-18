"""
Alpaca Options Client — Contract lookup, multi-leg orders, single-leg orders.

Uses the alpaca-py SDK for trading and raw REST for option contract queries
where the SDK doesn't yet expose convenience methods.
"""
import logging
import math
from dataclasses import dataclass
from datetime import date
from typing import Optional

import requests

from bot.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER

logger = logging.getLogger("bot.options")

# ─── API base URLs ────────────────────────────────────────────────────────────
PAPER_BASE = "https://paper-api.alpaca.markets"
LIVE_BASE = "https://api.alpaca.markets"


def _base_url() -> str:
    return PAPER_BASE if ALPACA_PAPER else LIVE_BASE


def _headers() -> dict:
    return {
        "accept": "application/json",
        "content-type": "application/json",
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class OptionContract:
    """Parsed option contract from Alpaca API."""
    id: str
    symbol: str               # e.g. "XSP260318C00575000"
    name: str
    status: str
    tradable: bool
    expiration_date: str       # "YYYY-MM-DD"
    root_symbol: str
    underlying_symbol: str
    type: str                  # "call" or "put"
    style: str                 # "american" or "european"
    strike_price: float
    size: int                  # multiplier (usually 100)
    open_interest: Optional[int] = None
    close_price: Optional[float] = None


@dataclass
class CondorLegs:
    """Four OCC symbols for an iron condor."""
    long_put: str       # Buy OTM put (lowest strike)
    short_put: str      # Sell put (closer to money)
    short_call: str     # Sell call (closer to money)
    long_call: str      # Buy OTM call (highest strike)

    long_put_strike: float = 0.0
    short_put_strike: float = 0.0
    short_call_strike: float = 0.0
    long_call_strike: float = 0.0


# ─── Contract lookup ──────────────────────────────────────────────────────────

def fetch_option_contracts(
    underlying: str,
    expiration_date: str,
    option_type: Optional[str] = None,
    strike_price_gte: Optional[float] = None,
    strike_price_lte: Optional[float] = None,
    limit: int = 1000,
) -> list[OptionContract]:
    """
    Fetch option contracts from /v2/options/contracts.

    Parameters
    ----------
    underlying : Underlying symbol (e.g. "XSP")
    expiration_date : "YYYY-MM-DD"
    option_type : "call" or "put" or None for both
    strike_price_gte / strike_price_lte : strike price range
    limit : max results per page
    """
    url = f"{_base_url()}/v2/options/contracts"
    params = {
        "underlying_symbols": underlying,
        "expiration_date": expiration_date,
        "status": "active",
        "limit": limit,
    }
    if option_type:
        params["type"] = option_type
    if strike_price_gte is not None:
        params["strike_price_gte"] = str(strike_price_gte)
    if strike_price_lte is not None:
        params["strike_price_lte"] = str(strike_price_lte)

    contracts = []
    page_token = None

    while True:
        if page_token:
            params["page_token"] = page_token

        resp = requests.get(url, headers=_headers(), params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        for c in data.get("option_contracts", []):
            contracts.append(OptionContract(
                id=c["id"],
                symbol=c["symbol"],
                name=c.get("name", ""),
                status=c["status"],
                tradable=c.get("tradable", False),
                expiration_date=c["expiration_date"],
                root_symbol=c.get("root_symbol", ""),
                underlying_symbol=c.get("underlying_symbol", ""),
                type=c["type"],
                style=c.get("style", ""),
                strike_price=float(c["strike_price"]),
                size=int(c.get("size", 100)),
                open_interest=int(c["open_interest"]) if c.get("open_interest") else None,
                close_price=float(c["close_price"]) if c.get("close_price") else None,
            ))

        page_token = data.get("page_token")
        if not page_token:
            break

    logger.info(
        "Fetched %d %s contracts for %s exp=%s",
        len(contracts), option_type or "all", underlying, expiration_date,
    )
    return contracts


def find_nearest_strike(contracts: list[OptionContract], target_strike: float) -> Optional[OptionContract]:
    """Find the contract whose strike is nearest to target_strike."""
    if not contracts:
        return None
    return min(contracts, key=lambda c: abs(c.strike_price - target_strike))


def get_todays_expiration() -> str:
    """Return today's date as YYYY-MM-DD for 0DTE."""
    return date.today().strftime("%Y-%m-%d")


# ─── Strike calculation ──────────────────────────────────────────────────────

def compute_condor_strikes(
    anchor_price: float,
    short_pct: float = 0.009,
    wing_pct: float = 0.010,
) -> dict:
    """
    Calculate iron condor strike targets from an anchor price.

    Returns dict with keys: short_put, long_put, short_call, long_call
    """
    short_put = anchor_price * (1 - short_pct)
    short_call = anchor_price * (1 + short_pct)
    long_put = anchor_price * (1 - short_pct - wing_pct)
    long_call = anchor_price * (1 + short_pct + wing_pct)

    return {
        "short_put": round(short_put, 2),
        "long_put": round(long_put, 2),
        "short_call": round(short_call, 2),
        "long_call": round(long_call, 2),
    }


def build_condor_legs(
    anchor_price: float,
    option_root: str,
    expiration_date: str,
    short_pct: float = 0.009,
    wing_pct: float = 0.010,
) -> Optional[CondorLegs]:
    """
    Fetch contracts and find the best 4 legs for an iron condor.

    Returns CondorLegs with OCC symbols, or None if contracts not found.
    """
    targets = compute_condor_strikes(anchor_price, short_pct, wing_pct)
    logger.info(
        "Condor strike targets: anchor=%.2f LP=%.2f SP=%.2f SC=%.2f LC=%.2f",
        anchor_price, targets["long_put"], targets["short_put"],
        targets["short_call"], targets["long_call"],
    )

    # Fetch puts and calls
    puts = fetch_option_contracts(
        option_root, expiration_date, option_type="put",
        strike_price_gte=targets["long_put"] - 5,
        strike_price_lte=targets["short_put"] + 5,
    )
    calls = fetch_option_contracts(
        option_root, expiration_date, option_type="call",
        strike_price_gte=targets["short_call"] - 5,
        strike_price_lte=targets["long_call"] + 5,
    )

    tradable_puts = [c for c in puts if c.tradable]
    tradable_calls = [c for c in calls if c.tradable]

    long_put = find_nearest_strike(tradable_puts, targets["long_put"])
    short_put = find_nearest_strike(tradable_puts, targets["short_put"])
    short_call = find_nearest_strike(tradable_calls, targets["short_call"])
    long_call = find_nearest_strike(tradable_calls, targets["long_call"])

    if not all([long_put, short_put, short_call, long_call]):
        logger.error(
            "Could not find all 4 condor legs. LP=%s SP=%s SC=%s LC=%s",
            long_put, short_put, short_call, long_call,
        )
        return None

    # Ensure puts are ordered correctly (long_put < short_put)
    if long_put.strike_price >= short_put.strike_price:
        logger.error("Long put strike (%.2f) must be < short put strike (%.2f)",
                      long_put.strike_price, short_put.strike_price)
        return None

    # Ensure calls are ordered correctly (short_call < long_call)
    if short_call.strike_price >= long_call.strike_price:
        logger.error("Short call strike (%.2f) must be < long call strike (%.2f)",
                      short_call.strike_price, long_call.strike_price)
        return None

    legs = CondorLegs(
        long_put=long_put.symbol,
        short_put=short_put.symbol,
        short_call=short_call.symbol,
        long_call=long_call.symbol,
        long_put_strike=long_put.strike_price,
        short_put_strike=short_put.strike_price,
        short_call_strike=short_call.strike_price,
        long_call_strike=long_call.strike_price,
    )

    logger.info(
        "Condor legs built: LP=%s(%.2f) SP=%s(%.2f) SC=%s(%.2f) LC=%s(%.2f)",
        legs.long_put, legs.long_put_strike,
        legs.short_put, legs.short_put_strike,
        legs.short_call, legs.short_call_strike,
        legs.long_call, legs.long_call_strike,
    )
    return legs


# ─── Order placement ─────────────────────────────────────────────────────────

def place_iron_condor_order(
    legs: CondorLegs,
    qty: int,
    limit_price: float,
    time_in_force: str = "day",
    dry_run: bool = False,
) -> Optional[dict]:
    """
    Place an iron condor as a single multi-leg order.

    The limit_price is the net credit we want to receive.
    Alpaca mleg orders: positive limit_price = net credit for a condor.
    """
    payload = {
        "order_class": "mleg",
        "qty": str(qty),
        "type": "limit",
        "limit_price": str(round(limit_price, 2)),
        "time_in_force": time_in_force,
        "legs": [
            {
                "symbol": legs.long_put,
                "ratio_qty": "1",
                "side": "buy",
                "position_intent": "buy_to_open",
            },
            {
                "symbol": legs.short_put,
                "ratio_qty": "1",
                "side": "sell",
                "position_intent": "sell_to_open",
            },
            {
                "symbol": legs.short_call,
                "ratio_qty": "1",
                "side": "sell",
                "position_intent": "sell_to_open",
            },
            {
                "symbol": legs.long_call,
                "ratio_qty": "1",
                "side": "buy",
                "position_intent": "buy_to_open",
            },
        ],
    }

    logger.info(
        "CONDOR ORDER: qty=%d credit=$%.2f legs=[%s %s %s %s] %s",
        qty, limit_price,
        legs.long_put, legs.short_put, legs.short_call, legs.long_call,
        "[DRY RUN]" if dry_run else "",
    )

    if dry_run:
        logger.info("DRY RUN — order not submitted")
        return {"dry_run": True, "payload": payload}

    url = f"{_base_url()}/v2/orders"
    resp = requests.post(url, headers=_headers(), json=payload, timeout=15)
    resp.raise_for_status()
    order = resp.json()
    logger.info("Condor order placed: id=%s status=%s", order.get("id"), order.get("status"))
    return order


def close_condor_order(
    legs: CondorLegs,
    qty: int,
    dry_run: bool = False,
) -> Optional[dict]:
    """
    Close an iron condor by placing the reverse multi-leg order at market.

    Defense exit — we want to close ASAP so we use market order.
    """
    payload = {
        "order_class": "mleg",
        "qty": str(qty),
        "type": "market",
        "time_in_force": "day",
        "legs": [
            {
                "symbol": legs.long_put,
                "ratio_qty": "1",
                "side": "sell",
                "position_intent": "sell_to_close",
            },
            {
                "symbol": legs.short_put,
                "ratio_qty": "1",
                "side": "buy",
                "position_intent": "buy_to_close",
            },
            {
                "symbol": legs.short_call,
                "ratio_qty": "1",
                "side": "buy",
                "position_intent": "buy_to_close",
            },
            {
                "symbol": legs.long_call,
                "ratio_qty": "1",
                "side": "sell",
                "position_intent": "sell_to_close",
            },
        ],
    }

    logger.info(
        "CONDOR CLOSE: qty=%d legs=[%s %s %s %s] %s",
        qty, legs.long_put, legs.short_put, legs.short_call, legs.long_call,
        "[DRY RUN]" if dry_run else "",
    )

    if dry_run:
        logger.info("DRY RUN — close order not submitted")
        return {"dry_run": True, "payload": payload}

    url = f"{_base_url()}/v2/orders"
    resp = requests.post(url, headers=_headers(), json=payload, timeout=15)
    resp.raise_for_status()
    order = resp.json()
    logger.info("Condor close order placed: id=%s status=%s", order.get("id"), order.get("status"))
    return order


def place_single_option_order(
    symbol: str,
    qty: int,
    side: str = "buy",
    order_type: str = "market",
    limit_price: Optional[float] = None,
    time_in_force: str = "day",
    dry_run: bool = False,
) -> Optional[dict]:
    """
    Place a single-leg option order (for directional trades).

    Parameters
    ----------
    symbol : OCC symbol (e.g. "XND260318C00020000")
    qty : Number of contracts
    side : "buy" or "sell"
    order_type : "market" or "limit"
    limit_price : Required if order_type is "limit"
    """
    position_intent = "buy_to_open" if side == "buy" else "sell_to_close"

    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force,
        "position_intent": position_intent,
    }
    if order_type == "limit" and limit_price is not None:
        payload["limit_price"] = str(round(limit_price, 2))

    logger.info(
        "OPTION ORDER: %s %s qty=%d type=%s intent=%s %s %s",
        side.upper(), symbol, qty, order_type, position_intent,
        f"@${limit_price:.2f}" if limit_price else "",
        "[DRY RUN]" if dry_run else "",
    )

    if dry_run:
        logger.info("DRY RUN — order not submitted")
        return {"dry_run": True, "payload": payload}

    url = f"{_base_url()}/v2/orders"
    resp = requests.post(url, headers=_headers(), json=payload, timeout=15)
    resp.raise_for_status()
    order = resp.json()
    logger.info("Option order placed: id=%s status=%s", order.get("id"), order.get("status"))
    return order


# ─── Order management ─────────────────────────────────────────────────────────

def get_option_snapshot(symbol: str) -> Optional[float]:
    """
    Get a live premium estimate for an option contract from Alpaca snapshots.

    Returns the mid-price (avg of latest bid/ask) if available, or the
    latest trade price as a fallback.  Returns None if no data.
    """
    # Alpaca option snapshots: data.alpaca.markets/v2/options/snapshots/{symbol}
    url = f"https://data.alpaca.markets/v1beta1/options/snapshots/{symbol}"
    try:
        resp = requests.get(url, headers=_headers(), timeout=10)
        resp.raise_for_status()
        snap = resp.json()

        # Try mid of latest quote first
        quote = snap.get("latestQuote", {})
        bid = float(quote.get("bp", 0))
        ask = float(quote.get("ap", 0))
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            logger.info("Option snapshot %s: bid=$%.2f ask=$%.2f mid=$%.2f", symbol, bid, ask, mid)
            return mid

        # Fallback to latest trade
        trade = snap.get("latestTrade", {})
        price = float(trade.get("p", 0))
        if price > 0:
            logger.info("Option snapshot %s: latest trade=$%.2f (no quote)", symbol, price)
            return price

    except Exception as e:
        logger.warning("Failed to get option snapshot for %s: %s", symbol, e)
    return None


def get_order(order_id: str) -> Optional[dict]:
    """Get order status by ID."""
    url = f"{_base_url()}/v2/orders/{order_id}"
    resp = requests.get(url, headers=_headers(), timeout=10)
    resp.raise_for_status()
    return resp.json()


def cancel_order(order_id: str) -> bool:
    """Cancel an order by ID."""
    url = f"{_base_url()}/v2/orders/{order_id}"
    resp = requests.delete(url, headers=_headers(), timeout=10)
    if resp.status_code in (200, 204):
        logger.info("Order %s cancelled", order_id)
        return True
    logger.warning("Failed to cancel order %s: %s", order_id, resp.text)
    return False


def get_all_orders(status: str = "open") -> list[dict]:
    """Get all orders with given status."""
    url = f"{_base_url()}/v2/orders"
    params = {"status": status, "limit": 500}
    resp = requests.get(url, headers=_headers(), params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def cancel_all_orders() -> bool:
    """Cancel all open orders."""
    url = f"{_base_url()}/v2/orders"
    resp = requests.delete(url, headers=_headers(), timeout=10)
    if resp.status_code in (200, 204, 207):
        logger.info("All orders cancelled")
        return True
    logger.warning("Failed to cancel all orders: %s", resp.text)
    return False


def get_option_positions() -> list[dict]:
    """Get all open option positions."""
    url = f"{_base_url()}/v2/positions"
    resp = requests.get(url, headers=_headers(), timeout=10)
    resp.raise_for_status()
    positions = resp.json()
    # Filter to option positions (symbol format matches OCC)
    return [p for p in positions if len(p.get("symbol", "")) > 10]


def get_account_info() -> dict:
    """Get account details including buying power."""
    url = f"{_base_url()}/v2/account"
    resp = requests.get(url, headers=_headers(), timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_buying_power() -> float:
    """Get current buying power from account."""
    acct = get_account_info()
    return float(acct.get("buying_power", 0))


def get_equity() -> float:
    """Get current equity from account."""
    acct = get_account_info()
    return float(acct.get("equity", 0))


def get_cash() -> float:
    """Get current cash from account."""
    acct = get_account_info()
    return float(acct.get("cash", 0))


# ─── Sizing helpers ───────────────────────────────────────────────────────────

def size_condor(buying_power: float, max_loss_per_contract: float) -> int:
    """
    Calculate number of condor contracts.
    contracts = floor(buying_power / max_loss_per_contract)
    max_loss_per_contract is in dollars (e.g. $3.70 * 100 = $370)
    """
    max_loss_dollars = max_loss_per_contract * 100  # Per-contract max loss in $
    if max_loss_dollars <= 0:
        return 0
    contracts = math.floor(buying_power / max_loss_dollars)
    logger.info(
        "Condor sizing: buying_power=$%.2f / max_loss=$%.2f = %d contracts",
        buying_power, max_loss_dollars, contracts,
    )
    return max(contracts, 0)


def size_directional(buying_power: float, equity_risk_pct: float, leverage: float) -> float:
    """
    Calculate notional for directional trade.
    notional = buying_power × equity_risk_pct × leverage
    Returns dollar amount to spend on premium.
    """
    notional = buying_power * equity_risk_pct * leverage
    logger.info(
        "Directional sizing: buying_power=$%.2f × %.4f × %.1fx = $%.2f notional",
        buying_power, equity_risk_pct, leverage, notional,
    )
    return notional
