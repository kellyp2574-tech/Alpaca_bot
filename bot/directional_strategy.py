"""
XND Directional Strategy — Conditional 0DTE directional play on Nasdaq-100.

Sleeve 2: If filters pass at 10:30 AM, buy a call or put at 10:45 AM
matching the morning trend direction. Hold to cash settlement at 4:00 PM.
"""
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from bot.condor_config import DirectionalConfig
from bot.options_client import (
    fetch_option_contracts,
    find_nearest_strike,
    find_strike_at_or_above,
    find_strike_at_or_below,
    get_buying_power,
    get_option_snapshot,
    get_todays_expiration,
    place_single_option_order,
    size_directional,
    get_order,
)
from bot.market_data import MorningTracker

logger = logging.getLogger("bot.directional")


@dataclass
class DirectionalState:
    """Runtime state for the directional play."""
    # Filter results
    filters_checked: bool = False
    qualified: bool = False
    direction: Optional[str] = None   # "up" or "down"

    # Filter values at assessment time
    vix_prev_close: float = 0.0
    qqq_range_pct: float = 0.0
    qqq_direction_pct: float = 0.0

    # Trade details
    option_symbol: Optional[str] = None
    option_type: Optional[str] = None  # "call" or "put"
    strike_price: float = 0.0
    qty: int = 0
    premium_spent: float = 0.0

    # Order tracking
    entry_order_id: Optional[str] = None
    entry_time: Optional[datetime] = None
    is_filled: bool = False
    is_open: bool = False
    entry_order_dead: bool = False  # terminal state — stop polling

    # Early exit (discretionary close before settlement)
    early_exit_order_id: Optional[str] = None
    early_exit_filled: bool = False
    early_exit_fill_price: Optional[float] = None
    early_exit_order_dead: bool = False  # terminal — stop polling

    # Settlement
    settled: bool = False
    settlement_value: float = 0.0
    realized_pnl: float = 0.0


class DirectionalStrategy:
    """
    Manages the XND directional sleeve.

    Lifecycle:
    1. assess_filters() — at 10:30 AM, check VIX/QQQ filters
    2. enter() — at 10:45 AM (if qualified), buy call or put
    3. on_settlement() — at 4:00 PM, record cash settlement
    """

    def __init__(self, config: DirectionalConfig, dry_run: bool = False):
        self.cfg = config
        self.dry_run = dry_run
        self.state = DirectionalState()

    def assess_filters(self, tracker: MorningTracker) -> bool:
        """
        Evaluate entry filters at 10:30 AM.

        All three must be true:
        1. Previous day VIX close >= 18
        2. QQQ morning range >= 0.40%
        3. |QQQ morning directional move| > 0.30%

        Returns True if all filters pass.
        """
        self.state.filters_checked = True

        # VIX filter
        vix = tracker.vix_prev_close
        if vix is None:
            logger.warning("DIRECTIONAL: VIX data unavailable — skipping today")
            self.state.qualified = False
            return False

        self.state.vix_prev_close = vix
        vix_pass = vix >= self.cfg.vix_threshold

        # QQQ range filter
        range_pct = tracker.qqq_morning_range_pct
        self.state.qqq_range_pct = range_pct
        range_pass = range_pct >= self.cfg.morning_range_pct

        # QQQ direction filter
        direction_pct = tracker.qqq_morning_direction_pct
        self.state.qqq_direction_pct = direction_pct
        direction_pass = abs(direction_pct) > self.cfg.morning_direction_pct

        self.state.qualified = vix_pass and range_pass and direction_pass
        self.state.direction = tracker.qqq_direction if self.state.qualified else None

        logger.info(
            "DIRECTIONAL ASSESSMENT: VIX=%.2f(%s) range=%.3f%%(%s) direction=%.3f%%(%s) → %s",
            vix, "PASS" if vix_pass else "FAIL",
            range_pct * 100, "PASS" if range_pass else "FAIL",
            direction_pct * 100, "PASS" if direction_pass else "FAIL",
            "QUALIFIED" if self.state.qualified else "NO TRADE",
        )

        if self.state.qualified:
            logger.info(
                "DIRECTIONAL: Will buy %s at %s",
                "CALL" if self.state.direction == "up" else "PUT",
                self.cfg.entry_time,
            )

        return self.state.qualified

    # QQQ ≈ NDX / 40.  XND is a mini-NDX product whose underlying
    # value is 1/100th of NDX.  So XND strikes ≈ NDX / 100 ≈ QQQ × 0.4.
    # We first convert QQQ → NDX estimate, then scale to the option
    # root's strike denomination.
    QQQ_TO_NDX_RATIO = 40.0

    # Strike scale divisors: NDX_estimate / divisor = strike-scale value
    _STRIKE_DIVISORS = {
        "XND": 100.0,   # XND strikes ≈ NDX / 100
        "NDX": 1.0,     # NDX strikes are in NDX index points
    }

    def enter(self, tracker: MorningTracker) -> bool:
        """
        Enter the directional trade at 10:45 AM.

        Buy a call (if morning up) or put (if morning down) on XND 0DTE.
        Strike selection uses an NDX-level estimate derived from QQQ × 40.

        Returns True if the entry order was submitted successfully.
        The position is NOT considered open until the fill is confirmed
        (see check_fill_status).
        """
        if not self.state.qualified:
            logger.info("DIRECTIONAL: Not qualified — no trade today")
            return False

        # Determine option type
        option_type = "call" if self.state.direction == "up" else "put"
        self.state.option_type = option_type

        qqq_price = tracker.qqq_last
        if qqq_price <= 0:
            logger.error("DIRECTIONAL: QQQ price unavailable")
            return False

        # Convert QQQ ETF price → approximate NDX index level,
        # then scale to the option root's strike denomination.
        ndx_estimate = qqq_price * self.QQQ_TO_NDX_RATIO
        root = self.cfg.option_root.upper()
        divisor = self._STRIKE_DIVISORS.get(root)
        if divisor is None:
            logger.error("DIRECTIONAL: Unsupported option root %s — cannot determine strike scale", root)
            return False
        strike_center = ndx_estimate / divisor
        logger.info(
            "DIRECTIONAL: QQQ=$%.2f → NDX≈%.2f → %s strike_center=%.2f (divisor=%.0f)",
            qqq_price, ndx_estimate, root, strike_center, divisor,
        )

        # Sanity-check strike_center against the option root
        if root == "XND" and strike_center > 1000:
            logger.error(
                "DIRECTIONAL: Refusing XND contract lookup — strike_center=%.2f "
                "is too high (expected ~200-300 for XND). Check scaling.",
                strike_center,
            )
            return False
        if root in {"NDX", "SPX"} and strike_center < 1000:
            logger.error(
                "DIRECTIONAL: Refusing %s contract lookup — strike_center=%.2f "
                "is suspiciously low. Check scaling.",
                root, strike_center,
            )
            return False

        expiration = get_todays_expiration()

        # Fetch contracts in a neighbourhood around the strike center
        strike_margin = strike_center * 0.02  # ±2 % to capture nearby strikes

        logger.info(
            "DIRECTIONAL: Contract lookup — root=%s strike_center=%.2f "
            "margin=%.2f gte=%.2f lte=%.2f exp=%s type=%s",
            root, strike_center, strike_margin,
            strike_center - strike_margin, strike_center + strike_margin,
            expiration, option_type,
        )

        try:
            contracts = fetch_option_contracts(
                underlying=self.cfg.option_root,
                expiration_date=expiration,
                option_type=option_type,
                strike_price_gte=strike_center - strike_margin,
                strike_price_lte=strike_center + strike_margin,
            )
        except Exception as e:
            logger.error("DIRECTIONAL: Contract fetch failed for %s exp=%s: %s",
                         self.cfg.option_root, expiration, e)
            return False

        tradable = [c for c in contracts if c.tradable]
        if not tradable:
            logger.error("DIRECTIONAL: No tradable %s contracts for %s exp=%s near strike=%.0f",
                         option_type, self.cfg.option_root, expiration, strike_center)
            return False

        # Select a slightly OTM strike in the signal direction.
        # Calls (up): strike at-or-above center * (1 + otm_offset)
        # Puts (down): strike at-or-below center * (1 - otm_offset)
        otm_offset = self.cfg.directional_otm_offset
        if option_type == "call":
            otm_target = strike_center * (1 + otm_offset)
            contract = find_strike_at_or_above(tradable, otm_target)
        else:
            otm_target = strike_center * (1 - otm_offset)
            contract = find_strike_at_or_below(tradable, otm_target)

        if not contract:
            logger.error("DIRECTIONAL: Could not find suitable %s contract (OTM target=%.2f)",
                         option_type, otm_target)
            return False

        logger.info(
            "DIRECTIONAL: OTM strike selection — strike_center=%.2f, otm_target=%.2f, "
            "chosen_strike=%.2f (%s)",
            strike_center, otm_target, contract.strike_price, option_type,
        )
        self.state.option_symbol = contract.symbol
        self.state.strike_price = contract.strike_price

        # Sizing — based on current available buying power at entry time
        buying_power = get_buying_power()
        notional = size_directional(
            buying_power, self.cfg.directional_bp_pct, self.cfg.directional_leverage_multiplier,
        )

        # Try to get a live premium quote for this 0DTE contract.
        # contract.close_price is yesterday's close — potentially very
        # stale for a 0DTE option.  Prefer a live snapshot if available.
        premium_is_stale = False
        live_premium = get_option_snapshot(contract.symbol)
        if live_premium and live_premium > 0:
            estimated_premium = live_premium
            logger.info("DIRECTIONAL: Using live premium $%.2f for %s", live_premium, contract.symbol)
        elif contract.close_price and contract.close_price > 0:
            estimated_premium = contract.close_price
            premium_is_stale = True
            logger.warning(
                "DIRECTIONAL: Live premium unavailable, falling back to prior close $%.2f for %s "
                "— qty will be capped at 1 (stale 0DTE premium risk)",
                estimated_premium, contract.symbol,
            )
        else:
            estimated_premium = 2.00
            premium_is_stale = True
            logger.warning(
                "DIRECTIONAL: No premium data — using hardcoded fallback $%.2f "
                "— qty will be capped at 1 (stale 0DTE premium risk)",
                estimated_premium,
            )
        premium_per_contract = estimated_premium * contract.size  # × 100
        qty = max(1, math.floor(notional / premium_per_contract)) if premium_per_contract > 0 else 1

        # Cap at 1 contract when premium is stale to limit sizing drift
        if premium_is_stale and qty > 1:
            logger.warning(
                "DIRECTIONAL: Capping qty from %d to 1 (stale premium — live quote unavailable)",
                qty,
            )
            qty = 1

        self.state.qty = qty
        self.state.premium_spent = estimated_premium * qty * contract.size

        logger.info(
            "DIRECTIONAL ENTRY: BUY %s %d contracts, strike=$%.2f (%s≈%.0f), est_premium=$%.2f",
            contract.symbol, qty, contract.strike_price, root, strike_center, estimated_premium,
        )

        # Place order — position is NOT open yet, only order_submitted
        try:
            order = place_single_option_order(
                symbol=contract.symbol,
                qty=qty,
                side="buy",
                order_type=self.cfg.order_type,
                time_in_force=self.cfg.time_in_force,
                dry_run=self.dry_run,
            )

            if order:
                self.state.entry_order_id = order.get("id")
                self.state.entry_time = datetime.now()
                # is_open stays False until fill is confirmed
                logger.info(
                    "DIRECTIONAL ORDER SUBMITTED: %s %d × %s, order_id=%s",
                    option_type.upper(), qty, contract.symbol,
                    self.state.entry_order_id,
                )
                return True
        except Exception as e:
            logger.error("DIRECTIONAL ENTRY FAILED: %s", e)

        return False

    def check_fill_status(self) -> bool:
        """Check if the entry order has been filled.  Sets is_open = True only on fill."""
        if self.state.is_filled:
            return True
        if not self.state.entry_order_id or self.state.entry_order_dead:
            return False

        if self.dry_run:
            self.state.is_filled = True
            self.state.is_open = True
            return True

        try:
            order = get_order(self.state.entry_order_id)
            status = order.get("status", "")

            if status == "filled":
                self.state.is_filled = True
                self.state.is_open = True
                filled_price = float(order.get("filled_avg_price", 0))
                if filled_price > 0:
                    self.state.premium_spent = filled_price * self.state.qty * 100
                logger.info(
                    "DIRECTIONAL FILLED: %d × %s @ $%.2f",
                    self.state.qty, self.state.option_symbol, filled_price,
                )
                return True
            elif status in ("cancelled", "expired", "rejected"):
                logger.warning("DIRECTIONAL ORDER %s: %s — marking terminal", status.upper(), self.state.entry_order_id)
                self.state.entry_order_dead = True
                return False
            else:
                logger.debug("Directional order status: %s", status)
        except Exception as e:
            logger.warning("Failed to check directional fill status: %s", e)

        return False

    def check_early_exit_fill(self) -> bool:
        """
        Poll the discretionary early-exit close order for a fill.

        Returns True once the fill is confirmed.  The orchestrator
        should record the PDT exit only after this returns True.
        """
        if self.state.early_exit_filled or not self.state.early_exit_order_id:
            return self.state.early_exit_filled

        if self.dry_run:
            self.state.early_exit_filled = True
            self.state.early_exit_fill_price = 0.0
            self.state.is_open = False
            return True

        try:
            order = get_order(self.state.early_exit_order_id)
            status = order.get("status", "")
            if status == "filled":
                self.state.early_exit_filled = True
                self.state.early_exit_fill_price = float(order.get("filled_avg_price", 0))
                self.state.is_open = False
                logger.info(
                    "DIRECTIONAL EARLY EXIT FILLED: %d × %s @ $%.2f",
                    self.state.qty, self.state.option_symbol,
                    self.state.early_exit_fill_price,
                )
                return True
            elif status in ("cancelled", "expired", "rejected"):
                logger.error(
                    "DIRECTIONAL EARLY EXIT ORDER %s — position may still be open!",
                    status.upper(),
                )
                self.state.early_exit_order_dead = True
            else:
                logger.debug("Directional early-exit order status: %s", status)
        except Exception as e:
            logger.warning("Failed to check directional early-exit fill: %s", e)
        return False

    def on_settlement(self, tracker: MorningTracker) -> dict:
        """
        Handle 4:00 PM cash settlement for the directional trade.

        XND is cash-settled European.  P&L = intrinsic value at settlement
        minus premium paid.  If the order never filled, report no_position.

        NOTE: Settlement uses QQQ × 40 as an NDX proxy.  Actual XND
        settlement is derived from a special index calculation, not
        literally QQQ × 40.  The reported P&L is therefore an *estimate*;
        reconcile against the broker statement for precise accounting.
        """
        if not self.state.is_filled:
            status = "never_filled" if self.state.entry_order_id else "no_position"
            return {"status": status}

        # Compute intrinsic value from QQQ → NDX → strike-scale settlement
        qqq_settle = tracker.qqq_last
        ndx_settle = qqq_settle * self.QQQ_TO_NDX_RATIO if qqq_settle > 0 else 0.0
        root = self.cfg.option_root.upper()
        divisor = self._STRIKE_DIVISORS.get(root)
        if divisor is None:
            logger.error("DIRECTIONAL SETTLEMENT: Unsupported option root %s", root)
            return {"status": "error", "reason": f"unsupported option root {root}"}
        settle_level = ndx_settle / divisor  # same scale as the strike

        strike = self.state.strike_price
        multiplier = 100  # standard option multiplier

        if self.state.option_type == "call":
            intrinsic_per = max(settle_level - strike, 0.0)
        else:  # put
            intrinsic_per = max(strike - settle_level, 0.0)

        settlement_value = intrinsic_per * self.state.qty * multiplier
        self.state.settlement_value = settlement_value
        self.state.realized_pnl = settlement_value - self.state.premium_spent

        self.state.settled = True
        self.state.is_open = False

        result = {
            "status": "settled",
            "direction": self.state.direction,
            "option_type": self.state.option_type,
            "symbol": self.state.option_symbol,
            "contracts": self.state.qty,
            "strike": self.state.strike_price,
            "settle_level": round(settle_level, 2),
            "ndx_settlement": round(ndx_settle, 2),
            "intrinsic_per_contract": round(intrinsic_per, 2),
            "settlement_value": round(settlement_value, 2),
            "premium_spent": round(self.state.premium_spent, 2),
            "realized_pnl": round(self.state.realized_pnl, 2),
        }

        logger.info(
            "DIRECTIONAL SETTLEMENT: %s %d × %s strike=$%.2f %s_settle=%.2f "
            "intrinsic=$%.2f settlement=$%.2f premium=$%.2f pnl=$%.2f",
            self.state.option_type, self.state.qty,
            self.state.option_symbol, self.state.strike_price,
            root, settle_level, intrinsic_per, settlement_value,
            self.state.premium_spent, self.state.realized_pnl,
        )
        return result

    def get_summary(self) -> dict:
        """Return current state summary for logging."""
        return {
            "filters_checked": self.state.filters_checked,
            "qualified": self.state.qualified,
            "direction": self.state.direction,
            "option_type": self.state.option_type,
            "symbol": self.state.option_symbol,
            "qty": self.state.qty,
            "is_open": self.state.is_open,
            "is_filled": self.state.is_filled,
            "vix": self.state.vix_prev_close,
            "qqq_range": f"{self.state.qqq_range_pct * 100:.3f}%",
            "qqq_direction": f"{self.state.qqq_direction_pct * 100:.3f}%",
        }
