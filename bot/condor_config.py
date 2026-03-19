"""
0DTE Options Strategy Configuration
Two sleeves: XSP Iron Condor (daily) + XND Directional (conditional)
"""
from dataclasses import dataclass, field


@dataclass
class CondorConfig:
    """XSP Iron Condor parameters."""

    # Underlying / instrument
    underlying: str = "SPY"           # Price reference for strike calc
    option_root: str = "XSP"         # Mini-SPX, European, cash-settled
    multiplier: int = 100             # Standard option multiplier

    # Strike geometry
    short_strike_pct: float = 0.0090  # 0.90% OTM from anchor
    wing_width_pct: float = 0.0100    # 1.00% wing width
    target_credit: float = 1.30       # Target net credit per contract
    max_loss_per_contract: float = 3.70  # Wing - credit ≈ max loss

    # Defense
    defense_trigger_pct: float = 0.0140  # 1.40% move from anchor → close all

    # Schedule (ET)
    entry_time: str = "10:45"
    expiry_time: str = "16:00"
    shutdown_time: str = "16:15"

    # Order parameters
    order_type: str = "limit"
    time_in_force: str = "day"
    credit_tolerance: float = 0.20    # Accept credit within ±$0.20 of target

    # Fill optimization — smart limit order entry
    fill_start_at_mid: bool = True           # Start limit at mid instead of target_credit
    fill_patience_secs: int = 30             # Seconds to wait at each price level
    fill_adjust_step_min: float = 0.02        # Floor for dynamic step size
    fill_adjust_step_frac: float = 0.25       # step = max(min, frac * (mid - natural))
    fill_max_adjustments: int = 6            # Max price adjustments before accepting natural
    fill_min_credit: float = 0.80            # Walk away if natural credit below this
    fill_deterioration_pct: float = 0.30     # Abort if natural drops >30% from peak natural
    min_credit_risk_ratio: float = 0.30      # Skip if credit/max_loss < 0.30 (edge integrity)
    fill_max_entry_time: str = "11:00"       # Cancel and skip if not filled by this time (ET)
    max_leg_spread_pct: float = 0.50         # Skip entry if any leg spread > 50% of mid
    wide_spread_size_reduce_pct: float = 0.30  # Reduce avg spread threshold: reduce qty by 50%
    wide_spread_size_factor: float = 0.50    # Multiply qty by this when spreads are wide


@dataclass
class DirectionalConfig:
    """XND Directional play parameters."""

    underlying: str = "QQQ"           # Price reference for morning range
    option_root: str = "XND"         # Mini-NDX, European, cash-settled
    multiplier: int = 100

    # Entry filters
    vix_threshold: float = 18.0       # Previous day VIX close >= 18
    morning_range_pct: float = 0.0040  # QQQ morning range >= 0.40%
    morning_direction_pct: float = 0.0030  # |QQQ direction| > 0.30%

    # Schedule (ET)
    assessment_time: str = "10:30"    # Evaluate filters
    entry_time: str = "10:45"         # Place trade if qualified

    # Strike selection — slightly OTM in signal direction
    directional_otm_offset: float = 0.001  # 0.10% OTM from NDX estimate

    # Sizing — based on current available buying power at entry time
    directional_bp_pct: float = 0.0125     # Percent of current buying power allocated to directional premium budget
    directional_leverage_multiplier: float = 5.0  # Directional premium budget is scaled by this leverage multiplier

    # Order parameters
    order_type: str = "market"
    time_in_force: str = "day"


@dataclass
class ScheduleConfig:
    """Daily timeline."""

    bot_start: str = "09:00"
    market_open: str = "09:30"
    morning_assessment: str = "10:30"
    directional_entry: str = "10:45"
    condor_entry: str = "10:45"
    market_close: str = "16:00"
    bot_shutdown: str = "16:15"

    # Defense monitoring interval (seconds)
    defense_check_interval: int = 60
    # Account refresh interval (seconds)
    account_refresh_interval: int = 300


@dataclass
class StrategyConfig:
    """Top-level config combining both sleeves + schedule."""

    condor: CondorConfig = field(default_factory=CondorConfig)
    directional: DirectionalConfig = field(default_factory=DirectionalConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)

    # VIX ticker for filter checks
    vix_ticker: str = "^VIX"

    # Logging
    log_level: str = "INFO"
