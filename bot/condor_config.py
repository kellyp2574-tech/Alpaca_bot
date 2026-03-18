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
    defense_trigger_pct: float = 0.0100  # 1.00% move from anchor → close all

    # Schedule (ET)
    entry_time: str = "11:30"
    expiry_time: str = "16:00"
    shutdown_time: str = "16:15"

    # Order parameters
    order_type: str = "limit"
    time_in_force: str = "day"
    credit_tolerance: float = 0.20    # Accept credit within ±$0.20 of target


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

    # Sizing
    equity_risk_pct: float = 0.0125   # 1.25% of cash risked
    leverage_multiplier: float = 5.0   # Approximate option leverage

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
    condor_entry: str = "11:30"
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
