# core/config.py
import os
from pathlib import Path
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from datetime import time, datetime
import logging
import json
import pytz
from enum import Enum
from dataclasses import dataclass, field, asdict

class DataSource(Enum):
    YAHOO = "YAHOO"
    NSE = "NSE"
    BSE = "BSE"
    OPENBB = "OPENBB"

class TradingMode(Enum):
    LIVE = "LIVE"
    PAPER = "PAPER"
    BACKTEST = "BACKTEST"

@dataclass
class RiskConfig:
    max_position_size: float = 0.10
    max_portfolio_risk: float = 0.25
    max_correlation: float = 0.60
    position_sizing_model: str = 'KELLY'
    max_drawdown: float = 0.15
    risk_free_rate: float = 0.05
    var_confidence: float = 0.99
    max_volatility: float = 0.30
    min_liquidity_ratio: float = 2.0
    max_volume_ratio: float = 0.10
    margin_requirement: float = 1.0
    max_leverage: float = 1.0
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.05
    trailing_stop_pct: float = 0.02
    max_single_order_notional: float = 1000000
    max_daily_drawdown: float = 0.05
    max_position_concentration: float = 0.20
    stress_test_scenarios: List[str] = field(default_factory=lambda: [
        "MARKET_CRASH",
        "SECTOR_CRISIS",
        "LIQUIDITY_EVENT",
        "VOLATILITY_SPIKE"
    ])

@dataclass
class DataConfig:
    primary_source: DataSource = DataSource.YAHOO
    backup_sources: List[DataSource] = field(default_factory=lambda: [
        DataSource.NSE,
        DataSource.BSE
    ])
    cache_expiry: int = 60  # seconds
    min_history: int = 100
    rate_limit: int = 5  # requests per second
    batch_size: int = 50
    price_precision: int = 2
    quantity_precision: int = 0
    update_interval: int = 30  # seconds
    retry_attempts: int = 3
    retry_delay: int = 5  # seconds
    warm_up_period: int = 100
    save_raw_data: bool = True
    compression: bool = True
    validate_data: bool = True

@dataclass
class MarketConfig:
    exchanges: List[str] = field(default_factory=lambda: ['NSE', 'BSE'])
    trading_hours: Dict[str, Dict[str, time]] = field(default_factory=lambda: {
        'NSE': {'start': time(9, 15), 'end': time(15, 30)},
        'BSE': {'start': time(9, 15), 'end': time(15, 30)}
    })
    holidays: List[str] = field(default_factory=list)
    margin_requirements: Dict[str, float] = field(default_factory=lambda: {'default': 1.0})
    tick_sizes: Dict[str, float] = field(default_factory=lambda: {'default': 0.05})
    lot_sizes: Dict[str, int] = field(default_factory=lambda: {'default': 1})
    circuit_limits: Dict[str, float] = field(default_factory=lambda: {'default': 0.20})
    timezone: str = 'Asia/Kolkata'
    settlement_days: int = 2
    pre_market_start: time = time(9, 0)
    post_market_end: time = time(15, 45)
    auction_duration: int = 15  # minutes

@dataclass
class SystemConfig:
    max_workers: int = 10
    queue_size: int = 1000
    cache_size: int = 500  # MB
    log_level: str = 'INFO'
    log_retention: int = 30  # days
    heartbeat_interval: int = 60  # seconds
    shutdown_timeout: int = 30  # seconds
    max_retries: int = 3
    backoff_factor: float = 2.0
    health_check_interval: int = 300  # seconds
    cleanup_interval: int = 3600  # seconds
    max_memory_usage: float = 0.8  # 80% of available RAM
    max_cpu_usage: float = 0.9  # 90% of CPU
    debug_mode: bool = False

class Config:
    def __init__(self):
        # Setup logging first
        self.logger = self._setup_logging()
        
        # Initialize paths
        self.BASE_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.BASE_DIR / 'data'
        self.LOG_DIR = self.DATA_DIR / 'logs'
        self.CACHE_DIR = self.DATA_DIR / 'cache'
        self.BACKTEST_DIR = self.DATA_DIR / 'backtest'
        self.CONFIG_DIR = self.BASE_DIR / 'config'
        
        # Create directories
        self._create_directories()
        
        # Trading parameters
        self.TRADING_MODE = TradingMode.PAPER
        self.CAPITAL = 100000
        self.MIN_TRADE_AMOUNT = 1000
        self.COMMISSION_RATE = 0.0003
        self.SLIPPAGE = 0.0002
        
        # Market parameters
        self.MARKET_START = time(9, 15)
        self.MARKET_END = time(15, 30)
        self.SCAN_INTERVAL = 30
        
        # Initialize configurations
        self.risk = RiskConfig()
        self.data = DataConfig()
        self.market = MarketConfig()
        self.system = SystemConfig()
        
        # Load configuration
        self.load_config()

    def _setup_logging(self) -> logging.Logger:
        """Setup configuration logging"""
        logger = logging.getLogger('config')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    def _create_directories(self):
        """Create necessary directories"""
        try:
            directories = [
                self.DATA_DIR,
                self.LOG_DIR,
                self.CACHE_DIR,
                self.BACKTEST_DIR,
                self.CONFIG_DIR
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                
        except Exception as e:
            self.logger.error(f"Error creating directories: {e}")
            raise

    def load_config(self):
        """Load configuration from YAML files"""
        try:
            # Load main config
            main_config = self.CONFIG_DIR / 'config.yaml'
            if main_config.exists():
                with open(main_config) as f:
                    config_data = yaml.safe_load(f)
                    if config_data:
                        self._update_config(config_data)
            
            # Load environment-specific config
            env = os.getenv('TRADING_ENV', 'development')
            env_config = self.CONFIG_DIR / f'config.{env}.yaml'
            if env_config.exists():
                with open(env_config) as f:
                    env_data = yaml.safe_load(f)
                    if env_data:
                        self._update_config(env_data)
                        
            self.logger.info(f"Configuration loaded for environment: {env}")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise

    def _update_config(self, config_data: dict):
        """Update configuration from dictionary"""
        try:
            for key, value in config_data.items():
                if hasattr(self, key):
                    if isinstance(value, dict):
                        current_value = getattr(self, key)
                        if isinstance(current_value, (RiskConfig, DataConfig, MarketConfig, SystemConfig)):
                            for k, v in value.items():
                                if hasattr(current_value, k):
                                    setattr(current_value, k, v)
                                else:
                                    self.logger.warning(f"Unknown config parameter: {k}")
                    else:
                        setattr(self, key, value)
                else:
                    self.logger.warning(f"Unknown config parameter: {key}")
                    
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            raise

    def validate(self) -> bool:
        """Validate configuration parameters"""
        try:
            # Validate capital and trade amounts
            if self.CAPITAL <= 0 or self.MIN_TRADE_AMOUNT <= 0:
                return False
                
            # Validate rates
            if not (0 < self.COMMISSION_RATE < 1 and 
                   0 < self.SLIPPAGE < 1):
                return False
                
            # Validate risk parameters
            if not (0 < self.risk.max_position_size < 1 and
                   0 < self.risk.max_portfolio_risk < 1):
                return False
                
            # Validate market hours
            if self.MARKET_END <= self.MARKET_START:
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    def save_config(self):
        """Save configuration to YAML file"""
        try:
            config_data = {
                'TRADING_MODE': self.TRADING_MODE.value,
                'CAPITAL': self.CAPITAL,
                'MIN_TRADE_AMOUNT': self.MIN_TRADE_AMOUNT,
                'COMMISSION_RATE': self.COMMISSION_RATE,
                'SLIPPAGE': self.SLIPPAGE,
                'MARKET_START': self.MARKET_START.strftime('%H:%M'),
                'MARKET_END': self.MARKET_END.strftime('%H:%M'),
                'SCAN_INTERVAL': self.SCAN_INTERVAL,
                'risk': asdict(self.risk),
                'data': asdict(self.data),
                'market': {
                    k: v for k, v in asdict(self.market).items()
                    if not isinstance(v, time)
                },
                'system': asdict(self.system)
            }
            
            config_file = self.CONFIG_DIR / 'config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
                
            self.logger.info("Configuration saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise

    def get_env_config(self) -> Dict:
        """Get environment-specific configuration"""
        try:
            env = os.getenv('TRADING_ENV', 'development')
            env_config = self.CONFIG_DIR / f'config.{env}.yaml'
            
            if env_config.exists():
                with open(env_config) as f:
                    return yaml.safe_load(f)
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting environment config: {e}")
            return {}

# Create global config instance
CONFIG = Config()
