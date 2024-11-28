# core/strategy.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Set, Type, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import talib
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.ensemble import IsolationForest
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import json
import uuid
import traceback

from .market_data import market_data
from .execution import execution_engine, OrderRequest, OrderType
from .events import event_manager, Event, EventType
from .config import CONFIG

@dataclass
class Signal:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    direction: str = ""  # 'LONG' or 'SHORT'
    strength: float = 0.0  # 0 to 1
    strategy: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    price: float = 0.0
    expiry: Optional[datetime] = None
    status: str = "PENDING"  # PENDING, ACTIVE, EXECUTED, EXPIRED
    indicators: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.expiry is None:
            self.expiry = self.timestamp + timedelta(minutes=15)

    def is_active(self) -> bool:
        return (
            self.status in ["PENDING", "ACTIVE"] and
            datetime.now() < self.expiry
        )

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'direction': self.direction,
            'strength': self.strength,
            'strategy': self.strategy,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'expiry': self.expiry.isoformat() if self.expiry else None,
            'status': self.status,
            'indicators': self.indicators,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Signal':
        """Create signal from dictionary"""
        if 'timestamp' in data:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'expiry' in data and data['expiry']:
            data['expiry'] = datetime.fromisoformat(data['expiry'])
        return cls(**data)

@dataclass
class StrategyState:
    name: str
    active: bool = True
    last_updated: datetime = field(default_factory=datetime.now)
    positions: Set[str] = field(default_factory=set)
    signals: List[Signal] = field(default_factory=list)
    performance: Dict = field(default_factory=dict)
    parameters: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'active': self.active,
            'last_updated': self.last_updated.isoformat(),
            'positions': list(self.positions),
            'signals': [s.to_dict() for s in self.signals],
            'performance': self.performance,
            'parameters': self.parameters,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyState':
        """Create strategy state from dictionary"""
        if 'last_updated' in data:
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        if 'signals' in data:
            data['signals'] = [Signal.from_dict(s) for s in data['signals']]
        if 'positions' in data:
            data['positions'] = set(data['positions'])
        return cls(**data)

class IndicatorCalculator:
    """Utility class for calculating technical indicators"""
    
    @staticmethod
    def calculate_moving_averages(data: pd.DataFrame, periods: List[int]) -> Dict[str, pd.Series]:
        """Calculate multiple moving averages"""
        mas = {}
        for period in periods:
            mas[f'SMA_{period}'] = talib.SMA(data['Close'], timeperiod=period)
            mas[f'EMA_{period}'] = talib.EMA(data['Close'], timeperiod=period)
        return mas

    @staticmethod
    def calculate_momentum_indicators(data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate momentum indicators"""
        indicators = {
            'RSI': talib.RSI(data['Close']),
            'MACD': talib.MACD(data['Close'])[0],
            'MACD_Signal': talib.MACD(data['Close'])[1],
            'MACD_Hist': talib.MACD(data['Close'])[2],
            'MOM': talib.MOM(data['Close']),
            'ROC': talib.ROC(data['Close'])
        }
        
        # Stochastic oscillator
        slowk, slowd = talib.STOCH(data['High'], data['Low'], data['Close'])
        indicators['STOCH_K'] = slowk
        indicators['STOCH_D'] = slowd
        
        return indicators

    @staticmethod
    def calculate_volatility_indicators(data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volatility indicators"""
        indicators = {
            'ATR': talib.ATR(data['High'], data['Low'], data['Close']),
            'NATR': talib.NATR(data['High'], data['Low'], data['Close'])
        }
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(data['Close'])
        indicators['BB_Upper'] = upper
        indicators['BB_Middle'] = middle
        indicators['BB_Lower'] = lower
        
        return indicators

    @staticmethod
    def calculate_volume_indicators(data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volume indicators"""
        indicators = {
            'OBV': talib.OBV(data['Close'], data['Volume']),
            'AD': talib.AD(data['High'], data['Low'], data['Close'], data['Volume']),
            'ADX': talib.ADX(data['High'], data['Low'], data['Close'])
        }
        
        # Volume moving averages
        indicators['Volume_SMA'] = talib.SMA(data['Volume'], timeperiod=20)
        indicators['Volume_Ratio'] = data['Volume'] / indicators['Volume_SMA']
        
        return indicators

    @staticmethod
    def calculate_trend_indicators(data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate trend indicators"""
        indicators = {
            'ADX': talib.ADX(data['High'], data['Low'], data['Close']),
            'PLUS_DI': talib.PLUS_DI(data['High'], data['Low'], data['Close']),
            'MINUS_DI': talib.MINUS_DI(data['High'], data['Low'], data['Close']),
            'AROON_UP': talib.AROON(data['High'], data['Low'])[0],
            'AROON_DOWN': talib.AROON(data['High'], data['Low'])[1]
        }
        return indicators
    
class BaseStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.state = StrategyState(name=name)
        self.logger = self._setup_logging()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.required_history = 100
        self.update_interval = 60  # seconds
        self._last_run = datetime.min
        self._initialized = False
        self.indicator_calculator = IndicatorCalculator()
        
        # Default parameters
        self.parameters = {
            'max_positions': CONFIG.MAX_POSITIONS,
            'position_size': CONFIG.risk.max_position_size,
            'min_signal_strength': CONFIG.MIN_SIGNAL_STRENGTH,
            'max_correlation': CONFIG.risk.max_correlation,
            'max_drawdown': CONFIG.risk.max_drawdown
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup strategy logging"""
        logger = logging.getLogger(f"strategy.{self.name}")
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(CONFIG.LOG_DIR / f'strategy_{self.name}.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    async def initialize(self):
        """Initialize strategy"""
        try:
            if self._initialized:
                return
                
            self.logger.info(f"Initializing strategy {self.name}...")
            
            # Load saved state
            await self.load_state()
            
            # Subscribe to events
            event_manager.subscribe(EventType.MARKET_UPDATE, self._handle_market_update)
            event_manager.subscribe(EventType.TRADE, self._handle_trade)
            
            # Initialize technical indicators
            await self._initialize_indicators()
            
            self._initialized = True
            self.logger.info(f"Strategy {self.name} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing strategy: {e}\n{traceback.format_exc()}")
            raise

    async def load_state(self):
        """Load strategy state from disk"""
        try:
            state_file = CONFIG.DATA_DIR / f'strategy_{self.name}_state.json'
            if state_file.exists():
                with open(state_file) as f:
                    state_data = json.load(f)
                    self.state = StrategyState.from_dict(state_data)
                    
                # Load parameters
                if 'parameters' in state_data:
                    self.parameters.update(state_data['parameters'])
                    
                self.logger.info(f"Loaded strategy state with {len(self.state.signals)} signals")
                
        except Exception as e:
            self.logger.error(f"Error loading strategy state: {e}")

    async def save_state(self):
        """Save strategy state to disk"""
        try:
            state_data = {
                **self.state.to_dict(),
                'parameters': self.parameters
            }
            
            state_file = CONFIG.DATA_DIR / f'strategy_{self.name}_state.json'
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving strategy state: {e}")

    async def _initialize_indicators(self):
        """Initialize technical indicators - implemented by specific strategies"""
        pass

    async def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals - implemented by specific strategies"""
        raise NotImplementedError

    async def validate_signal(self, signal: Signal) -> bool:
        """Validate trading signal"""
        try:
            # Check if strategy is active
            if not self.state.active:
                return False
                
            # Check if symbol already has active position
            if signal.symbol in self.state.positions:
                return False
                
            # Check signal strength
            if signal.strength < self.parameters['min_signal_strength']:
                return False
                
            # Check recent signals
            recent_signals = [
                s for s in self.state.signals
                if s.symbol == signal.symbol and s.is_active()
            ]
            if recent_signals:
                return False
                
            # Check market hours
            if not market_data.is_market_open():
                return False
                
            # Check position limits
            if len(self.state.positions) >= self.parameters['max_positions']:
                return False
                
            # Check correlation limits
            if not await self._check_correlation(signal.symbol):
                return False
                
            # Check drawdown limits
            if not await self._check_drawdown():
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False

    async def _check_correlation(self, symbol: str) -> bool:
        """Check correlation with existing positions"""
        try:
            if not self.state.positions:
                return True
                
            for pos_symbol in self.state.positions:
                correlation = await self._calculate_correlation(symbol, pos_symbol)
                if abs(correlation) > self.parameters['max_correlation']:
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking correlation: {e}")
            return True

    async def _calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols"""
        try:
            data1 = await market_data.get_stock_data(symbol1)
            data2 = await market_data.get_stock_data(symbol2)
            
            if data1 is None or data2 is None or data1.empty or data2.empty:
                return 0.0
                
            returns1 = data1['Close'].pct_change().dropna()
            returns2 = data2['Close'].pct_change().dropna()
            
            return returns1.corr(returns2)
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation: {e}")
            return 0.0

    async def _check_drawdown(self) -> bool:
        """Check if current drawdown exceeds limit"""
        try:
            if not self.state.performance:
                return True
                
            current_drawdown = self.state.performance.get('current_drawdown', 0)
            return current_drawdown <= self.parameters['max_drawdown']
            
        except Exception as e:
            self.logger.error(f"Error checking drawdown: {e}")
            return True

    async def _handle_market_update(self, event: Event):
        """Handle market update events"""
        try:
            if not self._should_run():
                return
                
            symbol = event.data['symbol']
            
            # Get market data
            data = await market_data.get_stock_data(symbol)
            if data is None:
                return
                
            # Generate signals
            signals = await self.generate_signals(data)
            
            # Process valid signals
            for signal in signals:
                if await self.validate_signal(signal):
                    await self._process_signal(signal)
                    
            self._last_run = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error handling market update: {e}")

    def _should_run(self) -> bool:
        """Check if strategy should run"""
        return (
            self.state.active and
            (datetime.now() - self._last_run).total_seconds() >= self.update_interval
        )
        
    async def _process_signal(self, signal: Signal):
        """Process and execute trading signal"""
        try:
            # Add signal to state
            self.state.signals.append(signal)
            signal.status = "ACTIVE"
            
            # Calculate position size
            position_size = await self._calculate_position_size(signal)
            if position_size <= 0:
                return
                
            # Create order request
            order_request = OrderRequest(
                symbol=signal.symbol,
                side="BUY" if signal.direction == "LONG" else "SELL",
                quantity=position_size,
                order_type=OrderType.MARKET,
                strategy_id=self.name,
                metadata={
                    'signal_id': signal.id,
                    'signal_strength': signal.strength,
                    'indicators': signal.indicators
                }
            )
            
            # Place order
            order_id = await execution_engine.place_order(order_request)
            
            if order_id:
                signal.status = "EXECUTED"
                self.state.positions.add(signal.symbol)
                
                # Publish signal event
                await event_manager.publish(Event(
                    type=EventType.SIGNAL,
                    data={
                        'signal': signal.to_dict(),
                        'order_id': order_id
                    }
                ))
                
                # Update state
                await self.save_state()
                
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}\n{traceback.format_exc()}")

    async def _calculate_position_size(self, signal: Signal) -> int:
        """Calculate optimal position size"""
        try:
            # Get available capital
            portfolio_value = await self._get_portfolio_value()
            max_position_value = portfolio_value * self.parameters['position_size']
            
            # Adjust for signal strength
            position_value = max_position_value * signal.strength
            
            # Adjust for volatility
            volatility_adj = await self._calculate_volatility_adjustment(signal.symbol)
            adjusted_value = position_value * volatility_adj
            
            # Convert to number of shares
            shares = int(adjusted_value / signal.price)
            
            # Check minimum trade amount
            if shares * signal.price < CONFIG.MIN_TRADE_AMOUNT:
                return 0
                
            return shares
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0

class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy"""
    
    def __init__(self):
        super().__init__("Momentum")
        
        # Strategy-specific parameters
        self.parameters.update({
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'volume_ma_period': 20,
            'momentum_period': 10,
            'trend_ma_period': 50,
            'min_adx': 25,
            'min_volume_ratio': 1.5
        })

    async def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate momentum-based trading signals"""
        try:
            signals = []
            
            # Calculate indicators in parallel
            indicator_tasks = [
                self._calculate_momentum_indicators(data),
                self._calculate_trend_indicators(data),
                self._calculate_volume_indicators(data)
            ]
            
            momentum_data, trend_data, volume_data = await asyncio.gather(*indicator_tasks)
            
            # Get latest values
            latest = data.index[-1]
            price = data['Close'].iloc[-1]
            
            # Check long signals
            long_conditions = [
                momentum_data['RSI'].iloc[-1] < self.parameters['rsi_oversold'],
                momentum_data['MACD'].iloc[-1] > momentum_data['MACD_Signal'].iloc[-1],
                trend_data['ADX'].iloc[-1] > self.parameters['min_adx'],
                volume_data['Volume_Ratio'].iloc[-1] > self.parameters['min_volume_ratio'],
                data['Close'].iloc[-1] > trend_data['SMA_50'].iloc[-1]
            ]
            
            if all(long_conditions):
                strength = await self._calculate_signal_strength(
                    data, momentum_data, trend_data, volume_data, 'LONG'
                )
                
                if strength > self.parameters['min_signal_strength']:
                    signals.append(Signal(
                        symbol=data.index[0],
                        direction='LONG',
                        strength=strength,
                        strategy=self.name,
                        price=price,
                        indicators={
                            'rsi': momentum_data['RSI'].iloc[-1],
                            'macd': momentum_data['MACD'].iloc[-1],
                            'adx': trend_data['ADX'].iloc[-1],
                            'volume_ratio': volume_data['Volume_Ratio'].iloc[-1]
                        }
                    ))
            
            # Check short signals
            short_conditions = [
                momentum_data['RSI'].iloc[-1] > self.parameters['rsi_overbought'],
                momentum_data['MACD'].iloc[-1] < momentum_data['MACD_Signal'].iloc[-1],
                trend_data['ADX'].iloc[-1] > self.parameters['min_adx'],
                volume_data['Volume_Ratio'].iloc[-1] > self.parameters['min_volume_ratio'],
                data['Close'].iloc[-1] < trend_data['SMA_50'].iloc[-1]
            ]
            
            if all(short_conditions):
                strength = await self._calculate_signal_strength(
                    data, momentum_data, trend_data, volume_data, 'SHORT'
                )
                
                if strength > self.parameters['min_signal_strength']:
                    signals.append(Signal(
                        symbol=data.index[0],
                        direction='SHORT',
                        strength=strength,
                        strategy=self.name,
                        price=price,
                        indicators={
                            'rsi': momentum_data['RSI'].iloc[-1],
                            'macd': momentum_data['MACD'].iloc[-1],
                            'adx': trend_data['ADX'].iloc[-1],
                            'volume_ratio': volume_data['Volume_Ratio'].iloc[-1]
                        }
                    ))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating momentum signals: {e}")
            return []

    async def _calculate_signal_strength(
        self,
        data: pd.DataFrame,
        momentum_data: pd.DataFrame,
        trend_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        direction: str
    ) -> float:
        """Calculate signal strength based on multiple factors"""
        try:
            # RSI factor
            rsi = momentum_data['RSI'].iloc[-1]
            if direction == 'LONG':
                rsi_factor = (self.parameters['rsi_oversold'] - rsi) / self.parameters['rsi_oversold']
            else:
                rsi_factor = (rsi - self.parameters['rsi_overbought']) / (100 - self.parameters['rsi_overbought'])
            
            # MACD factor
            macd_diff = abs(
                momentum_data['MACD'].iloc[-1] - momentum_data['MACD_Signal'].iloc[-1]
            )
            macd_factor = min(macd_diff / 2, 1.0)
            
            # ADX factor
            adx = trend_data['ADX'].iloc[-1]
            adx_factor = min(adx / 50, 1.0)
            
            # Volume factor
            volume_ratio = volume_data['Volume_Ratio'].iloc[-1]
            volume_factor = min(volume_ratio / 3.0, 1.0)
            
            # Trend alignment
            sma_50 = trend_data['SMA_50'].iloc[-1]
            price = data['Close'].iloc[-1]
            trend_factor = 1.0 if (
                (direction == 'LONG' and price > sma_50) or
                (direction == 'SHORT' and price < sma_50)
            ) else 0.5
            
            # Combine factors with weights
            strength = (
                rsi_factor * 0.3 +
                macd_factor * 0.2 +
                adx_factor * 0.2 +
                volume_factor * 0.15 +
                trend_factor * 0.15
            )
            
            return min(max(strength, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {e}")
            return 0.0

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy"""
    
    def __init__(self):
        super().__init__("MeanReversion")
        
        # Strategy-specific parameters
        self.parameters.update({
            'lookback': 20,
            'std_dev': 2.0,
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14,
            'min_reversal_rate': 0.6,
            'volume_ma_period': 20,
            'min_zscore': 2.0,
            'max_holding_period': 5  # days
        })

    async def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate mean reversion signals"""
        try:
            signals = []
            
            # Calculate indicators in parallel
            indicator_tasks = [
                self._calculate_mean_reversion_indicators(data),
                self._calculate_volatility_indicators(data),
                self._calculate_volume_indicators(data)
            ]
            
            mr_data, vol_data, volume_data = await asyncio.gather(*indicator_tasks)
            
            # Latest values
            latest = data.index[-1]
            price = data['Close'].iloc[-1]
            zscore = mr_data['ZScore'].iloc[-1]
            
            # Check long (oversold) signals
            long_conditions = [
                zscore < -self.parameters['min_zscore'],
                price < vol_data['BB_Lower'].iloc[-1],
                mr_data['RSI'].iloc[-1] < 30,
                volume_data['Volume_Ratio'].iloc[-1] > 1.2,
                mr_data['Reversal_Rate'].iloc[-1] > self.parameters['min_reversal_rate']
            ]
            
            if all(long_conditions):
                strength = await self._calculate_reversion_probability(
                    data, mr_data, vol_data, volume_data, 'LONG'
                )
                
                if strength > self.parameters['min_signal_strength']:
                    signals.append(Signal(
                        symbol=data.index[0],
                        direction='LONG',
                        strength=strength,
                        strategy=self.name,
                        price=price,
                        indicators={
                            'zscore': zscore,
                            'bb_position': (price - vol_data['BB_Lower'].iloc[-1]) /
                                         (vol_data['BB_Upper'].iloc[-1] - vol_data['BB_Lower'].iloc[-1]),
                            'rsi': mr_data['RSI'].iloc[-1],
                            'reversal_rate': mr_data['Reversal_Rate'].iloc[-1]
                        }
                    ))
            
            # Check short (overbought) signals
            short_conditions = [
                zscore > self.parameters['min_zscore'],
                price > vol_data['BB_Upper'].iloc[-1],
                mr_data['RSI'].iloc[-1] > 70,
                volume_data['Volume_Ratio'].iloc[-1] > 1.2,
                mr_data['Reversal_Rate'].iloc[-1] > self.parameters['min_reversal_rate']
            ]
            
            if all(short_conditions):
                strength = await self._calculate_reversion_probability(
                    data, mr_data, vol_data, volume_data, 'SHORT'
                )
                
                if strength > self.parameters['min_signal_strength']:
                    signals.append(Signal(
                        symbol=data.index[0],
                        direction='SHORT',
                        strength=strength,
                        strategy=self.name,
                        price=price,
                        indicators={
                            'zscore': zscore,
                            'bb_position': (price - vol_data['BB_Lower'].iloc[-1]) /
                                         (vol_data['BB_Upper'].iloc[-1] - vol_data['BB_Lower'].iloc[-1]),
                            'rsi': mr_data['RSI'].iloc[-1],
                            'reversal_rate': mr_data['Reversal_Rate'].iloc[-1]
                        }
                    ))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating mean reversion signals: {e}")
            return []

    async def _calculate_mean_reversion_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion indicators"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._calculate_mr_indicators_sync,
                data
            )
        except Exception as e:
            self.logger.error(f"Error calculating mean reversion indicators: {e}")
            return pd.DataFrame()

    def _calculate_mr_indicators_sync(self, data: pd.DataFrame) -> pd.DataFrame:
        """Synchronous mean reversion calculations"""
        df = pd.DataFrame(index=data.index)
        
        # Z-score
        rolling_mean = data['Close'].rolling(self.parameters['lookback']).mean()
        rolling_std = data['Close'].rolling(self.parameters['lookback']).std()
        df['ZScore'] = (data['Close'] - rolling_mean) / rolling_std
        
        # RSI
        df['RSI'] = talib.RSI(data['Close'], timeperiod=self.parameters['rsi_period'])
        
        # Reversal rate calculation
        returns = data['Close'].pct_change()
        signs = returns.apply(lambda x: 1 if x > 0 else -1)
        sign_changes = (signs != signs.shift(1)).astype(int)
        df['Reversal_Rate'] = sign_changes.rolling(self.parameters['lookback']).mean()
        
        return df

    async def _calculate_reversion_probability(
        self,
        data: pd.DataFrame,
        mr_data: pd.DataFrame,
        vol_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        direction: str
    ) -> float:
        """Calculate probability of mean reversion"""
        try:
            # Z-score factor
            zscore = abs(mr_data['ZScore'].iloc[-1])
            zscore_factor = min(zscore / self.parameters['min_zscore'], 1.0)
            
            # Bollinger Band factor
            price = data['Close'].iloc[-1]
            bb_range = vol_data['BB_Upper'].iloc[-1] - vol_data['BB_Lower'].iloc[-1]
            if direction == 'LONG':
                bb_factor = (vol_data['BB_Middle'].iloc[-1] - price) / bb_range
            else:
                bb_factor = (price - vol_data['BB_Middle'].iloc[-1]) / bb_range
            bb_factor = min(max(bb_factor, 0), 1)
            
            # RSI factor
            rsi = mr_data['RSI'].iloc[-1]
            if direction == 'LONG':
                rsi_factor = (30 - rsi) / 30 if rsi < 30 else 0
            else:
                rsi_factor = (rsi - 70) / 30 if rsi > 70 else 0
            
            # Volume factor
            volume_factor = min(volume_data['Volume_Ratio'].iloc[-1] / 2.0, 1.0)
            
            # Reversal factor
            reversal_factor = min(
                mr_data['Reversal_Rate'].iloc[-1] / self.parameters['min_reversal_rate'],
                1.0
            )
            
            # Historical success rate
            success_rate = await self._calculate_historical_success(
                data, direction, self.parameters['lookback']
            )
            
            # Combine factors
            strength = (
                zscore_factor * 0.25 +
                bb_factor * 0.25 +
                rsi_factor * 0.15 +
                volume_factor * 0.15 +
                reversal_factor * 0.1 +
                success_rate * 0.1
            )
            
            return min(max(strength, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating reversion probability: {e}")
            return 0.0
        
class BreakoutStrategy(BaseStrategy):
    """Breakout trading strategy"""
    
    def __init__(self):
        super().__init__("Breakout")
        
        # Strategy-specific parameters
        self.parameters.update({
            'breakout_window': 20,
            'volume_ma_period': 20,
            'atr_period': 14,
            'min_touches': 3,
            'price_tolerance': 0.001,  # 0.1%
            'volume_threshold': 1.5,
            'min_consolidation_days': 5,
            'max_volatility': 0.02,  # 2%
            'min_level_distance': 0.02  # 2%
        })

class StrategyManager:
    def __init__(self):
        self.logger = self._setup_logging()
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_strategies: Set[str] = set()
        self._initialized = False
        self._running = False
        self._tasks = set()

    def _setup_logging(self) -> logging.Logger:
        """Setup strategy manager logging"""
        logger = logging.getLogger('strategy_manager')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(CONFIG.LOG_DIR / 'strategy_manager.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    async def initialize(self):
        """Initialize strategy manager"""
        try:
            if self._initialized:
                return
                
            self.logger.info("Initializing strategy manager...")
            
            # Initialize default strategies
            self.strategies = {
                'momentum': MomentumStrategy(),
                'mean_reversion': MeanReversionStrategy(),
                'breakout': BreakoutStrategy()
            }
            
            # Initialize each strategy
            for strategy in self.strategies.values():
                await strategy.initialize()
                
            # Subscribe to events
            event_manager.subscribe(EventType.MARKET_UPDATE, self._handle_market_update)
            
            self._initialized = True
            self._running = True
            self.logger.info("Strategy manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing strategy manager: {e}\n{traceback.format_exc()}")
            raise

    def add_strategy(self, strategy: BaseStrategy):
        """Add new strategy"""
        try:
            strategy_name = strategy.name.lower()
            if strategy_name not in self.strategies:
                self.strategies[strategy_name] = strategy
                self.logger.info(f"Added strategy: {strategy.name}")
                
        except Exception as e:
            self.logger.error(f"Error adding strategy: {e}")

    def remove_strategy(self, strategy_name: str):
        """Remove strategy"""
        try:
            strategy_name = strategy_name.lower()
            if strategy_name in self.strategies:
                del self.strategies[strategy_name]
                self.active_strategies.discard(strategy_name)
                self.logger.info(f"Removed strategy: {strategy_name}")
                
        except Exception as e:
            self.logger.error(f"Error removing strategy: {e}")

    async def activate_strategy(self, strategy_name: str):
        """Activate strategy"""
        try:
            strategy_name = strategy_name.lower()
            if strategy_name in self.strategies:
                strategy = self.strategies[strategy_name]
                
                # Initialize if needed
                if not strategy._initialized:
                    await strategy.initialize()
                    
                strategy.state.active = True
                self.active_strategies.add(strategy_name)
                self.logger.info(f"Activated strategy: {strategy_name}")
                
                # Publish event
                await event_manager.publish(Event(
                    type=EventType.SYSTEM,
                    data={
                        'action': 'STRATEGY_ACTIVATED',
                        'strategy': strategy_name
                    }
                ))
                
        except Exception as e:
            self.logger.error(f"Error activating strategy: {e}")

    async def deactivate_strategy(self, strategy_name: str):
        """Deactivate strategy"""
        try:
            strategy_name = strategy_name.lower()
            if strategy_name in self.active_strategies:
                strategy = self.strategies[strategy_name]
                strategy.state.active = False
                self.active_strategies.discard(strategy_name)
                
                # Save strategy state
                await strategy.save_state()
                
                self.logger.info(f"Deactivated strategy: {strategy_name}")
                
                # Publish event
                await event_manager.publish(Event(
                    type=EventType.SYSTEM,
                    data={
                        'action': 'STRATEGY_DEACTIVATED',
                        'strategy': strategy_name
                    }
                ))
                
        except Exception as e:
            self.logger.error(f"Error deactivating strategy: {e}")

    async def _handle_market_update(self, event: Event):
        """Handle market update events"""
        try:
            symbol = event.data['symbol']
            
            # Get market data
            data = await market_data.get_stock_data(symbol)
            if data is None:
                return
                
            # Process active strategies
            active_strategies = [
                self.strategies[name] for name in self.active_strategies
            ]
            
            # Generate signals concurrently
            tasks = [
                strategy.generate_signals(data)
                for strategy in active_strategies
            ]
            
            signal_lists = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process signals from each strategy
            for strategy, signals in zip(active_strategies, signal_lists):
                if isinstance(signals, Exception):
                    self.logger.error(f"Error in strategy {strategy.name}: {signals}")
                    continue
                    
                for signal in signals:
                    await strategy._process_signal(signal)
                    
        except Exception as e:
            self.logger.error(f"Error handling market update: {e}")

    async def get_active_signals(self) -> Dict[str, List[Signal]]:
        """Get all active signals from all strategies"""
        try:
            signals = {}
            for strategy_name in self.active_strategies:
                strategy = self.strategies[strategy_name]
                active_signals = [
                    signal for signal in strategy.state.signals
                    if signal.is_active()
                ]
                if active_signals:
                    signals[strategy_name] = active_signals
            return signals
        except Exception as e:
            self.logger.error(f"Error getting active signals: {e}")
            return {}

    async def get_strategy_performance(self) -> Dict[str, Dict]:
        """Get performance metrics for all strategies"""
        try:
            performance = {}
            for strategy_name, strategy in self.strategies.items():
                performance[strategy_name] = strategy.state.performance
            return performance
        except Exception as e:
            self.logger.error(f"Error getting strategy performance: {e}")
            return {}

    async def shutdown(self):
        """Shutdown strategy manager"""
        try:
            self.logger.info("Shutting down strategy manager...")
            
            self._running = False
            
            # Save all strategy states
            for strategy in self.strategies.values():
                await strategy.save_state()
                
            # Cancel all tasks
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
            self.logger.info("Strategy manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Create singleton instance
strategy_manager = StrategyManager()
