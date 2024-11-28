# core/market_data.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import aiohttp
import asyncio
import logging
from functools import lru_cache
import yfinance as yf
import talib
from concurrent.futures import ThreadPoolExecutor
import json
import traceback
from pathlib import Path
from aiohttp import ClientTimeout, ClientSession
import pytz
import numpy as np
from collections import deque
import pickle

from .events import event_manager, Event, EventType
from .config import CONFIG

@dataclass
class MarketDepthLevel:
    price: float
    quantity: int
    orders: int = 0
    exchange: str = "NSE"
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class MarketDepth:
    symbol: str
    timestamp: datetime
    bid: List[MarketDepthLevel]
    ask: List[MarketDepthLevel]
    total_bid_qty: int = 0
    total_ask_qty: int = 0
    spread: float = 0.0
    num_bids: int = 0
    num_asks: int = 0
    exchange: str = "NSE"
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'bid': [asdict(level) for level in self.bid],
            'ask': [asdict(level) for level in self.ask],
            'total_bid_qty': self.total_bid_qty,
            'total_ask_qty': self.total_ask_qty,
            'spread': self.spread,
            'num_bids': self.num_bids,
            'num_asks': self.num_asks,
            'exchange': self.exchange
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MarketDepth':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['bid'] = [MarketDepthLevel(**level) for level in data['bid']]
        data['ask'] = [MarketDepthLevel(**level) for level in data['ask']]
        return cls(**data)

@dataclass
class StockInfo:
    symbol: str
    name: str = ""
    sector: str = "Unknown"
    industry: str = "Unknown"
    market_cap: float = 0.0
    float_shares: int = 0
    avg_volume: int = 0
    beta: float = 0.0
    pe_ratio: float = 0.0
    dividend_yield: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    exchange: str = "NSE"
    currency: str = "INR"
    is_index: bool = False
    lot_size: int = 1
    tick_size: float = 0.05
    margin_required: float = 1.0
    trading_hours: Dict[str, datetime] = field(default_factory=dict)
    circuit_limits: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

@dataclass
class CacheEntry:
    data: pd.DataFrame
    timestamp: datetime
    source: str
    metadata: Dict = field(default_factory=dict)

class DataCache:
    """Advanced caching system for market data"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.access_count: Dict[str, int] = {}
        self.last_access: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get data from cache"""
        async with self._lock:
            if key not in self.cache:
                return None
                
            entry = self.cache[key]
            self.access_count[key] = self.access_count.get(key, 0) + 1
            self.last_access[key] = datetime.now()
            
            # Check if entry is expired
            if (datetime.now() - entry.timestamp) > timedelta(minutes=CONFIG.data.cache_expiry):
                await self.remove(key)
                return None
                
            return entry

    async def set(self, key: str, entry: CacheEntry):
        """Add data to cache"""
        async with self._lock:
            # Enforce cache size limit
            if len(self.cache) >= self.max_size:
                await self._evict()
                
            self.cache[key] = entry
            self.access_count[key] = 1
            self.last_access[key] = datetime.now()

    async def remove(self, key: str):
        """Remove data from cache"""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                self.access_count.pop(key, None)
                self.last_access.pop(key, None)

    async def _evict(self):
        """Evict entries based on LRU and access frequency"""
        if not self.cache:
            return
            
        # Calculate scores for each entry
        scores = {}
        now = datetime.now()
        
        for key in self.cache:
            time_score = (now - self.last_access[key]).total_seconds()
            freq_score = 1 / (self.access_count[key] + 1)
            scores[key] = time_score * freq_score
            
        # Remove entry with highest score (least recently/frequently used)
        key_to_remove = max(scores.items(), key=lambda x: x[1])[0]
        await self.remove(key_to_remove)

    async def save_to_disk(self, path: Path):
        """Save cache to disk"""
        async with self._lock:
            with open(path, 'wb') as f:
                pickle.dump({
                    'cache': self.cache,
                    'access_count': self.access_count,
                    'last_access': self.last_access
                }, f)

    async def load_from_disk(self, path: Path):
        """Load cache from disk"""
        async with self._lock:
            if path.exists():
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    self.cache = data['cache']
                    self.access_count = data['access_count']
                    self.last_access = data['last_access']

class MarketDataManager:
    def __init__(self):
        self.logger = self._setup_logging()
        self.executor = ThreadPoolExecutor(max_workers=CONFIG.MAX_WORKERS)
        self.session: Optional[ClientSession] = None
        self.cache = DataCache(max_size=CONFIG.CACHE_SIZE)
        self.stock_info: Dict[str, StockInfo] = {}
        self.symbols: Set[str] = set()
        self.depth_cache: Dict[str, MarketDepth] = {}
        self.last_update: Dict[str, datetime] = {}
        self.update_intervals: Dict[str, int] = {}
        self._initialized = False
        self._running = False
        self._tasks = set()
        self.rate_limiter = asyncio.Semaphore(CONFIG.data.rate_limit)
        self.timezone = pytz.timezone('Asia/Kolkata')
        
    async def initialize(self):
        """Initialize market data manager"""
        try:
            if self._initialized:
                return
                
            self.logger.info("Initializing market data manager...")
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=ClientTimeout(total=30),
                headers={
                    'User-Agent': 'Mozilla/5.0',
                    'Accept': 'application/json',
                    'Connection': 'keep-alive'
                }
            )
            
            # Load cache from disk
            await self.cache.load_from_disk(CONFIG.DATA_DIR / 'market_data_cache.pkl')
            
            # Load stock info
            await self._load_stock_info()
            
            # Initialize symbols
            await self._initialize_symbols()
            
            # Start background tasks
            self._running = True
            self._tasks.add(
                asyncio.create_task(self._cache_cleanup_task())
            )
            self._tasks.add(
                asyncio.create_task(self._data_update_task())
            )
            
            self._initialized = True
            self.logger.info("Market data manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing market data manager: {e}\n{traceback.format_exc()}")
            raise

    async def _load_stock_info(self):
        """Load stock information"""
        try:
            info_file = CONFIG.DATA_DIR / 'stock_info.json'
            if info_file.exists():
                with open(info_file) as f:
                    data = json.load(f)
                    self.stock_info = {
                        symbol: StockInfo(**info) 
                        for symbol, info in data.items()
                    }
            
            # Update stock info in background
            self._tasks.add(
                asyncio.create_task(self._update_stock_info())
            )
            
        except Exception as e:
            self.logger.error(f"Error loading stock info: {e}")

    async def _initialize_symbols(self):
        """Initialize available trading symbols"""
        try:
            # Try loading from config first
            config_symbols = getattr(CONFIG, 'ALL_STOCKS', {}).get('symbols', [])
            if config_symbols:
                self.symbols.update(config_symbols)
                return
            
            # Load from exchange
            nse_symbols = await self._fetch_nse_symbols()
            if nse_symbols:
                self.symbols.update(nse_symbols)
                return
            
            # Fallback to default symbols
            default_symbols = [
                'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS',
                'ICICIBANK.NS', 'HDFC.NS', 'ITC.NS', 'KOTAKBANK.NS',
                'LT.NS', 'HINDUNILVR.NS'
            ]
            self.symbols.update(default_symbols)
            
        except Exception as e:
            self.logger.error(f"Error initializing symbols: {e}")

    async def get_stock_data(
        self,
        symbol: str,
        interval: str = '1d',
        lookback: int = 100,
        include_indicators: bool = True,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """Get stock data with failover mechanisms"""
        try:
            # Check cache if enabled
            if use_cache:
                cache_key = f"{symbol}_{interval}_{lookback}"
                cached_data = await self.cache.get(cache_key)
                if cached_data is not None:
                    return cached_data.data
            
            # Apply rate limiting
            async with self.rate_limiter:
                # Try primary data source first
                data = await self._fetch_primary_source(symbol, interval, lookback)
                
                # Try backup sources if primary fails
                if data is None or data.empty:
                    for source in CONFIG.data.backup_sources:
                        data = await self._fetch_backup_source(
                            symbol, interval, lookback, source
                        )
                        if data is not None and not data.empty:
                            break
                
                if data is not None and not data.empty:
                    # Clean and validate data
                    data = self._clean_data(data)
                    if not self._validate_data(data):
                        return None
                    
                    # Add indicators if requested
                    if include_indicators:
                        data = await self._add_indicators(data)
                    
                    # Update cache
                    if use_cache:
                        await self.cache.set(
                            cache_key,
                            CacheEntry(
                                data=data,
                                timestamp=datetime.now(),
                                source='primary'
                            )
                        )
                    
                    # Update last update time
                    self.last_update[symbol] = datetime.now()
                    
                    return data
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}\n{traceback.format_exc()}")
            return None

    async def _fetch_primary_source(
        self,
        symbol: str,
        interval: str,
        lookback: int
    ) -> Optional[pd.DataFrame]:
        """Fetch data from primary source"""
        try:
            if CONFIG.data.primary_source == 'YAHOO':
                return await self._fetch_yf_data(symbol, interval, lookback)
            elif CONFIG.data.primary_source == 'NSE':
                return await self._fetch_nse_data(symbol, interval, lookback)
            else:
                self.logger.error(f"Unknown primary source: {CONFIG.data.primary_source}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching from primary source: {e}")
            return None

    async def _fetch_yf_data(
        self,
        symbol: str,
        interval: str,
        lookback: int
    ) -> Optional[pd.DataFrame]:
        """Fetch data from yfinance"""
        try:
            # Convert interval to yfinance format
            yf_interval = self._convert_interval(interval)
            
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback)
            
            # Fetch data using ThreadPoolExecutor
            return await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._yf_download,
                symbol,
                start_date,
                end_date,
                yf_interval
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching from Yahoo Finance: {e}")
            return None

    def _yf_download(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Download data from yfinance (sync)"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if not data.empty:
                data.index = pd.to_datetime(data.index)
                return data
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error in yfinance download: {e}")
            return None
        
    async def _fetch_nse_data(
        self,
        symbol: str,
        interval: str,
        lookback: int
    ) -> Optional[pd.DataFrame]:
        """Fetch data from NSE"""
        try:
            # Remove .NS suffix if present
            clean_symbol = symbol.replace('.NS', '')
            
            # Build URL based on interval
            if interval == '1d':
                url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={clean_symbol}"
            else:
                url = f"https://www.nseindia.com/api/chart-dataapi?symbol={clean_symbol}&interval={interval}"
            
            # Fetch data with session
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_nse_data(data, interval)
                else:
                    self.logger.warning(f"NSE request failed with status {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error fetching from NSE: {e}")
            return None

    def _parse_nse_data(self, data: Dict, interval: str) -> Optional[pd.DataFrame]:
        """Parse NSE JSON data into DataFrame"""
        try:
            if interval == '1d':
                # Parse daily data
                records = data.get('data', [])
                df = pd.DataFrame(records)
                df['Date'] = pd.to_datetime(df['CH_TIMESTAMP'])
                df = df.set_index('Date')
                
                # Rename columns
                df = df.rename(columns={
                    'CH_OPENING_PRICE': 'Open',
                    'CH_TRADE_HIGH_PRICE': 'High',
                    'CH_TRADE_LOW_PRICE': 'Low',
                    'CH_CLOSING_PRICE': 'Close',
                    'CH_TOT_TRADED_QTY': 'Volume'
                })
                
            else:
                # Parse intraday data
                candles = data.get('candles', [])
                df = pd.DataFrame(candles, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            
            # Convert to numeric
            price_columns = ['Open', 'High', 'Low', 'Close']
            df[price_columns] = df[price_columns].apply(pd.to_numeric)
            df['Volume'] = pd.to_numeric(df['Volume'])
            
            return df.sort_index()
            
        except Exception as e:
            self.logger.error(f"Error parsing NSE data: {e}")
            return None

    async def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        try:
            # Create tasks for parallel calculation
            tasks = [
                self._calculate_trend_indicators(data),
                self._calculate_momentum_indicators(data),
                self._calculate_volatility_indicators(data),
                self._calculate_volume_indicators(data)
            ]
            
            # Execute tasks concurrently
            results = await asyncio.gather(*tasks)
            
            # Combine results
            df = data.copy()
            for result in results:
                df = pd.concat([df, result], axis=1)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding indicators: {e}")
            return data

    async def _calculate_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend indicators"""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._trend_indicators_sync,
            data
        )

    def _trend_indicators_sync(self, data: pd.DataFrame) -> pd.DataFrame:
        """Synchronous trend indicator calculations"""
        try:
            df = pd.DataFrame(index=data.index)
            
            # Moving Averages
            for period in [20, 50, 200]:
                df[f'SMA_{period}'] = talib.SMA(data['Close'], timeperiod=period)
                df[f'EMA_{period}'] = talib.EMA(data['Close'], timeperiod=period)
            
            # ADX
            df['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'])
            df['PLUS_DI'] = talib.PLUS_DI(data['High'], data['Low'], data['Close'])
            df['MINUS_DI'] = talib.MINUS_DI(data['High'], data['Low'], data['Close'])
            
            # MACD
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(data['Close'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating trend indicators: {e}")
            return pd.DataFrame()

    async def _calculate_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._momentum_indicators_sync,
            data
        )

    def _momentum_indicators_sync(self, data: pd.DataFrame) -> pd.DataFrame:
        """Synchronous momentum indicator calculations"""
        try:
            df = pd.DataFrame(index=data.index)
            
            # RSI
            df['RSI'] = talib.RSI(data['Close'])
            
            # Stochastic
            df['STOCH_K'], df['STOCH_D'] = talib.STOCH(
                data['High'],
                data['Low'],
                data['Close']
            )
            
            # ROC
            df['ROC'] = talib.ROC(data['Close'])
            
            # MFI
            df['MFI'] = talib.MFI(
                data['High'],
                data['Low'],
                data['Close'],
                data['Volume']
            )
            
            # Williams %R
            df['WILLR'] = talib.WILLR(
                data['High'],
                data['Low'],
                data['Close']
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum indicators: {e}")
            return pd.DataFrame()

    async def _calculate_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators"""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._volatility_indicators_sync,
            data
        )

    def _volatility_indicators_sync(self, data: pd.DataFrame) -> pd.DataFrame:
        """Synchronous volatility indicator calculations"""
        try:
            df = pd.DataFrame(index=data.index)
            
            # Bollinger Bands
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(
                data['Close'],
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2
            )
            
            # ATR
            df['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'])
            
            # Standard Deviation
            df['STD'] = data['Close'].rolling(window=20).std()
            
            # Historical Volatility
            returns = np.log(data['Close'] / data['Close'].shift(1))
            df['HV'] = returns.rolling(window=20).std() * np.sqrt(252)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility indicators: {e}")
            return pd.DataFrame()
        
    async def get_market_depth(self, symbol: str) -> Optional[MarketDepth]:
        """Get market depth data"""
        try:
            # Check cache first
            if symbol in self.depth_cache:
                depth = self.depth_cache[symbol]
                if (datetime.now() - depth.timestamp).total_seconds() < 5:  # 5-second cache
                    return depth
            
            # Apply rate limiting
            async with self.rate_limiter:
                # Fetch new depth data
                clean_symbol = symbol.replace('.NS', '')
                url = f"https://www.nseindia.com/api/quote-equity?symbol={clean_symbol}&section=trade_info"
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        depth = self._parse_market_depth(clean_symbol, data)
                        
                        if depth:
                            self.depth_cache[symbol] = depth
                            return depth
                    else:
                        self.logger.warning(f"Depth request failed with status {response.status}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching market depth for {symbol}: {e}")
            return None

    def _parse_market_depth(self, symbol: str, data: Dict) -> Optional[MarketDepth]:
        """Parse market depth data"""
        try:
            if 'marketDeptOrderBook' not in data:
                return None
                
            book = data['marketDeptOrderBook']
            
            # Process bids
            bids = []
            total_bid_qty = 0
            for bid in book.get('bids', []):
                level = MarketDepthLevel(
                    price=float(bid['price']),
                    quantity=int(bid['quantity']),
                    orders=int(bid['numberOfOrders'])
                )
                bids.append(level)
                total_bid_qty += level.quantity
                
            # Process asks
            asks = []
            total_ask_qty = 0
            for ask in book.get('asks', []):
                level = MarketDepthLevel(
                    price=float(ask['price']),
                    quantity=int(ask['quantity']),
                    orders=int(ask['numberOfOrders'])
                )
                asks.append(level)
                total_ask_qty += level.quantity
            
            # Calculate spread
            spread = asks[0].price - bids[0].price if bids and asks else 0.0
            
            return MarketDepth(
                symbol=symbol,
                timestamp=datetime.now(),
                bid=bids,
                ask=asks,
                total_bid_qty=total_bid_qty,
                total_ask_qty=total_ask_qty,
                spread=spread,
                num_bids=len(bids),
                num_asks=len(asks)
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing market depth: {e}")
            return None

    async def get_stock_info(self, symbol: str) -> Optional[StockInfo]:
        """Get stock information and metadata"""
        try:
            # Check cache first
            if symbol in self.stock_info:
                info = self.stock_info[symbol]
                if (datetime.now() - info.last_updated).days < 1:  # 1-day cache
                    return info
            
            # Fetch new info using ThreadPoolExecutor
            info = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._fetch_stock_info,
                symbol
            )
            
            if info:
                self.stock_info[symbol] = info
                await self._save_stock_info()
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error fetching stock info: {e}")
            return None

    def _fetch_stock_info(self, symbol: str) -> Optional[StockInfo]:
        """Fetch stock info from yfinance (sync)"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return StockInfo(
                symbol=symbol,
                name=info.get('longName', ''),
                sector=info.get('sector', 'Unknown'),
                industry=info.get('industry', 'Unknown'),
                market_cap=info.get('marketCap', 0),
                float_shares=info.get('floatShares', 0),
                avg_volume=info.get('averageVolume', 0),
                beta=info.get('beta', 0),
                pe_ratio=info.get('forwardPE', 0),
                dividend_yield=info.get('dividendYield', 0),
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching stock info: {e}")
            return None

    async def _save_stock_info(self):
        """Save stock information to disk"""
        try:
            info_file = CONFIG.DATA_DIR / 'stock_info.json'
            with open(info_file, 'w') as f:
                json.dump(
                    {symbol: asdict(info) for symbol, info in self.stock_info.items()},
                    f,
                    indent=2,
                    default=str
                )
        except Exception as e:
            self.logger.error(f"Error saving stock info: {e}")

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            now = datetime.now(self.timezone)
            
            # Check time
            current_time = now.time()
            if not CONFIG.MARKET_START <= current_time <= CONFIG.MARKET_END:
                return False
            
            # Check weekday
            if now.weekday() in [5, 6]:  # Saturday=5, Sunday=6
                return False
            
            # Check holidays
            if str(now.date()) in CONFIG.market.holidays:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return False

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data"""
        try:
            # Remove duplicates
            data = data[~data.index.duplicated(keep='last')]
            
            # Sort by index
            data.sort_index(inplace=True)
            
            # Forward fill missing values
            data.fillna(method='ffill', inplace=True)
            
            # Convert column names
            data.columns = [col.title() for col in data.columns]
            
            # Ensure required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    data[col] = 0
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error cleaning data: {e}")
            return data

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data quality"""
        try:
            # Check for minimum size
            if len(data) < CONFIG.data.min_history:
                return False
            
            # Check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                return False
            
            # Check for invalid values
            price_columns = ['Open', 'High', 'Low', 'Close']
            if (data[price_columns] <= 0).any().any():
                return False
            
            # Check high/low consistency
            if (data['High'] < data['Low']).any():
                return False
            
            # Check for excessive gaps
            returns = data['Close'].pct_change()
            if abs(returns).max() > 0.5:  # 50% price change
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data: {e}")
            return False

    async def shutdown(self):
        """Shutdown market data manager"""
        try:
            self.logger.info("Shutting down market data manager...")
            
            self._running = False
            
            # Cancel all tasks
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
            # Save cache
            await self.cache.save_to_disk(CONFIG.DATA_DIR / 'market_data_cache.pkl')
            
            # Close session
            if self.session and not self.session.closed:
                await self.session.close()
            
            # Clean up executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("Market data manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Create singleton instance
market_data = MarketDataManager()
