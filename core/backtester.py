# core/backtester.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Set, Tuple
from dataclasses import dataclass, field, asdict  # Added asdict here
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback


from .market_data import market_data
from .strategy import strategy_manager, Signal
from .events import event_manager, Event, EventType
from .config import CONFIG

@dataclass
class BacktestPosition:
    symbol: str
    quantity: int
    entry_price: float
    side: str
    entry_time: datetime
    strategy: str
    stop_loss: float = 0.0
    take_profit: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.stop_loss == 0.0:
            self.stop_loss = self.entry_price * (
                1 - CONFIG.STOP_LOSS_PCT if self.side == 'LONG'
                else 1 + CONFIG.STOP_LOSS_PCT
            )
        if self.take_profit == 0.0:
            self.take_profit = self.entry_price * (
                1 + CONFIG.TAKE_PROFIT_PCT if self.side == 'LONG'
                else 1 - CONFIG.TAKE_PROFIT_PCT
            )

@dataclass
class BacktestTrade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    side: str
    quantity: int
    entry_price: float
    exit_price: float
    pnl: float
    strategy: str
    return_pct: float
    commission: float
    slippage: float
    metadata: Dict = field(default_factory=dict)

@dataclass
class BacktestMetrics:
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    risk_reward_ratio: float = 0.0
    recovery_factor: float = 0.0
    calmar_ratio: float = 0.0
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trade_duration: timedelta = field(default_factory=lambda: timedelta())
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_mae: float = 0.0  # Maximum Adverse Excursion
    avg_mfe: float = 0.0  # Maximum Favorable Excursion

@dataclass
class BacktestResult:
    initial_capital: float
    final_capital: float
    trades: List[BacktestTrade]
    positions: Dict[str, BacktestPosition]
    equity_curve: pd.DataFrame
    drawdown_curve: pd.DataFrame
    returns: pd.Series
    metrics: BacktestMetrics
    strategy_metrics: Dict[str, BacktestMetrics]
    start_date: datetime
    end_date: datetime
    parameters: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'trades': [asdict(t) for t in self.trades],
            'positions': {k: asdict(v) for k, v in self.positions.items()},
            'equity_curve': self.equity_curve.to_dict(),
            'drawdown_curve': self.drawdown_curve.to_dict(),
            'returns': self.returns.to_dict(),
            'metrics': asdict(self.metrics),
            'strategy_metrics': {k: asdict(v) for k, v in self.strategy_metrics.items()},
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'parameters': self.parameters,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BacktestResult':
        data['trades'] = [BacktestTrade(**t) for t in data['trades']]
        data['positions'] = {k: BacktestPosition(**v) for k, v in data['positions'].items()}
        data['equity_curve'] = pd.DataFrame(data['equity_curve'])
        data['drawdown_curve'] = pd.DataFrame(data['drawdown_curve'])
        data['returns'] = pd.Series(data['returns'])
        data['metrics'] = BacktestMetrics(**data['metrics'])
        data['strategy_metrics'] = {k: BacktestMetrics(**v) for k, v in data['strategy_metrics'].items()}
        data['start_date'] = datetime.fromisoformat(data['start_date'])
        data['end_date'] = datetime.fromisoformat(data['end_date'])
        return cls(**data)

class Backtester:
    def __init__(self):
        self.logger = self._setup_logging()
        self.executor = ThreadPoolExecutor(max_workers=CONFIG.MAX_WORKERS)
        self.results_cache: Dict[str, BacktestResult] = {}
        self._running = False
        self._tasks = set()

    def _setup_logging(self) -> logging.Logger:
        """Setup backtester logging"""
        logger = logging.getLogger('backtester')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(CONFIG.LOG_DIR / 'backtester.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    async def run(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        strategies: Optional[List[str]] = None,
        initial_capital: Optional[float] = None,
        parameters: Optional[Dict] = None
    ) -> Optional[BacktestResult]:
        """Run backtest simulation"""
        try:
            self.logger.info(f"Starting backtest for {len(symbols)} symbols...")
            
            # Setup initial parameters
            initial_capital = initial_capital or CONFIG.CAPITAL
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
            
            # Get historical data
            data = await self._get_historical_data(symbols, start, end)
            if not data:
                raise ValueError("No data available for backtesting")
                
            # Initialize backtest state
            state = {
                'capital': initial_capital,
                'positions': {},
                'trades': [],
                'equity_curve': [],
                'current_time': start
            }
            
            # Run simulation
            await self._run_simulation(state, data, strategies, parameters)
            
            # Calculate results
            result = await self._calculate_results(
                state,
                start,
                end,
                strategies,
                parameters
            )
            
            # Cache results
            cache_key = f"{','.join(symbols)}_{start_date}_{end_date}"
            self.results_cache[cache_key] = result
            
            # Save results
            await self._save_results(result)
            
            self.logger.info("Backtest completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}\n{traceback.format_exc()}")
            return None

    async def _get_historical_data(
        self,
        symbols: List[str],
        start: pd.Timestamp,
        end: pd.Timestamp
    ) -> Optional[pd.DataFrame]:
        """Get historical data for all symbols"""
        try:
            all_data = {}
            
            # Fetch data for each symbol concurrently
            async def fetch_symbol_data(symbol: str):
                data = await market_data.get_stock_data(
                    symbol,
                    interval='1d',
                    start_date=start.strftime('%Y-%m-%d'),
                    end_date=end.strftime('%Y-%m-%d')
                )
                if data is not None:
                    return symbol, data
                return None
                
            tasks = [fetch_symbol_data(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks)
            
            # Combine results
            for result in results:
                if result:
                    symbol, data = result
                    all_data[symbol] = data
                    
            if not all_data:
                return None
                
            # Align data indices
            return self._align_data(all_data)
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return None

    def _align_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Align data from different symbols"""
        try:
            aligned = {}
            for symbol, df in data.items():
                for column in df.columns:
                    aligned[f"{symbol}_{column}"] = df[column]
            return pd.DataFrame(aligned)
        except Exception as e:
            self.logger.error(f"Error aligning data: {e}")
            return pd.DataFrame()

    async def _run_simulation(
        self,
        state: Dict,
        data: pd.DataFrame,
        strategies: Optional[List[str]],
        parameters: Optional[Dict]
    ):
        """Run backtest simulation"""
        try:
            for timestamp in data.index:
                state['current_time'] = timestamp
                
                # Update positions
                await self._update_positions(state, data, timestamp)
                
                # Generate signals
                signals = await self._generate_signals(
                    data.loc[:timestamp],
                    strategies,
                    parameters
                )
                
                # Process signals
                await self._process_signals(state, signals, data, timestamp)
                
                # Record equity
                equity = self._calculate_equity(state, data, timestamp)
                state['equity_curve'].append({
                    'timestamp': timestamp,
                    'equity': equity
                })
                
        except Exception as e:
            self.logger.error(f"Error in simulation: {e}\n{traceback.format_exc()}")
            raise

    async def _update_positions(
        self,
        state: Dict,
        data: pd.DataFrame,
        timestamp: pd.Timestamp
    ):
        """Update all positions"""
        try:
            for symbol, position in list(state['positions'].items()):
                current_price = data.loc[timestamp, f"{symbol}_Close"]
                
                # Update unrealized P&L
                position_value = position.quantity * position.entry_price
                current_value = position.quantity * current_price
                
                if position.side == 'LONG':
                    position.unrealized_pnl = current_value - position_value
                else:
                    position.unrealized_pnl = position_value - current_value
                    
                # Check stop loss
                if (position.side == 'LONG' and current_price <= position.stop_loss) or \
                   (position.side == 'SHORT' and current_price >= position.stop_loss):
                    await self._close_position(
                        state,
                        symbol,
                        current_price,
                        timestamp,
                        'Stop Loss'
                    )
                    continue
                    
                # Check take profit
                if (position.side == 'LONG' and current_price >= position.take_profit) or \
                   (position.side == 'SHORT' and current_price <= position.take_profit):
                    await self._close_position(
                        state,
                        symbol,
                        current_price,
                        timestamp,
                        'Take Profit'
                    )
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")

    async def _close_position(
        self,
        state: Dict,
        symbol: str,
        price: float,
        timestamp: pd.Timestamp,
        reason: str
    ):
        """Close a position"""
        try:
            position = state['positions'][symbol]
            
            # Calculate P&L
            if position.side == 'LONG':
                pnl = (price - position.entry_price) * position.quantity
            else:
                pnl = (position.entry_price - price) * position.quantity
                
            # Calculate commission and slippage
            commission = (position.entry_price + price) * position.quantity * CONFIG.COMMISSION_RATE
            slippage = abs(price - position.entry_price) * CONFIG.SLIPPAGE * position.quantity
            
            # Record trade
            trade = BacktestTrade(
                symbol=symbol,
                entry_time=position.entry_time,
                exit_time=timestamp,
                side=position.side,
                quantity=position.quantity,
                entry_price=position.entry_price,
                exit_price=price,
                pnl=pnl - commission - slippage,
                strategy=position.strategy,
                return_pct=(pnl / (position.entry_price * position.quantity)) * 100,
                commission=commission,
                slippage=slippage,
                metadata={
                    'reason': reason,
                    **position.metadata
                }
            )
            
            state['trades'].append(trade)
            
            # Update capital
            state['capital'] += pnl - commission - slippage
            
            # Remove position
            del state['positions'][symbol]
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            
    async def _generate_signals(
        self,
        data: pd.DataFrame,
        strategies: Optional[List[str]],
        parameters: Optional[Dict]
    ) -> List[Signal]:
        """Generate trading signals"""
        try:
            all_signals = []
            
            # Apply parameters if provided
            if parameters:
                await self._apply_strategy_parameters(parameters)
            
            # Generate signals for each symbol
            for column in data.columns:
                if '_Close' in column:
                    symbol = column.replace('_Close', '')
                    
                    # Create symbol-specific dataframe
                    symbol_data = pd.DataFrame({
                        'Open': data[f"{symbol}_Open"],
                        'High': data[f"{symbol}_High"],
                        'Low': data[f"{symbol}_Low"],
                        'Close': data[f"{symbol}_Close"],
                        'Volume': data[f"{symbol}_Volume"]
                    })
                    
                    # Generate signals from each strategy
                    for strategy_name, strategy in strategy_manager.strategies.items():
                        if strategies and strategy_name not in strategies:
                            continue
                            
                        signals = await strategy.generate_signals(symbol_data)
                        all_signals.extend(signals)
                        
            return all_signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return []

    async def _process_signals(
        self,
        state: Dict,
        signals: List[Signal],
        data: pd.DataFrame,
        timestamp: pd.Timestamp
    ):
        """Process trading signals"""
        try:
            for signal in signals:
                # Skip if already in position
                if signal.symbol in state['positions']:
                    continue
                    
                # Calculate position size
                position_size = self._calculate_position_size(
                    state['capital'],
                    signal.price,
                    signal.strength
                )
                
                if position_size <= 0:
                    continue
                    
                # Open position
                position = BacktestPosition(
                    symbol=signal.symbol,
                    quantity=position_size,
                    entry_price=signal.price,
                    side='LONG' if signal.direction == 'LONG' else 'SHORT',
                    entry_time=timestamp,
                    strategy=signal.strategy,
                    metadata=signal.metadata
                )
                
                # Update capital
                commission = signal.price * position_size * CONFIG.COMMISSION_RATE
                slippage = signal.price * CONFIG.SLIPPAGE * position_size
                
                if position.side == 'LONG':
                    state['capital'] -= (signal.price * position_size + commission + slippage)
                else:
                    state['capital'] += (signal.price * position_size - commission - slippage)
                    
                state['positions'][signal.symbol] = position
                
        except Exception as e:
            self.logger.error(f"Error processing signals: {e}")

    def _calculate_position_size(
        self,
        capital: float,
        price: float,
        signal_strength: float
    ) -> int:
        """Calculate position size"""
        try:
            # Calculate maximum position value
            max_position = capital * CONFIG.risk.max_position_size
            
            # Adjust for signal strength
            position_value = max_position * signal_strength
            
            # Convert to number of shares
            shares = int(position_value / price)
            
            # Check minimum trade amount
            if shares * price < CONFIG.MIN_TRADE_AMOUNT:
                return 0
                
            return shares
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0

    def _calculate_equity(
        self,
        state: Dict,
        data: pd.DataFrame,
        timestamp: pd.Timestamp
    ) -> float:
        """Calculate current equity"""
        try:
            equity = state['capital']
            
            # Add unrealized P&L from open positions
            for symbol, position in state['positions'].items():
                current_price = data.loc[timestamp, f"{symbol}_Close"]
                position_value = position.quantity * position.entry_price
                current_value = position.quantity * current_price
                
                if position.side == 'LONG':
                    equity += (current_value - position_value)
                else:
                    equity += (position_value - current_value)
                    
            return equity
            
        except Exception as e:
            self.logger.error(f"Error calculating equity: {e}")
            return state['capital']

    async def _calculate_results(
        self,
        state: Dict,
        start: pd.Timestamp,
        end: pd.Timestamp,
        strategies: Optional[List[str]],
        parameters: Optional[Dict]
    ) -> BacktestResult:
        """Calculate backtest results"""
        try:
            # Create equity curve
            equity_curve = pd.DataFrame(state['equity_curve'])
            equity_curve.set_index('timestamp', inplace=True)
            
            # Calculate returns
            returns = equity_curve['equity'].pct_change()
            
            # Calculate drawdown
            drawdown_curve = self._calculate_drawdown(equity_curve['equity'])
            
            # Calculate strategy-specific metrics
            strategy_metrics = await self._calculate_strategy_metrics(
                state['trades'],
                strategies
            )
            
            # Calculate overall metrics
            metrics = await self._calculate_performance_metrics(
                returns,
                state['trades'],
                equity_curve,
                drawdown_curve,
                state['capital']
            )
            
            return BacktestResult(
                initial_capital=CONFIG.CAPITAL,
                final_capital=state['capital'],
                trades=state['trades'],
                positions=state['positions'],
                equity_curve=equity_curve,
                drawdown_curve=drawdown_curve,
                returns=returns,
                metrics=metrics,
                strategy_metrics=strategy_metrics,
                start_date=start,
                end_date=end,
                parameters=parameters or {}
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating results: {e}")
            raise

    def _calculate_drawdown(self, equity: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        try:
            peak = equity.expanding().max()
            drawdown = (equity - peak) / peak
            return drawdown
        except Exception as e:
            self.logger.error(f"Error calculating drawdown: {e}")
            return pd.Series()

    async def _calculate_performance_metrics(
        self,
        returns: pd.Series,
        trades: List[BacktestTrade],
        equity_curve: pd.DataFrame,
        drawdown_curve: pd.Series,
        final_capital: float
    ) -> BacktestMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            
            # Basic metrics
            total_return = (final_capital / CONFIG.CAPITAL - 1) * 100
            days = (returns.index[-1] - returns.index[0]).days
            annual_return = ((1 + total_return/100) ** (365/days) - 1) * 100
            
            # Risk metrics
            volatility = returns.std() * np.sqrt(252)
            downside_returns = returns[returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252)
            
            sharpe_ratio = (annual_return - CONFIG.risk.risk_free_rate) / volatility \
                if volatility != 0 else 0
            sortino_ratio = (annual_return - CONFIG.risk.risk_free_rate) / downside_vol \
                if downside_vol != 0 else 0
                
            max_drawdown = drawdown_curve.min() * 100
            
            # Trade metrics
            win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0
            
            profit_factor = sum(t.pnl for t in winning_trades) / abs(sum(t.pnl for t in losing_trades)) \
                if losing_trades else float('inf')
                
            risk_reward_ratio = avg_win / avg_loss if avg_loss != 0 else float('inf')
            
            # Streak analysis
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            current_streak = 0
            
            for trade in trades:
                if trade.pnl > 0:
                    if current_streak > 0:
                        current_streak += 1
                    else:
                        current_streak = 1
                    max_consecutive_wins = max(max_consecutive_wins, current_streak)
                else:
                    if current_streak < 0:
                        current_streak -= 1
                    else:
                        current_streak = -1
                    max_consecutive_losses = max(max_consecutive_losses, abs(current_streak))
                    
            return BacktestMetrics(
                total_return=total_return,
                annual_return=annual_return,
                volatility=volatility * 100,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                risk_reward_ratio=risk_reward_ratio,
                trades_count=len(trades),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                max_consecutive_wins=max_consecutive_wins,
                max_consecutive_losses=max_consecutive_losses,
                largest_win=max(t.pnl for t in trades) if trades else 0,
                largest_loss=min(t.pnl for t in trades) if trades else 0
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return BacktestMetrics()
        
    async def optimize_parameters(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        param_ranges: Dict[str, List],
        optimization_target: str = 'sharpe_ratio',
        strategy: Optional[str] = None
    ) -> Dict:
        """Optimize strategy parameters"""
        try:
            self.logger.info("Starting parameter optimization...")
            
            best_metrics = None
            best_params = None
            best_value = float('-inf')
            results = []
            
            # Generate parameter combinations
            param_combinations = self._generate_param_combinations(param_ranges)
            total_combinations = len(param_combinations)
            
            # Run backtests with different parameters concurrently
            async def optimize_single(params: Dict) -> Tuple[Dict, float, Dict]:
                try:
                    # Run backtest with parameters
                    result = await self.run(
                        symbols=symbols,
                        start_date=start_date,
                        end_date=end_date,
                        strategies=[strategy] if strategy else None,
                        parameters=params
                    )
                    
                    if result is None:
                        return params, float('-inf'), {}
                        
                    # Get optimization metric
                    metric_value = getattr(result.metrics, optimization_target)
                    return params, metric_value, result.metrics.__dict__
                    
                except Exception as e:
                    self.logger.error(f"Error in optimization run: {e}")
                    return params, float('-inf'), {}
            
            # Process combinations in batches
            batch_size = min(CONFIG.MAX_WORKERS, total_combinations)
            for i in range(0, total_combinations, batch_size):
                batch = param_combinations[i:i + batch_size]
                batch_results = await asyncio.gather(*[
                    optimize_single(params) for params in batch
                ])
                
                # Process batch results
                for params, value, metrics in batch_results:
                    results.append({
                        'parameters': params,
                        'value': value,
                        'metrics': metrics
                    })
                    
                    if value > best_value:
                        best_value = value
                        best_params = params
                        best_metrics = metrics
                        
                # Log progress
                progress = min((i + batch_size) / total_combinations * 100, 100)
                self.logger.info(f"Optimization progress: {progress:.1f}%")
            
            return {
                'best_parameters': best_params,
                'best_value': best_value,
                'best_metrics': best_metrics,
                'all_results': results
            }
            
        except Exception as e:
            self.logger.error(f"Error in parameter optimization: {e}\n{traceback.format_exc()}")
            return None

    def _generate_param_combinations(self, param_ranges: Dict[str, List]) -> List[Dict]:
        """Generate all parameter combinations"""
        try:
            import itertools
            
            keys = param_ranges.keys()
            values = param_ranges.values()
            combinations = list(itertools.product(*values))
            
            return [dict(zip(keys, combo)) for combo in combinations]
            
        except Exception as e:
            self.logger.error(f"Error generating parameter combinations: {e}")
            return []

    async def _save_results(self, result: BacktestResult):
        """Save backtest results"""
        try:
            # Create results directory if needed
            results_dir = CONFIG.BACKTEST_DIR
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.json"
            filepath = results_dir / filename
            
            # Save results
            with open(filepath, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
                
            self.logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

    def plot_results(self, result: BacktestResult) -> go.Figure:
        """Plot backtest results"""
        try:
            # Create figure with secondary y-axis
            fig = make_subplots(
                rows=4,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(
                    'Portfolio Value',
                    'Drawdown',
                    'Daily Returns',
                    'Trade P&L'
                ),
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )
            
            # Plot equity curve
            fig.add_trace(
                go.Scatter(
                    x=result.equity_curve.index,
                    y=result.equity_curve['equity'],
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Plot drawdown
            fig.add_trace(
                go.Scatter(
                    x=result.drawdown_curve.index,
                    y=result.drawdown_curve * 100,
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color='red', width=1)
                ),
                row=2, col=1
            )
            
            # Plot daily returns
            fig.add_trace(
                go.Bar(
                    x=result.returns.index,
                    y=result.returns * 100,
                    name='Daily Returns'
                ),
                row=3, col=1
            )
            
            # Plot trade P&L
            trade_dates = [t.exit_time for t in result.trades]
            trade_pnl = [t.pnl for t in result.trades]
            colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnl]
            
            fig.add_trace(
                go.Bar(
                    x=trade_dates,
                    y=trade_pnl,
                    name='Trade P&L',
                    marker_color=colors
                ),
                row=4, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=1000,
                showlegend=True,
                title_text="Backtest Results",
                margin=dict(t=30)
            )
            
            # Add metrics annotations
            metrics_text = (
                f"Total Return: {result.metrics.total_return:.2f}%<br>"
                f"Annual Return: {result.metrics.annual_return:.2f}%<br>"
                f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}<br>"
                f"Max Drawdown: {result.metrics.max_drawdown:.2f}%<br>"
                f"Win Rate: {result.metrics.win_rate:.2f}%<br>"
                f"Profit Factor: {result.metrics.profit_factor:.2f}<br>"
                f"Total Trades: {result.metrics.trades_count}"
            )
            
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=1.02,
                y=0.98,
                text=metrics_text,
                showarrow=False,
                font=dict(size=10),
                align="left"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting results: {e}")
            return None

    async def load_results(self, filename: str) -> Optional[BacktestResult]:
        """Load saved backtest results"""
        try:
            filepath = CONFIG.BACKTEST_DIR / filename
            if not filepath.exists():
                self.logger.error(f"Results file not found: {filepath}")
                return None
                
            with open(filepath) as f:
                data = json.load(f)
                
            return BacktestResult.from_dict(data)
            
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            return None

    async def shutdown(self):
        """Shutdown backtester"""
        try:
            self.logger.info("Shutting down backtester...")
            
            # Cancel all tasks
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
            # Clean up executor
            self.executor.shutdown(wait=True)
            
            # Clear cache
            self.results_cache.clear()
            
            self.logger.info("Backtester shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Create singleton instance
backtester = Backtester()
