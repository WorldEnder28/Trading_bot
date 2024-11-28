# core/portfolio.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import traceback
import uuid

from .market_data import market_data
from .risk import risk_manager
from .events import event_manager, Event, EventType
from .config import CONFIG

@dataclass
class Position:
    symbol: str
    quantity: int
    average_price: float
    side: str  # 'LONG' or 'SHORT'
    entry_time: datetime
    last_update: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    strategy: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'average_price': self.average_price,
            'side': self.side,
            'entry_time': self.entry_time.isoformat(),
            'last_update': self.last_update.isoformat(),
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'strategy': self.strategy,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Position':
        data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        data['last_update'] = datetime.fromisoformat(data['last_update'])
        return cls(**data)

@dataclass
class Trade:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    side: str = ""  # 'LONG' or 'SHORT'
    quantity: int = 0
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    pnl: float = 0.0
    strategy: Optional[str] = None
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    @property
    def duration(self) -> timedelta:
        if self.exit_time:
            return self.exit_time - self.entry_time
        return datetime.now() - self.entry_time
    
    @property
    def return_pct(self) -> float:
        if not self.exit_price:
            return 0.0
        cost = self.quantity * self.entry_price
        if cost == 0:
            return 0.0
        return (self.pnl / cost) * 100
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'side': self.side,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'strategy': self.strategy,
            'commission': self.commission,
            'slippage': self.slippage,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Trade':
        data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        if data['exit_time']:
            data['exit_time'] = datetime.fromisoformat(data['exit_time'])
        return cls(**data)

class PortfolioManager:
    def __init__(self):
        self.logger = self._setup_logging()
        self.executor = ThreadPoolExecutor(max_workers=CONFIG.MAX_WORKERS)
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.cash: float = CONFIG.CAPITAL
        self.initial_capital: float = CONFIG.CAPITAL
        self.equity_curve: List[Dict] = []
        self._initialized = False
        self._running = False
        self._tasks = set()
        self.update_interval = 60  # seconds
        self._last_update = datetime.min

    def _setup_logging(self) -> logging.Logger:
        """Setup portfolio logging"""
        logger = logging.getLogger('portfolio_manager')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(CONFIG.LOG_DIR / 'portfolio.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    async def initialize(self):
        """Initialize portfolio manager"""
        try:
            if self._initialized:
                return
                
            self.logger.info("Initializing portfolio manager...")
            
            # Load saved state
            await self._load_state()
            
            # Subscribe to events
            event_manager.subscribe(EventType.MARKET_UPDATE, self._handle_market_update)
            event_manager.subscribe(EventType.TRADE, self._handle_trade)
            
            # Start portfolio monitoring
            self._running = True
            self._tasks.add(
                asyncio.create_task(self._monitor_portfolio())
            )
            
            self._initialized = True
            self.logger.info("Portfolio manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing portfolio manager: {e}\n{traceback.format_exc()}")
            raise
        
    async def _load_state(self):
        """Load portfolio state from disk"""
        try:
            state_file = CONFIG.DATA_DIR / 'portfolio_state.json'
            if state_file.exists():
                with open(state_file) as f:
                    state_data = json.load(f)
                    
                # Restore positions
                self.positions = {
                    symbol: Position.from_dict(pos_data)
                    for symbol, pos_data in state_data.get('positions', {}).items()
                }
                
                # Restore trades
                self.trades = [
                    Trade.from_dict(trade_data)
                    for trade_data in state_data.get('trades', [])
                ]
                
                # Restore other state
                self.cash = state_data.get('cash', self.cash)
                self.initial_capital = state_data.get('initial_capital', self.initial_capital)
                self.equity_curve = state_data.get('equity_curve', [])
                
                self.logger.info(f"Loaded portfolio state with {len(self.positions)} positions and {len(self.trades)} trades")
                
        except Exception as e:
            self.logger.error(f"Error loading portfolio state: {e}\n{traceback.format_exc()}")

    async def _save_state(self):
        """Save portfolio state to disk"""
        try:
            state_data = {
                'positions': {
                    symbol: position.to_dict()
                    for symbol, position in self.positions.items()
                },
                'trades': [trade.to_dict() for trade in self.trades],
                'cash': self.cash,
                'initial_capital': self.initial_capital,
                'equity_curve': self.equity_curve
            }
            
            state_file = CONFIG.DATA_DIR / 'portfolio_state.json'
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
            self.logger.debug("Portfolio state saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving portfolio state: {e}")

    async def _monitor_portfolio(self):
        """Monitor portfolio positions and metrics"""
        try:
            self.logger.info("Starting portfolio monitoring...")
            
            while self._running:
                try:
                    now = datetime.now()
                    if (now - self._last_update).total_seconds() >= self.update_interval:
                        # Update positions
                        await self._update_positions()
                        
                        # Update equity curve
                        await self._update_equity_curve()
                        
                        # Check risk limits
                        await self._check_risk_limits()
                        
                        # Save state
                        await self._save_state()
                        
                        self._last_update = now
                        
                    await asyncio.sleep(1)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in portfolio monitoring: {e}\n{traceback.format_exc()}")
                    await asyncio.sleep(10)  # Back off on error
                    
        except Exception as e:
            self.logger.error(f"Fatal error in portfolio monitoring: {e}\n{traceback.format_exc()}")
        finally:
            self.logger.info("Portfolio monitoring stopped")

    async def _update_positions(self):
        """Update all position values and metrics"""
        try:
            for symbol, position in list(self.positions.items()):
                # Get current price
                data = await market_data.get_stock_data(symbol)
                if data is None or data.empty:
                    continue
                    
                current_price = data['Close'].iloc[-1]
                
                # Update unrealized P&L
                position_value = position.quantity * position.average_price
                current_value = position.quantity * current_price
                
                if position.side == 'LONG':
                    position.unrealized_pnl = current_value - position_value
                else:  # SHORT
                    position.unrealized_pnl = position_value - current_value
                    
                position.last_update = datetime.now()
                
                # Check exit conditions
                await self._check_exit_conditions(symbol, position, current_price)
                
                # Publish position update event
                await event_manager.publish(Event(
                    type=EventType.POSITION,
                    data={
                        'symbol': symbol,
                        'position': position.to_dict()
                    }
                ))
                
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")

    async def _check_exit_conditions(
        self,
        symbol: str,
        position: Position,
        current_price: float
    ):
        """Check if position should be exited"""
        try:
            # Stop loss check
            if position.stop_loss:
                if (position.side == 'LONG' and current_price <= position.stop_loss) or \
                   (position.side == 'SHORT' and current_price >= position.stop_loss):
                    await self.close_position(symbol, reason="Stop Loss")
                    return
                    
            # Take profit check
            if position.take_profit:
                if (position.side == 'LONG' and current_price >= position.take_profit) or \
                   (position.side == 'SHORT' and current_price <= position.take_profit):
                    await self.close_position(symbol, reason="Take Profit")
                    return
                    
            # Risk limit check
            position_value = position.quantity * current_price
            position_limit = await risk_manager._calculate_position_limit(symbol)
            
            if position_value > position_limit:
                await self.close_position(symbol, reason="Risk Limit")
                return
                
            # Time-based exit (end of day)
            if datetime.now().time() > CONFIG.MARKET_END:
                await self.close_position(symbol, reason="End of Day")
                return
                
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")

    async def _update_equity_curve(self):
        """Update equity curve data"""
        try:
            total_value = self.cash + sum(
                pos.unrealized_pnl for pos in self.positions.values()
            )
            
            self.equity_curve.append({
                'timestamp': datetime.now(),
                'cash': self.cash,
                'positions_value': total_value - self.cash,
                'total_value': total_value,
                'returns': (total_value / self.initial_capital) - 1
            })
            
            # Keep last 1000 points
            if len(self.equity_curve) > 1000:
                self.equity_curve = self.equity_curve[-1000:]
                
        except Exception as e:
            self.logger.error(f"Error updating equity curve: {e}")

    async def _check_risk_limits(self):
        """Check portfolio against risk limits"""
        try:
            # Calculate portfolio metrics
            portfolio_var = await risk_manager._calculate_portfolio_var()
            max_var = CONFIG.CAPITAL * CONFIG.risk.max_portfolio_risk
            
            if portfolio_var > max_var:
                self.logger.warning(
                    f"Portfolio VaR ({portfolio_var:.2f}) exceeds limit ({max_var:.2f})"
                )
                await self._reduce_risk()
                
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")

    async def _reduce_risk(self):
        """Reduce portfolio risk by trimming positions"""
        try:
            # Sort positions by risk contribution
            position_risks = []
            for symbol, position in self.positions.items():
                risk_metrics = await risk_manager._calculate_position_metrics(symbol)
                if risk_metrics:
                    position_risks.append({
                        'symbol': symbol,
                        'position': position,
                        'var': risk_metrics.var_95,
                        'beta': risk_metrics.beta,
                        'volatility': risk_metrics.volatility
                    })
                    
            # Sort by risk metrics
            position_risks.sort(
                key=lambda x: (x['var'], x['beta'], x['volatility']),
                reverse=True
            )
            
            # Reduce highest risk positions
            for risk_item in position_risks[:2]:  # Reduce top 2 risky positions
                position = risk_item['position']
                reduction_qty = int(position.quantity * 0.3)  # Reduce by 30%
                
                if reduction_qty > 0:
                    await self.close_position(
                        risk_item['symbol'],
                        quantity=reduction_qty,
                        reason="Risk Reduction"
                    )
                    
        except Exception as e:
            self.logger.error(f"Error reducing portfolio risk: {e}")
            
            
    async def open_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        side: str,
        strategy: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Open a new position"""
        try:
            # Validate parameters
            if not await self._validate_position_params(
                symbol, quantity, price, side
            ):
                return False
                
            # Calculate position cost
            position_cost = quantity * price
            commission = position_cost * CONFIG.COMMISSION_RATE
            slippage = price * CONFIG.SLIPPAGE * quantity
            total_cost = position_cost + commission + slippage
            
            # Check if we have enough cash
            if side == 'LONG' and total_cost > self.cash:
                self.logger.warning(f"Insufficient cash for position in {symbol}")
                return False
                
            # Calculate stop loss and take profit
            stop_loss = price * (1 - CONFIG.STOP_LOSS_PCT) if side == 'LONG' \
                       else price * (1 + CONFIG.STOP_LOSS_PCT)
            take_profit = price * (1 + CONFIG.TAKE_PROFIT_PCT) if side == 'LONG' \
                         else price * (1 - CONFIG.TAKE_PROFIT_PCT)
                
            # Create position
            position = Position(
                symbol=symbol,
                quantity=quantity,
                average_price=price,
                side=side,
                entry_time=datetime.now(),
                last_update=datetime.now(),
                strategy=strategy,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata or {}
            )
            
            # Update cash
            if side == 'LONG':
                self.cash -= total_cost
            else:
                self.cash += (position_cost - commission - slippage)
                
            # Add position
            self.positions[symbol] = position
            
            # Create trade record
            trade = Trade(
                symbol=symbol,
                entry_time=position.entry_time,
                side=side,
                quantity=quantity,
                entry_price=price,
                strategy=strategy,
                commission=commission,
                slippage=slippage,
                metadata=metadata or {}
            )
            
            self.trades.append(trade)
            
            # Publish events
            await event_manager.publish(Event(
                type=EventType.POSITION,
                data={
                    'action': 'OPEN',
                    'symbol': symbol,
                    'position': position.to_dict()
                }
            ))
            
            await event_manager.publish(Event(
                type=EventType.TRADE,
                data={
                    'action': 'ENTRY',
                    'trade': trade.to_dict()
                }
            ))
            
            self.logger.info(f"""
            Opened {side} position:
            Symbol: {symbol}
            Quantity: {quantity}
            Price: {price:.2f}
            Cost: {total_cost:.2f}
            Stop Loss: {stop_loss:.2f}
            Take Profit: {take_profit:.2f}
            """)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error opening position: {e}\n{traceback.format_exc()}")
            return False

    async def close_position(
        self,
        symbol: str,
        quantity: Optional[int] = None,
        reason: Optional[str] = None
    ) -> bool:
        """Close position"""
        try:
            if symbol not in self.positions:
                return False
                
            position = self.positions[symbol]
            
            # Get current price
            data = await market_data.get_stock_data(symbol)
            if data is None or data.empty:
                return False
                
            exit_price = data['Close'].iloc[-1]
            
            # Handle partial close
            close_quantity = quantity or position.quantity
            if close_quantity > position.quantity:
                return False
                
            # Calculate P&L
            position_value = close_quantity * position.entry_price
            exit_value = close_quantity * exit_price
            
            if position.side == 'LONG':
                pnl = exit_value - position_value
            else:
                pnl = position_value - exit_value
                
            # Calculate commission and slippage
            commission = exit_value * CONFIG.COMMISSION_RATE
            slippage = exit_price * CONFIG.SLIPPAGE * close_quantity
            net_pnl = pnl - commission - slippage
            
            # Update cash
            if position.side == 'LONG':
                self.cash += (exit_value - commission - slippage)
            else:
                self.cash -= (exit_value + commission + slippage)
                
            # Update or remove position
            if close_quantity == position.quantity:
                position_closed = True
                del self.positions[symbol]
            else:
                position_closed = False
                position.quantity -= close_quantity
                position.realized_pnl += net_pnl
                position.last_update = datetime.now()
                
            # Update trade record
            for trade in reversed(self.trades):
                if trade.symbol == symbol and trade.exit_time is None:
                    trade.exit_time = datetime.now()
                    trade.exit_price = exit_price
                    trade.pnl = net_pnl
                    trade.metadata['close_reason'] = reason
                    break
                    
            # Publish events
            await event_manager.publish(Event(
                type=EventType.POSITION,
                data={
                    'action': 'CLOSE' if position_closed else 'PARTIAL_CLOSE',
                    'symbol': symbol,
                    'quantity': close_quantity,
                    'price': exit_price,
                    'pnl': net_pnl,
                    'reason': reason
                }
            ))
            
            await event_manager.publish(Event(
                type=EventType.TRADE,
                data={
                    'action': 'EXIT',
                    'symbol': symbol,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'reason': reason
                }
            ))
            
            self.logger.info(f"""
            Closed {"" if position_closed else "partial "}position:
            Symbol: {symbol}
            Quantity: {close_quantity}
            Exit Price: {exit_price:.2f}
            P&L: {net_pnl:.2f}
            Reason: {reason}
            """)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}\n{traceback.format_exc()}")
            return False

    async def _validate_position_params(
        self,
        symbol: str,
        quantity: int,
        price: float,
        side: str
    ) -> bool:
        """Validate position parameters"""
        try:
            # Basic parameter checks
            if quantity <= 0 or price <= 0:
                return False
                
            if side not in ['LONG', 'SHORT']:
                return False
                
            # Check if symbol already has position
            if symbol in self.positions:
                return False
                
            # Check maximum positions
            if len(self.positions) >= CONFIG.MAX_POSITIONS:
                return False
                
            # Check minimum trade amount
            if quantity * price < CONFIG.MIN_TRADE_AMOUNT:
                return False
                
            # Check risk limits
            position_value = quantity * price
            position_limit = await risk_manager._calculate_position_limit(symbol)
            
            if position_value > position_limit:
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating position parameters: {e}")
            return False
        
    async def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        try:
            total_value = self.cash
            
            for symbol, position in self.positions.items():
                data = await market_data.get_stock_data(symbol)
                if data is not None and not data.empty:
                    current_price = data['Close'].iloc[-1]
                    position_value = position.quantity * current_price
                    total_value += position_value
                    
            return total_value
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {e}")
            return self.cash

    async def get_portfolio_metrics(self) -> Dict:
        """Calculate portfolio performance metrics"""
        try:
            if not self.equity_curve:
                return {}
                
            # Convert equity curve to DataFrame
            df = pd.DataFrame(self.equity_curve)
            df.set_index('timestamp', inplace=True)
            
            # Calculate returns
            returns = df['total_value'].pct_change().dropna()
            
            # Calculate metrics
            total_return = (df['total_value'].iloc[-1] / self.initial_capital - 1) * 100
            
            # Annualized return
            days = (df.index[-1] - df.index[0]).days
            annual_return = ((1 + total_return/100) ** (365/days) - 1) * 100 if days > 0 else 0
            
            # Risk-adjusted returns
            volatility = returns.std() * np.sqrt(252)
            downside_returns = returns[returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            sharpe_ratio = (annual_return - CONFIG.risk.risk_free_rate) / (volatility * 100) \
                if volatility != 0 else 0
                
            sortino_ratio = (annual_return - CONFIG.risk.risk_free_rate) / (downside_vol * 100) \
                if downside_vol != 0 else 0
                
            # Drawdown analysis
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min()) * 100
            
            # Trade analysis
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl < 0]
            
            win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
            
            gross_profit = sum(t.pnl for t in winning_trades)
            gross_loss = abs(sum(t.pnl for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility * 100,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(self.trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'avg_trade_return': np.mean([t.return_pct for t in self.trades if t.return_pct]) \
                    if self.trades else 0,
                'best_trade': max((t.pnl for t in self.trades), default=0),
                'worst_trade': min((t.pnl for t in self.trades), default=0)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}\n{traceback.format_exc()}")
            return {}

    async def get_position_weights(self) -> Dict[str, float]:
        """Get position weights in portfolio"""
        try:
            total_value = await self.get_portfolio_value()
            if total_value <= 0:
                return {}
                
            weights = {}
            for symbol, position in self.positions.items():
                data = await market_data.get_stock_data(symbol)
                if data is not None and not data.empty:
                    current_price = data['Close'].iloc[-1]
                    position_value = position.quantity * current_price
                    weights[symbol] = position_value / total_value
                    
            return weights
            
        except Exception as e:
            self.logger.error(f"Error calculating position weights: {e}")
            return {}

    async def get_daily_pnl(self) -> float:
        """Get P&L for current trading day"""
        try:
            today = datetime.now().date()
            daily_pnl = 0.0
            
            # Add realized P&L from closed trades today
            daily_pnl += sum(
                trade.pnl for trade in self.trades
                if trade.exit_time and trade.exit_time.date() == today
            )
            
            # Add unrealized P&L from open positions
            daily_pnl += sum(pos.unrealized_pnl for pos in self.positions.values())
            
            return daily_pnl
            
        except Exception as e:
            self.logger.error(f"Error calculating daily P&L: {e}")
            return 0.0

    async def get_position_summary(self) -> List[Dict]:
        """Get summary of all positions"""
        try:
            summary = []
            
            for symbol, position in self.positions.items():
                data = await market_data.get_stock_data(symbol)
                if data is not None and not data.empty:
                    current_price = data['Close'].iloc[-1]
                    position_value = position.quantity * current_price
                    unrealized_return = (position.unrealized_pnl / 
                                       (position.quantity * position.average_price)) * 100
                    
                    summary.append({
                        'symbol': symbol,
                        'side': position.side,
                        'quantity': position.quantity,
                        'average_price': position.average_price,
                        'current_price': current_price,
                        'market_value': position_value,
                        'unrealized_pnl': position.unrealized_pnl,
                        'realized_pnl': position.realized_pnl,
                        'return_pct': unrealized_return,
                        'duration': (datetime.now() - position.entry_time).days,
                        'strategy': position.strategy
                    })
                    
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting position summary: {e}")
            return []

    async def shutdown(self):
        """Shutdown portfolio manager"""
        try:
            self.logger.info("Shutting down portfolio manager...")
            
            # Stop monitoring
            self._running = False
            
            # Cancel all tasks
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
            # Save final state
            await self._save_state()
            
            # Clean up executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("Portfolio manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Create singleton instance
portfolio_manager = PortfolioManager()
