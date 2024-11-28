# core/execution.py
import asyncio
from typing import Dict, List, Optional, Union, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import numpy as np
from enum import Enum
import logging
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
import traceback

from .market_data import market_data
from .events import event_manager, Event, EventType
from .config import CONFIG

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"

class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class TimeInForce(Enum):
    DAY = "DAY"
    GTC = "GOOD_TILL_CANCELLED"
    IOC = "IMMEDIATE_OR_CANCEL"
    FOK = "FILL_OR_KILL"

@dataclass
class Fill:
    quantity: int
    price: float
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0
    liquidity: str = "TAKER"  # TAKER or MAKER
    venue: str = "PRIMARY"
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'commission': self.commission,
            'slippage': self.slippage,
            'liquidity': self.liquidity,
            'venue': self.venue,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Fill':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    strategy_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    trailing_percent: Optional[float] = None
    min_quantity: Optional[int] = None
    max_show_quantity: Optional[int] = None
    expire_time: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.side, str):
            self.side = OrderSide(self.side)
        if isinstance(self.order_type, str):
            self.order_type = OrderType(self.order_type)
        if isinstance(self.time_in_force, str):
            self.time_in_force = TimeInForce(self.time_in_force)

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate order request parameters"""
        try:
            if self.quantity <= 0:
                return False, "Invalid quantity"

            if self.order_type != OrderType.MARKET and self.price is None:
                return False, "Price required for non-market orders"

            if self.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LIMIT] and self.stop_price is None:
                return False, "Stop price required for stop orders"

            if self.trailing_percent is not None and (self.trailing_percent <= 0 or self.trailing_percent > 100):
                return False, "Invalid trailing percentage"

            if self.min_quantity is not None and (self.min_quantity <= 0 or self.min_quantity > self.quantity):
                return False, "Invalid minimum quantity"

            if self.max_show_quantity is not None and (self.max_show_quantity <= 0 or self.max_show_quantity > self.quantity):
                return False, "Invalid max show quantity"

            if self.expire_time is not None and self.expire_time <= datetime.now():
                return False, "Invalid expiry time"

            return True, None

        except Exception as e:
            return False, f"Validation error: {str(e)}"

@dataclass
class Order:
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request: OrderRequest = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_price: float = 0.0
    fills: List[Fill] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    cancelled_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    parent_orders: List[str] = field(default_factory=list)
    child_orders: List[str] = field(default_factory=list)
    current_stop_price: Optional[float] = None
    total_commission: float = 0.0
    total_slippage: float = 0.0

    def remaining_quantity(self) -> int:
        return self.request.quantity - self.filled_quantity

    def is_active(self) -> bool:
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED
        ]

    def is_complete(self) -> bool:
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]

    def add_fill(self, fill: Fill) -> None:
        self.fills.append(fill)
        self.filled_quantity += fill.quantity
        self.total_commission += fill.commission
        self.total_slippage += fill.slippage
        
        # Update average price
        total_value = sum(f.price * f.quantity for f in self.fills)
        self.average_price = total_value / self.filled_quantity if self.filled_quantity > 0 else 0.0
        
        # Update status
        if self.filled_quantity == self.request.quantity:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
            
        self.updated_at = datetime.now()

    def update_trailing_stop(self, current_price: float) -> None:
        """Update trailing stop price"""
        if (self.request.order_type == OrderType.TRAILING_STOP and 
            self.request.trailing_percent is not None):
            
            if self.request.side == OrderSide.SELL:
                new_stop = current_price * (1 - self.request.trailing_percent / 100)
                if self.current_stop_price is None or new_stop > self.current_stop_price:
                    self.current_stop_price = new_stop
            else:  # BUY
                new_stop = current_price * (1 + self.request.trailing_percent / 100)
                if self.current_stop_price is None or new_stop < self.current_stop_price:
                    self.current_stop_price = new_stop

    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'request': asdict(self.request),
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'average_price': self.average_price,
            'fills': [fill.to_dict() for fill in self.fills],
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'cancelled_at': self.cancelled_at.isoformat() if self.cancelled_at else None,
            'expired_at': self.expired_at.isoformat() if self.expired_at else None,
            'error_message': self.error_message,
            'metadata': self.metadata,
            'parent_orders': self.parent_orders,
            'child_orders': self.child_orders,
            'current_stop_price': self.current_stop_price,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Order':
        data['request'] = OrderRequest(**data['request'])
        data['status'] = OrderStatus(data['status'])
        data['fills'] = [Fill.from_dict(f) for f in data['fills']]
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if data['cancelled_at']:
            data['cancelled_at'] = datetime.fromisoformat(data['cancelled_at'])
        if data['expired_at']:
            data['expired_at'] = datetime.fromisoformat(data['expired_at'])
        return cls(**data)
    
class ExecutionEngine:
    def __init__(self):
        self.logger = self._setup_logging()
        self.executor = ThreadPoolExecutor(max_workers=CONFIG.MAX_WORKERS)
        self.orders: Dict[str, Order] = {}
        self.active_orders: Set[str] = set()
        self.positions: Dict[str, Dict] = {}
        self._initialized = False
        self._running = False
        self._tasks = set()
        self._order_queue = asyncio.Queue()
        self.smart_routing_enabled = True
        self.rate_limiter = asyncio.Semaphore(10)  # Limit concurrent executions

    def _setup_logging(self) -> logging.Logger:
        """Setup execution logging"""
        logger = logging.getLogger('execution_engine')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(CONFIG.LOG_DIR / 'execution.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    async def initialize(self):
        """Initialize execution engine"""
        try:
            if self._initialized:
                return
                
            self.logger.info("Initializing execution engine...")
            
            # Load saved state
            await self._load_state()
            
            # Start order processing
            self._running = True
            self._tasks.add(
                asyncio.create_task(self._process_orders())
            )
            
            # Start order monitoring
            self._tasks.add(
                asyncio.create_task(self._monitor_orders())
            )
            
            # Subscribe to events
            event_manager.subscribe(EventType.MARKET_UPDATE, self._handle_market_update)
            
            self._initialized = True
            self.logger.info("Execution engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing execution engine: {e}\n{traceback.format_exc()}")
            raise

    async def _load_state(self):
        """Load execution state from disk"""
        try:
            state_file = CONFIG.DATA_DIR / 'execution_state.json'
            if state_file.exists():
                with open(state_file) as f:
                    state_data = json.load(f)
                    
                # Restore orders
                self.orders = {
                    order_id: Order.from_dict(order_data)
                    for order_id, order_data in state_data.get('orders', {}).items()
                }
                
                # Restore active orders
                self.active_orders = set(
                    order_id for order_id, order in self.orders.items()
                    if order.is_active()
                )
                
                # Restore positions
                self.positions = state_data.get('positions', {})
                
                self.logger.info(f"Loaded {len(self.orders)} orders and {len(self.positions)} positions")
                
        except Exception as e:
            self.logger.error(f"Error loading execution state: {e}")

    async def _save_state(self):
        """Save execution state to disk"""
        try:
            state_data = {
                'orders': {
                    order_id: order.to_dict()
                    for order_id, order in self.orders.items()
                },
                'positions': self.positions
            }
            
            state_file = CONFIG.DATA_DIR / 'execution_state.json'
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving execution state: {e}")

    async def place_order(self, request: OrderRequest) -> Optional[str]:
        """Place a new order"""
        try:
            # Validate request
            is_valid, error_message = request.validate()
            if not is_valid:
                self.logger.warning(f"Invalid order request: {error_message}")
                return None
                
            # Create order
            order = Order(request=request)
            
            # Additional validations
            if not await self._validate_order(order):
                return None
                
            # Register order
            self.orders[order.order_id] = order
            self.active_orders.add(order.order_id)
            
            # Queue order for processing
            await self._order_queue.put(order)
            
            # Publish order event
            await event_manager.publish(Event(
                type=EventType.ORDER,
                data={
                    'action': 'NEW',
                    'order': order.to_dict()
                }
            ))
            
            self.logger.info(f"""
            New order placed:
            ID: {order.order_id}
            Symbol: {request.symbol}
            Side: {request.side.value}
            Type: {request.order_type.value}
            Quantity: {request.quantity}
            Price: {request.price}
            """)
            
            return order.order_id
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}\n{traceback.format_exc()}")
            return None

    async def _validate_order(self, order: Order) -> bool:
        """Validate order against current market conditions"""
        try:
            request = order.request
            
            # Check market hours
            if not market_data.is_market_open():
                order.status = OrderStatus.REJECTED
                order.error_message = "Market is closed"
                return False
                
            # Get current price
            data = await market_data.get_stock_data(request.symbol)
            if data is None or data.empty:
                order.status = OrderStatus.REJECTED
                order.error_message = "Unable to get market data"
                return False
                
            current_price = data['Close'].iloc[-1]
            
            # Check minimum trade value
            trade_value = request.quantity * (request.price or current_price)
            if trade_value < CONFIG.MIN_TRADE_AMOUNT:
                order.status = OrderStatus.REJECTED
                order.error_message = "Trade value below minimum"
                return False
                
            # Check price reasonability
            if request.price is not None:
                price_diff = abs(request.price - current_price) / current_price
                if price_diff > 0.1:  # 10% price difference
                    order.status = OrderStatus.REJECTED
                    order.error_message = "Price too far from market"
                    return False
                    
            # Check position limits
            if len(self.positions) >= CONFIG.MAX_POSITIONS and request.symbol not in self.positions:
                order.status = OrderStatus.REJECTED
                order.error_message = "Position limit reached"
                return False
                
            # Check available capital
            if request.side == OrderSide.BUY:
                required_capital = trade_value * (1 + CONFIG.COMMISSION_RATE)
                available_capital = await self._get_available_capital()
                
                if required_capital > available_capital:
                    order.status = OrderStatus.REJECTED
                    order.error_message = "Insufficient capital"
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating order: {e}")
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            return False

    async def _process_orders(self):
        """Process orders from queue"""
        try:
            self.logger.info("Starting order processing...")
            
            while self._running:
                try:
                    # Get next order from queue
                    order = await self._order_queue.get()
                    
                    if not order.is_active():
                        continue
                        
                    # Apply rate limiting
                    async with self.rate_limiter:
                        # Process based on order type
                        if order.request.order_type == OrderType.MARKET:
                            await self._execute_market_order(order)
                        elif order.request.order_type == OrderType.LIMIT:
                            await self._execute_limit_order(order)
                        elif order.request.order_type == OrderType.STOP_LOSS:
                            await self._execute_stop_order(order)
                        elif order.request.order_type == OrderType.TRAILING_STOP:
                            await self._execute_trailing_stop_order(order)
                            
                    self._order_queue.task_done()
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error processing order: {e}\n{traceback.format_exc()}")
                    
        except Exception as e:
            self.logger.error(f"Fatal error in order processing: {e}\n{traceback.format_exc()}")
        finally:
            self.logger.info("Order processing stopped")
            
    async def _execute_market_order(self, order: Order):
        """Execute market order"""
        try:
            # Get market depth
            depth = await market_data.get_market_depth(order.request.symbol)
            if not depth:
                await self._reject_order(order, "No market depth available")
                return
                
            # Calculate execution price
            exec_price = await self._calculate_execution_price(order, depth)
            if not exec_price:
                await self._reject_order(order, "Could not determine execution price")
                return
                
            # Check slippage
            expected_price = await self._get_last_price(order.request.symbol)
            slippage = abs(exec_price - expected_price) / expected_price
            
            if slippage > CONFIG.SLIPPAGE:
                await self._reject_order(
                    order,
                    f"Slippage ({slippage:.4f}) exceeds limit"
                )
                return
                
            # Execute with smart routing if enabled
            if self.smart_routing_enabled:
                await self._smart_route_order(order, exec_price)
            else:
                await self._execute_single_venue(order, exec_price)
                
        except Exception as e:
            self.logger.error(f"Error executing market order: {e}\n{traceback.format_exc()}")
            await self._reject_order(order, str(e))

    async def _execute_limit_order(self, order: Order):
        """Execute limit order"""
        try:
            initial_wait = 0
            max_attempts = 5
            
            while initial_wait < max_attempts and order.is_active():
                # Get market depth
                depth = await market_data.get_market_depth(order.request.symbol)
                if not depth:
                    await asyncio.sleep(1)
                    initial_wait += 1
                    continue
                    
                # Check if limit price is executable
                if await self._can_execute_limit(order, depth):
                    exec_price = await self._get_improved_price(order, depth)
                    
                    if self.smart_routing_enabled:
                        await self._smart_route_order(order, exec_price)
                    else:
                        await self._execute_single_venue(order, exec_price)
                    return
                    
                # Check time in force
                if order.request.time_in_force == TimeInForce.IOC:
                    await self._cancel_order(order, "IOC order not immediately executable")
                    return
                    
                await asyncio.sleep(1)
                initial_wait += 1
                
            # Order couldn't be filled within attempts
            if order.request.time_in_force == TimeInForce.FOK:
                await self._cancel_order(order, "FOK order could not be completely filled")
            elif order.request.time_in_force == TimeInForce.DAY:
                await self._expire_order(order)
                
        except Exception as e:
            self.logger.error(f"Error executing limit order: {e}\n{traceback.format_exc()}")
            await self._reject_order(order, str(e))

    async def _smart_route_order(self, order: Order, exec_price: float):
        """Smart order routing implementation"""
        try:
            remaining_qty = order.remaining_quantity()
            
            # Calculate participation rate
            volume_data = await market_data.get_stock_data(
                order.request.symbol,
                interval='1m',
                lookback=5
            )
            avg_volume = volume_data['Volume'].mean()
            participation_qty = int(avg_volume * 0.1)  # 10% participation
            
            # Split order if necessary
            chunks = self._calculate_order_chunks(
                remaining_qty,
                participation_qty,
                exec_price
            )
            
            # Calculate venues and allocations
            venues = await self._get_best_venues(order, exec_price)
            
            # Execute chunks across venues
            for chunk in chunks:
                venue_allocations = self._allocate_to_venues(chunk, venues)
                
                # Execute on each venue
                execution_tasks = [
                    self._execute_on_venue(order, qty, price, venue)
                    for venue, (qty, price) in venue_allocations.items()
                ]
                
                results = await asyncio.gather(*execution_tasks, return_exceptions=True)
                
                # Process results
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.error(f"Venue execution error: {result}")
                    elif result:
                        fill_qty, fill_price, venue = result
                        await self._record_fill(
                            order,
                            fill_qty,
                            fill_price,
                            venue
                        )
                
                # Check if order is complete
                if order.filled_quantity == order.request.quantity:
                    break
                    
                await asyncio.sleep(1)  # Pause between chunks
                
        except Exception as e:
            self.logger.error(f"Error in smart routing: {e}\n{traceback.format_exc()}")
            if order.filled_quantity == 0:
                await self._reject_order(order, str(e))

    async def _execute_on_venue(
        self,
        order: Order,
        quantity: int,
        price: float,
        venue: str
    ) -> Optional[Tuple[int, float, str]]:
        """Execute order chunk on specific venue"""
        try:
            # Calculate commission and slippage
            commission = quantity * price * CONFIG.COMMISSION_RATE
            slippage = abs(price - await self._get_last_price(order.request.symbol))
            
            # Create fill record
            fill = Fill(
                quantity=quantity,
                price=price,
                timestamp=datetime.now(),
                commission=commission,
                slippage=slippage,
                venue=venue,
                liquidity="TAKER"  # Could be dynamic based on venue and order type
            )
            
            # Update order
            order.add_fill(fill)
            
            # Update position
            await self._update_position(order, fill)
            
            # Publish trade event
            await event_manager.publish(Event(
                type=EventType.TRADE,
                data={
                    'order_id': order.order_id,
                    'symbol': order.request.symbol,
                    'side': order.request.side.value,
                    'quantity': quantity,
                    'price': price,
                    'venue': venue,
                    'commission': commission,
                    'slippage': slippage
                }
            ))
            
            return quantity, price, venue
            
        except Exception as e:
            self.logger.error(f"Error executing on venue {venue}: {e}")
            return None

    def _calculate_order_chunks(
        self,
        quantity: int,
        participation_qty: int,
        price: float
    ) -> List[int]:
        """Calculate optimal order chunks"""
        try:
            if quantity <= participation_qty:
                return [quantity]
                
            # Calculate optimal number of chunks
            trade_value = quantity * price
            min_chunk_value = CONFIG.MIN_TRADE_AMOUNT
            max_chunks = int(trade_value / min_chunk_value)
            
            # Use TWAP/VWAP style chunking
            num_chunks = min(
                max_chunks,
                int(np.ceil(quantity / participation_qty))
            )
            
            base_chunk = quantity // num_chunks
            chunks = [base_chunk] * num_chunks
            
            # Add remainder to last chunk
            remainder = quantity - (base_chunk * num_chunks)
            if remainder > 0:
                chunks[-1] += remainder
                
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error calculating chunks: {e}")
            return [quantity]
        
    async def _update_position(self, order: Order, fill: Fill):
        """Update position after fill"""
        try:
            symbol = order.request.symbol
            
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'quantity': 0,
                    'average_price': 0,
                    'realized_pnl': 0,
                    'unrealized_pnl': 0,
                    'cost_basis': 0,
                    'last_update': datetime.now().isoformat(),
                    'strategy': order.request.strategy_id,
                    'trades': []
                }
                
            position = self.positions[symbol]
            
            # Update position
            if order.request.side == OrderSide.BUY:
                new_quantity = position['quantity'] + fill.quantity
                new_cost = (
                    position['quantity'] * position['average_price'] +
                    fill.quantity * fill.price
                )
                
                if new_quantity > 0:
                    position['average_price'] = new_cost / new_quantity
                    position['cost_basis'] = new_cost
                
                position['quantity'] = new_quantity
                
            else:  # SELL
                # Calculate realized P&L
                trade_pnl = (
                    fill.price - position['average_price']
                ) * min(fill.quantity, abs(position['quantity']))
                
                position['realized_pnl'] += trade_pnl
                position['quantity'] -= fill.quantity
                
                if position['quantity'] == 0:
                    position['average_price'] = 0
                    position['cost_basis'] = 0
                    
            # Record trade
            position['trades'].append({
                'order_id': order.order_id,
                'side': order.request.side.value,
                'quantity': fill.quantity,
                'price': fill.price,
                'commission': fill.commission,
                'timestamp': fill.timestamp.isoformat(),
                'pnl': trade_pnl if order.request.side == OrderSide.SELL else 0
            })
            
            position['last_update'] = datetime.now().isoformat()
            
            # Publish position update event
            await event_manager.publish(Event(
                type=EventType.POSITION,
                data={
                    'symbol': symbol,
                    'position': position
                }
            ))
            
            # Save state
            await self._save_state()
            
        except Exception as e:
            self.logger.error(f"Error updating position: {e}\n{traceback.format_exc()}")

    async def _get_best_venues(self, order: Order, price: float) -> List[Dict]:
        """Get best execution venues sorted by quality"""
        try:
            venues = [
                {
                    'name': 'PRIMARY',
                    'score': 1.0,
                    'liquidity': 1.0,
                    'latency': 0
                }
                # Add additional venues here
            ]
            
            # Sort venues by composite score
            return sorted(
                venues,
                key=lambda x: (x['score'] * x['liquidity']) / (1 + x['latency']),
                reverse=True
            )
            
        except Exception as e:
            self.logger.error(f"Error getting best venues: {e}")
            return [{'name': 'PRIMARY', 'score': 1.0}]

    def _allocate_to_venues(
        self,
        quantity: int,
        venues: List[Dict]
    ) -> Dict[str, Tuple[int, float]]:
        """Allocate order quantity across venues"""
        try:
            allocations = {}
            remaining_qty = quantity
            total_score = sum(venue['score'] for venue in venues)
            
            for venue in venues:
                if remaining_qty <= 0:
                    break
                    
                venue_qty = int(
                    (venue['score'] / total_score) * quantity
                )
                venue_qty = min(venue_qty, remaining_qty)
                
                if venue_qty > 0:
                    allocations[venue['name']] = (venue_qty, None)  # Price determined at execution
                    remaining_qty -= venue_qty
                    
            # Allocate any remaining quantity to primary venue
            if remaining_qty > 0:
                primary_qty = allocations.get('PRIMARY', (0, None))[0]
                allocations['PRIMARY'] = (primary_qty + remaining_qty, None)
                
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error allocating to venues: {e}")
            return {'PRIMARY': (quantity, None)}

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        try:
            if order_id not in self.orders:
                return False
                
            order = self.orders[order_id]
            if not order.is_active():
                return False
                
            await self._cancel_order(order, "User requested cancellation")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False

    async def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get current order status"""
        try:
            if order_id not in self.orders:
                return None
            return self.orders[order_id].status
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return None

    async def get_active_orders(
        self,
        symbol: Optional[str] = None,
        strategy_id: Optional[str] = None
    ) -> List[Order]:
        """Get active orders with optional filters"""
        try:
            orders = [
                self.orders[order_id]
                for order_id in self.active_orders
            ]
            
            if symbol:
                orders = [
                    order for order in orders
                    if order.request.symbol == symbol
                ]
                
            if strategy_id:
                orders = [
                    order for order in orders
                    if order.request.strategy_id == strategy_id
                ]
                
            return orders
        except Exception as e:
            self.logger.error(f"Error getting active orders: {e}")
            return []

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Order]:
        """Get order history with filters"""
        try:
            orders = list(self.orders.values())
            
            if symbol:
                orders = [o for o in orders if o.request.symbol == symbol]
                
            if start_time:
                orders = [o for o in orders if o.created_at >= start_time]
                
            if end_time:
                orders = [o for o in orders if o.created_at <= end_time]
                
            return sorted(orders, key=lambda x: x.created_at, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error getting order history: {e}")
            return []

    async def shutdown(self):
        """Shutdown execution engine"""
        try:
            self.logger.info("Shutting down execution engine...")
            
            # Stop processing
            self._running = False
            
            # Cancel all active orders
            for order_id in list(self.active_orders):
                await self.cancel_order(order_id)
                
            # Cancel all tasks
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
            # Save final state
            await self._save_state()
            
            # Clean up executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("Execution engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Create singleton instance
execution_engine = ExecutionEngine()
