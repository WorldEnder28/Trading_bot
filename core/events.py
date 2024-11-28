# core/events.py
from typing import Dict, List, Optional, Union, Set, Callable, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import asyncio
import logging
import json
import uuid
import traceback
from enum import Enum
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import CONFIG

class EventType(Enum):
    MARKET_UPDATE = "MARKET_UPDATE"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    TRADE = "TRADE"
    POSITION = "POSITION"
    RISK = "RISK"
    ERROR = "ERROR"
    SYSTEM = "SYSTEM"
    STRATEGY = "STRATEGY"
    PORTFOLIO = "PORTFOLIO"
    EXECUTION = "EXECUTION"
    NOTIFICATION = "NOTIFICATION"

class EventPriority(Enum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3

@dataclass
class Event:
    type: EventType
    data: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    priority: EventPriority = EventPriority.MEDIUM
    source: Optional[str] = None
    target: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    processed: bool = False
    error: Optional[str] = None
    processing_time: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.type.value,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.value,
            'source': self.source,
            'target': self.target,
            'correlation_id': self.correlation_id,
            'metadata': self.metadata,
            'processed': self.processed,
            'error': self.error,
            'processing_time': self.processing_time
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Event':
        data['type'] = EventType(data['type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['priority'] = EventPriority(data['priority'])
        return cls(**data)

class EventCallback:
    def __init__(
        self,
        callback: Callable,
        filter_condition: Optional[Callable] = None,
        priority: EventPriority = EventPriority.MEDIUM,
        error_handler: Optional[Callable] = None,
        timeout: Optional[float] = None,
        retries: int = 0
    ):
        self.callback = callback
        self.filter_condition = filter_condition
        self.priority = priority
        self.error_handler = error_handler
        self.timeout = timeout
        self.retries = retries
        self.failed_attempts = 0

    async def __call__(self, event: Event) -> bool:
        try:
            # Check filter condition
            if self.filter_condition and not self.filter_condition(event):
                return True

            # Apply timeout if specified
            if self.timeout:
                try:
                    async with asyncio.timeout(self.timeout):
                        await self._execute_callback(event)
                except asyncio.TimeoutError:
                    if self.error_handler:
                        await self.error_handler(event, "Callback timeout")
                    return False
            else:
                await self._execute_callback(event)

            return True

        except Exception as e:
            if self.error_handler:
                await self.error_handler(event, str(e))
            return False

    async def _execute_callback(self, event: Event):
        """Execute callback with retry logic"""
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                if asyncio.iscoroutinefunction(self.callback):
                    await self.callback(event)
                else:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.callback, event
                    )
                return
            except Exception as e:
                last_error = e
                self.failed_attempts += 1
                if attempt < self.retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise last_error

@dataclass
class EventStats:
    total_events: int = 0
    processed_events: int = 0
    failed_events: int = 0
    average_processing_time: float = 0.0
    events_by_type: Dict[EventType, int] = field(default_factory=lambda: defaultdict(int))
    events_by_priority: Dict[EventPriority, int] = field(default_factory=lambda: defaultdict(int))
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_updated: datetime = field(default_factory=datetime.now)

    def update(self, event: Event, success: bool, processing_time: float):
        """Update event statistics"""
        self.total_events += 1
        if success:
            self.processed_events += 1
        else:
            self.failed_events += 1
            if event.error:
                self.error_counts[event.error] += 1

        self.events_by_type[event.type] += 1
        self.events_by_priority[event.priority] += 1

        # Update average processing time
        self.average_processing_time = (
            (self.average_processing_time * (self.total_events - 1) + processing_time)
            / self.total_events
        )
        self.last_updated = datetime.now()

    def to_dict(self) -> Dict:
        return {
            'total_events': self.total_events,
            'processed_events': self.processed_events,
            'failed_events': self.failed_events,
            'average_processing_time': self.average_processing_time,
            'events_by_type': {k.value: v for k, v in self.events_by_type.items()},
            'events_by_priority': {k.value: v for k, v in self.events_by_priority.items()},
            'error_counts': dict(self.error_counts),
            'last_updated': self.last_updated.isoformat()
        }
        
class EventManager:
    def __init__(self):
        self.logger = self._setup_logging()
        self.executor = ThreadPoolExecutor(max_workers=CONFIG.MAX_WORKERS)
        self.listeners: Dict[EventType, Set[EventCallback]] = {
            event_type: set() for event_type in EventType
        }
        self._queue = asyncio.PriorityQueue()
        self._error_handlers: Set[Callable] = set()
        self._event_history: List[Event] = []
        self._stats = EventStats()
        self._running = False
        self._initialized = False
        self._tasks = set()
        self.max_history = 1000
        self.max_queue_size = CONFIG.QUEUE_SIZE
        self.processing_timeout = 10  # seconds
        self.max_retry_delay = 30  # seconds
        self.publish_stats = True

    def _setup_logging(self) -> logging.Logger:
        """Setup event manager logging"""
        logger = logging.getLogger('event_manager')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(CONFIG.LOG_DIR / 'event_manager.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    async def initialize(self):
        """Initialize event manager"""
        try:
            if self._initialized:
                return
                
            self.logger.info("Initializing event manager...")
            
            # Load saved state
            await self._load_state()
            
            # Start event processing
            self._running = True
            self._tasks.add(
                asyncio.create_task(self._process_events())
            )
            
            # Start stats monitoring
            if self.publish_stats:
                self._tasks.add(
                    asyncio.create_task(self._monitor_stats())
                )
            
            self._initialized = True
            self.logger.info("Event manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing event manager: {e}\n{traceback.format_exc()}")
            raise

    async def _load_state(self):
        """Load event manager state"""
        try:
            state_file = CONFIG.DATA_DIR / 'event_state.json'
            if state_file.exists():
                with open(state_file) as f:
                    state_data = json.load(f)
                    
                # Restore event history
                self._event_history = [
                    Event.from_dict(event_data)
                    for event_data in state_data.get('event_history', [])
                ][-self.max_history:]
                
                # Restore stats
                if 'stats' in state_data:
                    stats_data = state_data['stats']
                    stats_data['last_updated'] = datetime.fromisoformat(
                        stats_data['last_updated']
                    )
                    self._stats = EventStats(**stats_data)
                    
        except Exception as e:
            self.logger.error(f"Error loading event state: {e}")

    async def _save_state(self):
        """Save event manager state"""
        try:
            state_data = {
                'event_history': [
                    event.to_dict() for event in self._event_history
                ],
                'stats': self._stats.to_dict()
            }
            
            state_file = CONFIG.DATA_DIR / 'event_state.json'
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving event state: {e}")

    def subscribe(
        self,
        event_type: Union[EventType, List[EventType]],
        callback: Callable,
        filter_condition: Optional[Callable] = None,
        priority: EventPriority = EventPriority.MEDIUM,
        error_handler: Optional[Callable] = None,
        timeout: Optional[float] = None,
        retries: int = 0
    ):
        """Register event listener with advanced options"""
        try:
            event_callback = EventCallback(
                callback=callback,
                filter_condition=filter_condition,
                priority=priority,
                error_handler=error_handler or self._handle_error,
                timeout=timeout,
                retries=retries
            )
            
            if isinstance(event_type, list):
                for et in event_type:
                    self.listeners[et].add(event_callback)
            else:
                self.listeners[event_type].add(event_callback)
                
            self.logger.debug(
                f"Subscribed {callback.__name__} to {event_type} "
                f"with priority {priority.name}"
            )
            
        except Exception as e:
            self.logger.error(f"Error registering event listener: {e}\n{traceback.format_exc()}")
            self._handle_error(e)

    def unsubscribe(
        self,
        event_type: Union[EventType, List[EventType]],
        callback: Callable
    ):
        """Remove event listener"""
        try:
            if isinstance(event_type, list):
                for et in event_type:
                    self._unsubscribe_single(et, callback)
            else:
                self._unsubscribe_single(event_type, callback)
                
        except Exception as e:
            self.logger.error(f"Error removing event listener: {e}")
            self._handle_error(e)

    def _unsubscribe_single(self, event_type: EventType, callback: Callable):
        """Remove single event listener"""
        to_remove = None
        for listener in self.listeners[event_type]:
            if listener.callback == callback:
                to_remove = listener
                break
                
        if to_remove:
            self.listeners[event_type].discard(to_remove)
            self.logger.debug(f"Unsubscribed {callback.__name__} from {event_type}")

    async def publish(self, event: Event):
        """Publish event to queue"""
        try:
            if not self._running:
                self.logger.warning("Event manager not running, event discarded")
                return
                
            # Check queue size
            if self._queue.qsize() >= self.max_queue_size:
                self.logger.warning("Event queue full, event discarded")
                return
                
            # Add to queue with priority
            await self._queue.put((event.priority.value, event))
            
            # Add to history
            self._add_to_history(event)
            
        except Exception as e:
            self.logger.error(f"Error publishing event: {e}\n{traceback.format_exc()}")
            self._handle_error(e)
            
    async def _process_events(self):
        """Process events from queue"""
        try:
            self.logger.info("Event processing started")
            
            while self._running:
                try:
                    # Get next event with priority
                    _, event = await self._queue.get()
                    
                    start_time = datetime.now()
                    success = True
                    
                    try:
                        # Process event with timeout
                        async with asyncio.timeout(self.processing_timeout):
                            await self._process_single_event(event)
                            
                    except asyncio.TimeoutError:
                        self.logger.error(f"Event processing timeout: {event.id}")
                        event.error = "Processing timeout"
                        success = False
                        
                    except Exception as e:
                        self.logger.error(f"Error processing event: {e}\n{traceback.format_exc()}")
                        event.error = str(e)
                        success = False
                        
                    finally:
                        # Calculate processing time
                        processing_time = (datetime.now() - start_time).total_seconds()
                        event.processing_time = processing_time
                        event.processed = success
                        
                        # Update statistics
                        self._stats.update(event, success, processing_time)
                        
                        self._queue.task_done()
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in event processing loop: {e}\n{traceback.format_exc()}")
                    await asyncio.sleep(1)  # Prevent tight loop on error
                    
        except Exception as e:
            self.logger.error(f"Fatal error in event processing: {e}\n{traceback.format_exc()}")
        finally:
            self.logger.info("Event processing stopped")

    async def _process_single_event(self, event: Event):
        """Process a single event"""
        try:
            callbacks = sorted(
                self.listeners[event.type],
                key=lambda x: x.priority.value
            )
            
            if not callbacks:
                return
                
            # Process callbacks based on priority
            for callback in callbacks:
                try:
                    # Check if callback should be retried
                    if callback.failed_attempts >= callback.retries:
                        continue
                        
                    success = await callback(event)
                    if not success:
                        self.logger.warning(
                            f"Callback {callback.callback.__name__} failed "
                            f"for event {event.id}"
                        )
                        
                except Exception as e:
                    self.logger.error(
                        f"Error in callback {callback.callback.__name__}: {e}\n"
                        f"{traceback.format_exc()}"
                    )
                    
        except Exception as e:
            self.logger.error(f"Error processing event {event.id}: {e}")
            raise

    async def _monitor_stats(self):
        """Monitor and publish event statistics"""
        try:
            while self._running:
                try:
                    # Publish stats event
                    await self.publish(Event(
                        type=EventType.SYSTEM,
                        data={
                            'type': 'STATS',
                            'stats': self._stats.to_dict()
                        }
                    ))
                    
                    # Save state periodically
                    await self._save_state()
                    
                    # Clean up old events from history
                    self._cleanup_history()
                    
                    await asyncio.sleep(60)  # Update every minute
                    
                except Exception as e:
                    self.logger.error(f"Error in stats monitoring: {e}")
                    await asyncio.sleep(60)
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Fatal error in stats monitoring: {e}")

    def _add_to_history(self, event: Event):
        """Add event to history with size limit"""
        self._event_history.append(event)
        self._cleanup_history()

    def _cleanup_history(self):
        """Clean up old events from history"""
        if len(self._event_history) > self.max_history:
            self._event_history = self._event_history[-self.max_history:]

    def get_event_history(
        self,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        source: Optional[str] = None,
        success_only: bool = False
    ) -> List[Event]:
        """Get filtered event history"""
        events = self._event_history
        
        if event_type:
            events = [e for e in events if e.type == event_type]
            
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
            
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
            
        if source:
            events = [e for e in events if e.source == source]
            
        if success_only:
            events = [e for e in events if e.processed and not e.error]
            
        return events

    def register_error_handler(self, handler: Callable):
        """Register global error handler"""
        self._error_handlers.add(handler)

    def unregister_error_handler(self, handler: Callable):
        """Unregister global error handler"""
        self._error_handlers.discard(handler)

    def _handle_error(self, error: Exception):
        """Handle error through registered error handlers"""
        for handler in self._error_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(error))
                else:
                    handler(error)
            except Exception as e:
                self.logger.error(f"Error in error handler: {e}")
                
    async def get_stats(
        self,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict:
        """Get detailed event statistics"""
        try:
            # Get filtered events
            events = self.get_event_history(
                event_type=event_type,
                start_time=start_time,
                end_time=end_time
            )
            
            if not events:
                return {
                    'total_events': 0,
                    'processed_events': 0,
                    'failed_events': 0,
                    'average_processing_time': 0.0,
                    'success_rate': 0.0,
                    'events_by_type': {},
                    'events_by_priority': {},
                    'error_counts': {},
                    'processing_times': {
                        'min': 0.0,
                        'max': 0.0,
                        'avg': 0.0,
                        'median': 0.0
                    }
                }
            
            # Calculate statistics
            total_events = len(events)
            processed_events = sum(1 for e in events if e.processed)
            failed_events = sum(1 for e in events if not e.processed or e.error)
            
            # Processing times analysis
            processing_times = [
                e.processing_time for e in events 
                if e.processing_time is not None
            ]
            
            avg_processing_time = (
                sum(processing_times) / len(processing_times)
                if processing_times else 0.0
            )
            
            # Event type distribution
            events_by_type = defaultdict(int)
            for event in events:
                events_by_type[event.type.value] += 1
            
            # Priority distribution
            events_by_priority = defaultdict(int)
            for event in events:
                events_by_priority[event.priority.value] += 1
            
            # Error analysis
            error_counts = defaultdict(int)
            for event in events:
                if event.error:
                    error_counts[event.error] += 1
            
            # Processing times statistics
            processing_stats = {
                'min': min(processing_times) if processing_times else 0.0,
                'max': max(processing_times) if processing_times else 0.0,
                'avg': avg_processing_time,
                'median': sorted(processing_times)[len(processing_times)//2]
                if processing_times else 0.0
            }
            
            return {
                'total_events': total_events,
                'processed_events': processed_events,
                'failed_events': failed_events,
                'average_processing_time': avg_processing_time,
                'success_rate': (processed_events / total_events * 100) 
                    if total_events > 0 else 0.0,
                'events_by_type': dict(events_by_type),
                'events_by_priority': dict(events_by_priority),
                'error_counts': dict(error_counts),
                'processing_times': processing_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}\n{traceback.format_exc()}")
            return {}

    async def get_queue_status(self) -> Dict:
        """Get current queue status"""
        try:
            return {
                'queue_size': self._queue.qsize(),
                'queue_full': self._queue.qsize() >= self.max_queue_size,
                'max_queue_size': self.max_queue_size,
                'listeners_count': sum(len(l) for l in self.listeners.values()),
                'event_types': {
                    et.value: len(self.listeners[et])
                    for et in EventType
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting queue status: {e}")
            return {}

    async def clear_queue(self):
        """Clear all pending events from queue"""
        try:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except asyncio.QueueEmpty:
                    break
        except Exception as e:
            self.logger.error(f"Error clearing queue: {e}")

    async def shutdown(self):
        """Shutdown event manager"""
        try:
            self.logger.info("Shutting down event manager...")
            
            # Stop processing
            self._running = False
            
            # Process remaining events with timeout
            try:
                async with asyncio.timeout(30):  # 30-second timeout
                    await self._queue.join()
            except asyncio.TimeoutError:
                self.logger.warning("Timeout waiting for queue to empty")
            
            # Cancel all tasks
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
            # Clear queue
            await self.clear_queue()
            
            # Save final state
            await self._save_state()
            
            # Clean up executor
            self.executor.shutdown(wait=True)
            
            # Clear listeners
            self.listeners = {event_type: set() for event_type in EventType}
            self._error_handlers.clear()
            
            self.logger.info("Event manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}\n{traceback.format_exc()}")

    async def restart(self):
        """Restart event manager"""
        try:
            await self.shutdown()
            await self.initialize()
            self.logger.info("Event manager restarted successfully")
        except Exception as e:
            self.logger.error(f"Error restarting event manager: {e}")
            raise

    def health_check(self) -> Dict:
        """Perform health check"""
        try:
            return {
                'status': 'healthy' if self._running else 'stopped',
                'initialized': self._initialized,
                'queue_size': self._queue.qsize(),
                'queue_utilization': self._queue.qsize() / self.max_queue_size * 100,
                'active_listeners': sum(len(l) for l in self.listeners.values()),
                'event_history_size': len(self._event_history),
                'last_event_time': self._event_history[-1].timestamp.isoformat()
                    if self._event_history else None,
                'error_rate': (
                    self._stats.failed_events / self._stats.total_events * 100
                    if self._stats.total_events > 0 else 0.0
                ),
                'average_processing_time': self._stats.average_processing_time
            }
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

# Create singleton instance
event_manager = EventManager()
