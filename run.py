# run.py
import asyncio
import argparse
import logging
from datetime import datetime
import sys
import os
from pathlib import Path
import streamlit as st
from typing import Dict, List, Optional, Union
import async_timeout
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from core.config import CONFIG
from core.market_data import market_data
from core.execution import execution_engine
from core.strategy import strategy_manager
from core.risk import risk_manager
from core.portfolio import portfolio_manager
from core.backtester import backtester
from core.events import event_manager, EventType, Event

class TradingSystem:
    def __init__(self):
        self.logger = self._setup_logging()
        self.running = False
        
    def _setup_logging(self) -> logging.Logger:
        """Setup system logging"""
        logger = logging.getLogger('trading_system')
        logger.setLevel(logging.INFO)
        
        # Ensure log directory exists
        CONFIG.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = CONFIG.LOG_DIR / f'system_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        return logger

    async def start(self):
        """Start trading system"""
        try:
            self.logger.info("Starting trading system...")
            self.running = True
            
            # Initialize components with timeout
            async with async_timeout.timeout(60):
                await self._initialize_components()
            self.logger.info("Component initialization completed")
            
            # Start event processing
            self.logger.info("Starting event manager...")
            await event_manager.start()
            
            # Start system loop
            self.logger.info("System startup completed successfully")
            asyncio.create_task(self._run_system_loop())
            
        except asyncio.TimeoutError as e:
            self.logger.error(f"System initialization timed out: {str(e)}\n{traceback.format_exc()}")
            await self.cleanup()
            await self.stop()
            raise RuntimeError(f"System initialization timed out: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error starting system: {str(e)}\n{traceback.format_exc()}")
            await self.cleanup()
            await self.stop()
            raise

    async def _initialize_components(self):
        """Initialize system components"""
        try:
            self.logger.info("Starting component initialization...")
            
            # Initialize market data first
            self.logger.info("Initializing market data connection...")
            try:
                # Test with yfinance first
                self.logger.info("Testing market data access using yfinance...")
                import yfinance as yf
                test_symbol = "RELIANCE.NS"
                test_ticker = yf.Ticker(test_symbol)
                test_data = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: test_ticker.history(period="1d")
                )
                
                if test_data is not None and not test_data.empty:
                    self.logger.info("Yfinance market data test successful")
                else:
                    self.logger.warning("Failed to fetch data from yfinance")
                    raise Exception("Failed to fetch market data from yfinance")
                    
            except Exception as e:
                self.logger.error(f"Error testing market data: {str(e)}")
                raise
            
            # Subscribe to events
            self.logger.info("Setting up event subscriptions...")
            event_manager.subscribe(EventType.MARKET_UPDATE, self._handle_market_update)
            event_manager.subscribe(EventType.SIGNAL, self._handle_signal)
            event_manager.subscribe(EventType.ORDER, self._handle_order)
            event_manager.subscribe(EventType.TRADE, self._handle_trade)
            event_manager.subscribe(EventType.ERROR, self._handle_error)
            self.logger.info("Event subscriptions completed")
            
            # Initialize portfolio
            self.logger.info("Initializing portfolio...")
            portfolio_manager.cash = CONFIG.CAPITAL
            portfolio_manager.initial_capital = CONFIG.CAPITAL
            self.logger.info("Portfolio initialized")
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}\n{traceback.format_exc()}")
            raise

    async def _run_system_loop(self):
        """Main system loop"""
        try:
            self.logger.info("Starting main system loop...")
            
            while self.running:
                try:
                    await asyncio.sleep(CONFIG.SCAN_INTERVAL)
                except Exception as e:
                    self.logger.error(f"Error in system loop: {str(e)}\n{traceback.format_exc()}")
                    
        except Exception as e:
            self.logger.error(f"Fatal error in system loop: {str(e)}\n{traceback.format_exc()}")
            await self.stop()

    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.logger.info("Cleaning up resources...")
            await market_data.close()
            if event_manager._running:
                event_manager.stop()
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}\n{traceback.format_exc()}")

    async def stop(self):
        """Stop trading system"""
        self.running = False
        await self.cleanup()

    async def _handle_market_update(self, event: Event):
        pass

    async def _handle_signal(self, event: Event):
        pass

    async def _handle_order(self, event: Event):
        pass

    async def _handle_trade(self, event: Event):
        pass

    async def _handle_error(self, event: Event):
        pass

async def run_trading_system(gui: bool = True):
    """Run trading system with optional GUI"""
    try:
        if not hasattr(st.session_state, 'trading_system'):
            if gui:
                from app.main import TradingApp
                
                progress_text = "Initializing trading system..."
                progress_bar = st.progress(0, text=progress_text)
                
                try:
                    progress_bar.progress(25, text="Creating system components...")
                    system = TradingSystem()
                    st.session_state.trading_system = system
                    
                    progress_bar.progress(50, text="Starting trading system...")
                    await system.start()
                    
                    progress_bar.progress(75, text="Loading interface...")
                    app = TradingApp()
                    st.session_state.trading_app = app
                    
                    progress_bar.progress(90, text="Rendering interface...")
                    app.render()  # Use sync version
                    
                    progress_bar.progress(100, text="System ready!")
                    await asyncio.sleep(1)
                    
                    progress_bar.empty()
                    
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Initialization error: {str(e)}\n{traceback.format_exc()}")
                    raise
            else:
                print("Starting trading system in console mode...")
                system = TradingSystem()
                st.session_state.trading_system = system
                await system.start()
        else:
            if gui and hasattr(st.session_state, 'trading_app'):
                st.session_state.trading_app.render()  # Use sync version
                
    except Exception as e:
        error_msg = f"Error running system: {str(e)}\n{traceback.format_exc()}"
        if gui:
            st.error(error_msg)
        else:
            print(error_msg)

def main():
    parser = argparse.ArgumentParser(description='Trading System')
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Run without GUI (console mode)'
    )
    args = parser.parse_args()
    
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(run_trading_system(not args.no_gui))
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error running system: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
