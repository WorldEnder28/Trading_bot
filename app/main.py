# app/main.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import asyncio
from pathlib import Path
import logging
from typing import Dict
import traceback

from core.market_data import market_data
from core.execution import execution_engine
from core.strategy import strategy_manager
from core.risk import risk_manager
from core.portfolio import portfolio_manager
from core.events import event_manager, EventType, Event
from core.config import CONFIG

# Set Streamlit page configuration
st.set_page_config(
    page_title="Advanced Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.reportview-container {
    background: #0e1117;
}
.sidebar .sidebar-content {
    background: #262730;
}
.Widget>label {
    color: #ffffff;
}
.stProgress .st-bo {
    background-color: #21c354;
}
.stAlert {
    background-color: #262730;
    color: #ffffff;
}
.stMetric {
    background-color: #1c1c1c;
    padding: 15px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

class TradingApp:
    def __init__(self):
        self.logger = self._setup_logging()
        self._initialize_session_state()
        self._load_components()

    def _setup_logging(self) -> logging.Logger:
        """Setup application logging"""
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'active_page' not in st.session_state:
            st.session_state.active_page = 'Dashboard'
        if 'trading_active' not in st.session_state:
            st.session_state.trading_active = False
        if 'selected_symbols' not in st.session_state:
            st.session_state.selected_symbols = []
        if 'active_strategies' not in st.session_state:
            st.session_state.active_strategies = []
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = None

    def _load_components(self):
        """Load application components"""
        try:
            from app.pages.dashboard import DashboardPage
            from app.pages.portfolio import PortfolioPage
            from app.pages.strategy import StrategyPage
            from app.pages.backtesting import BacktestingPage
            from app.pages.settings import SettingsPage
            
            self.pages = {
                'Dashboard': DashboardPage(),
                'Portfolio': PortfolioPage(),
                'Strategy': StrategyPage(),
                'Backtesting': BacktestingPage(),
                'Settings': SettingsPage()
            }
        except Exception as e:
            self.logger.error(f"Error loading components: {e}")
            st.error("Failed to load application components")
            raise

    def get_risk_utilization_sync(self) -> Dict:
        """Get risk utilization metrics synchronously"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                current_var = loop.run_until_complete(risk_manager._calculate_portfolio_var())
                max_var = CONFIG.CAPITAL * CONFIG.risk.max_portfolio_risk
                current_usage = (current_var / max_var) * 100 if max_var != 0 else 0
                return {'current': current_usage, 'change': 0}
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"Error calculating risk utilization: {e}")
            return {'current': 0, 'change': 0}

    def get_margin_utilization_sync(self) -> Dict:
        """Get margin utilization metrics synchronously"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                available_margin = loop.run_until_complete(risk_manager._get_available_margin())
                used_margin = CONFIG.CAPITAL - available_margin
                current_usage = (used_margin / CONFIG.CAPITAL) * 100
                return {'current': current_usage, 'change': 0}
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"Error calculating margin utilization: {e}")
            return {'current': 0, 'change': 0}

    def get_win_rate_sync(self) -> Dict:
        """Get win rate metrics synchronously"""
        try:
            trades = portfolio_manager.trades
            if not trades:
                return {'current': 0, 'change': 0}
                
            winning_trades = sum(1 for t in trades if t.pnl > 0)
            current_rate = (winning_trades / len(trades)) * 100
            return {'current': current_rate, 'change': 0}
        except Exception as e:
            self.logger.error(f"Error calculating win rate: {e}")
            return {'current': 0, 'change': 0}

    def get_quick_stats(self) -> Dict:
        """Get quick statistics"""
        try:
            portfolio_value = portfolio_manager.get_total_value()
            daily_pnl = portfolio_manager.get_daily_pnl()
            
            return {
                'portfolio_value': portfolio_value,
                'daily_return': (daily_pnl / portfolio_value) * 100 if portfolio_value else 0,
                'open_positions': len(portfolio_manager.positions),
                'position_change': self.get_position_change(),
                'daily_pnl': daily_pnl,
                'pnl_change': self.get_pnl_change()
            }
        except Exception as e:
            self.logger.error(f"Error getting quick stats: {e}")
            return {
                'portfolio_value': 0,
                'daily_return': 0,
                'open_positions': 0,
                'position_change': 0,
                'daily_pnl': 0,
                'pnl_change': 0
            }

    def get_position_change(self) -> int:
        """Get change in number of positions today"""
        try:
            current_positions = len(portfolio_manager.positions)
            morning_positions = len([
                t for t in portfolio_manager.trades 
                if t.entry_time.date() == datetime.now().date()
            ])
            return current_positions - morning_positions
        except Exception as e:
            self.logger.error(f"Error calculating position change: {e}")
            return 0

    def get_pnl_change(self) -> float:
        """Get change in P&L from previous day"""
        try:
            today_pnl = portfolio_manager.get_daily_pnl()
            yesterday_pnl = sum(
                t.pnl for t in portfolio_manager.trades
                if t.exit_time.date() == (datetime.now().date() - timedelta(days=1))
            )
            return ((today_pnl / yesterday_pnl) - 1) * 100 if yesterday_pnl else 0
        except Exception as e:
            self.logger.error(f"Error calculating PnL change: {e}")
            return 0

    def start_trading_sync(self):
        """Start trading system synchronously"""
        try:
            event_manager.subscribe(EventType.TRADE, self.handle_trade)
            event_manager.subscribe(EventType.SIGNAL, self.handle_signal)
            event_manager.subscribe(EventType.ERROR, self.handle_error)
            st.session_state.trading_active = True
            self.add_notification("Trading system started")
        except Exception as e:
            self.logger.error(f"Error starting trading: {e}")
            st.error("Failed to start trading system")

    def stop_trading_sync(self):
        """Stop trading system synchronously"""
        try:
            st.session_state.trading_active = False
            self.add_notification("Trading system stopped")
        except Exception as e:
            self.logger.error(f"Error stopping trading: {e}")
            st.error("Failed to stop trading system")

    def add_notification(self, message: str):
        """Add a notification message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.notifications.append(f"[{timestamp}] {message}")
        if len(st.session_state.notifications) > 100:
            st.session_state.notifications = st.session_state.notifications[-100:]

    def handle_trade(self, event: Event):
        """Handle trade events"""
        try:
            trade_data = event.data
            self.add_notification(
                f"Trade executed: {trade_data['symbol']} {trade_data['side']} "
                f"@ ₹{trade_data['price']:.2f}"
            )
        except Exception as e:
            self.logger.error(f"Error handling trade event: {e}")

    def handle_signal(self, event: Event):
        """Handle signal events"""
        try:
            signal_data = event.data
            self.add_notification(
                f"Signal generated: {signal_data['symbol']} {signal_data['direction']} "
                f"({signal_data['strategy']})"
            )
        except Exception as e:
            self.logger.error(f"Error handling signal event: {e}")

    def handle_error(self, event: Event):
        """Handle error events"""
        try:
            error_data = event.data
            self.add_notification(f"Error: {error_data['message']}")
        except Exception as e:
            self.logger.error(f"Error handling error event: {e}")

    def render(self):
        """Render the complete application"""
        try:
            # Render sidebar
            with st.sidebar:
                st.title("Trading System")
                
                # Navigation
                for page_name in self.pages.keys():
                    if st.button(
                        page_name,
                        key=f"nav_{page_name}",
                        use_container_width=True,
                        type="secondary" if st.session_state.active_page != page_name else "primary"
                    ):
                        st.session_state.active_page = page_name
                        st.rerun()
                
                # Trading Controls section
                st.divider()
                st.subheader("Trading Controls")
                
                if not st.session_state.trading_active:
                    if st.button("Start Trading", type="primary", use_container_width=True):
                        self.start_trading_sync()
                else:
                    if st.button("Stop Trading", type="secondary", use_container_width=True):
                        self.stop_trading_sync()

                # Quick Stats section
                st.divider()
                st.subheader("Quick Stats")
                
                metrics = self.get_quick_stats()
                st.metric(
                    "Portfolio Value",
                    f"₹{metrics['portfolio_value']:,.2f}",
                    f"{metrics['daily_return']:.2f}%"
                )
                st.metric(
                    "Open Positions",
                    metrics['open_positions'],
                    f"{metrics['position_change']:+d} today"
                )
                st.metric(
                    "Day's P&L",
                    f"₹{metrics['daily_pnl']:,.2f}",
                    f"{metrics['pnl_change']:.2f}%"
                )

                # Notifications section
                st.divider()
                st.subheader("Notifications")
                for notification in st.session_state.notifications[-5:]:
                    st.info(notification)

            # Main content
            with st.container():
                # Header
                col1, col2, col3, col4 = st.columns([3,1,1,1])
                
                with col1:
                    st.title(st.session_state.active_page)
                
                with col2:
                    risk_usage = self.get_risk_utilization_sync()
                    st.metric(
                        "Risk Usage",
                        f"{risk_usage['current']:.1f}%",
                        f"{risk_usage['change']:.1f}%"
                    )
                
                with col3:
                    margin_usage = self.get_margin_utilization_sync()
                    st.metric(
                        "Margin Usage",
                        f"{margin_usage['current']:.1f}%",
                        f"{margin_usage['change']:.1f}%"
                    )
                
                with col4:
                    win_rate = self.get_win_rate_sync()
                    st.metric(
                        "Win Rate",
                        f"{win_rate['current']:.1f}%",
                        f"{win_rate['change']:.1f}%"
                    )

                st.divider()
                
                # Render active page
                active_page = st.session_state.active_page
                if active_page in self.pages:
                    self.pages[active_page].render()

        except Exception as e:
            self.logger.error(f"Error rendering app: {str(e)}\n{traceback.format_exc()}")
            st.error("An error occurred while rendering the application")

def main():
    """Main application entry point"""
    try:
        app = TradingApp()
        app.render()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logging.error(f"Application error: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
