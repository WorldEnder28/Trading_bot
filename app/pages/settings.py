# app/pages/settings.py
import streamlit as st
import yaml
from datetime import datetime
import os
from pathlib import Path

from core.config import CONFIG
from core.market_data import market_data
from core.strategy import strategy_manager
from core.risk import risk_manager

class SettingsPage:
    def __init__(self):
        self.initialize_state()
        
    def initialize_state(self):
        if 'settings_changed' not in st.session_state:
            st.session_state.settings_changed = False
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = 'Trading'
            
    def render(self):
        """Render settings page"""
        # Settings tabs
        tabs = st.tabs([
            'Trading',
            'Risk Management',
            'Data Sources',
            'Notifications',
            'System'
        ])
        
        # Trading Settings
        with tabs[0]:
            self.render_trading_settings()
            
        # Risk Management Settings
        with tabs[1]:
            self.render_risk_settings()
            
        # Data Source Settings
        with tabs[2]:
            self.render_data_settings()
            
        # Notification Settings
        with tabs[3]:
            self.render_notification_settings()
            
        # System Settings
        with tabs[4]:
            self.render_system_settings()
            
        # Save Settings
        if st.session_state.settings_changed:
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Save Settings", type="primary"):
                    self.save_settings()
                    st.success("Settings saved successfully!")
                    st.session_state.settings_changed = False
                    
            with col2:
                if st.button("Reset to Default"):
                    self.reset_settings()
                    st.success("Settings reset to default values!")
                    st.experimental_rerun()
                    
    def render_trading_settings(self):
        """Render trading settings section"""
        st.subheader("Trading Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Capital Settings
            st.write("Capital Settings")
            
            new_capital = st.number_input(
                "Initial Capital",
                min_value=10000,
                value=CONFIG.CAPITAL,
                step=10000,
                help="Initial trading capital"
            )
            
            if new_capital != CONFIG.CAPITAL:
                CONFIG.CAPITAL = new_capital
                st.session_state.settings_changed = True
                
            min_trade = st.number_input(
                "Minimum Trade Amount",
                min_value=1000,
                value=CONFIG.MIN_TRADE_AMOUNT,
                step=1000,
                help="Minimum amount per trade"
            )
            
            if min_trade != CONFIG.MIN_TRADE_AMOUNT:
                CONFIG.MIN_TRADE_AMOUNT = min_trade
                st.session_state.settings_changed = True
                
        with col2:
            # Trading Schedule
            st.write("Trading Schedule")
            
            market_start = st.time_input(
                "Market Start Time",
                value=CONFIG.MARKET_START,
                help="Market opening time"
            )
            
            if market_start != CONFIG.MARKET_START:
                CONFIG.MARKET_START = market_start
                st.session_state.settings_changed = True
                
            market_end = st.time_input(
                "Market End Time",
                value=CONFIG.MARKET_END,
                help="Market closing time"
            )
            
            if market_end != CONFIG.MARKET_END:
                CONFIG.MARKET_END = market_end
                st.session_state.settings_changed = True
                
        # Position Settings
        st.write("Position Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_positions = st.number_input(
                "Maximum Positions",
                min_value=1,
                value=CONFIG.MAX_POSITIONS,
                help="Maximum number of concurrent positions"
            )
            
            if max_positions != CONFIG.MAX_POSITIONS:
                CONFIG.MAX_POSITIONS = max_positions
                st.session_state.settings_changed = True
                
        with col2:
            position_size = st.slider(
                "Position Size (%)",
                min_value=1,
                max_value=100,
                value=int(CONFIG.POSITION_SIZE * 100),
                help="Maximum position size as percentage of capital"
            )
            
            if position_size/100 != CONFIG.POSITION_SIZE:
                CONFIG.POSITION_SIZE = position_size/100
                st.session_state.settings_changed = True
                
        with col3:
            pyramiding = st.checkbox(
                "Enable Pyramiding",
                value=CONFIG.PYRAMIDING['ENABLED'],
                help="Allow adding to winning positions"
            )
            
            if pyramiding != CONFIG.PYRAMIDING['ENABLED']:
                CONFIG.PYRAMIDING['ENABLED'] = pyramiding
                st.session_state.settings_changed = True
                
        if pyramiding:
            col1, col2 = st.columns(2)
            
            with col1:
                max_adds = st.number_input(
                    "Maximum Adds",
                    min_value=1,
                    value=CONFIG.PYRAMIDING['MAX_ADDS'],
                    help="Maximum number of position additions"
                )
                
                if max_adds != CONFIG.PYRAMIDING['MAX_ADDS']:
                    CONFIG.PYRAMIDING['MAX_ADDS'] = max_adds
                    st.session_state.settings_changed = True
                    
            with col2:
                min_profit = st.slider(
                    "Minimum Profit for Adding (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=float(CONFIG.PYRAMIDING['MIN_PROFIT'] * 100),
                    help="Minimum profit required before adding"
                )
                
                if min_profit/100 != CONFIG.PYRAMIDING['MIN_PROFIT']:
                    CONFIG.PYRAMIDING['MIN_PROFIT'] = min_profit/100
                    st.session_state.settings_changed = True
                    
    def render_risk_settings(self):
        """Render risk management settings section"""
        st.subheader("Risk Management Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Position Risk Settings
            st.write("Position Risk Settings")
            
            stop_loss = st.slider(
                "Stop Loss (%)",
                min_value=0.1,
                max_value=10.0,
                value=float(CONFIG.STOP_LOSS_PCT * 100),
                help="Default stop loss percentage"
            )
            
            if stop_loss/100 != CONFIG.STOP_LOSS_PCT:
                CONFIG.STOP_LOSS_PCT = stop_loss/100
                st.session_state.settings_changed = True
                
            take_profit = st.slider(
                "Take Profit (%)",
                min_value=0.1,
                max_value=20.0,
                value=float(CONFIG.TAKE_PROFIT_PCT * 100),
                help="Default take profit percentage"
            )
            
            if take_profit/100 != CONFIG.TAKE_PROFIT_PCT:
                CONFIG.TAKE_PROFIT_PCT = take_profit/100
                st.session_state.settings_changed = True
                
        with col2:
            # Portfolio Risk Settings
            st.write("Portfolio Risk Settings")
            
            max_risk = st.slider(
                "Maximum Portfolio Risk (%)",
                min_value=1.0,
                max_value=50.0,
                value=float(CONFIG.risk.max_portfolio_risk * 100),
                help="Maximum portfolio risk exposure"
            )
            
            if max_risk/100 != CONFIG.risk.max_portfolio_risk:
                CONFIG.risk.max_portfolio_risk = max_risk/100
                st.session_state.settings_changed = True
                
            daily_loss = st.slider(
                "Maximum Daily Loss (%)",
                min_value=1.0,
                max_value=20.0,
                value=float(CONFIG.RISK['MAX_DAILY_LOSS'] * 100),
                help="Maximum allowed daily loss"
            )
            
            if daily_loss/100 != CONFIG.RISK['MAX_DAILY_LOSS']:
                CONFIG.RISK['MAX_DAILY_LOSS'] = daily_loss/100
                st.session_state.settings_changed = True
                
        # Advanced Risk Settings
        st.write("Advanced Risk Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_corr = st.slider(
                "Maximum Correlation",
                min_value=0.0,
                max_value=1.0,
                value=float(CONFIG.risk.max_correlation),
                help="Maximum allowed position correlation"
            )
            
            if max_corr != CONFIG.risk.max_correlation:
                CONFIG.risk.max_correlation = max_corr
                st.session_state.settings_changed = True
                
        with col2:
            trailing_stop = st.slider(
                "Trailing Stop (%)",
                min_value=0.1,
                max_value=10.0,
                value=float(CONFIG.RISK['TRAILING_STOP'] * 100),
                help="Trailing stop percentage"
            )
            
            if trailing_stop/100 != CONFIG.RISK['TRAILING_STOP']:
                CONFIG.RISK['TRAILING_STOP'] = trailing_stop/100
                st.session_state.settings_changed = True
                
        with col3:
            profit_lock = st.slider(
                "Profit Lock (%)",
                min_value=0.1,
                max_value=10.0,
                value=float(CONFIG.RISK['PROFIT_LOCK'] * 100),
                help="Profit locking percentage"
            )
            
            if profit_lock/100 != CONFIG.RISK['PROFIT_LOCK']:
                CONFIG.RISK['PROFIT_LOCK'] = profit_lock/100
                st.session_state.settings_changed = True
                
    def render_data_settings(self):
        """Render data source settings section"""
        st.subheader("Data Source Settings")
        
        # Data Sources
        st.write("Data Sources")
        
        col1, col2 = st.columns(2)
        
        with col1:
            primary_source = st.selectbox(
                "Primary Data Source",
                ['OPENBB', 'NSE', 'BSE', 'YAHOO'],
                index=['OPENBB', 'NSE', 'BSE', 'YAHOO'].index(CONFIG.data.primary_source),
                help="Primary market data source"
            )
            
            if primary_source != CONFIG.data.primary_source:
                CONFIG.data.primary_source = primary_source
                st.session_state.settings_changed = True
                
        with col2:
            backup_sources = st.multiselect(
                "Backup Data Sources",
                ['OPENBB', 'NSE', 'BSE', 'YAHOO'],
                default=CONFIG.data.backup_sources or [],
                help="Backup data sources"
            )
            
            if backup_sources != CONFIG.data.backup_sources:
                CONFIG.data.backup_sources = backup_sources
                st.session_state.settings_changed = True
                
        # Data Update Settings
        st.write("Data Update Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            update_interval = st.number_input(
                "Update Interval (seconds)",
                min_value=1,
                value=CONFIG.SCAN_INTERVAL,
                help="Market data update interval"
            )
            
            if update_interval != CONFIG.SCAN_INTERVAL:
                CONFIG.SCAN_INTERVAL = update_interval
                st.session_state.settings_changed = True
                
        with col2:
            min_history = st.number_input(
                "Minimum History (days)",
                min_value=1,
                value=CONFIG.data.min_history,
                help="Minimum required historical data"
            )
            
            if min_history != CONFIG.data.min_history:
                CONFIG.data.min_history = min_history
                st.session_state.settings_changed = True
                
    def render_notification_settings(self):
        """Render notification settings section"""
        st.subheader("Notification Settings")
        
        # Email Notifications
        st.write("Email Notifications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            email_enabled = st.checkbox(
                "Enable Email Notifications",
                value=CONFIG.get('EMAIL_ENABLED', False),
                help="Enable email notifications"
            )
            
            if email_enabled != CONFIG.get('EMAIL_ENABLED', False):
                CONFIG.EMAIL_ENABLED = email_enabled
                st.session_state.settings_changed = True
                
            if email_enabled:
                email = st.text_input(
                    "Email Address",
                    value=CONFIG.get('EMAIL_ADDRESS', ''),
                    help="Notification email address"
                )
                
                if email != CONFIG.get('EMAIL_ADDRESS', ''):
                    CONFIG.EMAIL_ADDRESS = email
                    st.session_state.settings_changed = True
                    
        with col2:
            if email_enabled:
                st.write("Notification Types")
                
                trade_notify = st.checkbox(
                    "Trade Notifications",
                    value=CONFIG.get('TRADE_NOTIFICATIONS', True),
                    help="Notify on trade execution"
                )
                
                if trade_notify != CONFIG.get('TRADE_NOTIFICATIONS', True):
                    CONFIG.TRADE_NOTIFICATIONS = trade_notify
                    st.session_state.settings_changed = True
                    
                signal_notify = st.checkbox(
                    "Signal Notifications",
                    value=CONFIG.get('SIGNAL_NOTIFICATIONS', True),
                    help="Notify on signal generation"
                )
                
                if signal_notify != CONFIG.get('SIGNAL_NOTIFICATIONS', True):
                    CONFIG.SIGNAL_NOTIFICATIONS = signal_notify
                    st.session_state.settings_changed = True
                    
                error_notify = st.checkbox(
                    "Error Notifications",
                    value=CONFIG.get('ERROR_NOTIFICATIONS', True),
                    help="Notify on system errors"
                )
                
                if error_notify != CONFIG.get('ERROR_NOTIFICATIONS', True):
                    CONFIG.ERROR_NOTIFICATIONS = error_notify
                    st.session_state.settings_changed = True
                    
    def render_system_settings(self):
        """Render system settings section"""
        st.subheader("System Settings")
        
        # Logging Settings
        st.write("Logging Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            log_level = st.selectbox(
                "Log Level",
                ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                index=['DEBUG', 'INFO', 'WARNING', 'ERROR'].index(
                    CONFIG.get('LOG_LEVEL', 'INFO')
                ),
                help="Logging detail level"
            )
            
            if log_level != CONFIG.get('LOG_LEVEL', 'INFO'):
                CONFIG.LOG_LEVEL = log_level
                st.session_state.settings_changed = True
                
        with col2:
            log_retention = st.number_input(
                "Log Retention (days)",
                min_value=1,
                value=CONFIG.get('LOG_RETENTION', 30),
                help="Number of days to keep logs"
            )
            
            
            if log_retention != CONFIG.get('LOG_RETENTION', 30):
                CONFIG.LOG_RETENTION = log_retention
                st.session_state.settings_changed = True
                
        # Data Storage Settings
        st.write("Data Storage Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            data_dir = st.text_input(
                "Data Directory",
                value=str(CONFIG.DATA_DIR),
                help="Directory for storing data files"
            )
            
            if Path(data_dir) != CONFIG.DATA_DIR:
                CONFIG.DATA_DIR = Path(data_dir)
                st.session_state.settings_changed = True
                
        with col2:
            cache_size = st.number_input(
                "Cache Size (MB)",
                min_value=100,
                value=CONFIG.get('CACHE_SIZE', 500),
                help="Maximum cache size in megabytes"
            )
            
            if cache_size != CONFIG.get('CACHE_SIZE', 500):
                CONFIG.CACHE_SIZE = cache_size
                st.session_state.settings_changed = True
                
        # Performance Settings
        st.write("Performance Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_workers = st.number_input(
                "Max Worker Threads",
                min_value=1,
                max_value=32,
                value=CONFIG.get('MAX_WORKERS', 10),
                help="Maximum number of worker threads"
            )
            
            if max_workers != CONFIG.get('MAX_WORKERS', 10):
                CONFIG.MAX_WORKERS = max_workers
                st.session_state.settings_changed = True
                
        with col2:
            queue_size = st.number_input(
                "Queue Size",
                min_value=100,
                value=CONFIG.get('QUEUE_SIZE', 1000),
                help="Maximum event queue size"
            )
            
            if queue_size != CONFIG.get('QUEUE_SIZE', 1000):
                CONFIG.QUEUE_SIZE = queue_size
                st.session_state.settings_changed = True
                
    def save_settings(self):
        """Save current settings to config file"""
        try:
            config_data = {
                'CAPITAL': CONFIG.CAPITAL,
                'MIN_TRADE_AMOUNT': CONFIG.MIN_TRADE_AMOUNT,
                'MARKET_START': CONFIG.MARKET_START.strftime('%H:%M'),
                'MARKET_END': CONFIG.MARKET_END.strftime('%H:%M'),
                'MAX_POSITIONS': CONFIG.MAX_POSITIONS,
                'POSITION_SIZE': CONFIG.POSITION_SIZE,
                'PYRAMIDING': CONFIG.PYRAMIDING,
                'STOP_LOSS_PCT': CONFIG.STOP_LOSS_PCT,
                'TAKE_PROFIT_PCT': CONFIG.TAKE_PROFIT_PCT,
                'RISK': CONFIG.risk.__dict__,
                'DATA': CONFIG.data.__dict__,
                'SCAN_INTERVAL': CONFIG.SCAN_INTERVAL,
                'LOG_LEVEL': CONFIG.get('LOG_LEVEL', 'INFO'),
                'LOG_RETENTION': CONFIG.get('LOG_RETENTION', 30),
                'CACHE_SIZE': CONFIG.get('CACHE_SIZE', 500),
                'MAX_WORKERS': CONFIG.get('MAX_WORKERS', 10),
                'QUEUE_SIZE': CONFIG.get('QUEUE_SIZE', 1000),
                'EMAIL_ENABLED': CONFIG.get('EMAIL_ENABLED', False),
                'EMAIL_ADDRESS': CONFIG.get('EMAIL_ADDRESS', ''),
                'TRADE_NOTIFICATIONS': CONFIG.get('TRADE_NOTIFICATIONS', True),
                'SIGNAL_NOTIFICATIONS': CONFIG.get('SIGNAL_NOTIFICATIONS', True),
                'ERROR_NOTIFICATIONS': CONFIG.get('ERROR_NOTIFICATIONS', True)
            }
            
            # Save to YAML file
            config_path = CONFIG.BASE_DIR / 'config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
                
        except Exception as e:
            st.error(f"Error saving settings: {e}")
            
    def reset_settings(self):
        """Reset settings to default values"""
        try:
            # Delete existing config file
            config_path = CONFIG.BASE_DIR / 'config.yaml'
            if config_path.exists():
                os.remove(config_path)
                
            # Reinitialize config
            CONFIG.__init__()
            
        except Exception as e:
            st.error(f"Error resetting settings: {e}")
