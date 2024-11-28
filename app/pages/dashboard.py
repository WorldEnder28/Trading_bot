# app/pages/dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

from core.portfolio import portfolio_manager
from core.market_data import market_data
from core.risk import risk_manager
from core.config import CONFIG

class DashboardPage:
    def __init__(self):
        self.initialize_state()
        
    def initialize_state(self):
        if 'selected_timeframe' not in st.session_state:
            st.session_state.selected_timeframe = '1D'
            
    def render(self):
        """Render dashboard page"""
        # Market Overview
        self.render_market_overview()
        
        # Portfolio Summary
        col1, col2 = st.columns([2,1])
        with col1:
            self.render_portfolio_chart()
        with col2:
            self.render_portfolio_metrics()
            
        # Trading Activity
        self.render_trading_activity()
        
    def render_market_overview(self):
        """Render market overview section"""
        st.subheader("Market Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Mock data for demonstration
        with col1:
            st.metric(
                "Nifty 50",
                "19,674.25",
                "0.75%"
            )
            
        with col2:
            st.metric(
                "Bank Nifty",
                "44,237.90",
                "0.92%"
            )
            
        with col3:
            st.metric(
                "Market Breadth",
                "1.45",
                "0.12"
            )
            
        with col4:
            st.metric(
                "VIX",
                "13.25",
                "-0.45"
            )
            
    def render_portfolio_chart(self):
        """Render portfolio performance chart"""
        st.subheader("Portfolio Performance")
        
        # Get equity curve data
        equity_data = pd.DataFrame(portfolio_manager.equity_curve)
        
        if not equity_data.empty:
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('Portfolio Value', 'Daily Returns'),
                row_heights=[0.7, 0.3]
            )
            
            # Portfolio value
            fig.add_trace(
                go.Scatter(
                    x=equity_data['timestamp'],
                    y=equity_data['equity'],
                    name='Portfolio Value'
                ),
                row=1, col=1
            )
            
            # Daily returns
            returns = pd.Series(
                equity_data['equity'].values,
                index=equity_data['timestamp']
            ).pct_change()
            
            fig.add_trace(
                go.Bar(
                    x=returns.index,
                    y=returns * 100,
                    name='Daily Returns'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=500,
                showlegend=True,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No portfolio data available")
            
    def render_portfolio_metrics(self):
        """Render portfolio metrics"""
        st.subheader("Portfolio Metrics")
        
        metrics = portfolio_manager.get_performance_metrics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Total Return",
                f"{metrics.get('total_return', 0):.2f}%"
            )
            st.metric(
                "Sharpe Ratio",
                f"{metrics.get('sharpe_ratio', 0):.2f}"
            )
            st.metric(
                "Max Drawdown",
                f"{metrics.get('max_drawdown', 0):.2f}%"
            )
            
        with col2:
            st.metric(
                "Win Rate",
                f"{metrics.get('win_rate', 0):.2f}%"
            )
            st.metric(
                "Profit Factor",
                f"{metrics.get('profit_factor', 0):.2f}"
            )
            st.metric(
                "Volatility",
                f"{metrics.get('volatility', 0):.2f}%"
            )
            
    def render_trading_activity(self):
        """Render trading activity section"""
        st.subheader("Recent Trading Activity")
        
        trades = portfolio_manager.trades[-10:]  # Last 10 trades
        if trades:
            trade_data = []
            for trade in trades:
                trade_data.append({
                    'Time': trade.exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                    'Symbol': trade.symbol,
                    'Side': trade.side,
                    'Quantity': trade.quantity,
                    'Entry': f"₹{trade.entry_price:,.2f}",
                    'Exit': f"₹{trade.exit_price:,.2f}",
                    'P&L': f"₹{trade.pnl:,.2f}",
                    'Return': f"{trade.percent_return:.2f}%",
                    'Strategy': trade.strategy
                })
                
            df = pd.DataFrame(trade_data)
            
            # Style dataframe
            def highlight_pnl(val):
                if 'P&L' in val.name or 'Return' in val.name:
                    try:
                        value = float(val.strip('%').strip('₹').replace(',', ''))
                        return ['color: red' if value < 0 else 'color: green' for _ in val]
                    except:
                        return [''] * len(val)
                return [''] * len(val)
                
            styled_df = df.style.apply(highlight_pnl)
            
            st.dataframe(
                styled_df,
                height=400,
                use_container_width=True
            )
        else:
            st.info("No recent trades")
