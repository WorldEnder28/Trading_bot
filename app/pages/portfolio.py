# app/pages/portfolio.py
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

class PortfolioPage:
    def __init__(self):
        self.initialize_state()
        
    def initialize_state(self):
        if 'selected_period' not in st.session_state:
            st.session_state.selected_period = '1M'
        if 'show_closed_positions' not in st.session_state:
            st.session_state.show_closed_positions = False
            
    def render(self):
        """Render portfolio page"""
        # Portfolio Overview
        self.render_portfolio_overview()
        
        # Holdings Breakdown
        self.render_holdings_breakdown()
        
        # Performance Analysis
        col1, col2 = st.columns([2,1])
        with col1:
            self.render_performance_analysis()
        with col2:
            self.render_allocation_analysis()
            
        # Trade History
        self.render_trade_history()
        
    def render_portfolio_overview(self):
        """Render portfolio overview section"""
        st.subheader("Portfolio Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        portfolio_value = portfolio_manager.get_total_value()
        total_pnl = portfolio_value - CONFIG.CAPITAL
        
        with col1:
            st.metric(
                "Total Value",
                f"₹{portfolio_value:,.2f}",
                f"{(total_pnl / CONFIG.CAPITAL) * 100:.2f}%"
            )
            
        with col2:
            daily_pnl = portfolio_manager.get_daily_pnl()
            st.metric(
                "Day's P&L",
                f"₹{daily_pnl:,.2f}",
                f"{(daily_pnl / portfolio_value) * 100:.2f}%"
            )
            
        with col3:
            cash = portfolio_manager.cash
            st.metric(
                "Available Cash",
                f"₹{cash:,.2f}",
                f"{(cash / portfolio_value) * 100:.2f}%"
            )
            
        with col4:
            margin_used = portfolio_value - cash
            st.metric(
                "Margin Used",
                f"₹{margin_used:,.2f}",
                f"{(margin_used / portfolio_value) * 100:.2f}%"
            )
            
        # Portfolio chart
        self.render_portfolio_chart()
        
    def render_portfolio_chart(self):
        """Render portfolio value chart"""
        equity_curve = pd.DataFrame(portfolio_manager.equity_curve)
        if equity_curve.empty:
            st.info("No portfolio data available")
            return
            
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=('Portfolio Value', 'Drawdown')
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=equity_curve['timestamp'],
                y=equity_curve['equity'],
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Calculate and plot drawdown
        peak = equity_curve['equity'].expanding().max()
        drawdown = ((equity_curve['equity'] - peak) / peak) * 100
        
        fig.add_trace(
            go.Scatter(
                x=equity_curve['timestamp'],
                y=drawdown,
                name='Drawdown',
                fill='tonexty',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def render_holdings_breakdown(self):
        """Render holdings breakdown section"""
        st.subheader("Holdings Breakdown")
        
        positions = portfolio_manager.positions
        if not positions:
            st.info("No active positions")
            return
            
        holdings_data = []
        for symbol, position in positions.items():
            data = market_data.get_stock_data(symbol)
            if data is not None and not data.empty:
                current_price = data['Close'].iloc[-1]
                market_value = position.quantity * current_price
                cost_basis = position.quantity * position.average_price
                unrealized_pnl = position.unrealized_pnl
                
                holdings_data.append({
                    'Symbol': symbol,
                    'Quantity': position.quantity,
                    'Avg Cost': f"₹{position.average_price:,.2f}",
                    'Current Price': f"₹{current_price:,.2f}",
                    'Market Value': f"₹{market_value:,.2f}",
                    'Cost Basis': f"₹{cost_basis:,.2f}",
                    'Unrealized P&L': f"₹{unrealized_pnl:,.2f}",
                    'Return %': f"{(unrealized_pnl / cost_basis) * 100:.2f}%",
                    'Weight %': f"{(market_value / portfolio_manager.get_total_value()) * 100:.2f}%",
                    'Strategy': position.strategy
                })
                
        df = pd.DataFrame(holdings_data)
        
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
        
    def render_performance_analysis(self):
        """Render performance analysis section"""
        st.subheader("Performance Analysis")
        
        # Time period selector
        period = st.selectbox(
            "Select Period",
            ['1D', '1W', '1M', '3M', 'YTD', '1Y', 'ALL'],
            index=2
        )
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(period)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Return", f"{metrics['total_return']:.2f}%")
            st.metric("Annualized Return", f"{metrics['annual_return']:.2f}%")
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            
        with col2:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
            st.metric("Recovery Time", f"{metrics['recovery_time']} days")
            st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
            
        with col3:
            st.metric("Win Rate", f"{metrics['win_rate']:.2f}%")
            st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
            st.metric("Average Trade", f"₹{metrics['avg_trade']:,.2f}")
            
        # Performance chart
        self.render_performance_chart(period)
        
    def render_performance_chart(self, period: str):
        """Render performance analysis chart"""
        equity_curve = pd.DataFrame(portfolio_manager.equity_curve)
        if equity_curve.empty:
            return
            
        # Filter data based on period
        start_date = self.get_start_date(period)
        equity_curve = equity_curve[equity_curve['timestamp'] >= start_date]
        
        # Calculate returns
        returns = pd.Series(
            equity_curve['equity'].values,
            index=equity_curve['timestamp']
        ).pct_change()
        
        # Create figure
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Cumulative Returns', 'Daily Returns'),
            row_heights=[0.7, 0.3]
        )
        
        # Cumulative returns
        cumulative_returns = (1 + returns).cumprod() - 1
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns * 100,
                name='Cumulative Returns'
            ),
            row=1, col=1
        )
        
        # Daily returns
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
        
    def render_allocation_analysis(self):
        """Render allocation analysis section"""
        st.subheader("Portfolio Allocation")
        
        positions = portfolio_manager.positions
        if not positions:
            st.info("No positions to analyze")
            return
            
        # Calculate allocations
        allocations = self.calculate_allocations()
        
        # Strategy allocation
        fig1 = go.Figure(data=[go.Pie(
            labels=list(allocations['strategy'].keys()),
            values=list(allocations['strategy'].values()),
            hole=.3
        )])
        
        fig1.update_layout(
            title="Strategy Allocation",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Sector allocation
        fig2 = go.Figure(data=[go.Pie(
            labels=list(allocations['sector'].keys()),
            values=list(allocations['sector'].values()),
            hole=.3
        )])
        
        fig2.update_layout(
            title="Sector Allocation",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
    def render_trade_history(self):
        """Render trade history section"""
        st.subheader("Trade History")
        
        # Show closed positions toggle
        show_closed = st.checkbox(
            "Show Closed Positions",
            value=st.session_state.show_closed_positions
        )
        st.session_state.show_closed_positions = show_closed
        
        trades = portfolio_manager.trades
        if not trades:
            st.info("No trade history available")
            return
            
        trades_data = []
        for trade in trades:
            if show_closed or trade.exit_time is None:
                trades_data.append({
                    'Exit Time': trade.exit_time.strftime("%Y-%m-%d %H:%M:%S") if trade.exit_time else "Open",
                    'Symbol': trade.symbol,
                    'Side': trade.side,
                    'Quantity': trade.quantity,
                    'Entry Price': f"₹{trade.entry_price:,.2f}",
                    'Exit Price': f"₹{trade.exit_price:,.2f}" if trade.exit_price else "N/A",
                    'P&L': f"₹{trade.pnl:,.2f}" if trade.pnl else "N/A",
                    'Return %': f"{trade.percent_return:.2f}%" if trade.percent_return else "N/A",
                    'Strategy': trade.strategy
                })
                
        df = pd.DataFrame(trades_data)
        
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
        
        # Trade analysis
        self.render_trade_analysis()
        
    def render_trade_analysis(self):
        """Render trade analysis charts"""
        trades = portfolio_manager.trades
        if not trades:
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            # Profit distribution
            profits = [t.pnl for t in trades if t.pnl is not None]
            
            fig1 = go.Figure(data=[go.Histogram(
                x=profits,
                nbinsx=20,
                name='P&L Distribution'
            )])
            
            fig1.update_layout(
                title="P&L Distribution",
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            # Trade returns distribution
            returns = [t.percent_return for t in trades if t.percent_return is not None]
            
            fig2 = go.Figure(data=[go.Histogram(
                x=returns,
                nbinsx=20,
                name='Returns Distribution'
            )])
            
            fig2.update_layout(
                title="Returns Distribution",
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
    def calculate_performance_metrics(self, period: str) -> dict:
        """Calculate performance metrics for selected period"""
        try:
            start_date = self.get_start_date(period)
            
            # Filter trades
            trades = [
                t for t in portfolio_manager.trades
                if t.exit_time >= start_date
            ]
            
            if not trades:
                return {
                    'total_return': 0,
                    'annual_return': 0,
                    'sharpe_ratio': 0,
                    'sortino_ratio': 0,
                    'max_drawdown': 0,
                    'recovery_time': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'avg_trade': 0
                }
                
            # Calculate metrics
            total_pnl = sum(t.pnl for t in trades)
            total_return = (total_pnl / CONFIG.CAPITAL) * 100
            
            # Annualized return
            days = (datetime.now() - start_date).days
            annual_return = ((1 + total_return/100) ** (365/days) - 1) * 100 if days > 0 else 0
            
            # Calculate daily returns
            daily_returns = pd.Series([t.pnl for t in trades]).resample('D').sum()
            
            # Sharpe ratio
            excess_returns = daily_returns - (CONFIG.risk.risk_free_rate/252)
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if len(daily_returns) > 1 else 0
            
            # Sortino ratio
            downside_returns = excess_returns[excess_returns < 0]
            sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std() if len(downside_returns) > 1 else 0
            
            # Max drawdown and recovery
            cumulative = daily_returns.cumsum()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max)
            max_drawdown = (drawdown / running_max).min() * 100 if len(running_max) > 0 else 0
            
            # Recovery time calculation
            underwater = drawdown < 0
            recovery_time = underwater.groupby((~underwater).cumsum()).sum().max()
            
            # Win rate and profit factor
            winning_trades = sum(1 for t in trades if t.pnl > 0)
            win_rate = (winning_trades / len(trades)) * 100
            
            gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
            # Average trade
            avg_trade = total_pnl / len(trades)
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'recovery_time': int(recovery_time),
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_trade': avg_trade
            }
            
        except Exception as e:
            st.error(f"Error calculating performance metrics: {e}")
            return {
                'total_return': 0,
                'annual_return': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'recovery_time': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_trade': 0
            }

    def calculate_allocations(self) -> dict:
        """Calculate portfolio allocations"""
        try:
            positions = portfolio_manager.positions
            if not positions:
                return {'strategy': {}, 'sector': {}}
                
            total_value = portfolio_manager.get_total_value()
            
            # Strategy allocation
            strategy_allocation = {}
            sector_allocation = {}
            
            for symbol, position in positions.items():
                market_value = position.quantity * position.average_price
                weight = market_value / total_value
                
                # Strategy allocation
                strategy = position.strategy or 'Unknown'
                strategy_allocation[strategy] = strategy_allocation.get(strategy, 0) + weight
                
                # Get sector information
                stock_info = market_data.get_stock_info(symbol)
                sector = stock_info.get('sector', 'Unknown')
                sector_allocation[sector] = sector_allocation.get(sector, 0) + weight
                
            return {
                'strategy': strategy_allocation,
                'sector': sector_allocation
            }
            
        except Exception as e:
            st.error(f"Error calculating allocations: {e}")
            return {'strategy': {}, 'sector': {}}

    def get_start_date(self, period: str) -> datetime:
        """Get start date based on selected period"""
        now = datetime.now()
        
        if period == '1D':
            return now - timedelta(days=1)
        elif period == '1W':
            return now - timedelta(weeks=1)
        elif period == '1M':
            return now - timedelta(days=30)
        elif period == '3M':
            return now - timedelta(days=90)
        elif period == 'YTD':
            return datetime(now.year, 1, 1)
        elif period == '1Y':
            return now - timedelta(days=365)
        else:  # ALL
            return datetime(2000, 1, 1)
