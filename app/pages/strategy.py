# app/pages/strategy.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

from core.strategy import strategy_manager
from core.market_data import market_data
from core.portfolio import portfolio_manager
from core.config import CONFIG

class StrategyPage:
    def __init__(self):
        self.initialize_state()
        
    def initialize_state(self):
        if 'selected_strategy' not in st.session_state:
            st.session_state.selected_strategy = 'All'
        if 'selected_timeframe' not in st.session_state:
            st.session_state.selected_timeframe = '1D'
        if 'show_inactive' not in st.session_state:
            st.session_state.show_inactive = False
            
    def render(self):
        """Render strategy page"""
        # Strategy Controls
        self.render_strategy_controls()
        
        # Strategy Performance
        self.render_strategy_performance()
        
        # Signal Analysis
        col1, col2 = st.columns([2,1])
        with col1:
            self.render_signal_analysis()
        with col2:
            self.render_strategy_metrics()
            
        # Strategy Settings
        self.render_strategy_settings()
        
    def render_strategy_controls(self):
        """Render strategy control section"""
        st.subheader("Strategy Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            strategies = ['All'] + list(strategy_manager.strategies.keys())
            selected_strategy = st.selectbox(
                "Select Strategy",
                strategies,
                index=strategies.index(st.session_state.selected_strategy)
            )
            st.session_state.selected_strategy = selected_strategy
            
        with col2:
            timeframes = ['1D', '1W', '1M', '3M', 'YTD', '1Y', 'ALL']
            selected_timeframe = st.selectbox(
                "Select Timeframe",
                timeframes,
                index=timeframes.index(st.session_state.selected_timeframe)
            )
            st.session_state.selected_timeframe = selected_timeframe
            
        with col3:
            show_inactive = st.checkbox(
                "Show Inactive Strategies",
                value=st.session_state.show_inactive
            )
            st.session_state.show_inactive = show_inactive
            
        with col4:
            if st.button("Reset Strategy"):
                self.reset_strategy()
                
    def render_strategy_performance(self):
        """Render strategy performance section"""
        st.subheader("Strategy Performance")
        
        # Get performance data
        performance_data = self.get_strategy_performance()
        if not performance_data:
            st.info("No performance data available")
            return
            
        # Create metrics table
        metrics_data = []
        for strategy, metrics in performance_data.items():
            if not st.session_state.show_inactive and metrics['status'] == 'Inactive':
                continue
                
            metrics_data.append({
                'Strategy': strategy,
                'Total Return': f"{metrics['total_return']:.2f}%",
                'Win Rate': f"{metrics['win_rate']:.2f}%",
                'Profit Factor': f"{metrics['profit_factor']:.2f}",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{metrics['max_drawdown']:.2f}%",
                'Total Trades': metrics['total_trades'],
                'Active Signals': metrics['active_signals'],
                'Status': metrics['status']
            })
            
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            
            # Style dataframe
            def highlight_status(val):
                if val.name == 'Status':
                    return ['background-color: #28a745' if x == 'Active' else 'background-color: #dc3545' for x in val]
                return [''] * len(val)
                
            styled_df = df.style.apply(highlight_status)
            
            st.dataframe(
                styled_df,
                height=400,
                use_container_width=True
            )
            
            # Render performance chart
            self.render_performance_chart(performance_data)
            
    def render_performance_chart(self, performance_data: dict):
        """Render strategy performance comparison chart"""
        # Create figure
        fig = go.Figure()
        
        for strategy, metrics in performance_data.items():
            if not st.session_state.show_inactive and metrics['status'] == 'Inactive':
                continue
                
            equity_curve = metrics['equity_curve']
            if not equity_curve.empty:
                fig.add_trace(go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve['returns'] * 100,
                    name=strategy,
                    mode='lines'
                ))
                
        fig.update_layout(
            title="Strategy Performance Comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def render_signal_analysis(self):
        """Render signal analysis section"""
        st.subheader("Signal Analysis")
        
        # Get signal data
        signals = self.get_strategy_signals()
        if not signals:
            st.info("No signals available")
            return
            
        # Create signals table
        signals_data = []
        for signal in signals:
            signals_data.append({
                'Time': signal.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'Symbol': signal.symbol,
                'Direction': signal.direction,
                'Strength': f"{signal.strength:.2f}",
                'Strategy': signal.strategy,
                'Price': f"₹{signal.price:,.2f}",
                'Status': 'Active' if self.is_signal_active(signal) else 'Expired'
            })
            
        df = pd.DataFrame(signals_data)
        
        # Style dataframe
        def highlight_direction(val):
            if val.name == 'Direction':
                return ['color: green' if x == 'LONG' else 'color: red' for x in val]
            return [''] * len(val)
            
        styled_df = df.style.apply(highlight_direction)
        
        st.dataframe(
            styled_df,
            height=400,
            use_container_width=True
        )
        
        # Signal distribution chart
        self.render_signal_distribution(signals)
        
    def render_signal_distribution(self, signals: list):
        """Render signal distribution charts"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Direction distribution
            direction_counts = pd.Series([s.direction for s in signals]).value_counts()
            
            fig1 = go.Figure(data=[go.Pie(
                labels=direction_counts.index,
                values=direction_counts.values,
                hole=.3
            )])
            
            fig1.update_layout(
                title="Signal Direction Distribution",
                height=300
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            # Strength distribution
            strengths = [s.strength for s in signals]
            
            fig2 = go.Figure(data=[go.Histogram(
                x=strengths,
                nbinsx=20,
                name='Signal Strength'
            )])
            
            fig2.update_layout(
                title="Signal Strength Distribution",
                height=300
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
    def render_strategy_metrics(self):
        """Render detailed strategy metrics"""
        st.subheader("Strategy Metrics")
        
        strategy = st.session_state.selected_strategy
        if strategy == 'All':
            return
            
        metrics = self.get_detailed_metrics(strategy)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Signal Accuracy", f"{metrics['signal_accuracy']:.2f}%")
            st.metric("Average Return per Trade", f"₹{metrics['avg_return']:,.2f}")
            st.metric("Average Holding Time", f"{metrics['avg_holding_time']:.1f} hours")
            
        with col2:
            st.metric("Signal Count", metrics['signal_count'])
            st.metric("Active Signals", metrics['active_signals'])
            st.metric("Success Rate", f"{metrics['success_rate']:.2f}%")
            
        # Time analysis chart
        self.render_time_analysis(strategy)
        
    def render_time_analysis(self, strategy: str):
        """Render time-based analysis chart"""
        trades = self.get_strategy_trades(strategy)
        if not trades:
            return
            
        # Analyze trade timing
        trade_times = pd.Series([t.entry_time.hour for t in trades])
        trade_returns = pd.Series([t.pnl for t in trades])
        
        # Create figure with dual y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Trade count by hour
        fig.add_trace(
            go.Bar(
                x=range(24),
                y=[trade_times.value_counts().get(h, 0) for h in range(24)],
                name="Trade Count"
            ),
            secondary_y=False
        )
        
        # Average return by hour
        avg_returns = []
        for hour in range(24):
            hour_returns = trade_returns[trade_times == hour]
            avg_returns.append(hour_returns.mean() if len(hour_returns) > 0 else 0)
            
        fig.add_trace(
            go.Scatter(
                x=range(24),
                y=avg_returns,
                name="Average Return",
                line=dict(color='red')
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title="Trade Analysis by Hour",
            xaxis_title="Hour of Day",
            height=300
        )
        
        fig.update_yaxes(title_text="Trade Count", secondary_y=False)
        fig.update_yaxes(title_text="Average Return (₹)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
    def render_strategy_settings(self):
        """Render strategy settings section"""
        st.subheader("Strategy Settings")
        
        strategy = st.session_state.selected_strategy
        if strategy == 'All':
            return
            
        # Get strategy parameters
        params = self.get_strategy_parameters(strategy)
        if not params:
            return
            
        # Create parameter inputs
        updated_params = {}
        cols = st.columns(3)
        
        for i, (param, value) in enumerate(params.items()):
            with cols[i % 3]:
                if isinstance(value, bool):
                    updated_params[param] = st.checkbox(param, value=value)
                elif isinstance(value, (int, float)):
                    updated_params[param] = st.number_input(param, value=value)
                elif isinstance(value, str):
                    updated_params[param] = st.text_input(param, value=value)
                    
        # Save button
        if st.button("Save Settings"):
            self.update_strategy_parameters(strategy, updated_params)
            st.success("Strategy settings updated successfully")
            
    def get_strategy_performance(self) -> dict:
        """Get strategy performance data"""
        try:
            performance = {}
            start_date = self.get_start_date(st.session_state.selected_timeframe)
            
            strategies = [st.session_state.selected_strategy] \
                if st.session_state.selected_strategy != 'All' \
                else strategy_manager.strategies.keys()
                
            for strategy in strategies:
                trades = self.get_strategy_trades(strategy)
                trades = [t for t in trades if t.exit_time >= start_date]
                
                if not trades and not st.session_state.show_inactive:
                    continue
                    
                # Calculate metrics
                total_pnl = sum(t.pnl for t in trades)
                winning_trades = sum(1 for t in trades if t.pnl > 0)
                
                performance[strategy] = {
                    'total_return': (total_pnl / CONFIG.CAPITAL) * 100,
                    'win_rate': (winning_trades / len(trades)) * 100 if trades else 0,
                    'profit_factor': self.calculate_profit_factor(trades),
                    'sharpe_ratio': self.calculate_sharpe_ratio(trades),
                    'max_drawdown': self.calculate_max_drawdown(trades),
                    'total_trades': len(trades),
                    'active_signals': len(self.get_active_signals(strategy)),
                    'status': 'Active' if strategy in strategy_manager.active_strategies else 'Inactive',
                    'equity_curve': self.calculate_equity_curve(trades)
                }
                
            return performance
            
        except Exception as e:
            st.error(f"Error calculating strategy performance: {e}")
            return {}

    def get_strategy_signals(self) -> list:
        """Get strategy signals"""
        try:
            strategy = st.session_state.selected_strategy
            if strategy == 'All':
                signals = []
                for s in strategy_manager.strategies.values():
                    signals.extend(s.signals)
            else:
                signals = strategy_manager.strategies[strategy].signals
                
            # Filter by timeframe
            start_date = self.get_start_date(st.session_state.selected_timeframe)
            signals = [s for s in signals if s.timestamp >= start_date]
            
            return signals
            
        except Exception as e:
            st.error(f"Error getting strategy signals: {e}")
            return []

    def get_strategy_trades(self, strategy: str) -> list:
        """Get trades for specific strategy"""
        try:
            return [t for t in portfolio_manager.trades if t.strategy == strategy]
        except Exception as e:
            st.error(f"Error getting strategy trades: {e}")
            return []

    def get_detailed_metrics(self, strategy: str) -> dict:
        """Get detailed strategy metrics"""
        try:
            trades = self.get_strategy_trades(strategy)
            signals = strategy_manager.strategies[strategy].signals
            
            # Calculate metrics
            signal_count = len(signals)
            active_signals = len(self.get_active_signals(strategy))
            
            if trades:
                success_rate = sum(1 for t in trades if t.pnl > 0) / len(trades) * 100
                avg_return = sum(t.pnl for t in trades) / len(trades)
                avg_holding_time = sum(
                    (t.exit_time - t.entry_time).total_seconds() / 3600 
                    for t in trades
                ) / len(trades)
            else:
                success_rate = 0
                avg_return = 0
                avg_holding_time = 0
                
            # Calculate signal accuracy
            if signals:
                accurate_signals = sum(
                    1 for s in signals 
                    if self.is_signal_accurate(s)
                )
                signal_accuracy = (accurate_signals / len(signals)) * 100
            else:
                signal_accuracy = 0
                
                return {
                'signal_count': signal_count,
                'active_signals': active_signals,
                'signal_accuracy': signal_accuracy,
                'success_rate': success_rate,
                'avg_return': avg_return,
                'avg_holding_time': avg_holding_time
            }
            
        except Exception as e:
            st.error(f"Error calculating detailed metrics: {e}")
            return {
                'signal_count': 0,
                'active_signals': 0,
                'signal_accuracy': 0,
                'success_rate': 0,
                'avg_return': 0,
                'avg_holding_time': 0
            }

    def get_strategy_parameters(self, strategy: str) -> dict:
        """Get strategy parameters"""
        try:
            strategy_obj = strategy_manager.strategies[strategy]
            return {
                'RSI Period': strategy_obj.rsi_period,
                'RSI Oversold': strategy_obj.rsi_oversold,
                'RSI Overbought': strategy_obj.rsi_overbought,
                'MACD Fast': strategy_obj.macd_fast,
                'MACD Slow': strategy_obj.macd_slow,
                'MACD Signal': strategy_obj.macd_signal,
                'BB Period': strategy_obj.bb_period,
                'BB StdDev': strategy_obj.bb_std,
                'Volume MA Period': strategy_obj.volume_ma_period,
                'Min Signal Strength': strategy_obj.min_signal_strength
            }
        except Exception as e:
            st.error(f"Error getting strategy parameters: {e}")
            return {}

    def update_strategy_parameters(self, strategy: str, params: dict):
        """Update strategy parameters"""
        try:
            strategy_obj = strategy_manager.strategies[strategy]
            for param, value in params.items():
                param_name = param.lower().replace(' ', '_')
                setattr(strategy_obj, param_name, value)
                
        except Exception as e:
            st.error(f"Error updating strategy parameters: {e}")

    def reset_strategy(self):
        """Reset strategy to default settings"""
        try:
            strategy = st.session_state.selected_strategy
            if strategy != 'All':
                strategy_obj = strategy_manager.strategies[strategy]
                strategy_obj.__init__()
                st.success("Strategy reset to default settings")
                
        except Exception as e:
            st.error(f"Error resetting strategy: {e}")

    def get_active_signals(self, strategy: str) -> list:
        """Get active signals for strategy"""
        try:
            signals = strategy_manager.strategies[strategy].signals
            return [s for s in signals if self.is_signal_active(s)]
        except Exception as e:
            st.error(f"Error getting active signals: {e}")
            return []

    def is_signal_active(self, signal) -> bool:
        """Check if signal is still active"""
        try:
            # Signal expires after 1 day
            return (datetime.now() - signal.timestamp).total_seconds() < 86400
        except Exception:
            return False

    def is_signal_accurate(self, signal) -> bool:
        """Check if signal prediction was accurate"""
        try:
            # Get current price
            data = market_data.get_stock_data(signal.symbol)
            if data is None or data.empty:
                return False
                
            current_price = data['Close'].iloc[-1]
            price_change = (current_price - signal.price) / signal.price
            
            # Signal is accurate if price moved in predicted direction
            if signal.direction == 'LONG':
                return price_change > 0
            else:
                return price_change < 0
                
        except Exception:
            return False

    def calculate_profit_factor(self, trades: list) -> float:
        """Calculate profit factor"""
        try:
            gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
            return gross_profit / gross_loss if gross_loss != 0 else float('inf')
        except Exception:
            return 0

    def calculate_sharpe_ratio(self, trades: list) -> float:
        """Calculate Sharpe ratio"""
        try:
            if not trades:
                return 0
                
            returns = pd.Series([t.pnl for t in trades])
            excess_returns = returns - (CONFIG.risk.risk_free_rate/252)
            return np.sqrt(252) * excess_returns.mean() / excess_returns.std() if len(returns) > 1 else 0
            
        except Exception:
            return 0

    def calculate_max_drawdown(self, trades: list) -> float:
        """Calculate maximum drawdown"""
        try:
            if not trades:
                return 0
                
            equity_curve = pd.Series([t.pnl for t in trades]).cumsum()
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak
            return drawdown.min() * 100 if len(drawdown) > 0 else 0
            
        except Exception:
            return 0

    def calculate_equity_curve(self, trades: list) -> pd.DataFrame:
        """Calculate equity curve"""
        try:
            if not trades:
                return pd.DataFrame()
                
            # Create daily returns series
            returns = pd.Series(
                index=[t.exit_time for t in trades],
                data=[t.pnl / CONFIG.CAPITAL for t in trades]
            )
            
            # Resample to daily and calculate cumulative returns
            daily_returns = returns.resample('D').sum()
            cumulative_returns = (1 + daily_returns).cumprod() - 1
            
            return pd.DataFrame({
                'returns': cumulative_returns
            })
            
        except Exception as e:
            st.error(f"Error calculating equity curve: {e}")
            return pd.DataFrame()

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
