# app/pages/backtesting.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import asyncio

from core.backtester import backtester
from core.strategy import strategy_manager
from core.market_data import market_data
from core.config import CONFIG

class BacktestingPage:
    def __init__(self):
        self.initialize_state()
        
    def initialize_state(self):
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = None
        if 'optimization_results' not in st.session_state:
            st.session_state.optimization_results = None
            
    def render(self):
        """Render backtesting page"""
        # Backtest Controls
        self.render_backtest_controls()
        
        # Results Analysis
        if st.session_state.backtest_results:
            self.render_backtest_results()
            
        # Strategy Optimization
        st.divider()
        self.render_optimization_controls()
        
        if st.session_state.optimization_results:
            self.render_optimization_results()

    async def run_backtest(
            self,
            symbols: list,
            strategies: list, 
            start_date: datetime,
            end_date: datetime,
            capital: float,
            risk_per_trade: float
        ) -> dict:
        """Run backtest with given parameters"""
        try:
            # Configure backtester
            backtester.capital = capital
            backtester.risk_per_trade = risk_per_trade/100
            
            # Run backtest
            results = await backtester.run(
                symbols=symbols,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                strategies=strategies
            )
            
            if results:
                # Save results
                self.save_backtest_results(results)
                return results
                
            return None
            
        except Exception as e:
            st.error(f"Error running backtest: {e}")
            return None
            
# app/pages/backtesting.py

    def render_backtest_controls(self):
        """Render backtesting control section"""
        st.subheader("Backtest Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Symbol selection
            symbols = market_data.get_available_symbols()
            selected_symbols = st.multiselect(
                "Select Symbols",
                symbols,
                default=symbols[:5] if len(symbols) > 5 else symbols
            )
            
            # Strategy selection
            strategies = list(strategy_manager.strategies.keys())
            selected_strategies = st.multiselect(
                "Select Strategies",
                strategies,
                default=strategies
            )
            
        with col2:
            # Date range
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365)
            )
            
            end_date = st.date_input(
                "End Date",
                value=datetime.now()
            )
            
        with col3:
            # Capital settings
            initial_capital = st.number_input(
                "Initial Capital",
                min_value=10000,
                value=CONFIG.CAPITAL,
                step=10000,
                help="Initial trading capital"
            )
            
            # Risk settings
            risk_per_trade = st.slider(
                "Risk Per Trade (%)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1
            )
            
        # Run backtest button
        if st.button("Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                # Use sync version instead of async
                results = self.run_backtest_sync(
                    symbols=selected_symbols,
                    strategies=selected_strategies,
                    start_date=start_date,
                    end_date=end_date,
                    capital=initial_capital,
                    risk_per_trade=risk_per_trade
                )
                
                if results:
                    st.session_state.backtest_results = results
                    st.success("Backtest completed successfully!")

    def run_backtest_sync(
            self,
            symbols: list,
            strategies: list, 
            start_date: datetime,
            end_date: datetime,
            capital: float,
            risk_per_trade: float
        ) -> dict:
        """Synchronous version of run_backtest"""
        try:
            # Configure backtester
            backtester.capital = capital
            backtester.risk_per_trade = risk_per_trade/100
            
            # Run backtest synchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(backtester.run(
                symbols=symbols,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                strategies=strategies
            ))
            loop.close()
            
            if results:
                # Save results
                self.save_backtest_results(results)
                return results
                
            return None
            
        except Exception as e:
            st.error(f"Error running backtest: {e}")
            return None
                            
    def render_backtest_results(self):
        """Render backtest results analysis"""
        results = st.session_state.backtest_results
        
        # Performance Summary
        st.subheader("Performance Summary")
        self.render_performance_summary(results)
        
        # Performance Charts
        self.render_performance_charts(results)
        
        # Detailed Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_trade_analysis(results)
            
        with col2:
            self.render_strategy_analysis(results)
            
        # Trade List
        self.render_trade_list(results)
        
    def render_performance_summary(self, results: dict):
        """Render performance summary metrics"""
        metrics = results['summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{metrics['total_return']:.2f}%",
                f"₹{metrics['total_profit']:,.2f}"
            )
            
        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{metrics['sharpe_ratio']:.2f}",
                f"Sortino: {metrics['sortino_ratio']:.2f}"
            )
            
        with col3:
            st.metric(
                "Max Drawdown",
                f"{metrics['max_drawdown']:.2f}%",
                f"Recovery: {metrics['recovery_period']} days"
            )
            
        with col4:
            st.metric(
                "Win Rate",
                f"{metrics['win_rate']:.2f}%",
                f"Profit Factor: {metrics['profit_factor']:.2f}"
            )
            
    def render_performance_charts(self, results: dict):
        """Render performance analysis charts"""
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                'Portfolio Value',
                'Drawdown',
                'Daily Returns'
            ),
            row_heights=[0.5, 0.2, 0.3]
        )
        
        # Portfolio value
        equity_curve = results['equity_curve']
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve['equity'],
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Drawdown
        drawdown = results['drawdown']
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown * 100,
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        # Daily returns
        daily_returns = results['daily_returns']
        fig.add_trace(
            go.Bar(
                x=daily_returns.index,
                y=daily_returns * 100,
                name='Daily Returns'
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def render_trade_analysis(self, results: dict):
        """Render trade analysis section"""
        st.subheader("Trade Analysis")
        
        trades = results['trades']
        if not trades:
            st.info("No trades to analyze")
            return
            
        # Trade distribution
        fig1 = go.Figure()
        
        # P&L distribution
        pnl_values = [t['pnl'] for t in trades]
        fig1.add_trace(go.Histogram(
            x=pnl_values,
            nbinsx=50,
            name='P&L Distribution'
        ))
        
        fig1.update_layout(
            title="P&L Distribution",
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Trade metrics
        col1, col2 = st.columns(2)
        
        with col1:
            avg_win = np.mean([pnl for pnl in pnl_values if pnl > 0])
            avg_loss = abs(np.mean([pnl for pnl in pnl_values if pnl < 0]))
            
            st.metric("Average Win", f"₹{avg_win:,.2f}")
            st.metric("Average Loss", f"₹{avg_loss:,.2f}")
            st.metric("Win/Loss Ratio", f"{avg_win/avg_loss:.2f}" if avg_loss > 0 else "∞")
            
        with col2:
            st.metric("Total Trades", len(trades))
            st.metric("Winning Trades", sum(1 for t in trades if t['pnl'] > 0))
            st.metric("Losing Trades", sum(1 for t in trades if t['pnl'] < 0))
            
    def render_strategy_analysis(self, results: dict):
        """Render strategy analysis section"""
        st.subheader("Strategy Analysis")
        
        strategy_results = results['strategy_results']
        if not strategy_results:
            st.info("No strategy results to analyze")
            return
            
        # Strategy comparison chart
        fig = go.Figure()
        
        for strategy, data in strategy_results.items():
            equity_curve = data['equity_curve']
            fig.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_curve['returns'] * 100,
                name=strategy
            ))
            
        fig.update_layout(
            title="Strategy Performance Comparison",
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategy metrics
        metrics_data = []
        for strategy, data in strategy_results.items():
            metrics_data.append({
                'Strategy': strategy,
                'Total Return': f"{data['total_return']:.2f}%",
                'Win Rate': f"{data['win_rate']:.2f}%",
                'Profit Factor': f"{data['profit_factor']:.2f}",
                'Sharpe Ratio': f"{data['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{data['max_drawdown']:.2f}%",
                'Total Trades': data['total_trades']
            })
            
        if metrics_data:
            st.dataframe(
                pd.DataFrame(metrics_data),
                height=200,
                use_container_width=True
            )
            
    def render_trade_list(self, results: dict):
        """Render detailed trade list"""
        st.subheader("Trade List")
        
        trades = results['trades']
        if not trades:
            st.info("No trades to display")
            return
            
        trades_data = []
        for trade in trades:
            trades_data.append({
                'Entry Time': trade['entry_time'],
                'Exit Time': trade['exit_time'],
                'Symbol': trade['symbol'],
                'Strategy': trade['strategy'],
                'Side': trade['side'],
                'Entry Price': f"₹{trade['entry_price']:,.2f}",
                'Exit Price': f"₹{trade['exit_price']:,.2f}",
                'Quantity': trade['quantity'],
                'P&L': f"₹{trade['pnl']:,.2f}",
                'Return %': f"{trade['return']:.2f}%"
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
        
    def render_optimization_controls(self):
        """Render optimization control section"""
        st.subheader("Strategy Optimization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Strategy selection
            strategy = st.selectbox(
                "Select Strategy to Optimize",
                list(strategy_manager.strategies.keys())
            )
            
            # Optimization target
            target = st.selectbox(
                "Optimization Target",
                ['Sharpe Ratio', 'Total Return', 'Profit Factor', 'Win Rate']
            )
            
        with col2:
            # Parameter ranges
            st.write("Parameter Ranges")
            
            params = {}
            strategy_obj = strategy_manager.strategies[strategy]
            
            # Add parameter inputs based on strategy type
            if hasattr(strategy_obj, 'rsi_period'):
                params['RSI Period'] = {
                    'min': 5,
                    'max': 30,
                    'step': 1
                }
                
            if hasattr(strategy_obj, 'macd_fast'):
                params['MACD Fast'] = {
                    'min': 8,
                    'max': 20,
                    'step': 1
                }
                
            # Add inputs for parameter ranges
            param_ranges = {}
            for param, range_dict in params.items():
                min_val = st.number_input(
                    f"{param} Min",
                    value=range_dict['min'],
                    step=range_dict['step']
                )
                max_val = st.number_input(
                    f"{param} Max",
                    value=range_dict['max'],
                    step=range_dict['step']
                )
                param_ranges[param] = np.arange(min_val, max_val + range_dict['step'], range_dict['step'])
                
        # Run optimization button
        if st.button("Run Optimization", type="primary"):
            with st.spinner("Running optimization..."):
                results = self.run_optimization(
                    strategy=strategy,
                    param_ranges=param_ranges,
                    target=target
                )
                
                if results:
                    st.session_state.optimization_results = results
                    st.success("Optimization completed successfully!")
                    
    def render_optimization_results(self):
        """Render optimization results"""
        results = st.session_state.optimization_results
        
        # Best parameters
        st.subheader("Optimal Parameters")
        
        best_params = results['best_params']
        metrics = results['metrics']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Parameter values
            for param, value in best_params.items():
                st.metric(param, value)
                
        with col2:
            # Performance metrics
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            st.metric("Total Return", f"{metrics['total_return']:.2f}%")
            st.metric("Win Rate", f"{metrics['win_rate']:.2f}%")
            
        # Parameter impact analysis
        st.subheader("Parameter Impact Analysis")
        
        param_analysis = results['param_analysis']
        if param_analysis:
            fig = go.Figure()
            
            for param, data in param_analysis.items():
                fig.add_trace(go.Scatter(
                    x=data['values'],
                    y=data['performance'],
                    name=param
                ))
                
            fig.update_layout(
                title="Parameter Sensitivity Analysis",
                xaxis_title="Parameter Value",
                yaxis_title="Performance Metric",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

    async def run_optimization(
        self,
        strategy: str,
        param_ranges: dict,
        target: str
    ) -> dict:
        """Run strategy optimization"""
        try:
            # Convert target metric name
            target_map = {
                'Sharpe Ratio': 'sharpe_ratio',
                'Total Return': 'total_return',
                'Profit Factor': 'profit_factor',
                'Win Rate': 'win_rate'
            }
            optimization_target = target_map[target]
            
            # Run optimization
            results = await backtester.optimize_parameters(
                strategy=strategy,
                param_ranges=param_ranges,
                optimization_target=optimization_target
            )
            
            if results:
                # Calculate parameter impact
                param_analysis = await self.analyze_parameter_impact(
                    strategy,
                    param_ranges,
                    optimization_target
                )
                
                results['param_analysis'] = param_analysis
                
                # Save results
                self.save_optimization_results(results)
                return results
                
            return None
            
        except Exception as e:
            st.error(f"Error running optimization: {e}")
            return None

    async def analyze_parameter_impact(
        self,
        strategy: str,
        param_ranges: dict,
        target: str
    ) -> dict:
        """Analyze how each parameter impacts performance"""
        try:
            param_analysis = {}
            
            # Analyze each parameter separately
            for param, values in param_ranges.items():
                performances = []
                
                # Test each parameter value
                for value in values:
                    # Set parameter value
                    strategy_obj = strategy_manager.strategies[strategy]
                    param_name = param.lower().replace(' ', '_')
                    original_value = getattr(strategy_obj, param_name)
                    setattr(strategy_obj, param_name, value)
                    
                    # Run backtest
                    results = await backtester.run_quick_backtest(strategy)
                    if results:
                        performances.append(results['metrics'][target])
                        
                    # Restore original value
                    setattr(strategy_obj, param_name, original_value)
                    
                param_analysis[param] = {
                    'values': values.tolist(),
                    'performance': performances
                }
                
            return param_analysis
            
        except Exception as e:
            st.error(f"Error analyzing parameter impact: {e}")
            return {}

    def save_backtest_results(self, results: dict):
        """Save backtest results to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.json"
            
            # Convert data for JSON serialization
            serializable_results = {
                'summary': {
                    k: float(v) if isinstance(v, np.float64) else v
                    for k, v in results['summary'].items()
                },
                'trades': results['trades'],
                'strategy_results': {
                    strategy: {
                        k: float(v) if isinstance(v, np.float64) else v
                        for k, v in metrics.items()
                        if k != 'equity_curve'
                    }
                    for strategy, metrics in results['strategy_results'].items()
                }
            }
            
            filepath = CONFIG.BACKTEST_DIR / filename
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=4, default=str)
                
        except Exception as e:
            st.error(f"Error saving backtest results: {e}")

    def save_optimization_results(self, results: dict):
        """Save optimization results to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.json"
            
            # Convert data for JSON serialization
            serializable_results = {
                'best_params': results['best_params'],
                'metrics': {
                    k: float(v) if isinstance(v, np.float64) else v
                    for k, v in results['metrics'].items()
                },
                'param_analysis': {
                    param: {
                        'values': values.tolist() if isinstance(values, np.ndarray) else values,
                        'performance': performance
                    }
                    for param, data in results['param_analysis'].items()
                    for values, performance in [data['values'], data['performance']]
                }
            }
            
            filepath = CONFIG.BACKTEST_DIR / filename
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=4, default=str)
                
        except Exception as e:
            st.error(f"Error saving optimization results: {e}")
