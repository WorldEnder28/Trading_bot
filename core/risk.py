# core/risk.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import traceback
from pathlib import Path
import yfinance as yf

from .market_data import market_data
from .events import event_manager, Event, EventType
from .config import CONFIG

@dataclass
class RiskMetrics:
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    expected_shortfall: float = 0.0
    beta: float = 0.0
    correlation: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    treynor_ratio: float = 0.0
    information_ratio: float = 0.0
    max_drawdown: float = 0.0
    daily_var: float = 0.0
    weekly_var: float = 0.0
    monthly_var: float = 0.0
    stress_loss: float = 0.0
    liquidity_score: float = 0.0
    concentration_score: float = 0.0
    leverage: float = 1.0
    position_size: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            k: str(v) if isinstance(v, datetime) else v
            for k, v in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'RiskMetrics':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class PortfolioRiskMetrics:
    timestamp: datetime = field(default_factory=datetime.now)
    total_exposure: float = 0.0
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    leverage: float = 1.0
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    expected_shortfall: float = 0.0
    portfolio_beta: float = 0.0
    portfolio_correlation: float = 0.0
    portfolio_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    treynor_ratio: float = 0.0
    information_ratio: float = 0.0
    max_drawdown: float = 0.0
    concentration_risk: float = 0.0
    liquidity_risk: float = 0.0
    systematic_risk: float = 0.0
    idiosyncratic_risk: float = 0.0
    stress_test_loss: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            k: str(v) if isinstance(v, datetime) else v
            for k, v in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PortfolioRiskMetrics':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class StressScenario:
    name: str
    shocks: Dict[str, float]  # symbol -> price shock percentage
    volume_impact: float = 0.0
    volatility_impact: float = 0.0
    correlation_impact: float = 0.0
    description: str = ""
    probability: float = 0.0
    metadata: Dict = field(default_factory=dict)

class RiskManager:
    def __init__(self):
        self.logger = self._setup_logging()
        self.executor = ThreadPoolExecutor(max_workers=CONFIG.MAX_WORKERS)
        self.position_metrics: Dict[str, RiskMetrics] = {}
        self.portfolio_metrics = PortfolioRiskMetrics()
        self.position_limits: Dict[str, float] = {}
        self.var_window = 252  # Days for VaR calculation
        self.stress_scenarios: List[StressScenario] = []
        self._initialized = False
        self._running = False
        self._tasks = set()
        self.update_interval = 60  # seconds
        self._last_update = datetime.min

        # Initialize stress scenarios
        self._initialize_stress_scenarios()

    def _setup_logging(self) -> logging.Logger:
        """Setup risk manager logging"""
        logger = logging.getLogger('risk_manager')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(CONFIG.LOG_DIR / 'risk_manager.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    def _initialize_stress_scenarios(self):
        """Initialize default stress scenarios"""
        self.stress_scenarios = [
            StressScenario(
                name="Market Crash",
                shocks={
                    "DEFAULT": -0.15,  # -15% for all symbols
                    "DEFENSIVE": -0.10,  # -10% for defensive sectors
                    "VOLATILE": -0.20   # -20% for volatile sectors
                },
                volume_impact=2.0,      # 2x normal volume
                volatility_impact=2.5,   # 2.5x volatility spike
                correlation_impact=0.8,  # 0.8 correlation increase
                probability=0.05
            ),
            StressScenario(
                name="Sector Crisis",
                shocks={
                    "DEFAULT": -0.05,
                    "AFFECTED": -0.25,
                    "RELATED": -0.15
                },
                volume_impact=1.5,
                volatility_impact=2.0,
                correlation_impact=0.6,
                probability=0.10
            ),
            StressScenario(
                name="Liquidity Crisis",
                shocks={
                    "DEFAULT": -0.10,
                    "ILLIQUID": -0.30,
                    "LIQUID": -0.05
                },
                volume_impact=0.3,
                volatility_impact=1.8,
                correlation_impact=0.7,
                probability=0.08
            ),
            StressScenario(
                name="Volatility Spike",
                shocks={
                    "DEFAULT": -0.08,
                    "HIGH_BETA": -0.20,
                    "LOW_BETA": -0.05
                },
                volume_impact=1.8,
                volatility_impact=3.0,
                correlation_impact=0.5,
                probability=0.12
            )
        ]
        
    async def initialize(self):
        """Initialize risk manager"""
        try:
            if self._initialized:
                return
                
            self.logger.info("Initializing risk manager...")
            
            # Load saved state
            await self._load_state()
            
            # Subscribe to events
            event_manager.subscribe(EventType.MARKET_UPDATE, self._handle_market_update)
            event_manager.subscribe(EventType.POSITION, self._handle_position_update)
            
            # Start risk monitoring
            self._running = True
            self._tasks.add(
                asyncio.create_task(self._monitor_risk())
            )
            
            self._initialized = True
            self.logger.info("Risk manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing risk manager: {e}\n{traceback.format_exc()}")
            raise

    async def _monitor_risk(self):
        """Monitor risk metrics"""
        try:
            while self._running:
                try:
                    now = datetime.now()
                    if (now - self._last_update).total_seconds() >= self.update_interval:
                        # Update position metrics
                        positions = await self._get_all_positions()
                        position_tasks = [
                            self.update_position_risk(symbol, position)
                            for symbol, position in positions.items()
                        ]
                        await asyncio.gather(*position_tasks)
                        
                        # Update portfolio metrics
                        await self.update_portfolio_risk()
                        
                        # Check risk limits
                        violations = await self._check_risk_limits()
                        if violations:
                            await self._handle_risk_violations(violations)
                        
                        self._last_update = now
                        
                    await asyncio.sleep(1)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in risk monitoring: {e}\n{traceback.format_exc()}")
                    await asyncio.sleep(10)
                    
        except Exception as e:
            self.logger.error(f"Fatal error in risk monitoring: {e}\n{traceback.format_exc()}")
            
    async def update_position_risk(self, symbol: str, position: Dict):
        """Update risk metrics for a position"""
        try:
            # Get historical data
            data = await market_data.get_stock_data(
                symbol,
                interval='1d',
                lookback=self.var_window
            )
            
            if data is None or data.empty:
                return
            
            # Calculate metrics in parallel
            tasks = [
                self._calculate_var_metrics(data, position),
                self._calculate_beta_metrics(data),
                self._calculate_volatility_metrics(data),
                self._calculate_drawdown_metrics(data),
                self._calculate_liquidity_metrics(data, position),
                self._calculate_stress_metrics(symbol, position)
            ]
            
            results = await asyncio.gather(*tasks)
            var_metrics, beta_metrics, vol_metrics, dd_metrics, liq_metrics, stress_metrics = results
            
            # Combine metrics
            metrics = RiskMetrics(
                symbol=symbol,
                timestamp=datetime.now(),
                **var_metrics,
                **beta_metrics,
                **vol_metrics,
                **dd_metrics,
                **liq_metrics,
                **stress_metrics,
                position_size=position['quantity'] * data['Close'].iloc[-1]
            )
            
            # Update position metrics
            self.position_metrics[symbol] = metrics
            
            # Calculate position limit
            self.position_limits[symbol] = await self._calculate_position_limit(
                symbol, metrics
            )
            
            # Publish risk event
            await event_manager.publish(Event(
                type=EventType.RISK,
                data={
                    'type': 'POSITION',
                    'symbol': symbol,
                    'metrics': metrics.to_dict()
                }
            ))
            
        except Exception as e:
            self.logger.error(f"Error updating position risk for {symbol}: {e}\n{traceback.format_exc()}")

    async def _calculate_var_metrics(
        self,
        data: pd.DataFrame,
        position: Dict
    ) -> Dict:
        """Calculate Value at Risk metrics"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._var_metrics_sync,
                data,
                position
            )
        except Exception as e:
            self.logger.error(f"Error calculating VaR metrics: {e}")
            return {
                'var_95': 0.0,
                'var_99': 0.0,
                'cvar_95': 0.0,
                'cvar_99': 0.0,
                'expected_shortfall': 0.0
            }

    def _var_metrics_sync(
        self,
        data: pd.DataFrame,
        position: Dict
    ) -> Dict:
        """Synchronous VaR calculations"""
        try:
            returns = data['Close'].pct_change().dropna()
            position_value = position['quantity'] * data['Close'].iloc[-1]
            
            # Calculate VaR using historical simulation
            var_95 = np.percentile(returns, 5) * position_value
            var_99 = np.percentile(returns, 1) * position_value
            
            # Calculate CVaR (Expected Shortfall)
            cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * position_value
            cvar_99 = returns[returns <= np.percentile(returns, 1)].mean() * position_value
            
            # Calculate Expected Shortfall
            threshold = np.percentile(returns, 5)
            expected_shortfall = returns[returns <= threshold].mean() * position_value
            
            return {
                'var_95': abs(var_95),
                'var_99': abs(var_99),
                'cvar_95': abs(cvar_95),
                'cvar_99': abs(cvar_99),
                'expected_shortfall': abs(expected_shortfall),
                'daily_var': abs(var_95),
                'weekly_var': abs(var_95 * np.sqrt(5)),
                'monthly_var': abs(var_95 * np.sqrt(21))
            }
            
        except Exception as e:
            self.logger.error(f"Error in VaR calculations: {e}")
            return {
                'var_95': 0.0,
                'var_99': 0.0,
                'cvar_95': 0.0,
                'cvar_99': 0.0,
                'expected_shortfall': 0.0,
                'daily_var': 0.0,
                'weekly_var': 0.0,
                'monthly_var': 0.0
            }

    async def _calculate_beta_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate beta and correlation metrics"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._beta_metrics_sync,
                data
            )
        except Exception as e:
            self.logger.error(f"Error calculating beta metrics: {e}")
            return {
                'beta': 1.0,
                'correlation': 0.0,
                'systematic_risk': 0.0,
                'idiosyncratic_risk': 0.0
            }
            
    def _beta_metrics_sync(self, data: pd.DataFrame) -> Dict:
        """Synchronous beta and correlation calculations"""
        try:
            # Get market data
            market_data = yf.download('^NSEI', start=(datetime.now() - timedelta(days=self.var_window)))
            if market_data.empty:
                return {
                    'beta': 1.0,
                    'correlation': 0.0,
                    'systematic_risk': 0.0,
                    'idiosyncratic_risk': 0.0
                }
            
            # Calculate returns
            stock_returns = data['Close'].pct_change().dropna()
            market_returns = market_data['Close'].pct_change().dropna()
            
            # Align data
            aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
            if aligned_data.empty:
                return {
                    'beta': 1.0,
                    'correlation': 0.0,
                    'systematic_risk': 0.0,
                    'idiosyncratic_risk': 0.0
                }
            
            stock_returns = aligned_data.iloc[:, 0]
            market_returns = aligned_data.iloc[:, 1]
            
            # Calculate beta
            covariance = stock_returns.cov(market_returns)
            market_variance = market_returns.var()
            beta = covariance / market_variance if market_variance != 0 else 1.0
            
            # Calculate correlation
            correlation = stock_returns.corr(market_returns)
            
            # Calculate systematic and idiosyncratic risk
            stock_variance = stock_returns.var()
            systematic_risk = (beta ** 2) * market_variance
            idiosyncratic_risk = stock_variance - systematic_risk
            
            return {
                'beta': beta,
                'correlation': correlation,
                'systematic_risk': systematic_risk,
                'idiosyncratic_risk': idiosyncratic_risk
            }
            
        except Exception as e:
            self.logger.error(f"Error in beta calculations: {e}")
            return {
                'beta': 1.0,
                'correlation': 0.0,
                'systematic_risk': 0.0,
                'idiosyncratic_risk': 0.0
            }

    async def update_portfolio_risk(self):
        """Update portfolio risk metrics"""
        try:
            positions = await self._get_all_positions()
            if not positions:
                return
            
            # Calculate portfolio exposures
            exposures = await self._calculate_portfolio_exposures(positions)
            
            # Calculate portfolio metrics in parallel
            tasks = [
                self._calculate_portfolio_var(positions),
                self._calculate_portfolio_beta(positions),
                self._calculate_portfolio_correlations(positions),
                self._calculate_portfolio_concentration(positions),
                self._calculate_portfolio_liquidity(positions),
                self._calculate_portfolio_stress(positions)
            ]
            
            results = await asyncio.gather(*tasks)
            
            var_metrics, beta_metrics, corr_metrics, conc_metrics, liq_metrics, stress_metrics = results
            
            # Update portfolio metrics
            self.portfolio_metrics = PortfolioRiskMetrics(
                timestamp=datetime.now(),
                **exposures,
                **var_metrics,
                **beta_metrics,
                **corr_metrics,
                **conc_metrics,
                **liq_metrics,
                **stress_metrics
            )
            
            # Publish portfolio risk event
            await event_manager.publish(Event(
                type=EventType.RISK,
                data={
                    'type': 'PORTFOLIO',
                    'metrics': self.portfolio_metrics.to_dict()
                }
            ))
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio risk: {e}\n{traceback.format_exc()}")

    async def _calculate_portfolio_exposures(self, positions: Dict) -> Dict:
        """Calculate portfolio exposures and leverage"""
        try:
            total_exposure = 0.0
            long_exposure = 0.0
            short_exposure = 0.0
            
            for symbol, position in positions.items():
                data = await market_data.get_stock_data(symbol)
                if data is None or data.empty:
                    continue
                    
                price = data['Close'].iloc[-1]
                exposure = position['quantity'] * price
                
                total_exposure += abs(exposure)
                if position['quantity'] > 0:
                    long_exposure += exposure
                else:
                    short_exposure += abs(exposure)
            
            net_exposure = long_exposure - short_exposure
            gross_exposure = long_exposure + short_exposure
            leverage = gross_exposure / CONFIG.CAPITAL if CONFIG.CAPITAL > 0 else 0
            
            return {
                'total_exposure': total_exposure,
                'net_exposure': net_exposure,
                'gross_exposure': gross_exposure,
                'long_exposure': long_exposure,
                'short_exposure': short_exposure,
                'leverage': leverage
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio exposures: {e}")
            return {
                'total_exposure': 0.0,
                'net_exposure': 0.0,
                'gross_exposure': 0.0,
                'long_exposure': 0.0,
                'short_exposure': 0.0,
                'leverage': 1.0
            }

    async def _calculate_portfolio_stress(self, positions: Dict) -> Dict:
        """Calculate portfolio stress test results"""
        try:
            total_stress_loss = 0.0
            scenario_results = {}
            
            for scenario in self.stress_scenarios:
                scenario_loss = 0.0
                
                for symbol, position in positions.items():
                    # Get stock info for sector classification
                    stock_info = await market_data.get_stock_info(symbol)
                    sector = stock_info.sector if stock_info else "DEFAULT"
                    
                    # Determine shock percentage
                    shock = scenario.shocks.get(sector, scenario.shocks['DEFAULT'])
                    
                    # Calculate position stress loss
                    data = await market_data.get_stock_data(symbol)
                    if data is None or data.empty:
                        continue
                        
                    price = data['Close'].iloc[-1]
                    position_value = position['quantity'] * price
                    
                    # Apply shock with volatility and correlation adjustments
                    adjusted_shock = shock * scenario.volatility_impact
                    if position['quantity'] > 0:  # Long position
                        position_loss = position_value * adjusted_shock
                    else:  # Short position
                        position_loss = position_value * (-adjusted_shock)
                        
                    scenario_loss += position_loss
                
                scenario_results[scenario.name] = {
                    'loss': scenario_loss,
                    'probability': scenario.probability
                }
                
                total_stress_loss += abs(scenario_loss * scenario.probability)
            
            return {
                'stress_test_loss': total_stress_loss,
                'scenario_results': scenario_results
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio stress: {e}")
            return {
                'stress_test_loss': 0.0,
                'scenario_results': {}
            }
            
    async def _check_risk_limits(self) -> List[Dict]:
        """Check all risk limits"""
        try:
            violations = []
            
            # Portfolio risk limits
            if self.portfolio_metrics.var_95 > CONFIG.risk.max_portfolio_risk * CONFIG.CAPITAL:
                violations.append({
                    'type': 'PORTFOLIO_VAR',
                    'severity': 'HIGH',
                    'current': self.portfolio_metrics.var_95,
                    'limit': CONFIG.risk.max_portfolio_risk * CONFIG.CAPITAL,
                    'action_required': True
                })
                
            if self.portfolio_metrics.leverage > CONFIG.risk.max_leverage:
                violations.append({
                    'type': 'LEVERAGE',
                    'severity': 'HIGH',
                    'current': self.portfolio_metrics.leverage,
                    'limit': CONFIG.risk.max_leverage,
                    'action_required': True
                })
                
            # Correlation risk
            if self.portfolio_metrics.portfolio_correlation > CONFIG.risk.max_correlation:
                violations.append({
                    'type': 'CORRELATION',
                    'severity': 'MEDIUM',
                    'current': self.portfolio_metrics.portfolio_correlation,
                    'limit': CONFIG.risk.max_correlation,
                    'action_required': True
                })
                
            # Concentration risk
            if self.portfolio_metrics.concentration_risk > CONFIG.risk.max_position_size:
                violations.append({
                    'type': 'CONCENTRATION',
                    'severity': 'MEDIUM',
                    'current': self.portfolio_metrics.concentration_risk,
                    'limit': CONFIG.risk.max_position_size,
                    'action_required': True
                })
                
            # Position-specific limits
            for symbol, metrics in self.position_metrics.items():
                position_limit = self.position_limits.get(symbol, float('inf'))
                
                if metrics.position_size > position_limit:
                    violations.append({
                        'type': 'POSITION_LIMIT',
                        'severity': 'MEDIUM',
                        'symbol': symbol,
                        'current': metrics.position_size,
                        'limit': position_limit,
                        'action_required': True
                    })
                    
                if metrics.volatility > CONFIG.risk.max_volatility:
                    violations.append({
                        'type': 'VOLATILITY',
                        'severity': 'MEDIUM',
                        'symbol': symbol,
                        'current': metrics.volatility,
                        'limit': CONFIG.risk.max_volatility,
                        'action_required': False
                    })
                    
                if metrics.liquidity_score < CONFIG.risk.min_liquidity_ratio:
                    violations.append({
                        'type': 'LIQUIDITY',
                        'severity': 'LOW',
                        'symbol': symbol,
                        'current': metrics.liquidity_score,
                        'limit': CONFIG.risk.min_liquidity_ratio,
                        'action_required': False
                    })
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}\n{traceback.format_exc()}")
            return []

    async def _handle_risk_violations(self, violations: List[Dict]):
        """Handle risk limit violations"""
        try:
            # Log violations
            for violation in violations:
                self.logger.warning(
                    f"Risk limit violation: {violation['type']} - "
                    f"Current: {violation['current']}, Limit: {violation['limit']}"
                )
                
            # Publish risk event
            await event_manager.publish(Event(
                type=EventType.RISK,
                data={
                    'type': 'VIOLATION',
                    'violations': violations
                }
            ))
            
            # Handle high severity violations that require action
            high_severity_violations = [
                v for v in violations
                if v['severity'] == 'HIGH' and v['action_required']
            ]
            
            if high_severity_violations:
                await self._implement_risk_reduction(high_severity_violations)
                
        except Exception as e:
            self.logger.error(f"Error handling risk violations: {e}")

    async def _implement_risk_reduction(self, violations: List[Dict]):
        """Implement risk reduction measures"""
        try:
            positions = await self._get_all_positions()
            
            # Calculate risk contribution for each position
            risk_contributions = []
            for symbol, position in positions.items():
                metrics = self.position_metrics.get(symbol)
                if metrics:
                    risk_contributions.append({
                        'symbol': symbol,
                        'position': position,
                        'metrics': metrics,
                        'risk_score': self._calculate_risk_score(metrics)
                    })
                    
            # Sort positions by risk score
            risk_contributions.sort(key=lambda x: x['risk_score'], reverse=True)
            
            # Determine reduction targets based on violation types
            for violation in violations:
                if violation['type'] == 'PORTFOLIO_VAR':
                    reduction_target = 0.3  # Reduce top positions by 30%
                elif violation['type'] == 'LEVERAGE':
                    reduction_target = 0.4  # Reduce top positions by 40%
                else:
                    reduction_target = 0.2  # Reduce top positions by 20%
                
                # Reduce highest risk positions
                for risk_item in risk_contributions[:3]:  # Top 3 risky positions
                    position = risk_item['position']
                    reduction_qty = int(position['quantity'] * reduction_target)
                    
                    if reduction_qty > 0:
                        # Publish position reduction recommendation
                        await event_manager.publish(Event(
                            type=EventType.RISK,
                            data={
                                'type': 'REDUCTION',
                                'symbol': risk_item['symbol'],
                                'current_size': position['quantity'],
                                'reduction_size': reduction_qty,
                                'reason': f"Risk reduction - {violation['type']}"
                            }
                        ))
                        
        except Exception as e:
            self.logger.error(f"Error implementing risk reduction: {e}")

    def _calculate_risk_score(self, metrics: RiskMetrics) -> float:
        """Calculate composite risk score for position"""
        try:
            # Normalize and weight different risk factors
            var_score = min(metrics.var_95 / (CONFIG.CAPITAL * 0.02), 1.0) * 0.3
            vol_score = min(metrics.volatility / CONFIG.risk.max_volatility, 1.0) * 0.2
            beta_score = min(abs(metrics.beta), 2.0) / 2.0 * 0.2
            liq_score = (1 - metrics.liquidity_score) * 0.15
            size_score = min(metrics.position_size / (CONFIG.CAPITAL * CONFIG.risk.max_position_size), 1.0) * 0.15
            
            return var_score + vol_score + beta_score + liq_score + size_score
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 1.0

    async def shutdown(self):
        """Shutdown risk manager"""
        try:
            self.logger.info("Shutting down risk manager...")
            
            self._running = False
            
            # Cancel all tasks
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
            # Save final state
            await self._save_state()
            
            # Clean up executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("Risk manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Create singleton instance
risk_manager = RiskManager()
