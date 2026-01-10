"""
Tests for quant module.
"""

import unittest
import numpy as np


class TestStatistics(unittest.TestCase):
    """Test statistical models."""
    
    def test_imports(self):
        """Test statistics module imports."""
        try:
            from zipline.quant import statistics
        except ImportError as e:
            self.skipTest(f"Statistics dependencies not installed: {e}")
    
    def test_arima_forecaster(self):
        """Test ARIMA forecaster."""
        try:
            from zipline.quant.statistics import ARIMAForecaster
            
            forecaster = ARIMAForecaster(order=(1, 1, 1))
            self.assertEqual(forecaster.order, (1, 1, 1))
        except ImportError:
            self.skipTest("Statistics dependencies not installed")
    
    def test_garch_model(self):
        """Test GARCH model."""
        try:
            from zipline.quant.statistics import GARCHModel
            
            garch = GARCHModel(p=1, q=1)
            self.assertEqual(garch.p, 1)
            self.assertEqual(garch.q, 1)
        except ImportError:
            self.skipTest("Statistics dependencies not installed")


class TestStochastic(unittest.TestCase):
    """Test stochastic models."""
    
    def test_imports(self):
        """Test stochastic module imports."""
        try:
            from zipline.quant import stochastic
        except ImportError as e:
            self.skipTest(f"Stochastic dependencies not installed: {e}")
    
    def test_gbm_simulator(self):
        """Test GBM simulator."""
        try:
            from zipline.quant.stochastic import GBMSimulator
            
            gbm = GBMSimulator(mu=0.1, sigma=0.2)
            paths = gbm.simulate(S0=100, T=1, dt=0.01, n_paths=10)
            
            self.assertEqual(paths.shape[0], 10)
            self.assertEqual(paths[0, 0], 100)
        except ImportError:
            self.skipTest("Stochastic dependencies not installed")
    
    def test_black_scholes(self):
        """Test Black-Scholes model."""
        try:
            from zipline.quant.stochastic import BlackScholesModel
            
            bs = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2)
            price = bs.price()
            delta = bs.delta()
            
            self.assertIsInstance(price, (int, float))
            self.assertIsInstance(delta, (int, float))
            self.assertTrue(0 <= delta <= 1)  # Delta should be between 0 and 1 for call
        except ImportError:
            self.skipTest("Stochastic dependencies not installed")


class TestOptimization(unittest.TestCase):
    """Test portfolio optimization."""
    
    def test_imports(self):
        """Test optimization module imports."""
        try:
            from zipline.quant import optimization
        except ImportError as e:
            self.skipTest(f"Optimization dependencies not installed: {e}")
    
    def test_mean_variance_optimizer(self):
        """Test mean-variance optimizer."""
        try:
            from zipline.quant.optimization import MeanVarianceOptimizer
            
            expected_returns = np.array([0.1, 0.15, 0.12])
            cov_matrix = np.array([[0.04, 0.01, 0.02],
                                   [0.01, 0.09, 0.03],
                                   [0.02, 0.03, 0.06]])
            
            optimizer = MeanVarianceOptimizer(expected_returns, cov_matrix)
            self.assertEqual(len(optimizer.expected_returns), 3)
        except ImportError:
            self.skipTest("Optimization dependencies not installed")
    
    def test_risk_parity(self):
        """Test risk parity optimizer."""
        try:
            from zipline.quant.optimization import RiskParityOptimizer
            
            cov_matrix = np.array([[0.04, 0.01], [0.01, 0.09]])
            optimizer = RiskParityOptimizer(cov_matrix)
            weights = optimizer.optimize()
            
            self.assertEqual(len(weights), 2)
            self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        except ImportError:
            self.skipTest("Optimization dependencies not installed")


class TestRiskModels(unittest.TestCase):
    """Test advanced risk models."""
    
    def test_imports(self):
        """Test risk models imports."""
        try:
            from zipline.finance.risk import advanced
        except ImportError as e:
            self.skipTest(f"Risk model dependencies not installed: {e}")
    
    def test_expected_shortfall(self):
        """Test CVaR calculation."""
        try:
            from zipline.finance.risk.advanced import ExpectedShortfall
            
            returns = np.random.randn(1000) * 0.02
            es = ExpectedShortfall()
            cvar = es.historical_cvar(returns, confidence=0.95)
            
            self.assertIsInstance(cvar, (int, float))
            self.assertTrue(cvar < 0)  # CVaR should be negative
        except ImportError:
            self.skipTest("Risk model dependencies not installed")
    
    def test_drawdown_analyzer(self):
        """Test drawdown analyzer."""
        try:
            from zipline.finance.risk.advanced import DrawdownAnalyzer
            
            returns = np.array([0.01, 0.02, -0.05, -0.03, 0.04, 0.03])
            analyzer = DrawdownAnalyzer()
            
            max_dd = analyzer.max_drawdown(returns)
            self.assertIsInstance(max_dd, (int, float))
            self.assertTrue(max_dd <= 0)
        except ImportError:
            self.skipTest("Risk model dependencies not installed")


class TestAlternativeData(unittest.TestCase):
    """Test alternative data module."""
    
    def test_imports(self):
        """Test alternative data imports."""
        try:
            from zipline.data import alternative
        except ImportError as e:
            self.skipTest(f"Alternative data dependencies not installed: {e}")
    
    def test_sentiment_analyzer(self):
        """Test sentiment analyzer."""
        try:
            from zipline.data.alternative import SentimentAnalyzer
            
            analyzer = SentimentAnalyzer()
            # Just test initialization
            self.assertIsNotNone(analyzer)
        except ImportError:
            self.skipTest("Alternative data dependencies not installed")


if __name__ == '__main__':
    unittest.main()
