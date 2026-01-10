"""
Tests for risk management module.
"""
import unittest
import numpy as np


class TestVaR(unittest.TestCase):
    """Test VaR calculations."""
    
    def test_historical_var(self):
        """Test historical VaR calculation."""
        from zipline.finance.risk.var import HistoricalVaR
        
        var_calc = HistoricalVaR(confidence_level=0.95)
        
        # Generate sample returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)
        
        var = var_calc.calculate(returns, portfolio_value=100000)
        
        # VaR should be positive and reasonable
        self.assertGreater(var, 0)
        self.assertLess(var, 100000)  # Less than portfolio value
    
    def test_parametric_var(self):
        """Test parametric VaR calculation."""
        from zipline.finance.risk.var import ParametricVaR
        
        var_calc = ParametricVaR(confidence_level=0.95)
        
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)
        
        var = var_calc.calculate(returns, portfolio_value=100000)
        
        self.assertGreater(var, 0)


class TestRiskLimits(unittest.TestCase):
    """Test risk limits."""
    
    def test_max_drawdown_limit(self):
        """Test maximum drawdown limit."""
        from zipline.finance.risk.limits import MaxDrawdownLimit
        
        limit = MaxDrawdownLimit(max_drawdown=0.20)
        
        # Portfolio that doesn't breach limit
        portfolio_values = np.array([100000, 105000, 103000, 108000])
        self.assertFalse(limit.check(portfolio_values))
        
        # Portfolio that breaches limit
        portfolio_values = np.array([100000, 105000, 80000])
        self.assertTrue(limit.check(portfolio_values))
    
    def test_volatility_limit(self):
        """Test volatility limit."""
        from zipline.finance.risk.limits import VolatilityLimit
        
        limit = VolatilityLimit(max_volatility=0.25, window=20)
        
        # Low volatility returns
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 100)
        self.assertFalse(limit.check(returns))
        
        # High volatility returns
        returns = np.random.normal(0, 0.05, 100)
        self.assertTrue(limit.check(returns))


class TestRiskMetrics(unittest.TestCase):
    """Test risk metrics."""
    
    def test_rolling_sharpe(self):
        """Test rolling Sharpe ratio."""
        from zipline.finance.risk.metrics import RollingSharp
        
        sharpe_calc = RollingSharp(window=252)
        
        # Generate positive returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 500)
        
        sharpe = sharpe_calc.calculate(returns)
        
        # Sharpe should be positive for positive mean returns
        self.assertGreater(sharpe, 0)
    
    def test_max_drawdown_tracker(self):
        """Test maximum drawdown tracker."""
        from zipline.finance.risk.metrics import MaxDrawdownTracker
        
        tracker = MaxDrawdownTracker()
        
        # Update with portfolio values
        values = [100000, 105000, 103000, 108000, 102000, 110000]
        for value in values:
            tracker.update(value)
        
        max_dd = tracker.get_max_drawdown()
        
        # Should have some drawdown
        self.assertGreaterEqual(max_dd, 0)
        
        stats = tracker.get_statistics()
        self.assertIn('max_drawdown', stats)
        self.assertIn('current_drawdown', stats)


if __name__ == '__main__':
    unittest.main()
