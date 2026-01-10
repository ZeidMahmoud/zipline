"""
Tests for deep learning module.
"""

import unittest
import numpy as np


class TestDeepLearning(unittest.TestCase):
    """Test deep learning module imports and basic functionality."""
    
    def test_imports(self):
        """Test that deep learning modules can be imported."""
        try:
            from zipline.ml import deep_learning
        except ImportError as e:
            self.skipTest(f"Deep learning dependencies not installed: {e}")
    
    def test_lstm_predictor_init(self):
        """Test LSTM predictor initialization."""
        try:
            from zipline.ml.deep_learning import LSTMPredictor
            
            predictor = LSTMPredictor(input_dim=5, hidden_units=50)
            self.assertEqual(predictor.input_dim, 5)
            self.assertEqual(predictor.hidden_units, 50)
        except ImportError:
            self.skipTest("Deep learning dependencies not installed")
    
    def test_transformer_predictor_init(self):
        """Test Transformer predictor initialization."""
        try:
            from zipline.ml.deep_learning import TransformerPredictor
            
            predictor = TransformerPredictor(input_dim=10, d_model=64)
            self.assertEqual(predictor.input_dim, 10)
            self.assertEqual(predictor.d_model, 64)
        except ImportError:
            self.skipTest("Deep learning dependencies not installed")


class TestReinforcementLearning(unittest.TestCase):
    """Test reinforcement learning module."""
    
    def test_imports(self):
        """Test RL module imports."""
        try:
            from zipline.ml import reinforcement
        except ImportError as e:
            self.skipTest(f"RL dependencies not installed: {e}")
    
    def test_dqn_trader_init(self):
        """Test DQN trader initialization."""
        try:
            from zipline.ml.reinforcement import DQNTrader
            
            trader = DQNTrader(state_dim=20, action_dim=3)
            self.assertEqual(trader.state_dim, 20)
            self.assertEqual(trader.action_dim, 3)
        except ImportError:
            self.skipTest("RL dependencies not installed")
    
    def test_trading_environment(self):
        """Test trading environment."""
        try:
            from zipline.ml.reinforcement import TradingEnvironment
            
            data = np.random.randn(100, 5)
            env = TradingEnvironment(data)
            
            state = env.reset()
            self.assertIsInstance(state, np.ndarray)
            
            action = 1  # hold
            next_state, reward, done, info = env.step(action)
            self.assertIsInstance(next_state, np.ndarray)
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(done, bool)
        except ImportError:
            self.skipTest("RL dependencies not installed")


class TestEnsemble(unittest.TestCase):
    """Test ensemble methods."""
    
    def test_imports(self):
        """Test ensemble imports."""
        try:
            from zipline.ml import ensemble
        except ImportError as e:
            self.skipTest(f"Ensemble dependencies not installed: {e}")
    
    def test_gradient_boosting(self):
        """Test gradient boosting predictor."""
        try:
            from zipline.ml.ensemble import GradientBoostingPredictor
            from sklearn.datasets import make_regression
            
            X, y = make_regression(n_samples=100, n_features=10, random_state=42)
            
            # Test with XGBoost backend
            predictor = GradientBoostingPredictor(backend='xgboost', n_estimators=10)
            # Just test initialization, not actual fitting
            self.assertEqual(predictor.n_estimators, 10)
        except ImportError:
            self.skipTest("Ensemble dependencies not installed")


if __name__ == '__main__':
    unittest.main()
