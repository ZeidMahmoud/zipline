"""
Tests for ML module.
"""
import unittest
import numpy as np


class TestMLModels(unittest.TestCase):
    """Test ML model wrappers."""
    
    def test_sklearn_wrapper_init(self):
        """Test SklearnModelWrapper initialization."""
        from zipline.ml.models import SklearnModelWrapper
        
        # Create a mock model
        class MockModel:
            def predict(self, X):
                return np.zeros(len(X))
        
        model = MockModel()
        wrapper = SklearnModelWrapper(model=model)
        
        self.assertIsNotNone(wrapper.model)
        self.assertTrue(hasattr(wrapper, 'predict'))
    
    def test_model_registry(self):
        """Test ModelRegistry."""
        from zipline.ml.models import ModelRegistry
        
        registry = ModelRegistry()
        
        # Test registration
        class MockModel:
            def predict(self, X):
                return np.zeros(len(X))
        
        model = MockModel()
        registry.register('test_model', model)
        
        # Test retrieval
        retrieved = registry.get('test_model')
        self.assertIsNotNone(retrieved)
        
        # Test list
        models = registry.list_models()
        self.assertIn('test_model', models)


class TestFeatures(unittest.TestCase):
    """Test feature engineering utilities."""
    
    def test_compute_sma(self):
        """Test simple moving average calculation."""
        from zipline.ml.features import compute_sma
        
        prices = np.array([1, 2, 3, 4, 5])
        sma = compute_sma(prices, window=3)
        
        self.assertEqual(len(sma), len(prices))
        # Last value should be mean of [3, 4, 5] = 4
        self.assertAlmostEqual(sma[-1], 4.0)
    
    def test_compute_rsi(self):
        """Test RSI calculation."""
        from zipline.ml.features import compute_rsi
        
        # Create price series with upward trend
        prices = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        rsi = compute_rsi(prices, window=10)
        
        # RSI should be high for uptrend
        self.assertGreater(rsi, 50)


if __name__ == '__main__':
    unittest.main()
