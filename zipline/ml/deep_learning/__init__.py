"""
Deep Learning Module for Zipline.

This module provides deep learning models for time-series prediction,
chart pattern recognition, and scenario generation in algorithmic trading.
"""

try:
    from .lstm import LSTMPredictor
    from .transformer import TransformerPredictor, TemporalFusionTransformer
    from .cnn_charts import ChartPatternCNN
    from .gan_scenarios import MarketGAN, ConditionalGAN
    
    __all__ = [
        'LSTMPredictor',
        'TransformerPredictor',
        'TemporalFusionTransformer',
        'ChartPatternCNN',
        'MarketGAN',
        'ConditionalGAN',
    ]
except ImportError:
    # Deep learning dependencies are optional
    __all__ = []
