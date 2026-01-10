"""
Statistical Models for Time Series Analysis.

Provides ARIMA, GARCH, Kalman filtering, HMM, and cointegration analysis.
"""

try:
    from .arima import ARIMAForecaster, SARIMAForecaster
    from .garch import GARCHModel, EGARCHModel, GJRGARCHModel
    from .kalman import KalmanFilter, KalmanSmoother
    from .hmm import MarketRegimeHMM, GaussianHMM
    from .cointegration import (
        EngleGrangerTest, JohansenTest, PairsFinder,
        SpreadCalculator
    )
    
    __all__ = [
        'ARIMAForecaster', 'SARIMAForecaster',
        'GARCHModel', 'EGARCHModel', 'GJRGARCHModel',
        'KalmanFilter', 'KalmanSmoother',
        'MarketRegimeHMM', 'GaussianHMM',
        'EngleGrangerTest', 'JohansenTest', 'PairsFinder', 'SpreadCalculator',
    ]
except ImportError:
    __all__ = []
