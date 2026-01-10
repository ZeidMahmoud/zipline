"""
Feature engineering utilities for ML in Zipline.

This module provides common technical indicators and feature
transformations for use with machine learning models.
"""
from typing import List
import numpy as np
import pandas as pd

try:
    from zipline.pipeline.factors import CustomFactor
    from zipline.pipeline.data import USEquityPricing
except ImportError:
    CustomFactor = object
    USEquityPricing = None


class TechnicalFeatures(CustomFactor):
    """
    Pipeline factor that computes common technical indicators as features.
    
    This factor computes multiple technical indicators in a single pass,
    making it efficient for feature engineering in ML pipelines.
    
    Parameters
    ----------
    window_length : int, optional
        Number of days of historical data to use.
    indicators : list of str, optional
        List of indicators to compute. Options:
        - 'returns': Simple returns
        - 'log_returns': Log returns
        - 'momentum': Price momentum
        - 'volatility': Historical volatility
        - 'volume_change': Volume change
        Default is all indicators.
    
    Examples
    --------
    >>> from zipline.ml import TechnicalFeatures
    >>> from zipline.pipeline.data import USEquityPricing
    >>> 
    >>> features = TechnicalFeatures(
    ...     inputs=[USEquityPricing.close, USEquityPricing.volume],
    ...     window_length=20,
    ...     indicators=['returns', 'momentum', 'volatility']
    ... )
    """
    
    inputs = []
    window_length = 20
    
    def __init__(self, inputs=None, window_length=20, 
                 indicators=None, **kwargs):
        self.indicators = indicators or [
            'returns', 'log_returns', 'momentum', 
            'volatility', 'volume_change'
        ]
        
        if inputs is None and USEquityPricing:
            inputs = [USEquityPricing.close, USEquityPricing.volume]
        
        super().__init__(
            inputs=inputs or [],
            window_length=window_length,
            **kwargs
        )
    
    def compute(self, today, assets, out, *inputs):
        """
        Compute technical indicators.
        
        Parameters
        ----------
        today : pd.Timestamp
            The current simulation date.
        assets : pd.Index
            The assets in the universe.
        out : np.array
            Output array to populate with feature values.
        *inputs : tuple of np.array
            Input arrays from the pipeline.
        """
        if not inputs or len(inputs) < 1:
            out[:] = np.nan
            return
        
        prices = inputs[0]  # Shape: (window_length, n_assets)
        volumes = inputs[1] if len(inputs) > 1 else None
        
        # Compute features for each asset
        n_assets = len(assets)
        features = []
        
        for i in range(n_assets):
            asset_features = {}
            price_series = prices[:, i]
            
            # Skip if all NaN
            if np.isnan(price_series).all():
                features.append(np.nan)
                continue
            
            # Returns
            if 'returns' in self.indicators:
                returns = np.diff(price_series) / price_series[:-1]
                asset_features['returns'] = np.nanmean(returns)
            
            # Log returns
            if 'log_returns' in self.indicators:
                log_returns = np.diff(np.log(price_series))
                asset_features['log_returns'] = np.nanmean(log_returns)
            
            # Momentum (change from first to last)
            if 'momentum' in self.indicators:
                momentum = (price_series[-1] - price_series[0]) / price_series[0]
                asset_features['momentum'] = momentum
            
            # Volatility (std of returns)
            if 'volatility' in self.indicators:
                returns = np.diff(price_series) / price_series[:-1]
                asset_features['volatility'] = np.nanstd(returns)
            
            # Volume change
            if 'volume_change' in self.indicators and volumes is not None:
                volume_series = volumes[:, i]
                if not np.isnan(volume_series).all() and len(volume_series) > 1:
                    volume_change = (volume_series[-1] - volume_series[0]) / volume_series[0]
                    asset_features['volume_change'] = volume_change
                else:
                    asset_features['volume_change'] = np.nan
            
            # Combine features (use first feature value for simplicity)
            if asset_features:
                features.append(list(asset_features.values())[0])
            else:
                features.append(np.nan)
        
        out[:] = np.array(features)


def compute_sma(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Compute Simple Moving Average.
    
    Parameters
    ----------
    prices : np.ndarray
        Price array.
    window : int
        Window length for MA.
        
    Returns
    -------
    np.ndarray
        Moving average values.
    """
    if len(prices) < window:
        return np.full(len(prices), np.nan)
    
    return pd.Series(prices).rolling(window=window).mean().values


def compute_ema(prices: np.ndarray, span: int) -> np.ndarray:
    """
    Compute Exponential Moving Average.
    
    Parameters
    ----------
    prices : np.ndarray
        Price array.
    span : int
        Span for EMA calculation.
        
    Returns
    -------
    np.ndarray
        EMA values.
    """
    return pd.Series(prices).ewm(span=span, adjust=False).mean().values


def compute_rsi(prices: np.ndarray, window: int = 14) -> float:
    """
    Compute Relative Strength Index.
    
    Parameters
    ----------
    prices : np.ndarray
        Price array.
    window : int, optional
        Window length for RSI.
        
    Returns
    -------
    float
        RSI value.
    """
    if len(prices) < window + 1:
        return np.nan
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-window:])
    avg_loss = np.mean(losses[-window:])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_bollinger_bands(prices: np.ndarray, window: int = 20, 
                           num_std: float = 2.0) -> tuple:
    """
    Compute Bollinger Bands.
    
    Parameters
    ----------
    prices : np.ndarray
        Price array.
    window : int, optional
        Window length for bands.
    num_std : float, optional
        Number of standard deviations.
        
    Returns
    -------
    tuple
        (middle_band, upper_band, lower_band)
    """
    if len(prices) < window:
        return np.nan, np.nan, np.nan
    
    middle = compute_sma(prices, window)[-1]
    std = pd.Series(prices).rolling(window=window).std().iloc[-1]
    
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    
    return middle, upper, lower


def compute_macd(prices: np.ndarray, fast: int = 12, 
                slow: int = 26, signal: int = 9) -> tuple:
    """
    Compute MACD (Moving Average Convergence Divergence).
    
    Parameters
    ----------
    prices : np.ndarray
        Price array.
    fast : int, optional
        Fast EMA period.
    slow : int, optional
        Slow EMA period.
    signal : int, optional
        Signal line period.
        
    Returns
    -------
    tuple
        (macd_line, signal_line, histogram)
    """
    if len(prices) < slow:
        return np.nan, np.nan, np.nan
    
    fast_ema = compute_ema(prices, fast)
    slow_ema = compute_ema(prices, slow)
    
    macd_line = fast_ema - slow_ema
    signal_line = compute_ema(macd_line[~np.isnan(macd_line)], signal)
    
    # Pad signal line to match length
    signal_padded = np.full(len(macd_line), np.nan)
    signal_padded[-len(signal_line):] = signal_line
    
    histogram = macd_line - signal_padded
    
    return macd_line[-1], signal_padded[-1], histogram[-1]
