"""
Cointegration Analysis and Pairs Trading.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class EngleGrangerTest:
    """
    Engle-Granger two-step cointegration test.
    
    Tests whether two time series are cointegrated.
    
    Examples
    --------
    >>> test = EngleGrangerTest()
    >>> is_cointegrated, p_value = test.test(series1, series2)
    """
    
    def __init__(self):
        pass
    
    def test(self, y1, y2, significance=0.05):
        """
        Perform Engle-Granger cointegration test.
        
        Returns
        -------
        is_cointegrated : bool
            Whether series are cointegrated at given significance level
        p_value : float
            P-value of the test
        hedge_ratio : float
            Estimated hedge ratio
        """
        try:
            from statsmodels.tsa.stattools import coint
            
            score, p_value, _ = coint(y1, y2)
            
            # Estimate hedge ratio using OLS
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(y2.reshape(-1, 1), y1)
            hedge_ratio = lr.coef_[0]
            
            is_cointegrated = p_value < significance
            
            return is_cointegrated, p_value, hedge_ratio
            
        except ImportError:
            logger.error("statsmodels and sklearn required")
            raise


class JohansenTest:
    """
    Johansen cointegration test for multiple time series.
    
    Tests cointegration relationships among multiple assets.
    """
    
    def __init__(self):
        pass
    
    def test(self, data, det_order=0, k_ar_diff=1):
        """
        Perform Johansen test.
        
        Parameters
        ----------
        data : np.ndarray
            Multiple time series as columns
        det_order : int
            Deterministic trend order
        k_ar_diff : int
            Number of lagged differences
        
        Returns
        -------
        dict
            Test results including trace statistics and critical values
        """
        try:
            from statsmodels.tsa.vector_ar.vecm import coint_johansen
            
            result = coint_johansen(data, det_order, k_ar_diff)
            
            return {
                'trace_statistic': result.lr1,
                'max_eigen_statistic': result.lr2,
                'critical_values_trace': result.cvt,
                'critical_values_max_eigen': result.cvm,
                'eigenvectors': result.evec,
            }
            
        except ImportError:
            logger.error("statsmodels required")
            raise


class PairsFinder:
    """
    Find cointegrated pairs in a universe of assets.
    
    Screens multiple assets to identify cointegrated pairs suitable
    for pairs trading.
    
    Examples
    --------
    >>> finder = PairsFinder()
    >>> pairs = finder.find_pairs(price_data, significance=0.05)
    """
    
    def __init__(self):
        self.eg_test = EngleGrangerTest()
    
    def find_pairs(self, price_data, significance=0.05):
        """
        Find all cointegrated pairs.
        
        Parameters
        ----------
        price_data : dict or DataFrame
            Dictionary or DataFrame with price series for each asset
        significance : float
            Significance level for cointegration test
        
        Returns
        -------
        list
            List of tuples (asset1, asset2, p_value, hedge_ratio)
        """
        import pandas as pd
        
        if isinstance(price_data, dict):
            price_data = pd.DataFrame(price_data)
        
        assets = price_data.columns
        pairs = []
        
        for i, asset1 in enumerate(assets):
            for asset2 in assets[i+1:]:
                y1 = price_data[asset1].values
                y2 = price_data[asset2].values
                
                is_coint, p_val, hedge_ratio = self.eg_test.test(y1, y2, significance)
                
                if is_coint:
                    pairs.append((asset1, asset2, p_val, hedge_ratio))
        
        logger.info(f"Found {len(pairs)} cointegrated pairs")
        return pairs


class SpreadCalculator:
    """
    Calculate and track spreads for pairs trading.
    
    Computes the spread between cointegrated pairs and generates
    trading signals based on spread deviation.
    
    Examples
    --------
    >>> calc = SpreadCalculator()
    >>> spread = calc.calculate_spread(price1, price2, hedge_ratio)
    >>> z_score = calc.calculate_z_score(spread)
    """
    
    def __init__(self, lookback=20):
        self.lookback = lookback
    
    def calculate_spread(self, y1, y2, hedge_ratio):
        """
        Calculate spread between two series.
        
        Parameters
        ----------
        y1 : np.ndarray
            First price series
        y2 : np.ndarray
            Second price series
        hedge_ratio : float
            Hedge ratio from cointegration test
        
        Returns
        -------
        np.ndarray
            Spread series
        """
        return y1 - hedge_ratio * y2
    
    def calculate_z_score(self, spread, window=None):
        """
        Calculate z-score of spread for mean reversion signals.
        
        Parameters
        ----------
        spread : np.ndarray
            Spread series
        window : int, optional
            Rolling window size (uses self.lookback if None)
        
        Returns
        -------
        np.ndarray
            Z-score of spread
        """
        if window is None:
            window = self.lookback
        
        import pandas as pd
        spread_series = pd.Series(spread)
        
        mean = spread_series.rolling(window=window).mean()
        std = spread_series.rolling(window=window).std()
        
        z_score = (spread_series - mean) / std
        return z_score.values
    
    def estimate_half_life(self, spread):
        """
        Estimate half-life of mean reversion.
        
        Parameters
        ----------
        spread : np.ndarray
            Spread series
        
        Returns
        -------
        float
            Half-life in number of periods
        """
        from sklearn.linear_model import LinearRegression
        
        # Fit AR(1) model
        y = np.diff(spread)
        X = spread[:-1].reshape(-1, 1)
        
        lr = LinearRegression()
        lr.fit(X, y)
        
        # Half-life = -log(2) / log(1 + beta)
        beta = lr.coef_[0]
        half_life = -np.log(2) / np.log(1 + beta) if beta < 0 else np.inf
        
        return half_life
