"""
GARCH Models for Volatility Forecasting.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class GARCHModel:
    """
    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model.
    
    Parameters
    ----------
    p : int, optional
        GARCH order (default: 1)
    q : int, optional
        ARCH order (default: 1)
    
    Examples
    --------
    >>> garch = GARCHModel(p=1, q=1)
    >>> garch.fit(returns)
    >>> vol_forecast = garch.forecast_volatility(horizon=10)
    """
    
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.model = None
        self.is_fitted = False
    
    def fit(self, returns, **kwargs):
        """Fit GARCH model to returns."""
        try:
            from arch import arch_model
            self.model = arch_model(returns, vol='Garch', p=self.p, q=self.q)
            self.fitted_model = self.model.fit(**kwargs)
            self.is_fitted = True
            logger.info(f"GARCH({self.p},{self.q}) fitted")
        except ImportError:
            logger.error("arch package required. Install with: pip install arch")
            raise
        return self
    
    def forecast_volatility(self, horizon=1):
        """Forecast volatility."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        forecasts = self.fitted_model.forecast(horizon=horizon)
        return np.sqrt(forecasts.variance.values[-1, :])
    
    def calculate_var(self, alpha=0.05):
        """Calculate Value at Risk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        from scipy import stats
        vol_forecast = self.forecast_volatility(horizon=1)[0]
        var = stats.norm.ppf(alpha) * vol_forecast
        return var


class EGARCHModel:
    """
    EGARCH (Exponential GARCH) model for asymmetric volatility.
    
    Captures leverage effects where negative returns increase volatility
    more than positive returns.
    """
    
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.is_fitted = False
    
    def fit(self, returns, **kwargs):
        """Fit EGARCH model."""
        try:
            from arch import arch_model
            self.model = arch_model(returns, vol='EGARCH', p=self.p, q=self.q)
            self.fitted_model = self.model.fit(**kwargs)
            self.is_fitted = True
        except ImportError:
            logger.error("arch package required")
            raise
        return self
    
    def forecast_volatility(self, horizon=1):
        """Forecast volatility."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        forecasts = self.fitted_model.forecast(horizon=horizon)
        return np.sqrt(forecasts.variance.values[-1, :])


class GJRGARCHModel:
    """
    GJR-GARCH model for leverage effects.
    
    Extension of GARCH that allows for asymmetric response to shocks.
    """
    
    def __init__(self, p=1, o=1, q=1):
        self.p = p
        self.o = o
        self.q = q
        self.is_fitted = False
    
    def fit(self, returns, **kwargs):
        """Fit GJR-GARCH model."""
        try:
            from arch import arch_model
            self.model = arch_model(returns, vol='GARCH', p=self.p, o=self.o, q=self.q)
            self.fitted_model = self.model.fit(**kwargs)
            self.is_fitted = True
        except ImportError:
            logger.error("arch package required")
            raise
        return self
    
    def forecast_volatility(self, horizon=1):
        """Forecast volatility."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        forecasts = self.fitted_model.forecast(horizon=horizon)
        return np.sqrt(forecasts.variance.values[-1, :])
