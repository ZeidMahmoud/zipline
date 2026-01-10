"""
ARIMA and SARIMA Forecasting Models.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class ARIMAForecaster:
    """
    ARIMA (AutoRegressive Integrated Moving Average) forecaster.
    
    Parameters
    ----------
    order : tuple
        (p, d, q) order of the ARIMA model
    
    Examples
    --------
    >>> forecaster = ARIMAForecaster(order=(2, 1, 2))
    >>> forecaster.fit(price_series)
    >>> forecast = forecaster.forecast(steps=10)
    """
    
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.is_fitted = False
    
    def fit(self, data, **kwargs):
        """Fit ARIMA model to data."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            self.model = ARIMA(data, order=self.order)
            self.fitted_model = self.model.fit(**kwargs)
            self.is_fitted = True
            logger.info(f"ARIMA{self.order} fitted successfully")
        except ImportError:
            logger.error("statsmodels required for ARIMA. Install with: pip install statsmodels")
            raise
        return self
    
    def forecast(self, steps=1):
        """Generate forecast."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        return self.fitted_model.forecast(steps=steps)
    
    def get_confidence_intervals(self, steps=1, alpha=0.05):
        """Get forecast confidence intervals."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        return forecast_result.conf_int(alpha=alpha)


class SARIMAForecaster:
    """
    SARIMA (Seasonal ARIMA) forecaster.
    
    Parameters
    ----------
    order : tuple
        (p, d, q) order
    seasonal_order : tuple
        (P, D, Q, s) seasonal order
    
    Examples
    --------
    >>> forecaster = SARIMAForecaster(
    ...     order=(1, 1, 1),
    ...     seasonal_order=(1, 1, 1, 12)
    ... )
    >>> forecaster.fit(monthly_data)
    >>> forecast = forecaster.forecast(steps=12)
    """
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.is_fitted = False
    
    def fit(self, data, **kwargs):
        """Fit SARIMA model."""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            self.model = SARIMAX(data, order=self.order, 
                                seasonal_order=self.seasonal_order)
            self.fitted_model = self.model.fit(**kwargs)
            self.is_fitted = True
            logger.info(f"SARIMA{self.order}x{self.seasonal_order} fitted")
        except ImportError:
            logger.error("statsmodels required")
            raise
        return self
    
    def forecast(self, steps=1):
        """Generate forecast."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        return self.fitted_model.forecast(steps=steps)
