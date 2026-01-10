"""
Advanced risk metrics for portfolio analysis.

This module provides calculations for various risk-adjusted performance
metrics including Sharpe, Sortino, and Calmar ratios.
"""
from typing import Optional, Union
import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)


class RollingSharp:
    """
    Rolling Sharpe Ratio calculator.
    
    Calculates the Sharpe ratio using a rolling window of returns.
    
    Parameters
    ----------
    window : int, optional
        Rolling window length in days. Default is 252 (1 year).
    risk_free_rate : float, optional
        Annual risk-free rate. Default is 0.0.
    annualization_factor : int, optional
        Factor to annualize returns and volatility. Default is 252.
    
    Examples
    --------
    >>> sharpe = RollingSharp(window=60)
    >>> returns = np.random.randn(1000) * 0.01
    >>> ratios = sharpe.calculate(returns)
    """
    
    def __init__(self, window: int = 252, risk_free_rate: float = 0.0,
                 annualization_factor: int = 252):
        self.window = window
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
    
    def calculate(self, returns: Union[np.ndarray, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate rolling Sharpe ratio.
        
        Parameters
        ----------
        returns : array-like
            Historical returns.
            
        Returns
        -------
        float or Series
            Sharpe ratio(s). If returns is a Series, returns a Series of rolling values.
        """
        if isinstance(returns, pd.Series):
            return self._calculate_series(returns)
        else:
            return self._calculate_single(np.asarray(returns))
    
    def _calculate_single(self, returns: np.ndarray) -> float:
        """Calculate single Sharpe ratio value."""
        if len(returns) < 2:
            return np.nan
        
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 2:
            return np.nan
        
        # Use most recent window
        recent_returns = returns[-self.window:]
        
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns, ddof=1)
        
        if std_return == 0:
            return np.nan
        
        # Annualize
        annual_return = mean_return * self.annualization_factor
        annual_std = std_return * np.sqrt(self.annualization_factor)
        
        sharpe = (annual_return - self.risk_free_rate) / annual_std
        
        return sharpe
    
    def _calculate_series(self, returns: pd.Series) -> pd.Series:
        """Calculate rolling Sharpe ratio series."""
        # Calculate rolling mean and std
        rolling_mean = returns.rolling(window=self.window).mean()
        rolling_std = returns.rolling(window=self.window).std()
        
        # Annualize
        annual_return = rolling_mean * self.annualization_factor
        annual_std = rolling_std * np.sqrt(self.annualization_factor)
        
        # Calculate Sharpe
        sharpe = (annual_return - self.risk_free_rate) / annual_std
        
        return sharpe


class SortinoRatio:
    """
    Sortino Ratio calculator.
    
    Similar to Sharpe ratio but only penalizes downside volatility.
    
    Parameters
    ----------
    window : int, optional
        Rolling window length in days. Default is 252.
    risk_free_rate : float, optional
        Annual risk-free rate. Default is 0.0.
    target_return : float, optional
        Target return for downside calculation. Default is 0.0.
    annualization_factor : int, optional
        Factor to annualize returns. Default is 252.
    
    Examples
    --------
    >>> sortino = SortinoRatio(window=60)
    >>> returns = np.random.randn(1000) * 0.01
    >>> ratio = sortino.calculate(returns)
    """
    
    def __init__(self, window: int = 252, risk_free_rate: float = 0.0,
                 target_return: float = 0.0, annualization_factor: int = 252):
        self.window = window
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.annualization_factor = annualization_factor
    
    def calculate(self, returns: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate Sortino ratio.
        
        Parameters
        ----------
        returns : array-like
            Historical returns.
            
        Returns
        -------
        float
            Sortino ratio.
        """
        if len(returns) < 2:
            return np.nan
        
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 2:
            return np.nan
        
        # Use most recent window
        recent_returns = returns[-self.window:]
        
        mean_return = np.mean(recent_returns)
        
        # Calculate downside deviation
        downside_returns = recent_returns[recent_returns < self.target_return]
        
        if len(downside_returns) == 0:
            return np.inf if mean_return > self.risk_free_rate else np.nan
        
        downside_std = np.std(downside_returns, ddof=1)
        
        if downside_std == 0:
            return np.nan
        
        # Annualize
        annual_return = mean_return * self.annualization_factor
        annual_downside_std = downside_std * np.sqrt(self.annualization_factor)
        
        sortino = (annual_return - self.risk_free_rate) / annual_downside_std
        
        return sortino


class CalmarRatio:
    """
    Calmar Ratio calculator.
    
    Ratio of annualized return to maximum drawdown.
    
    Parameters
    ----------
    window : int, optional
        Rolling window length in days. Default is 252 (1 year).
    annualization_factor : int, optional
        Factor to annualize returns. Default is 252.
    
    Examples
    --------
    >>> calmar = CalmarRatio(window=252)
    >>> returns = np.random.randn(1000) * 0.01
    >>> ratio = calmar.calculate(returns)
    """
    
    def __init__(self, window: int = 252, annualization_factor: int = 252):
        self.window = window
        self.annualization_factor = annualization_factor
    
    def calculate(self, returns: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate Calmar ratio.
        
        Parameters
        ----------
        returns : array-like
            Historical returns.
            
        Returns
        -------
        float
            Calmar ratio.
        """
        if len(returns) < 2:
            return np.nan
        
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 2:
            return np.nan
        
        # Use most recent window
        recent_returns = returns[-self.window:]
        
        mean_return = np.mean(recent_returns)
        annual_return = mean_return * self.annualization_factor
        
        # Calculate maximum drawdown
        cumulative = np.cumprod(1 + recent_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        if max_drawdown == 0:
            return np.inf if annual_return > 0 else np.nan
        
        calmar = annual_return / abs(max_drawdown)
        
        return calmar


class MaxDrawdownTracker:
    """
    Maximum drawdown tracker for portfolio monitoring.
    
    Tracks and reports maximum drawdown metrics over time.
    
    Examples
    --------
    >>> tracker = MaxDrawdownTracker()
    >>> portfolio_values = [100000, 105000, 103000, 108000, 102000]
    >>> for value in portfolio_values:
    ...     tracker.update(value)
    >>> print(f"Max DD: {tracker.max_drawdown:.2%}")
    """
    
    def __init__(self):
        self.peak = None
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.drawdown_history = []
    
    def update(self, portfolio_value: float) -> None:
        """
        Update tracker with new portfolio value.
        
        Parameters
        ----------
        portfolio_value : float
            Current portfolio value.
        """
        if self.peak is None or portfolio_value > self.peak:
            self.peak = portfolio_value
        
        self.current_drawdown = (self.peak - portfolio_value) / self.peak
        
        if self.current_drawdown > self.max_drawdown:
            self.max_drawdown = self.current_drawdown
        
        self.drawdown_history.append(self.current_drawdown)
    
    def get_max_drawdown(self) -> float:
        """
        Get maximum drawdown observed.
        
        Returns
        -------
        float
            Maximum drawdown as a decimal.
        """
        return self.max_drawdown
    
    def get_current_drawdown(self) -> float:
        """
        Get current drawdown.
        
        Returns
        -------
        float
            Current drawdown as a decimal.
        """
        return self.current_drawdown
    
    def get_drawdown_duration(self) -> int:
        """
        Get duration of current drawdown.
        
        Returns
        -------
        int
            Number of periods in current drawdown.
        """
        if self.current_drawdown == 0:
            return 0
        
        # Count consecutive periods with non-zero drawdown
        duration = 0
        for dd in reversed(self.drawdown_history):
            if dd > 0:
                duration += 1
            else:
                break
        
        return duration
    
    def reset(self) -> None:
        """Reset the tracker."""
        self.peak = None
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.drawdown_history = []
    
    def get_statistics(self) -> dict:
        """
        Get comprehensive drawdown statistics.
        
        Returns
        -------
        dict
            Dictionary with drawdown statistics.
        """
        return {
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'drawdown_duration': self.get_drawdown_duration(),
            'peak_value': self.peak,
            'avg_drawdown': np.mean(self.drawdown_history) if self.drawdown_history else 0.0,
        }
