"""Digital Filters."""
import numpy as np
import logging
logger = logging.getLogger(__name__)

class KalmanTrendFilter:
    """Kalman filter for trend extraction."""
    
    def __init__(self):
        pass
    
    def filter(self, signal):
        """Extract trend using Kalman filter."""
        from zipline.quant.statistics.kalman import KalmanFilter
        
        kf = KalmanFilter(
            transition_matrices=[[1, 1], [0, 1]],
            observation_matrices=[[1, 0]]
        )
        
        filtered, _ = kf.filter(signal.reshape(-1, 1))
        return filtered[:, 0] if filtered is not None else signal

class HodrickPrescottFilter:
    """Hodrick-Prescott filter for trend/cycle decomposition."""
    
    def __init__(self, lamb=1600):
        self.lamb = lamb
    
    def filter(self, signal):
        """Apply HP filter."""
        try:
            from statsmodels.tsa.filters.hp_filter import hpfilter
            trend, cycle = hpfilter(signal, lamb=self.lamb)
            return trend, cycle
        except ImportError:
            logger.warning("statsmodels required for HP filter")
            return signal, np.zeros_like(signal)
