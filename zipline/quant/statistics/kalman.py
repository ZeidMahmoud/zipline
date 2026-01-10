"""
Kalman Filtering for State-Space Models.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class KalmanFilter:
    """
    Kalman Filter for state estimation and trend extraction.
    
    Parameters
    ----------
    transition_matrices : np.ndarray
        State transition matrix
    observation_matrices : np.ndarray
        Observation matrix
    initial_state_mean : np.ndarray, optional
        Initial state estimate
    initial_state_covariance : np.ndarray, optional
        Initial state covariance
    
    Examples
    --------
    >>> kf = KalmanFilter(
    ...     transition_matrices=[[1, 1], [0, 1]],
    ...     observation_matrices=[[1, 0]]
    ... )
    >>> filtered_state = kf.filter(observations)
    """
    
    def __init__(self, transition_matrices, observation_matrices,
                 initial_state_mean=None, initial_state_covariance=None):
        self.transition_matrices = np.asarray(transition_matrices)
        self.observation_matrices = np.asarray(observation_matrices)
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.is_fitted = False
    
    def filter(self, observations):
        """Apply Kalman filter to observations."""
        try:
            from pykalman import KalmanFilter as PyKalmanFilter
            
            kf = PyKalmanFilter(
                transition_matrices=self.transition_matrices,
                observation_matrices=self.observation_matrices,
                initial_state_mean=self.initial_state_mean,
                initial_state_covariance=self.initial_state_covariance,
            )
            
            filtered_state_means, filtered_state_covariances = kf.filter(observations)
            self.is_fitted = True
            return filtered_state_means, filtered_state_covariances
            
        except ImportError:
            logger.warning("pykalman not available, using basic implementation")
            return self._basic_filter(observations)
    
    def _basic_filter(self, observations):
        """Basic Kalman filter implementation."""
        n = len(observations)
        state_dim = self.transition_matrices.shape[0]
        
        # Initialize
        if self.initial_state_mean is None:
            state = np.zeros(state_dim)
        else:
            state = self.initial_state_mean.copy()
        
        if self.initial_state_covariance is None:
            P = np.eye(state_dim)
        else:
            P = self.initial_state_covariance.copy()
        
        # Process and observation noise (simplified)
        Q = np.eye(state_dim) * 0.01
        R = np.eye(1) * 1.0
        
        filtered_states = np.zeros((n, state_dim))
        
        for t in range(n):
            # Prediction
            state = self.transition_matrices @ state
            P = self.transition_matrices @ P @ self.transition_matrices.T + Q
            
            # Update
            y = observations[t] - self.observation_matrices @ state
            S = self.observation_matrices @ P @ self.observation_matrices.T + R
            K = P @ self.observation_matrices.T @ np.linalg.inv(S)
            
            state = state + K @ y
            P = (np.eye(state_dim) - K @ self.observation_matrices) @ P
            
            filtered_states[t] = state
        
        return filtered_states, None


class KalmanSmoother:
    """
    Kalman Smoother for smoothed state estimates.
    
    Performs forward-backward pass for better state estimation.
    """
    
    def __init__(self, transition_matrices, observation_matrices):
        self.kf = KalmanFilter(transition_matrices, observation_matrices)
    
    def smooth(self, observations):
        """Apply Kalman smoother."""
        try:
            from pykalman import KalmanFilter as PyKalmanFilter
            
            kf = PyKalmanFilter(
                transition_matrices=self.kf.transition_matrices,
                observation_matrices=self.kf.observation_matrices,
            )
            
            smoothed_state_means, smoothed_state_covariances = kf.smooth(observations)
            return smoothed_state_means, smoothed_state_covariances
            
        except ImportError:
            logger.warning("pykalman not available")
            # Fallback to filtering
            return self.kf.filter(observations)
