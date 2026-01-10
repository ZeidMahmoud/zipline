"""
Hidden Markov Models for Market Regime Detection.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class MarketRegimeHMM:
    """
    Hidden Markov Model for detecting market regimes.
    
    Identifies bull, bear, and sideways markets using HMM.
    
    Parameters
    ----------
    n_states : int, optional
        Number of hidden states/regimes (default: 3)
    n_iter : int, optional
        Number of EM iterations (default: 100)
    
    Examples
    --------
    >>> hmm = MarketRegimeHMM(n_states=3)
    >>> hmm.fit(returns)
    >>> regimes = hmm.predict(returns)
    >>> regime_probs = hmm.predict_proba(returns)
    """
    
    def __init__(self, n_states=3, n_iter=100):
        self.n_states = n_states
        self.n_iter = n_iter
        self.model = None
        self.is_fitted = False
    
    def fit(self, returns, **kwargs):
        """Fit HMM to returns data."""
        try:
            from hmmlearn import hmm
            
            # Reshape for hmmlearn
            X = returns.reshape(-1, 1) if len(returns.shape) == 1 else returns
            
            self.model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.n_iter
            )
            self.model.fit(X)
            self.is_fitted = True
            logger.info(f"HMM with {self.n_states} states fitted")
            
        except ImportError:
            logger.error("hmmlearn required. Install with: pip install hmmlearn")
            raise
        
        return self
    
    def predict(self, returns):
        """Predict most likely regime sequence."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X = returns.reshape(-1, 1) if len(returns.shape) == 1 else returns
        return self.model.predict(X)
    
    def predict_proba(self, returns):
        """Predict regime probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X = returns.reshape(-1, 1) if len(returns.shape) == 1 else returns
        return self.model.predict_proba(X)
    
    def get_regime_characteristics(self):
        """Get mean and covariance for each regime."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return {
            'means': self.model.means_,
            'covariances': self.model.covars_,
            'transition_matrix': self.model.transmat_
        }


class GaussianHMM:
    """
    Gaussian HMM with custom emission distributions.
    
    More flexible HMM implementation allowing custom specifications.
    """
    
    def __init__(self, n_states=2, n_features=1, n_iter=100):
        self.n_states = n_states
        self.n_features = n_features
        self.n_iter = n_iter
        self.is_fitted = False
    
    def fit(self, X, lengths=None):
        """Fit Gaussian HMM."""
        try:
            from hmmlearn import hmm
            
            self.model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.n_iter
            )
            
            if lengths is not None:
                self.model.fit(X, lengths=lengths)
            else:
                self.model.fit(X)
            
            self.is_fitted = True
            
        except ImportError:
            logger.error("hmmlearn required")
            raise
        
        return self
    
    def predict(self, X):
        """Predict state sequence."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.predict(X)
    
    def score(self, X):
        """Compute log-likelihood."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.score(X)
