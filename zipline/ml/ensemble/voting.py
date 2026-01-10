"""
Voting Ensemble Methods.
"""

import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class VotingPredictor:
    """
    Ensemble voting predictor.
    
    Combines predictions from multiple models using voting.
    
    Parameters
    ----------
    models : list
        List of models to ensemble
    weights : list, optional
        Weights for each model (default: equal weights)
    voting : str, optional
        'hard' or 'soft' voting (default: 'soft')
    
    Examples
    --------
    >>> voter = VotingPredictor(models=[model1, model2, model3])
    >>> voter.fit(X_train, y_train)
    >>> predictions = voter.predict(X_test)
    """
    
    def __init__(self, models: List, weights=None, voting='soft'):
        self.models = models
        self.weights = weights if weights is not None else [1/len(models)] * len(models)
        self.voting = voting
        self.is_fitted = False
    
    def fit(self, X, y):
        """Train all models."""
        for model in self.models:
            model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions using voting."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.voting == 'soft':
            # Weighted average of predictions
            predictions = np.array([
                model.predict(X) * weight
                for model, weight in zip(self.models, self.weights)
            ])
            return np.sum(predictions, axis=0)
        else:
            # Hard voting (majority vote)
            predictions = np.array([model.predict(X) for model in self.models])
            return np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(),
                axis=0,
                arr=predictions
            )
    
    def predict_proba(self, X):
        """Predict probabilities (soft voting only)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.voting != 'soft':
            raise ValueError("predict_proba only available for soft voting")
        
        probas = np.array([
            model.predict_proba(X) * weight
            for model, weight in zip(self.models, self.weights)
        ])
        return np.sum(probas, axis=0)
