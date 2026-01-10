"""
Model Stacking Ensemble.
"""

import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class StackingPredictor:
    """
    Stacked ensemble predictor.
    
    Combines multiple base models with a meta-learner.
    
    Parameters
    ----------
    base_models : list
        List of base models
    meta_model : object
        Meta-learner model
    use_cv : bool, optional
        Whether to use cross-validation (default: True)
    
    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> stacker = StackingPredictor(
    ...     base_models=[LinearRegression(), RandomForestRegressor()],
    ...     meta_model=LinearRegression()
    ... )
    >>> stacker.fit(X_train, y_train)
    >>> predictions = stacker.predict(X_test)
    """
    
    def __init__(self, base_models: List, meta_model, use_cv: bool = True):
        self.base_models = base_models
        self.meta_model = meta_model
        self.use_cv = use_cv
        self.is_fitted = False
    
    def fit(self, X, y, cv=5):
        """Train stacked ensemble."""
        if self.use_cv:
            # Generate out-of-fold predictions for meta-learner
            from sklearn.model_selection import cross_val_predict
            meta_features = np.column_stack([
                cross_val_predict(model, X, y, cv=cv, method='predict')
                for model in self.base_models
            ])
        else:
            # Simple approach: train base models and use predictions
            meta_features = np.column_stack([
                model.fit(X, y).predict(X)
                for model in self.base_models
            ])
        
        # Train all base models on full data
        for model in self.base_models:
            model.fit(X, y)
        
        # Train meta-model
        self.meta_model.fit(meta_features, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions using stacked ensemble."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get base model predictions
        base_predictions = np.column_stack([
            model.predict(X) for model in self.base_models
        ])
        
        # Meta-model prediction
        return self.meta_model.predict(base_predictions)
