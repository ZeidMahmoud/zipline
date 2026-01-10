"""
Gradient Boosting Ensemble Methods.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class GradientBoostingPredictor:
    """
    Wrapper for gradient boosting models (XGBoost, LightGBM, CatBoost).
    
    Parameters
    ----------
    backend : str, optional
        Boosting backend: 'xgboost', 'lightgbm', or 'catboost' (default: 'xgboost')
    n_estimators : int, optional
        Number of boosting rounds (default: 100)
    learning_rate : float, optional
        Learning rate (default: 0.1)
    max_depth : int, optional
        Maximum tree depth (default: 6)
    **kwargs
        Additional parameters for the backend
    
    Examples
    --------
    >>> booster = GradientBoostingPredictor(backend='xgboost', n_estimators=200)
    >>> booster.fit(X_train, y_train)
    >>> predictions = booster.predict(X_test)
    >>> importance = booster.get_feature_importance()
    """
    
    def __init__(self, backend='xgboost', n_estimators=100, 
                 learning_rate=0.1, max_depth=6, **kwargs):
        self.backend = backend
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.kwargs = kwargs
        self.model = None
        self.is_fitted = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the boosting model."""
        if self.backend == 'xgboost':
            try:
                import xgboost as xgb
                self.model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    **self.kwargs
                )
            except ImportError:
                logger.warning("XGBoost not installed")
        elif self.backend == 'lightgbm':
            try:
                import lightgbm as lgb
                self.model = lgb.LGBMRegressor(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    **self.kwargs
                )
            except ImportError:
                logger.warning("LightGBM not installed")
        elif self.backend == 'catboost':
            try:
                from catboost import CatBoostRegressor
                self.model = CatBoostRegressor(
                    iterations=self.n_estimators,
                    learning_rate=self.learning_rate,
                    depth=self.max_depth,
                    verbose=False,
                    **self.kwargs
                )
            except ImportError:
                logger.warning("CatBoost not installed")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def fit(self, X, y, **fit_params):
        """Train the boosting model."""
        if self.model is None:
            raise ImportError(f"Backend {self.backend} not available")
        
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def get_feature_importance(self, importance_type='gain'):
        """
        Get feature importance scores.
        
        Parameters
        ----------
        importance_type : str, optional
            Type of importance: 'gain', 'weight', 'cover' (default: 'gain')
        
        Returns
        -------
        np.ndarray
            Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            logger.warning("Feature importance not available for this model")
            return None
