"""
Model wrappers and registry for ML models in Zipline.

This module provides wrappers for machine learning models and a
registry system for managing trained models.
"""
from typing import Any, Dict, Optional
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class SklearnModelWrapper:
    """
    Wrapper for scikit-learn models to use in Zipline.
    
    This wrapper provides a consistent interface for using scikit-learn
    models in Zipline pipelines and algorithms.
    
    Parameters
    ----------
    model : sklearn model
        A trained scikit-learn model.
    feature_names : list of str, optional
        Names of the features used by the model.
    
    Attributes
    ----------
    model : sklearn model
        The wrapped model.
    feature_names : list of str
        Names of features.
    
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from zipline.ml import SklearnModelWrapper
    >>> 
    >>> # Train a model
    >>> model = RandomForestRegressor(n_estimators=100)
    >>> model.fit(X_train, y_train)
    >>> 
    >>> # Wrap it
    >>> wrapped_model = SklearnModelWrapper(
    ...     model=model,
    ...     feature_names=['feature1', 'feature2']
    ... )
    >>> 
    >>> # Use for predictions
    >>> predictions = wrapped_model.predict(X_test)
    """
    
    def __init__(self, model: Any, feature_names: Optional[list] = None):
        self.model = model
        self.feature_names = feature_names or []
        
        # Validate model has required methods
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have a predict() method")
    
    def predict(self, X):
        """
        Make predictions using the wrapped model.
        
        Parameters
        ----------
        X : array-like
            Feature matrix for prediction.
            
        Returns
        -------
        array
            Model predictions.
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities (for classifiers).
        
        Parameters
        ----------
        X : array-like
            Feature matrix for prediction.
            
        Returns
        -------
        array
            Class probabilities.
        """
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError("Model does not support predict_proba")
        return self.model.predict_proba(X)
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Parameters
        ----------
        path : str
            Path to save the model.
        """
        try:
            import joblib
            joblib.dump(self.model, path)
            log.info(f"Model saved to {path}")
        except ImportError:
            raise ImportError(
                "joblib is required to save models. "
                "Install it with: pip install joblib"
            )
    
    @classmethod
    def load(cls, path: str, feature_names: Optional[list] = None):
        """
        Load a model from disk.
        
        Parameters
        ----------
        path : str
            Path to load the model from.
        feature_names : list of str, optional
            Names of features.
            
        Returns
        -------
        SklearnModelWrapper
            Loaded model wrapper.
        """
        try:
            import joblib
            model = joblib.load(path)
            log.info(f"Model loaded from {path}")
            return cls(model=model, feature_names=feature_names)
        except ImportError:
            raise ImportError(
                "joblib is required to load models. "
                "Install it with: pip install joblib"
            )


class ModelRegistry:
    """
    Registry for managing trained ML models.
    
    This registry provides a central place to store and retrieve
    trained models for use in Zipline algorithms.
    
    Examples
    --------
    >>> from zipline.ml import ModelRegistry, SklearnModelWrapper
    >>> 
    >>> registry = ModelRegistry()
    >>> 
    >>> # Register a model
    >>> registry.register('my_model', wrapped_model)
    >>> 
    >>> # Retrieve a model
    >>> model = registry.get('my_model')
    >>> 
    >>> # List all models
    >>> models = registry.list_models()
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the model registry.
        
        Parameters
        ----------
        storage_path : str, optional
            Path to directory for storing models on disk.
        """
        self._models: Dict[str, Any] = {}
        self.storage_path = Path(storage_path) if storage_path else None
        
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def register(self, name: str, model: Any, 
                 save_to_disk: bool = False) -> None:
        """
        Register a model in the registry.
        
        Parameters
        ----------
        name : str
            Name to register the model under.
        model : Any
            The model to register.
        save_to_disk : bool, optional
            Whether to save the model to disk.
        """
        self._models[name] = model
        log.info(f"Registered model: {name}")
        
        if save_to_disk and self.storage_path:
            model_path = self.storage_path / f"{name}.joblib"
            if isinstance(model, SklearnModelWrapper):
                model.save(str(model_path))
            else:
                try:
                    import joblib
                    joblib.dump(model, model_path)
                except ImportError:
                    log.warning("joblib not available, skipping disk save")
    
    def get(self, name: str) -> Optional[Any]:
        """
        Retrieve a model from the registry.
        
        Parameters
        ----------
        name : str
            Name of the model to retrieve.
            
        Returns
        -------
        model or None
            The registered model, or None if not found.
        """
        model = self._models.get(name)
        
        # If not in memory but storage path exists, try loading from disk
        if model is None and self.storage_path:
            model_path = self.storage_path / f"{name}.joblib"
            if model_path.exists():
                try:
                    import joblib
                    model = joblib.load(model_path)
                    self._models[name] = model
                    log.info(f"Loaded model from disk: {name}")
                except ImportError:
                    log.warning("joblib not available, cannot load from disk")
        
        return model
    
    def unregister(self, name: str) -> bool:
        """
        Remove a model from the registry.
        
        Parameters
        ----------
        name : str
            Name of the model to remove.
            
        Returns
        -------
        bool
            True if model was removed, False if not found.
        """
        if name in self._models:
            del self._models[name]
            log.info(f"Unregistered model: {name}")
            return True
        return False
    
    def list_models(self) -> list:
        """
        List all registered models.
        
        Returns
        -------
        list of str
            Names of all registered models.
        """
        return list(self._models.keys())
    
    def clear(self) -> None:
        """Clear all models from the registry."""
        self._models.clear()
        log.info("Cleared all models from registry")


# Global model registry instance
_global_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """
    Get the global model registry instance.
    
    Returns
    -------
    ModelRegistry
        The global registry.
    """
    return _global_registry
