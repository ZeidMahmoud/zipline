"""
ML-based factors for Zipline Pipeline API.

This module provides pipeline factors that integrate machine learning
models for generating trading signals.
"""
from typing import Optional, List, Any
import numpy as np
import pandas as pd

try:
    from zipline.pipeline.factors import CustomFactor
    from zipline.pipeline.data import USEquityPricing
except ImportError:
    # Fallback for development/testing
    CustomFactor = object
    USEquityPricing = None


class MLPredictionFactor(CustomFactor):
    """
    Pipeline factor that uses a trained ML model for predictions.
    
    This factor applies a trained machine learning model to generate
    predictions for each asset in the pipeline.
    
    Parameters
    ----------
    model : object
        Trained model with a predict() method.
    inputs : list, optional
        List of BoundColumn objects to use as features.
    window_length : int, optional
        Number of days of historical data to use.
    
    Examples
    --------
    >>> from zipline.ml import MLPredictionFactor, SklearnModelWrapper
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> 
    >>> # Train a model
    >>> model = RandomForestRegressor()
    >>> # ... training code ...
    >>> 
    >>> # Use in pipeline
    >>> ml_factor = MLPredictionFactor(
    ...     model=model,
    ...     inputs=[USEquityPricing.close],
    ...     window_length=20
    ... )
    """
    
    def __init__(self, model: Any, inputs: Optional[List] = None, 
                 window_length: int = 20, **kwargs):
        self.model = model
        super().__init__(
            inputs=inputs or [],
            window_length=window_length,
            **kwargs
        )
    
    def compute(self, today, assets, out, *inputs):
        """
        Compute predictions for all assets.
        
        Parameters
        ----------
        today : pd.Timestamp
            The current simulation date.
        assets : pd.Index
            The assets in the universe.
        out : np.array
            Output array to populate with predictions.
        *inputs : tuple of np.array
            Input arrays from the pipeline.
        """
        if not inputs:
            out[:] = np.nan
            return
        
        # Reshape inputs for prediction
        # Each input is (window_length, n_assets)
        n_assets = len(assets)
        
        # Stack all inputs and transpose to get features for each asset
        # Shape: (n_assets, window_length * n_inputs)
        features = np.column_stack([
            inp.T.reshape(n_assets, -1) for inp in inputs
        ])
        
        # Handle NaN values
        mask = ~np.isnan(features).any(axis=1)
        
        try:
            # Make predictions for valid rows
            if mask.any():
                predictions = self.model.predict(features[mask])
                out[mask] = predictions
                out[~mask] = np.nan
            else:
                out[:] = np.nan
        except Exception as e:
            import logging
            logging.error(f"Error in ML prediction: {e}")
            out[:] = np.nan


class FeatureUnion(CustomFactor):
    """
    Combine multiple pipeline factors into a single feature vector.
    
    This factor stacks multiple input factors to create a combined
    feature representation for ML models.
    
    Parameters
    ----------
    factors : list
        List of Factor objects to combine.
    
    Examples
    --------
    >>> from zipline.pipeline.factors import SimpleMovingAverage, RSI
    >>> from zipline.ml import FeatureUnion
    >>> 
    >>> features = FeatureUnion([
    ...     SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=20),
    ...     RSI(),
    ... ])
    """
    
    def __init__(self, factors: List, **kwargs):
        self.factors = factors
        # Use the longest window length from all factors
        max_window = max(f.window_length for f in factors)
        super().__init__(
            inputs=[],
            window_length=max_window,
            **kwargs
        )
    
    def compute(self, today, assets, out, *inputs):
        """
        Combine features from all factors.
        
        Parameters
        ----------
        today : pd.Timestamp
            The current simulation date.
        assets : pd.Index
            The assets in the universe.
        out : np.array
            Output array to populate with combined features.
        *inputs : tuple of np.array
            Input arrays from the pipeline.
        """
        # This is a simplified implementation
        # In practice, would need more sophisticated feature combination
        if inputs:
            combined = np.concatenate([inp.flatten() for inp in inputs])
            out[:] = combined[:len(out)]
        else:
            out[:] = np.nan


class RollingRegression(CustomFactor):
    """
    Rolling window regression factor.
    
    Computes regression coefficients or predictions using a rolling
    window of historical data.
    
    Parameters
    ----------
    target : BoundColumn
        Target variable for regression.
    predictors : list of BoundColumn
        Predictor variables for regression.
    window_length : int
        Number of days in rolling window.
    return_type : str, optional
        What to return: 'predictions' or 'coefficients'.
        Default is 'predictions'.
    
    Examples
    --------
    >>> from zipline.ml import RollingRegression
    >>> from zipline.pipeline.data import USEquityPricing
    >>> 
    >>> regression = RollingRegression(
    ...     target=USEquityPricing.close,
    ...     predictors=[USEquityPricing.volume],
    ...     window_length=30,
    ...     return_type='predictions'
    ... )
    """
    
    def __init__(self, target, predictors: List, window_length: int = 30,
                 return_type: str = 'predictions', **kwargs):
        self.target = target
        self.predictors = predictors
        self.return_type = return_type
        
        inputs = [target] + predictors
        super().__init__(
            inputs=inputs,
            window_length=window_length,
            **kwargs
        )
    
    def compute(self, today, assets, out, *inputs):
        """
        Compute rolling regression.
        
        Parameters
        ----------
        today : pd.Timestamp
            The current simulation date.
        assets : pd.Index
            The assets in the universe.
        out : np.array
            Output array to populate with predictions/coefficients.
        *inputs : tuple of np.array
            Input arrays: first is target, rest are predictors.
        """
        if len(inputs) < 2:
            out[:] = np.nan
            return
        
        target_data = inputs[0]  # Shape: (window_length, n_assets)
        predictor_data = np.array(inputs[1:])  # Shape: (n_predictors, window_length, n_assets)
        
        n_assets = len(assets)
        
        for i in range(n_assets):
            y = target_data[:, i]
            X = predictor_data[:, :, i].T  # Shape: (window_length, n_predictors)
            
            # Check for NaN values
            if np.isnan(y).any() or np.isnan(X).any():
                out[i] = np.nan
                continue
            
            try:
                # Simple linear regression using numpy
                # Add intercept term
                X_with_intercept = np.column_stack([np.ones(len(X)), X])
                
                # Compute coefficients: (X'X)^-1 X'y
                coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                
                if self.return_type == 'predictions':
                    # Return prediction for the last time point
                    out[i] = np.dot(X_with_intercept[-1], coeffs)
                else:
                    # Return the slope coefficient (skip intercept)
                    out[i] = coeffs[1] if len(coeffs) > 1 else coeffs[0]
                    
            except (np.linalg.LinAlgError, Exception) as e:
                # Singular matrix or other error
                out[i] = np.nan
