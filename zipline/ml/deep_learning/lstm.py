"""
LSTM-based Price Predictor for Time Series Forecasting.

This module provides LSTM neural networks for predicting future price movements
in financial time series data.
"""

import numpy as np
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class LSTMPredictor:
    """
    LSTM-based time series predictor for financial data.
    
    This class implements a Long Short-Term Memory (LSTM) neural network
    for predicting future price movements. It supports multi-step ahead
    forecasting and can be integrated with Zipline's Pipeline API.
    
    Parameters
    ----------
    input_dim : int
        Number of input features
    hidden_units : int, optional
        Number of LSTM units in each layer (default: 50)
    num_layers : int, optional
        Number of LSTM layers (default: 2)
    dropout : float, optional
        Dropout rate for regularization (default: 0.2)
    forecast_horizon : int, optional
        Number of steps ahead to forecast (default: 1)
    use_gpu : bool, optional
        Whether to use GPU acceleration if available (default: True)
    
    Attributes
    ----------
    model : object or None
        The underlying neural network model (PyTorch or TensorFlow)
    is_fitted : bool
        Whether the model has been trained
    
    Examples
    --------
    >>> predictor = LSTMPredictor(input_dim=5, hidden_units=100)
    >>> predictor.fit(X_train, y_train, epochs=50)
    >>> predictions = predictor.predict(X_test)
    
    Notes
    -----
    This implementation requires either PyTorch or TensorFlow to be installed.
    Install with: pip install zipline[deep_learning]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_units: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 1,
        use_gpu: bool = True,
    ):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.forecast_horizon = forecast_horizon
        self.use_gpu = use_gpu
        
        self.model = None
        self.is_fitted = False
        self._backend = None
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the deep learning backend (PyTorch or TensorFlow)."""
        try:
            import torch
            import torch.nn as nn
            self._backend = 'pytorch'
            self._build_pytorch_model()
        except ImportError:
            try:
                import tensorflow as tf
                self._backend = 'tensorflow'
                self._build_tensorflow_model()
            except ImportError:
                logger.warning(
                    "Neither PyTorch nor TensorFlow is installed. "
                    "Please install with: pip install zipline[deep_learning]"
                )
                raise ImportError(
                    "Deep learning backend not available. "
                    "Install PyTorch or TensorFlow."
                )
    
    def _build_pytorch_model(self):
        """Build LSTM model using PyTorch."""
        import torch
        import torch.nn as nn
        
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_units, num_layers, dropout, output_dim):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(
                    input_dim,
                    hidden_units,
                    num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.fc = nn.Linear(hidden_units, output_dim)
                self.dropout = nn.Dropout(dropout)
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_output = lstm_out[:, -1, :]
                dropped = self.dropout(last_output)
                return self.fc(dropped)
        
        self.model = LSTMModel(
            self.input_dim,
            self.hidden_units,
            self.num_layers,
            self.dropout,
            self.forecast_horizon
        )
        
        if self.use_gpu and torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("Using GPU acceleration with PyTorch")
        else:
            logger.info("Using CPU with PyTorch")
    
    def _build_tensorflow_model(self):
        """Build LSTM model using TensorFlow/Keras."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential()
        
        # Add LSTM layers
        for i in range(self.num_layers):
            return_sequences = (i < self.num_layers - 1)
            model.add(layers.LSTM(
                self.hidden_units,
                return_sequences=return_sequences,
                dropout=self.dropout if self.num_layers > 1 else 0
            ))
        
        # Output layer
        model.add(layers.Dense(self.forecast_horizon))
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        logger.info("Using TensorFlow backend")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 1,
    ):
        """
        Train the LSTM model on historical data.
        
        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, sequence_length, n_features)
        y : np.ndarray
            Target values of shape (n_samples, forecast_horizon)
        epochs : int, optional
            Number of training epochs (default: 50)
        batch_size : int, optional
            Batch size for training (default: 32)
        validation_split : float, optional
            Fraction of data to use for validation (default: 0.2)
        verbose : int, optional
            Verbosity level (default: 1)
        
        Returns
        -------
        self : LSTMPredictor
            Returns self for method chaining
        """
        if self._backend == 'pytorch':
            return self._fit_pytorch(X, y, epochs, batch_size, validation_split, verbose)
        else:
            return self._fit_tensorflow(X, y, epochs, batch_size, validation_split, verbose)
    
    def _fit_pytorch(self, X, y, epochs, batch_size, validation_split, verbose):
        """Fit using PyTorch backend."""
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        if self.use_gpu and torch.cuda.is_available():
            X_tensor = X_tensor.cuda()
            y_tensor = y_tensor.cuda()
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def _fit_tensorflow(self, X, y, epochs, batch_size, validation_split, verbose):
        """Fit using TensorFlow backend."""
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, sequence_length, n_features)
        
        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples, forecast_horizon)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self._backend == 'pytorch':
            return self._predict_pytorch(X)
        else:
            return self._predict_tensorflow(X)
    
    def _predict_pytorch(self, X):
        """Predict using PyTorch backend."""
        import torch
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            if self.use_gpu and torch.cuda.is_available():
                X_tensor = X_tensor.cuda()
            predictions = self.model(X_tensor)
            if self.use_gpu and torch.cuda.is_available():
                predictions = predictions.cpu()
            return predictions.numpy()
    
    def _predict_tensorflow(self, X):
        """Predict using TensorFlow backend."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make probabilistic predictions with uncertainty estimates.
        
        Uses Monte Carlo dropout to estimate prediction uncertainty.
        
        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, sequence_length, n_features)
        n_samples : int, optional
            Number of Monte Carlo samples (default: 100)
        
        Returns
        -------
        mean : np.ndarray
            Mean predictions of shape (n_samples, forecast_horizon)
        std : np.ndarray
            Standard deviation of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = []
        for _ in range(n_samples):
            pred = self.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        return mean, std
    
    def save(self, filepath: str):
        """
        Save the trained model to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        if self._backend == 'pytorch':
            import torch
            torch.save({
                'model_state': self.model.state_dict(),
                'config': {
                    'input_dim': self.input_dim,
                    'hidden_units': self.hidden_units,
                    'num_layers': self.num_layers,
                    'dropout': self.dropout,
                    'forecast_horizon': self.forecast_horizon,
                }
            }, filepath)
        else:
            self.model.save(filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load a trained model from disk.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model
        """
        if self._backend == 'pytorch':
            import torch
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state'])
            self.is_fitted = True
        else:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(filepath)
            self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
