"""
Transformer-based Models for Time Series Prediction.

This module provides transformer architectures including attention mechanisms
for market pattern recognition and time series forecasting.
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TransformerPredictor:
    """
    Transformer-based predictor for financial time series.
    
    Uses multi-head self-attention mechanism for capturing long-range
    dependencies in market data.
    
    Parameters
    ----------
    input_dim : int
        Number of input features
    d_model : int, optional
        Dimension of the model (default: 64)
    nhead : int, optional
        Number of attention heads (default: 8)
    num_layers : int, optional
        Number of transformer layers (default: 3)
    dropout : float, optional
        Dropout rate (default: 0.1)
    forecast_horizon : int, optional
        Number of steps ahead to forecast (default: 1)
    use_gpu : bool, optional
        Whether to use GPU if available (default: True)
    
    Examples
    --------
    >>> predictor = TransformerPredictor(input_dim=10, d_model=128)
    >>> predictor.fit(X_train, y_train, epochs=100)
    >>> predictions = predictor.predict(X_test)
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        forecast_horizon: int = 1,
        use_gpu: bool = True,
    ):
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.forecast_horizon = forecast_horizon
        self.use_gpu = use_gpu
        
        self.model = None
        self.is_fitted = False
        self._backend = None
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the deep learning backend."""
        try:
            import torch
            self._backend = 'pytorch'
            self._build_pytorch_model()
        except ImportError:
            try:
                import tensorflow as tf
                self._backend = 'tensorflow'
                self._build_tensorflow_model()
            except ImportError:
                raise ImportError(
                    "Deep learning backend not available. "
                    "Install PyTorch or TensorFlow."
                )
    
    def _build_pytorch_model(self):
        """Build transformer model using PyTorch."""
        import torch
        import torch.nn as nn
        import math
        
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe)
            
            def forward(self, x):
                return x + self.pe[:, :x.size(1)]
        
        class TransformerModel(nn.Module):
            def __init__(self, input_dim, d_model, nhead, num_layers, dropout, output_dim):
                super().__init__()
                self.embedding = nn.Linear(input_dim, d_model)
                self.pos_encoder = PositionalEncoding(d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward=d_model*4, dropout=dropout
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.fc = nn.Linear(d_model, output_dim)
                self.d_model = d_model
            
            def forward(self, x):
                x = self.embedding(x) * math.sqrt(self.d_model)
                x = self.pos_encoder(x)
                x = x.transpose(0, 1)  # (batch, seq, features) -> (seq, batch, features)
                x = self.transformer(x)
                x = x.mean(dim=0)  # Global average pooling
                return self.fc(x)
        
        self.model = TransformerModel(
            self.input_dim,
            self.d_model,
            self.nhead,
            self.num_layers,
            self.dropout,
            self.forecast_horizon
        )
        
        if self.use_gpu and torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def _build_tensorflow_model(self):
        """Build transformer model using TensorFlow."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        class TransformerBlock(layers.Layer):
            def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
                super().__init__()
                self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
                self.ffn = keras.Sequential([
                    layers.Dense(dff, activation='relu'),
                    layers.Dense(d_model),
                ])
                self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
                self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
                self.dropout1 = layers.Dropout(dropout_rate)
                self.dropout2 = layers.Dropout(dropout_rate)
            
            def call(self, inputs, training):
                attn_output = self.att(inputs, inputs)
                attn_output = self.dropout1(attn_output, training=training)
                out1 = self.layernorm1(inputs + attn_output)
                ffn_output = self.ffn(out1)
                ffn_output = self.dropout2(ffn_output, training=training)
                return self.layernorm2(out1 + ffn_output)
        
        inputs = keras.Input(shape=(None, self.input_dim))
        x = layers.Dense(self.d_model)(inputs)
        
        for _ in range(self.num_layers):
            x = TransformerBlock(self.d_model, self.nhead, self.d_model*4, self.dropout)(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(self.forecast_horizon)(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    def fit(self, X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=1):
        """Train the transformer model."""
        if self._backend == 'pytorch':
            return self._fit_pytorch(X, y, epochs, batch_size, validation_split, verbose)
        else:
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size,
                          validation_split=validation_split, verbose=verbose)
            self.is_fitted = True
            return self
    
    def _fit_pytorch(self, X, y, epochs, batch_size, validation_split, verbose):
        """Fit using PyTorch backend."""
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        if self.use_gpu and torch.cuda.is_available():
            X_tensor = X_tensor.cuda()
            y_tensor = y_tensor.cuda()
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
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
    
    def predict(self, X):
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self._backend == 'pytorch':
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
        else:
            return self.model.predict(X)


class TemporalFusionTransformer:
    """
    Temporal Fusion Transformer for multi-horizon forecasting.
    
    State-of-the-art architecture combining attention with gating mechanisms
    for interpretable multi-horizon time series predictions.
    
    Parameters
    ----------
    input_dim : int
        Number of input features
    hidden_dim : int, optional
        Hidden layer dimension (default: 128)
    num_heads : int, optional
        Number of attention heads (default: 4)
    num_layers : int, optional
        Number of layers (default: 2)
    dropout : float, optional
        Dropout rate (default: 0.1)
    forecast_horizon : int, optional
        Number of steps to forecast (default: 10)
    
    Examples
    --------
    >>> tft = TemporalFusionTransformer(input_dim=15, forecast_horizon=30)
    >>> tft.fit(X_train, y_train, epochs=200)
    >>> predictions = tft.predict(X_test)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        forecast_horizon: int = 10,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.forecast_horizon = forecast_horizon
        
        self.model = None
        self.is_fitted = False
        
        logger.info(
            "TemporalFusionTransformer initialized. "
            "This is a simplified implementation. "
            "For full TFT, consider using specialized libraries like PyTorch Forecasting."
        )
        
        # Use the TransformerPredictor as the underlying model
        self._predictor = TransformerPredictor(
            input_dim=input_dim,
            d_model=hidden_dim,
            nhead=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            forecast_horizon=forecast_horizon,
        )
    
    def fit(self, X, y, **kwargs):
        """Train the TFT model."""
        self._predictor.fit(X, y, **kwargs)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make multi-horizon predictions."""
        return self._predictor.predict(X)
    
    def get_attention_weights(self, X):
        """
        Extract attention weights for interpretability.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
        
        Returns
        -------
        np.ndarray
            Attention weights showing which time steps are most important
        
        Notes
        -----
        This is a placeholder for full attention weight extraction.
        Full implementation requires hooks into the attention layers.
        """
        logger.warning(
            "Attention weight extraction not fully implemented. "
            "Use PyTorch Forecasting library for complete TFT implementation."
        )
        return np.ones((X.shape[0], X.shape[1])) / X.shape[1]
