"""
Convolutional Neural Networks for Chart Pattern Recognition.

This module provides CNN models for detecting technical chart patterns
in financial data by treating price charts as images.
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ChartPatternCNN:
    """
    CNN-based chart pattern recognizer.
    
    Detects classic technical analysis patterns like head and shoulders,
    double tops/bottoms, triangles, etc., by converting OHLCV data into
    candlestick chart images.
    
    Parameters
    ----------
    image_size : tuple, optional
        Size of chart images (height, width) (default: (64, 64))
    num_patterns : int, optional
        Number of pattern classes to detect (default: 10)
    pretrained : bool, optional
        Whether to use pretrained weights (default: False)
    
    Attributes
    ----------
    patterns : list
        List of pattern names the model can detect
    
    Examples
    --------
    >>> cnn = ChartPatternCNN(image_size=(128, 128))
    >>> cnn.fit(X_images, y_labels, epochs=100)
    >>> pattern_probs = cnn.predict_proba(new_charts)
    """
    
    PATTERN_NAMES = [
        'head_and_shoulders',
        'inverse_head_and_shoulders',
        'double_top',
        'double_bottom',
        'ascending_triangle',
        'descending_triangle',
        'symmetrical_triangle',
        'bullish_flag',
        'bearish_flag',
        'wedge',
    ]
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (64, 64),
        num_patterns: int = 10,
        pretrained: bool = False,
    ):
        self.image_size = image_size
        self.num_patterns = num_patterns
        self.pretrained = pretrained
        self.patterns = self.PATTERN_NAMES[:num_patterns]
        
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
        """Build CNN model using PyTorch."""
        import torch
        import torch.nn as nn
        
        class ChartCNN(nn.Module):
            def __init__(self, image_size, num_classes):
                super().__init__()
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                
                # Calculate flattened size
                h, w = image_size
                flatten_size = 128 * (h // 8) * (w // 8)
                
                self.fc_layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(flatten_size, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes),
                )
            
            def forward(self, x):
                x = self.conv_layers(x)
                x = self.fc_layers(x)
                return x
        
        self.model = ChartCNN(self.image_size, self.num_patterns)
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def _build_tensorflow_model(self):
        """Build CNN model using TensorFlow."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential([
            layers.Conv2D(32, 3, activation='relu', padding='same',
                         input_shape=(*self.image_size, 3)),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_patterns, activation='softmax'),
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
    
    def ohlcv_to_image(
        self,
        ohlcv: np.ndarray,
        style: str = 'candlestick'
    ) -> np.ndarray:
        """
        Convert OHLCV data to chart image.
        
        Parameters
        ----------
        ohlcv : np.ndarray
            OHLCV data of shape (n_periods, 5) containing
            open, high, low, close, volume
        style : str, optional
            Chart style: 'candlestick', 'line', or 'bar' (default: 'candlestick')
        
        Returns
        -------
        np.ndarray
            Image array of shape (height, width, 3) representing the chart
        
        Notes
        -----
        This is a simplified implementation. For production use, consider
        using matplotlib or plotly to generate high-quality chart images.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            import io
            from PIL import Image
        except ImportError:
            logger.warning("matplotlib and PIL required for chart generation")
            # Return a placeholder image
            return np.random.rand(*self.image_size, 3).astype(np.float32)
        
        fig, ax = plt.subplots(figsize=(self.image_size[1]/10, self.image_size[0]/10))
        
        if style == 'candlestick':
            for i, (o, h, l, c, v) in enumerate(ohlcv):
                color = 'green' if c >= o else 'red'
                ax.plot([i, i], [l, h], color='black', linewidth=0.5)
                width = 0.6
                height = abs(c - o)
                bottom = min(o, c)
                rect = Rectangle((i - width/2, bottom), width, height,
                                facecolor=color, edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
        elif style == 'line':
            close_prices = ohlcv[:, 3]
            ax.plot(close_prices, linewidth=2)
        else:  # bar
            ax.bar(range(len(ohlcv)), ohlcv[:, 3])
        
        ax.set_xlim(-0.5, len(ohlcv) - 0.5)
        ax.set_ylim(ohlcv[:, [1, 2]].min() * 0.995, ohlcv[:, [1, 2]].max() * 1.005)
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        # Convert to image array
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        img = img.resize(self.image_size)
        img_array = np.array(img)[:, :, :3] / 255.0  # Normalize to [0, 1]
        plt.close()
        
        return img_array.astype(np.float32)
    
    def fit(self, X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=1):
        """
        Train the CNN on chart images.
        
        Parameters
        ----------
        X : np.ndarray
            Chart images of shape (n_samples, height, width, 3)
        y : np.ndarray
            Pattern labels of shape (n_samples, num_patterns)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        validation_split : float
            Validation split ratio
        verbose : int
            Verbosity level
        """
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
        
        # Reshape for PyTorch (batch, channels, height, width)
        X = X.transpose(0, 3, 1, 2)
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(np.argmax(y, axis=1))
        
        if torch.cuda.is_available():
            X_tensor = X_tensor.cuda()
            y_tensor = y_tensor.cuda()
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            if verbose and (epoch + 1) % 10 == 0:
                accuracy = 100 * correct / total
                avg_loss = total_loss / len(loader)
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, "
                          f"Accuracy: {accuracy:.2f}%")
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict pattern classes for chart images."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X):
        """
        Predict pattern probabilities for chart images.
        
        Parameters
        ----------
        X : np.ndarray
            Chart images of shape (n_samples, height, width, 3)
        
        Returns
        -------
        np.ndarray
            Pattern probabilities of shape (n_samples, num_patterns)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self._backend == 'pytorch':
            import torch
            import torch.nn.functional as F
            
            X = X.transpose(0, 3, 1, 2)  # Convert to PyTorch format
            self.model.eval()
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                if torch.cuda.is_available():
                    X_tensor = X_tensor.cuda()
                outputs = self.model(X_tensor)
                proba = F.softmax(outputs, dim=1)
                if torch.cuda.is_available():
                    proba = proba.cpu()
                return proba.numpy()
        else:
            return self.model.predict(X)
    
    def detect_patterns(self, ohlcv_data: np.ndarray, threshold: float = 0.7) -> List[str]:
        """
        Detect patterns in OHLCV data.
        
        Parameters
        ----------
        ohlcv_data : np.ndarray
            OHLCV data of shape (n_periods, 5)
        threshold : float, optional
            Confidence threshold for pattern detection (default: 0.7)
        
        Returns
        -------
        list
            List of detected pattern names with confidence above threshold
        """
        image = self.ohlcv_to_image(ohlcv_data)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        proba = self.predict_proba(image)[0]
        
        detected = []
        for i, prob in enumerate(proba):
            if prob >= threshold:
                detected.append((self.patterns[i], prob))
        
        return detected
