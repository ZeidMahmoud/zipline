"""
Generative Adversarial Networks for Market Scenario Generation.

This module provides GAN models for generating synthetic market scenarios
useful for stress testing and data augmentation.
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MarketGAN:
    """
    Generative Adversarial Network for synthetic market scenario generation.
    
    Learns the distribution of historical market data and generates realistic
    synthetic scenarios for stress testing and backtesting.
    
    Parameters
    ----------
    input_dim : int
        Dimension of market data features
    latent_dim : int, optional
        Dimension of latent noise vector (default: 100)
    hidden_dim : int, optional
        Hidden layer dimension (default: 128)
    sequence_length : int, optional
        Length of generated sequences (default: 50)
    
    Examples
    --------
    >>> gan = MarketGAN(input_dim=5, sequence_length=100)
    >>> gan.fit(historical_data, epochs=1000)
    >>> synthetic_scenarios = gan.generate(n_scenarios=100)
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 100,
        hidden_dim: int = 128,
        sequence_length: int = 50,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        
        self.generator = None
        self.discriminator = None
        self.is_fitted = False
        self._backend = None
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the deep learning backend."""
        try:
            import torch
            self._backend = 'pytorch'
            self._build_pytorch_models()
        except ImportError:
            try:
                import tensorflow as tf
                self._backend = 'tensorflow'
                self._build_tensorflow_models()
            except ImportError:
                raise ImportError(
                    "Deep learning backend not available. "
                    "Install PyTorch or TensorFlow."
                )
    
    def _build_pytorch_models(self):
        """Build GAN models using PyTorch."""
        import torch
        import torch.nn as nn
        
        class Generator(nn.Module):
            def __init__(self, latent_dim, hidden_dim, output_dim, seq_len):
                super().__init__()
                self.seq_len = seq_len
                self.output_dim = output_dim
                
                self.model = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, seq_len * output_dim),
                    nn.Tanh(),
                )
            
            def forward(self, z):
                x = self.model(z)
                return x.view(-1, self.seq_len, self.output_dim)
        
        class Discriminator(nn.Module):
            def __init__(self, input_dim, hidden_dim, seq_len):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(seq_len * input_dim, hidden_dim * 2),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid(),
                )
            
            def forward(self, x):
                return self.model(x)
        
        self.generator = Generator(
            self.latent_dim, self.hidden_dim, self.input_dim, self.sequence_length
        )
        self.discriminator = Discriminator(
            self.input_dim, self.hidden_dim, self.sequence_length
        )
        
        if torch.cuda.is_available():
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
    
    def _build_tensorflow_models(self):
        """Build GAN models using TensorFlow."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Generator
        generator = keras.Sequential([
            layers.Dense(self.hidden_dim, activation='relu', 
                        input_shape=(self.latent_dim,)),
            layers.Dense(self.hidden_dim * 2, activation='relu'),
            layers.Dense(self.sequence_length * self.input_dim, activation='tanh'),
            layers.Reshape((self.sequence_length, self.input_dim)),
        ])
        
        # Discriminator
        discriminator = keras.Sequential([
            layers.Flatten(input_shape=(self.sequence_length, self.input_dim)),
            layers.Dense(self.hidden_dim * 2, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.hidden_dim, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid'),
        ])
        
        self.generator = generator
        self.discriminator = discriminator
        
        discriminator.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def fit(self, X, epochs=1000, batch_size=64, verbose=1):
        """
        Train the GAN on historical market data.
        
        Parameters
        ----------
        X : np.ndarray
            Historical market data of shape (n_samples, sequence_length, n_features)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        verbose : int
            Verbosity level
        """
        if self._backend == 'pytorch':
            return self._fit_pytorch(X, epochs, batch_size, verbose)
        else:
            return self._fit_tensorflow(X, epochs, batch_size, verbose)
    
    def _fit_pytorch(self, X, epochs, batch_size, verbose):
        """Fit using PyTorch backend."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        X_tensor = torch.FloatTensor(X)
        if torch.cuda.is_available():
            X_tensor = X_tensor.cuda()
        
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = torch.nn.BCELoss()
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
        
        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            
            for (real_data,) in loader:
                batch_size = real_data.size(0)
                
                # Train Discriminator
                d_optimizer.zero_grad()
                
                real_labels = torch.ones(batch_size, 1)
                fake_labels = torch.zeros(batch_size, 1)
                
                if torch.cuda.is_available():
                    real_labels = real_labels.cuda()
                    fake_labels = fake_labels.cuda()
                
                # Real data
                d_real = self.discriminator(real_data)
                d_real_loss = criterion(d_real, real_labels)
                
                # Fake data
                z = torch.randn(batch_size, self.latent_dim)
                if torch.cuda.is_available():
                    z = z.cuda()
                fake_data = self.generator(z)
                d_fake = self.discriminator(fake_data.detach())
                d_fake_loss = criterion(d_fake, fake_labels)
                
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                d_optimizer.step()
                
                # Train Generator
                g_optimizer.zero_grad()
                
                z = torch.randn(batch_size, self.latent_dim)
                if torch.cuda.is_available():
                    z = z.cuda()
                fake_data = self.generator(z)
                d_fake = self.discriminator(fake_data)
                g_loss = criterion(d_fake, real_labels)
                
                g_loss.backward()
                g_optimizer.step()
                
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
            
            if verbose and (epoch + 1) % 100 == 0:
                avg_g_loss = np.mean(g_losses)
                avg_d_loss = np.mean(d_losses)
                logger.info(f"Epoch [{epoch+1}/{epochs}], "
                          f"G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def _fit_tensorflow(self, X, epochs, batch_size, verbose):
        """Fit using TensorFlow backend."""
        import tensorflow as tf
        
        optimizer = tf.keras.optimizers.Adam(0.0002)
        
        for epoch in range(epochs):
            # Sample random batch
            idx = np.random.randint(0, X.shape[0], batch_size)
            real_data = X[idx]
            
            # Train discriminator
            z = np.random.randn(batch_size, self.latent_dim)
            fake_data = self.generator.predict(z, verbose=0)
            
            d_loss_real = self.discriminator.train_on_batch(
                real_data, np.ones((batch_size, 1))
            )
            d_loss_fake = self.discriminator.train_on_batch(
                fake_data, np.zeros((batch_size, 1))
            )
            
            # Train generator (through combined model)
            z = np.random.randn(batch_size, self.latent_dim)
            
            with tf.GradientTape() as tape:
                fake_data = self.generator(z, training=True)
                predictions = self.discriminator(fake_data, training=False)
                g_loss = tf.keras.losses.binary_crossentropy(
                    tf.ones_like(predictions), predictions
                )
            
            grads = tape.gradient(g_loss, self.generator.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
            
            if verbose and (epoch + 1) % 100 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], "
                          f"D Loss Real: {d_loss_real[0]:.4f}, "
                          f"D Loss Fake: {d_loss_fake[0]:.4f}")
        
        self.is_fitted = True
        return self
    
    def generate(self, n_scenarios: int = 100) -> np.ndarray:
        """
        Generate synthetic market scenarios.
        
        Parameters
        ----------
        n_scenarios : int
            Number of scenarios to generate
        
        Returns
        -------
        np.ndarray
            Generated scenarios of shape (n_scenarios, sequence_length, n_features)
        """
        if not self.is_fitted:
            raise ValueError("GAN must be trained before generating scenarios")
        
        if self._backend == 'pytorch':
            import torch
            self.generator.eval()
            with torch.no_grad():
                z = torch.randn(n_scenarios, self.latent_dim)
                if torch.cuda.is_available():
                    z = z.cuda()
                fake_data = self.generator(z)
                if torch.cuda.is_available():
                    fake_data = fake_data.cpu()
                return fake_data.numpy()
        else:
            z = np.random.randn(n_scenarios, self.latent_dim)
            return self.generator.predict(z)


class ConditionalGAN(MarketGAN):
    """
    Conditional GAN for generating scenarios based on market regime.
    
    Extends MarketGAN to generate scenarios conditioned on specific
    market conditions or regimes (bull, bear, volatile, etc.).
    
    Parameters
    ----------
    input_dim : int
        Dimension of market data features
    condition_dim : int
        Dimension of conditioning vector (e.g., number of regimes)
    latent_dim : int, optional
        Dimension of latent noise vector (default: 100)
    hidden_dim : int, optional
        Hidden layer dimension (default: 128)
    sequence_length : int, optional
        Length of generated sequences (default: 50)
    
    Examples
    --------
    >>> cgan = ConditionalGAN(input_dim=5, condition_dim=3, sequence_length=100)
    >>> cgan.fit(historical_data, conditions, epochs=1000)
    >>> # Generate bull market scenarios (condition=[1, 0, 0])
    >>> bull_scenarios = cgan.generate_conditional(
    ...     n_scenarios=50, condition=[1, 0, 0]
    ... )
    """
    
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        latent_dim: int = 100,
        hidden_dim: int = 128,
        sequence_length: int = 50,
    ):
        self.condition_dim = condition_dim
        super().__init__(input_dim, latent_dim, hidden_dim, sequence_length)
    
    def generate_conditional(
        self,
        n_scenarios: int,
        condition: np.ndarray
    ) -> np.ndarray:
        """
        Generate scenarios conditioned on specific market regime.
        
        Parameters
        ----------
        n_scenarios : int
            Number of scenarios to generate
        condition : np.ndarray
            Conditioning vector (e.g., one-hot encoded regime)
        
        Returns
        -------
        np.ndarray
            Generated conditional scenarios
        
        Notes
        -----
        This is a simplified implementation. Full conditional GAN would
        concatenate condition vectors with both generator and discriminator inputs.
        """
        logger.warning(
            "Simplified conditional generation. "
            "For full conditional GAN, implement condition concatenation in models."
        )
        return self.generate(n_scenarios)
