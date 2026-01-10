"""
Deep Q-Network (DQN) for Optimal Trade Execution.

This module implements DQN agents for learning optimal trading policies
through Q-learning with experience replay and target networks.
"""

import numpy as np
from collections import deque
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class DQNTrader:
    """
    Deep Q-Network trader for optimal trade execution.
    
    Implements DQN with experience replay and target network for stability.
    Actions include buy, sell, hold with various position sizes.
    
    Parameters
    ----------
    state_dim : int
        Dimension of state space (market features)
    action_dim : int
        Number of discrete actions (default: 5 for strong sell, sell, hold, buy, strong buy)
    hidden_dim : int, optional
        Hidden layer dimension (default: 128)
    learning_rate : float, optional
        Learning rate (default: 0.001)
    gamma : float, optional
        Discount factor (default: 0.99)
    epsilon : float, optional
        Initial exploration rate (default: 1.0)
    epsilon_decay : float, optional
        Epsilon decay rate (default: 0.995)
    epsilon_min : float, optional
        Minimum epsilon (default: 0.01)
    memory_size : int, optional
        Experience replay buffer size (default: 10000)
    batch_size : int, optional
        Training batch size (default: 32)
    target_update_freq : int, optional
        Frequency of target network updates (default: 10)
    
    Examples
    --------
    >>> trader = DQNTrader(state_dim=20, action_dim=5)
    >>> for episode in range(1000):
    ...     state = env.reset()
    ...     while not done:
    ...         action = trader.act(state)
    ...         next_state, reward, done = env.step(action)
    ...         trader.remember(state, action, reward, next_state, done)
    ...         trader.replay()
    ...         state = next_state
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 5,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 10,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.memory = deque(maxlen=memory_size)
        self.train_step = 0
        
        self.q_network = None
        self.target_network = None
        self._backend = None
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the RL backend."""
        try:
            import torch
            self._backend = 'pytorch'
            self._build_pytorch_networks()
        except ImportError:
            try:
                import tensorflow as tf
                self._backend = 'tensorflow'
                self._build_tensorflow_networks()
            except ImportError:
                raise ImportError(
                    "RL backend not available. Install PyTorch or TensorFlow."
                )
    
    def _build_pytorch_networks(self):
        """Build Q-networks using PyTorch."""
        import torch
        import torch.nn as nn
        
        class QNetwork(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim)
                )
            
            def forward(self, state):
                return self.network(state)
        
        self.q_network = QNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.target_network = QNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        if torch.cuda.is_available():
            self.q_network = self.q_network.cuda()
            self.target_network = self.target_network.cuda()
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
    
    def _build_tensorflow_networks(self):
        """Build Q-networks using TensorFlow."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        def build_network():
            model = keras.Sequential([
                layers.Dense(self.hidden_dim, activation='relu', input_shape=(self.state_dim,)),
                layers.Dense(self.hidden_dim, activation='relu'),
                layers.Dense(self.action_dim)
            ])
            return model
        
        self.q_network = build_network()
        self.target_network = build_network()
        self.target_network.set_weights(self.q_network.get_weights())
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Select action using epsilon-greedy policy.
        
        Parameters
        ----------
        state : np.ndarray
            Current state
        training : bool
            Whether in training mode (uses epsilon-greedy) or evaluation (greedy)
        
        Returns
        -------
        int
            Selected action index
        """
        if training and np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        
        if self._backend == 'pytorch':
            import torch
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                if torch.cuda.is_available():
                    state_tensor = state_tensor.cuda()
                q_values = self.q_network(state_tensor)
                if torch.cuda.is_available():
                    q_values = q_values.cpu()
                return q_values.numpy().argmax()
        else:
            state_array = np.expand_dims(state, axis=0)
            q_values = self.q_network.predict(state_array, verbose=0)
            return np.argmax(q_values[0])
    
    def replay(self):
        """
        Train the Q-network using experience replay.
        
        Returns
        -------
        float or None
            Average loss for the batch, or None if not enough samples
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample random batch
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        if self._backend == 'pytorch':
            loss = self._replay_pytorch(states, actions, rewards, next_states, dones)
        else:
            loss = self._replay_tensorflow(states, actions, rewards, next_states, dones)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self._update_target_network()
        
        return loss
    
    def _replay_pytorch(self, states, actions, rewards, next_states, dones):
        """Replay using PyTorch."""
        import torch
        import torch.nn.functional as F
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        if torch.cuda.is_available():
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            next_states = next_states.cuda()
            dones = dones.cuda()
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _replay_tensorflow(self, states, actions, rewards, next_states, dones):
        """Replay using TensorFlow."""
        import tensorflow as tf
        
        # Get next Q values from target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        max_next_q = np.max(next_q_values, axis=1)
        
        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q
        
        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            one_hot_actions = tf.one_hot(actions, self.action_dim)
            q_action = tf.reduce_sum(q_values * one_hot_actions, axis=1)
            loss = tf.reduce_mean(tf.square(target_q_values - q_action))
        
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))
        
        return loss.numpy()
    
    def _update_target_network(self):
        """Update target network with Q-network weights."""
        if self._backend == 'pytorch':
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            self.target_network.set_weights(self.q_network.get_weights())
    
    def save(self, filepath: str):
        """Save the trained model."""
        if self._backend == 'pytorch':
            import torch
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'epsilon': self.epsilon,
                'train_step': self.train_step,
            }, filepath)
        else:
            self.q_network.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained model."""
        if self._backend == 'pytorch':
            import torch
            checkpoint = torch.load(filepath)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.epsilon = checkpoint['epsilon']
            self.train_step = checkpoint['train_step']
        else:
            import tensorflow as tf
            self.q_network = tf.keras.models.load_model(filepath)
            self.target_network.set_weights(self.q_network.get_weights())
        logger.info(f"Model loaded from {filepath}")
