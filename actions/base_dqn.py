"""Base DQN module providing common functionality for learning-based actions."""

import logging
from collections import deque
from typing import Any, Deque, Optional

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseDQNConfig:
    """Base configuration for DQN modules."""

    target_update_freq: int = 100
    memory_size: int = 10000
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    dqn_hidden_size: int = 64
    batch_size: int = 32
    tau: float = 0.005


class BaseQNetwork(nn.Module):
    """Base neural network architecture for Q-value approximation."""

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
            result = self.network(x)
            return result.squeeze(0)
        return self.network(x)


class BaseDQNModule:
    """Base class for DQN-based learning modules."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: BaseDQNConfig,
        device: torch.device = DEVICE,
    ) -> None:
        self.device = device
        self.config = config
        self._setup_networks(input_dim, output_dim, config)
        self._setup_training(config)
        self.losses = []
        self.episode_rewards = []

    def _setup_networks(
        self, input_dim: int, output_dim: int, config: BaseDQNConfig
    ) -> None:
        """Initialize Q-networks and optimizer."""
        self.q_network = BaseQNetwork(input_dim, output_dim, config.dqn_hidden_size).to(
            self.device
        )
        self.target_network = BaseQNetwork(
            input_dim, output_dim, config.dqn_hidden_size
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=config.learning_rate
        )
        self.criterion = nn.SmoothL1Loss()

    def _setup_training(self, config: BaseDQNConfig) -> None:
        """Initialize training parameters."""
        self.memory: Deque = deque(maxlen=config.memory_size)
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.tau = config.tau
        self.steps = 0

    def store_experience(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
        self.episode_rewards.append(reward)

    def train(self, batch: list) -> Optional[float]:
        """Train the network using a batch of experiences."""
        if len(batch) < self.config.batch_size:
            return None

        # Unpack batch
        states = torch.stack([x[0] for x in batch])
        actions = torch.tensor([x[1] for x in batch], device=self.device).unsqueeze(1)
        rewards = torch.tensor([x[2] for x in batch], device=self.device)
        next_states = torch.stack([x[3] for x in batch])
        dones = torch.tensor(
            [x[4] for x in batch], device=self.device, dtype=torch.float
        )

        # Get current Q values
        current_q_values = self.q_network(states).gather(1, actions)

        # Compute target Q values using Double Q-Learning
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = (
                rewards.unsqueeze(1)
                + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
            )

        # Compute loss and update network
        loss = self.criterion(current_q_values, target_q_values)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        self._soft_update_target_network()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        loss_value = loss.item()
        self.losses.append(loss_value)
        return loss_value

    def _soft_update_target_network(self) -> None:
        """Soft update target network weights using tau parameter."""
        for target_param, local_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def get_state_dict(self) -> dict[str, Any]:
        """Get the current state of the DQN module.

        Returns:
            dict: State dictionary containing network weights and training parameters
        """
        return {
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'losses': self.losses,
            'episode_rewards': self.episode_rewards,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load a state dictionary into the DQN module.

        Args:
            state_dict (dict): State dictionary containing network weights and training parameters
        """
        self.q_network.load_state_dict(state_dict['q_network_state'])
        self.target_network.load_state_dict(state_dict['target_network_state'])
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.epsilon = state_dict['epsilon']
        self.steps = state_dict['steps']
        self.losses = state_dict['losses']
        self.episode_rewards = state_dict['episode_rewards']
