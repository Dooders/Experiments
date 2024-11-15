"""Attack learning and execution module using Deep Q-Learning (DQN).

This module implements a Deep Q-Learning approach for agents to learn optimal attack
policies in a multi-agent environment. It provides both the neural network architecture
and the training/execution logic.

Key Components:
    - AttackQNetwork: Neural network architecture for Q-value approximation
    - AttackModule: Main class handling training, action selection, and attack execution
    - Experience Replay: Stores and samples past experiences for stable learning
    - Target Network: Separate network for computing stable Q-value targets

Technical Details:
    - State Space: N-dimensional vector representing agent's current state
    - Action Space: 5 discrete actions (attack up/down/left/right, defend)
    - Learning Algorithm: Deep Q-Learning with experience replay
    - Exploration: Epsilon-greedy strategy with decay
    - Hardware Acceleration: Automatic GPU usage when available
"""

import logging
from collections import deque
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttackConfig:
    attack_target_update_freq: int = 100  # Frequency of target network updates
    attack_memory_size: int = 10000  # Size of experience replay memory
    attack_learning_rate: float = 0.001  # Learning rate for Q-network updates
    attack_gamma: float = 0.99  # Discount factor for future rewards
    attack_epsilon_start: float = 1.0  # Initial epsilon for exploration
    attack_epsilon_min: float = 0.01  # Minimum epsilon value
    attack_epsilon_decay: float = 0.995  # Epsilon decay rate
    attack_dqn_hidden_size: int = 64  # Size of hidden layers in DQN
    attack_batch_size: int = 32  # Batch size for training
    attack_tau: float = 0.005  # Soft update parameter for target network
    attack_base_cost: float = -0.2  # Base cost for any attack action
    attack_success_reward: float = 1.0  # Reward for successful attack
    attack_failure_penalty: float = -0.3  # Penalty for failed attack
    attack_defense_threshold: float = 0.3  # Health threshold for defense boost
    attack_defense_boost: float = (
        2.0  # Multiplier for defense action when health is low
    )


DEFAULT_ATTACK_CONFIG = AttackConfig()


class AttackActionSpace:
    """Define available attack actions."""

    ATTACK_RIGHT: int = 0
    ATTACK_LEFT: int = 1
    ATTACK_UP: int = 2
    ATTACK_DOWN: int = 3
    DEFEND: int = 4


class AttackQNetwork(nn.Module):
    """Neural network architecture for Q-value approximation in attack learning.

    This network maps state observations to Q-values for each possible attack action.
    Uses an enhanced architecture with batch normalization and dropout for stable learning.

    Architecture:
        Input Layer: state_dimension neurons
        Hidden Layer 1: hidden_size neurons with ReLU activation
        Hidden Layer 2: hidden_size neurons with ReLU activation
        Output Layer: 5 neurons (four attack directions + defend)
    """

    def __init__(self, input_dim: int, hidden_size: int = 64) -> None:
        super(AttackQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 5),  # 5 actions: 4 attack directions + defend
        )

        # Initialize weights using Xavier/Glorot initialization
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


class AttackModule:
    """Attack learning and execution module using enhanced Deep Q-Learning.

    This module implements Double Q-Learning with experience replay and adaptive
    defense mechanisms to learn optimal attack policies in a multi-agent environment.

    Key Features:
        - Double Q-Learning: Reduces overestimation bias
        - Adaptive Defense: Increases defense probability when health is low
        - Experience Replay: Stores transitions for stable batch learning
        - Target Network: Separate network for stable Q-value targets
    """

    def __init__(
        self,
        config: AttackConfig = DEFAULT_ATTACK_CONFIG,
        device: torch.device = DEVICE,
    ) -> None:
        self.device = device
        self.config = config
        self._setup_networks(config)
        self._setup_training(config)
        self._setup_action_space()

        # Add tracking for metrics
        self.losses = []
        self.episode_rewards = []

    def _setup_networks(self, config: AttackConfig) -> None:
        """Initialize Q-networks and optimizer."""
        self.q_network = AttackQNetwork(input_dim=6).to(
            self.device
        )  # Increased state dim
        self.target_network = AttackQNetwork(input_dim=6).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=config.attack_learning_rate
        )
        self.criterion = nn.SmoothL1Loss()

    def _setup_training(self, config: AttackConfig) -> None:
        """Initialize training parameters."""
        self.memory = deque(maxlen=config.attack_memory_size)
        self.gamma = config.attack_gamma
        self.epsilon = config.attack_epsilon_start
        self.epsilon_min = config.attack_epsilon_min
        self.epsilon_decay = config.attack_epsilon_decay
        self.tau = config.attack_tau
        self.steps = 0

    def _setup_action_space(self) -> None:
        """Initialize action space mapping."""
        # Define action space constants as class attributes
        self.ATTACK_RIGHT = 0
        self.ATTACK_LEFT = 1
        self.ATTACK_UP = 2
        self.ATTACK_DOWN = 3
        self.DEFEND = 4

        # Map actions to direction vectors
        self.action_space: Dict[int, Tuple[int, int]] = {
            self.ATTACK_RIGHT: (1, 0),
            self.ATTACK_LEFT: (-1, 0),
            self.ATTACK_UP: (0, 1),
            self.ATTACK_DOWN: (0, -1),
            self.DEFEND: (0, 0),  # No movement for defend action
        }

    def select_action(self, state: torch.Tensor, health_ratio: float) -> int:
        """Select action using epsilon-greedy with defense boost when health is low."""
        if torch.rand(1).item() > self.epsilon:
            with torch.no_grad():
                q_values = self.q_network(state)

                # Boost defense action probability when health is low
                if health_ratio < self.config.attack_defense_threshold:
                    q_values[
                        AttackActionSpace.DEFEND
                    ] *= self.config.attack_defense_boost

                return q_values.argmax().item()
        return torch.randint(0, len(self.action_space), (1,)).item()

    def store_experience(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Store experience in replay memory.

        Args:
            state: Current state tensor
            action: Action taken
            reward: Reward received
            next_state: Resulting state tensor
            done: Whether the episode ended
        """
        self.memory.append((state, action, reward, next_state, done))
        self.episode_rewards.append(reward)

    def train(self, batch: list) -> Optional[float]:
        """Train the network using a batch of experiences.

        Args:
            batch: List of (state, action, reward, next_state, done) tuples

        Returns:
            Optional[float]: Loss value if training occurred, None otherwise
        """
        if len(batch) < self.config.attack_batch_size:
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
            # Get actions from main network
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            # Get Q-values from target network for those actions
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            # Compute target Q values
            target_q_values = (
                rewards.unsqueeze(1)
                + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
            )

        # Compute loss and update network
        loss = self.criterion(current_q_values, target_q_values)

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        self._soft_update_target_network()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Store loss for tracking
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


def attack_action(agent: "BaseAgent") -> None:
    """Execute attack action using the AttackModule.

    This function handles the complete attack action workflow for an agent, including:
    - State evaluation and action selection
    - Target identification and selection
    - Damage calculation and application
    - Experience storage for learning

    The function implements several key mechanics:
    - Defense stance when DEFEND action is selected
    - Reduced damage when agent has low resources
    - Closest target selection within attack range
    - Experience replay storage for reinforcement learning

    Args:
        agent (BaseAgent): The agent executing the attack action

    Returns:
        None

    Side Effects:
        - Updates agent's defensive stance
        - Modifies target agent's health
        - Stores experience in agent's attack module
        - Logs attack outcomes
    """
    # Get current state and health ratio
    state = agent.get_state()
    health_ratio = agent.current_health / agent.max_health

    # Select attack action
    action = agent.attack_module.select_action(
        state.to_tensor(agent.attack_module.device), health_ratio
    )

    # Handle defense action
    if action == agent.attack_module.DEFEND:
        agent.is_defending = True
        logger.debug(f"Agent {id(agent)} took defensive stance")
        return

    # Calculate attack target position
    target_pos = agent.calculate_attack_position(action)

    # Find potential targets
    targets = [
        other
        for other in agent.environment.agents
        if other != agent
        and other.alive
        and np.sqrt(((np.array(other.position) - np.array(target_pos)) ** 2).sum())
        < agent.config.attack_range
    ]
    if not targets:
        logger.debug(f"Agent {id(agent)} attack found no targets")
        return

    # Select closest target
    target = min(
        targets,
        key=lambda t: np.sqrt(
            ((np.array(t.position) - np.array(target_pos)) ** 2).sum()
        ),
    )

    # Calculate base damage
    base_damage = agent.config.attack_base_damage
    if agent.resource_level < agent.config.min_reproduction_resources:
        base_damage *= 0.7  # Reduced damage when low on resources

    # Execute attack
    damage_dealt = target.handle_combat(agent, base_damage)

    # Calculate reward and store experience
    reward = agent.calculate_attack_reward(target, damage_dealt, action)

    # Store experience
    next_state = agent.get_state()
    agent.attack_module.store_experience(
        state.to_tensor(agent.attack_module.device),
        action,
        reward,
        next_state.to_tensor(agent.attack_module.device),
        not target.alive,
    )

    logger.info(
        f"Agent {id(agent)} attacked Agent {id(target)} for {damage_dealt:.1f} damage. "
        f"Target health: {target.current_health:.1f}/{target.max_health}"
    )
