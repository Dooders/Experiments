"""Movement learning and execution module using Deep Q-Learning (DQN).

This module implements a Deep Q-Learning approach for agents to learn optimal movement
policies in a 2D environment. It provides both the neural network architecture and
the training/execution logic.

Key Components:
    - MoveQNetwork: Neural network architecture for Q-value approximation
    - MoveModule: Main class handling training, action selection, and movement execution
    - Experience Replay: Stores and samples past experiences for stable learning
    - Target Network: Separate network for computing stable Q-value targets

Technical Details:
    - State Space: 4-dimensional vector representing agent's current state
    - Action Space: 4 discrete actions (right, left, up, down)
    - Learning Algorithm: Deep Q-Learning with experience replay
    - Exploration: Epsilon-greedy strategy with decay
    - Hardware Acceleration: Automatic GPU usage when available

Example Usage:
    ```python
    # Initialize configuration
    config = MovementConfig(
        learning_rate=0.001,
        memory_size=10000,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        target_update_freq=100
    )
    
    # Create and use module
    move_module = MoveModule(config)
    next_position = move_module.get_movement(agent, current_state)
    move_module.store_experience(state, action, reward, next_state)
    loss = move_module.train(batch)
    ```

Dependencies:
    - torch: Deep learning framework for neural network implementation
    - random: For exploration strategy implementation
    - collections.deque: For experience replay buffer management
"""

import logging
import random
from collections import deque
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import ModelState only during type checking
if TYPE_CHECKING:
    from resource import Resource

    from agents.base_agent import BaseAgent
    from environment import Environment
    from state import ModelState

logger = logging.getLogger(__name__)

# Move device check outside class for single evaluation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MoveConfig:
    move_target_update_freq: int = 100
    move_memory_size: int = 10000
    move_learning_rate: float = 0.001
    move_gamma: float = 0.99
    move_epsilon_start: float = 1.0
    move_epsilon_min: float = 0.01
    move_epsilon_decay: float = 0.995
    move_dqn_hidden_size: int = 64
    move_batch_size: int = 32
    move_reward_history_size: int = 100
    move_epsilon_adapt_threshold: float = 0.1
    move_epsilon_adapt_factor: float = 1.5
    move_min_reward_samples: int = 10
    move_tau: float = 0.005  # Soft update parameter


DEFAULT_MOVE_CONFIG = MoveConfig()


class MoveActionSpace:
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3


class MoveQNetwork(nn.Module):
    """Neural network architecture for Q-value approximation in movement learning.

    This network maps state observations to Q-values for each possible movement action.
    Uses an enhanced architecture with batch normalization, dropout, and proper weight
    initialization for stable and efficient learning.

    Architecture:
        Input Layer: state_dimension neurons
        Hidden Layer 1: hidden_size neurons with:
            - ReLU activation
            - 10% Dropout for regularization
        Hidden Layer 2: hidden_size neurons with:
            - ReLU activation
            - 10% Dropout for regularization
        Output Layer: 4 neurons (one for each movement action)

    Optimization Features:
        - Xavier/Glorot initialization for better gradient flow
        - Dropout layers for preventing overfitting
        - Layer normalization for training stability

    Args:
        input_dim (int): Dimension of the input state vector
        hidden_size (int, optional): Number of neurons in hidden layers. Defaults to 64.

    Forward Pass:
        Input: State tensor of shape (batch_size, input_dim) or (input_dim,)
        Output: Q-values tensor of shape (batch_size, 4) or (4,)
    """

    def __init__(self, input_dim, hidden_size=64):
        super(MoveQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),  # Layer norm instead of batch norm
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),  # Layer norm instead of batch norm
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 4),  # 4 actions: right, left, up, down
        )

        # Initialize weights using Xavier/Glorot initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Handle both batched and unbatched inputs
        if x.dim() == 1:
            x = x.unsqueeze(0)
            result = self.network(x)
            return result.squeeze(0)
        return self.network(x)


class MoveModule:
    """Movement learning and execution module using enhanced Deep Q-Learning.

    This module implements Double Q-Learning with experience replay, soft target
    network updates, and adaptive exploration to learn optimal movement policies
    in a 2D environment.

    Key Enhancements:
        - Double Q-Learning: Reduces overestimation bias by decoupling action selection
          and evaluation using two networks
        - Soft Target Updates: Gradually updates target network for stability
        - Adaptive Exploration: Dynamically adjusts exploration rate based on learning progress
        - Temperature-based Action Selection: Uses softmax with adaptive temperature for
          smarter exploration
        - Huber Loss: More robust to outliers than MSE
        - Gradient Clipping: Prevents exploding gradients

    Features:
        - Experience Replay: Stores transitions for stable batch learning
        - Target Network: Separate network for stable Q-value targets
        - Modular Setup: Separated network, training, and action space initialization
        - Automatic Device Selection: Uses GPU if available

    Args:
        config: Configuration object containing:
            - learning_rate (float): Learning rate for Adam optimizer
            - memory_size (int): Maximum size of experience replay buffer
            - gamma (float): Discount factor for future rewards [0,1]
            - epsilon_start (float): Initial exploration rate [0,1]
            - epsilon_min (float): Minimum exploration rate [0,1]
            - epsilon_decay (float): Base rate for epsilon decay
            - target_update_freq (int): Steps between target network updates

    Attributes:
        device (torch.device): CPU or CUDA device for computations
        q_network (MoveQNetwork): Main Q-network for action selection
        target_network (MoveQNetwork): Target network for stable learning
        memory (deque): Experience replay buffer
        epsilon (float): Current exploration rate
        reward_history (deque): Recent reward history for adaptive exploration
        epsilon_adapt_threshold (float): Minimum improvement threshold for adaptation
        epsilon_adapt_factor (float): Factor to adjust decay rate
    """

    def __init__(
        self, config: MoveConfig = DEFAULT_MOVE_CONFIG, device: torch.device = DEVICE
    ) -> None:
        self.device: torch.device = device
        self._setup_networks(config)
        self._setup_training(config)
        self._setup_action_space()

    def _setup_networks(self, config):
        """Initialize Q-networks and optimizer."""
        self.q_network = MoveQNetwork(input_dim=4).to(self.device)
        self.target_network = MoveQNetwork(input_dim=4).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=config.learning_rate
        )
        self.criterion = nn.SmoothL1Loss()

    def _setup_training(self, config):
        """Initialize training parameters with adaptive exploration."""
        self.memory: Deque = deque(maxlen=config.move_memory_size)
        self.gamma = config.move_gamma
        self.epsilon = config.move_epsilon_start
        self.epsilon_min = config.move_epsilon_min
        self.epsilon_decay = config.move_epsilon_decay
        self.target_update_freq = config.move_target_update_freq
        self.tau = config.move_tau
        self.steps = 0

        # Adaptive exploration parameters from config
        self.reward_history = deque(maxlen=config.move_reward_history_size)
        self.epsilon_adapt_threshold = config.move_epsilon_adapt_threshold
        self.epsilon_adapt_factor = config.move_epsilon_adapt_factor
        self.min_reward_samples = config.move_min_reward_samples

    def _setup_action_space(self) -> None:
        """Initialize action space mapping."""
        self.action_space: Dict[int, Tuple[int, int]] = {
            MoveActionSpace.RIGHT: (1, 0),
            MoveActionSpace.LEFT: (-1, 0),
            MoveActionSpace.UP: (0, 1),
            MoveActionSpace.DOWN: (0, -1),
        }

    def _update_epsilon(self, current_reward: float) -> float:
        """Update epsilon using adaptive decay strategy based on learning progress."""
        self.reward_history.append(current_reward)

        # Only adapt if we have enough reward samples for both recent and older averages
        if (
            len(self.reward_history) >= self.min_reward_samples + 10
        ):  # Need at least min_samples + 10
            # Calculate rolling averages
            recent_rewards = list(self.reward_history)[-10:]
            older_rewards = list(self.reward_history)[:-10]

            if recent_rewards and older_rewards:  # Extra safety check
                recent_avg = sum(recent_rewards) / len(recent_rewards)
                older_avg = sum(older_rewards) / len(older_rewards)

                # Calculate improvement
                improvement = recent_avg - older_avg

                # Adjust epsilon decay based on improvement
                if improvement < self.epsilon_adapt_threshold:
                    # Slow down decay if learning has plateaued
                    effective_decay = self.epsilon_decay ** (
                        1.0 / self.epsilon_adapt_factor
                    )
                else:
                    # Use normal decay if improving well
                    effective_decay = self.epsilon_decay
            else:
                effective_decay = self.epsilon_decay
        else:
            # Use default decay until we have enough samples
            effective_decay = self.epsilon_decay

        # Update epsilon with bounds
        self.epsilon = max(
            self.epsilon_min,
            min(
                self.epsilon * effective_decay,
                self.epsilon,  # Ensure we don't increase above current value
            ),
        )

        return self.epsilon

    def train(self, batch) -> Optional[float]:
        """Train Q-network using Double Q-Learning and adaptive exploration."""
        if len(batch) < 2:
            return None

        # States are already tensors from memory, just stack them
        states = torch.stack([state for state, _, _, _, _ in batch])
        actions = torch.LongTensor([[x[1]] for x in batch]).to(self.device)
        rewards = torch.FloatTensor([x[2] for x in batch]).to(self.device)
        next_states = torch.stack([next_state for _, _, _, next_state, _ in batch])
        dones = torch.FloatTensor([x[4] for x in batch]).to(self.device)

        # Get current Q values
        current_q_values = self.q_network(states).gather(1, actions)

        # Implement Double Q-Learning
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

        # Compute loss using Huber Loss
        loss = self.criterion(current_q_values, target_q_values)

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        self._soft_update_target_network()

        # Use mean reward from batch for epsilon adaptation
        mean_reward = rewards.mean().item()
        self._update_epsilon(mean_reward)

        return loss.cpu().item()

    def _soft_update_target_network(self):
        """Soft update target network weights using tau parameter."""
        for target_param, local_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def get_movement(
        self, agent: "BaseAgent", state: torch.Tensor
    ) -> Tuple[float, float]:
        """Determine next movement position using learned policy."""
        # Convert state to tensor if needed and store for later use
        if not isinstance(state, torch.Tensor):
            if hasattr(state, "to_tensor"):
                state = state.to_tensor(self.device)
            else:
                state = torch.FloatTensor(state).to(self.device)

        action = self.select_action(state)
        dx, dy = self.action_space[action]

        # Scale movement by agent's max_movement
        dx *= agent.max_movement
        dy *= agent.max_movement

        # Calculate new position within bounds
        new_x = max(0, min(agent.environment.width, agent.position[0] + dx))
        new_y = max(0, min(agent.environment.height, agent.position[1] + dy))

        # Store state for learning (already a tensor)
        self.last_state = state
        self.last_action = action

        return (new_x, new_y)

    def store_experience(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Store experience in replay memory with efficient tensor conversion."""

        # Convert states to tensors only once when storing
        def ensure_tensor(x) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                return x.to(self.device)
            elif hasattr(x, "to_tensor"):
                return x.to_tensor(self.device)
            elif isinstance(x, np.ndarray):
                return torch.FloatTensor(x).to(self.device)
            else:
                raise TypeError(f"Unexpected state type: {type(x)}")

        # Convert and store experience tuple with tensors
        state_tensor = ensure_tensor(state)
        next_state_tensor = ensure_tensor(next_state)
        self.memory.append((state_tensor, action, reward, next_state_tensor, done))

    def get_state(self) -> "ModelState":
        """Get current state of the move module.

        Returns:
            ModelState: Current state including learning parameters and metrics

        Example:
            >>> state = move_module.get_state()
            >>> print(f"Current epsilon: {state.epsilon}")
        """
        from state import ModelState  # Import locally to avoid circle

        return ModelState.from_move_module(self)

    def select_action(
        self, state_tensor: torch.Tensor, epsilon: Optional[float] = None
    ) -> int:
        """Select action using epsilon-greedy with adaptive temperature.

        Uses a temperature-based exploration strategy where the temperature is derived
        from the current epsilon value. This creates smoother exploration behavior
        compared to uniform random selection.

        Args:
            state_tensor (torch.Tensor): Current state observation
            epsilon (Optional[float]): Override default epsilon value

        Returns:
            int: Selected action index from MoveActionSpace

        Notes:
            - During exploration, uses softmax with temperature for weighted random selection
            - Temperature scales with epsilon for adaptive exploration behavior
            - During exploitation, selects action with highest Q-value
        """
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            # Temperature-based random action selection
            q_values = self.q_network(state_tensor)
            temperature = max(0.1, epsilon)  # Use epsilon as temperature

            # Apply softmax with temperature
            probabilities = torch.softmax(q_values / temperature, dim=0)

            # Convert to numpy for random choice
            action_probs = probabilities.detach().cpu().numpy()
            return np.random.choice(len(action_probs), p=action_probs)

        with torch.no_grad():
            return self.q_network(state_tensor).argmax().item()


def move_action(agent):
    """Execute movement using optimized Deep Q-Learning based policy."""
    # Get state and ensure it's a tensor
    state = _ensure_tensor(agent.get_state(), agent.move_module.device)

    # Get movement and update position
    initial_position = agent.position
    new_position = agent.move_module.get_movement(agent, state)
    agent.position = new_position

    # Calculate reward and store experience
    reward = _calculate_movement_reward(agent, initial_position, new_position)
    _store_and_train(agent, state, reward)

    logger.debug(
        f"Agent {id(agent)} moved from {initial_position} to {new_position}. "
        f"Reward: {reward:.3f}, Epsilon: {agent.move_module.epsilon:.3f}"
    )


def _calculate_movement_reward(
    agent: "BaseAgent",
    initial_position: Tuple[float, float],
    new_position: Tuple[float, float],
) -> float:
    """Calculate reward for movement based on resource proximity."""
    # Base cost for moving
    reward = -0.1

    # Calculate movement distance
    distance_moved = np.sqrt(
        (new_position[0] - initial_position[0]) ** 2
        + (new_position[1] - initial_position[1]) ** 2
    )

    if distance_moved > 0:
        closest_resource = _find_closest_resource(agent.environment, new_position)
        if closest_resource:
            old_distance = _calculate_distance(
                closest_resource.position, initial_position
            )
            new_distance = _calculate_distance(closest_resource.position, new_position)
            reward += 0.3 if new_distance < old_distance else -0.2

    return reward


def _find_closest_resource(
    environment: "Environment", position: Tuple[float, float]
) -> Optional["Resource"]:
    """Find the closest non-depleted resource."""
    active_resources = [r for r in environment.resources if not r.is_depleted()]
    if not active_resources:
        return None

    return min(
        active_resources, key=lambda r: _calculate_distance(r.position, position)
    )


def _calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two positions."""
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def _ensure_tensor(state, device) -> torch.Tensor:
    """Ensure state is a tensor on the correct device."""
    if isinstance(state, torch.Tensor):
        return state.to(device)
    if hasattr(state, "to_tensor"):
        return state.to_tensor(device)
    return torch.FloatTensor(state).to(device)


def _store_and_train(agent, state, reward):
    """Store experience and perform training if possible."""
    if agent.move_module.last_state is not None:
        next_state = _ensure_tensor(agent.get_state(), agent.move_module.device)

        agent.move_module.store_experience(
            state=agent.move_module.last_state,
            action=agent.move_module.last_action,
            reward=reward,
            next_state=next_state,
            done=False,
        )

        if len(agent.move_module.memory) >= 2:
            batch_size = min(32, len(agent.move_module.memory))
            agent.move_module.train(random.sample(agent.move_module.memory, batch_size))
