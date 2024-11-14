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
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import ModelState only during type checking
if TYPE_CHECKING:
    from state import ModelState

logger = logging.getLogger(__name__)

# Move device check outside class for single evaluation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MoveConfig:
    target_update_freq: int = 100
    memory_size: int = 10000
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    dqn_hidden_size: int = 64
    batch_size: int = 32


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

    This module implements Double Q-Learning with experience replay and soft target
    network updates to learn optimal movement policies in a 2D environment.

    Key Enhancements:
        - Double Q-Learning: Reduces overestimation bias by decoupling action selection
          and evaluation using two networks
        - Soft Target Updates: Gradually updates target network for stability
        - Huber Loss: More robust to outliers than MSE
        - Gradient Clipping: Prevents exploding gradients
        - Experience Replay: Stores transitions for stable batch learning

    Features:
        - Experience Replay: Stores transitions for stable batch learning
        - Target Network: Separate network for stable Q-value targets
        - Epsilon-greedy Exploration: Balance between exploration and exploitation
        - Automatic Device Selection: Uses GPU if available

    Args:
        config: Configuration object containing:
            - learning_rate (float): Learning rate for Adam optimizer
            - memory_size (int): Maximum size of experience replay buffer
            - gamma (float): Discount factor for future rewards [0,1]
            - epsilon_start (float): Initial exploration rate [0,1]
            - epsilon_min (float): Minimum exploration rate [0,1]
            - epsilon_decay (float): Rate at which epsilon decreases
            - target_update_freq (int): Steps between target network updates

    Attributes:
        device (torch.device): CPU or CUDA device for computations
        q_network (MoveQNetwork): Main Q-network for action selection
        target_network (MoveQNetwork): Target network for stable learning
        memory (deque): Experience replay buffer
        epsilon (float): Current exploration rate
        steps (int): Total steps taken for target network updates
        tau (float): Soft update parameter for target network
    """

    def __init__(
        self, config: MoveConfig = DEFAULT_MOVE_CONFIG, device: torch.device = DEVICE
    ):
        self.device = device
        self.q_network = MoveQNetwork(input_dim=4).to(self.device)
        self.target_network = MoveQNetwork(input_dim=4).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=config.learning_rate
        )
        self.criterion = nn.SmoothL1Loss()
        self.memory = deque(maxlen=config.memory_size)

        # Q-learning parameters
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.target_update_freq = config.target_update_freq
        self.steps = 0

        # Action space mapping
        self.action_space = {
            MoveActionSpace.RIGHT: (1, 0),
            MoveActionSpace.LEFT: (-1, 0),
            MoveActionSpace.UP: (0, 1),
            MoveActionSpace.DOWN: (0, -1),
        }

        # Add tau parameter for soft target network updates
        self.tau = 0.005  # Soft update parameter

    def _soft_update_target_network(self):
        """Soft update target network weights using tau parameter."""
        for target_param, local_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def select_action(self, state, epsilon=None):
        """Select movement action using epsilon-greedy policy.

        Chooses between random exploration and learned policy based on epsilon value.
        Handles both numpy arrays and torch tensors as input states.

        Args:
            state (Union[np.ndarray, torch.Tensor]): Current state observation
            epsilon (float, optional): Override default epsilon value. Defaults to None.

        Returns:
            int: Selected action index:
                MoveActionSpace.RIGHT: Move Right
                MoveActionSpace.LEFT: Move Left
                MoveActionSpace.UP: Move Up
                MoveActionSpace.DOWN: Move Down
        """
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            return random.randint(0, 3)

        with torch.no_grad():
            # State is already a tensor on correct device from store_experience
            q_values = self.q_network(state)
            return q_values.cpu().argmax().item()

    def train(self, batch) -> Optional[float]:
        """Train Q-network using Double Q-Learning and Huber Loss.

        Implements several optimizations for stable and efficient learning:
        1. Double Q-Learning: Uses main network for action selection and target
           network for value estimation to reduce overestimation bias
        2. Huber Loss: More robust to outliers than MSE loss
        3. Gradient Clipping: Prevents exploding gradients
        4. Soft Target Updates: Gradually updates target network weights

        Args:
            batch: List of experience tuples (state, action, reward, next_state, done)

        Returns:
            Optional[float]: The loss value if training occurred, None if batch too small

        Note:
            The training process includes several steps:
            1. Convert experiences to tensor format
            2. Compute current Q-values from main network
            3. Use Double Q-Learning to compute target Q-values
            4. Apply Huber loss and update main network
            5. Perform soft update of target network
            6. Update exploration rate (epsilon)
        """
        if len(batch) < 2:  # Minimum batch size check
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

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.cpu().item()

    def get_movement(self, agent, state):
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

    def store_experience(self, state, action, reward, next_state, done):
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


def move_action(agent):
    """Execute movement using optimized Deep Q-Learning based policy.

    This function handles the complete movement action cycle including:
    - State observation and conversion
    - Action selection using DQN
    - Movement execution
    - Reward calculation
    - Experience storage and training

    Reward Structure:
        Base: -0.1 (movement cost)
        Resource Proximity:
            +0.3: Moving closer to resources
            -0.2: Moving away from resources

    Training Features:
        - Double Q-Learning for stable value estimation
        - Experience replay for decorrelated training samples
        - Soft target network updates for stability
        - Huber loss for robustness to outliers

    Args:
        agent: Agent object with following required attributes:
            - move_module: MoveModule instance
            - position: Current (x, y) position
            - environment: Reference to environment
            - get_state(): Method returning current state

    Returns:
        None: Updates agent position and trains move_module in-place

    Example:
        >>> initial_pos = agent.position
        >>> move_action(agent)
        >>> print(f"Agent moved from {initial_pos} to {agent.position}")
    """
    # Get state and convert to tensor
    state = agent.get_state()
    if not isinstance(state, torch.Tensor):
        state = state.to_tensor(agent.move_module.device)

    initial_position = agent.position
    new_position = agent.move_module.get_movement(agent, state)
    agent.position = new_position

    # Calculate movement distance for reward
    distance_moved = np.sqrt(
        (new_position[0] - initial_position[0]) ** 2
        + (new_position[1] - initial_position[1]) ** 2
    )

    # Simple reward based on movement and resource proximity
    reward = -0.1  # Base cost for moving
    if distance_moved > 0:
        # Add reward for moving closer to resources
        closest_resource = min(
            [r for r in agent.environment.resources if not r.is_depleted()],
            key=lambda r: np.sqrt(
                (r.position[0] - new_position[0]) ** 2
                + (r.position[1] - new_position[1]) ** 2
            ),
            default=None,
        )
        if closest_resource:
            old_distance = np.sqrt(
                (closest_resource.position[0] - initial_position[0]) ** 2
                + (closest_resource.position[1] - initial_position[1]) ** 2
            )
            new_distance = np.sqrt(
                (closest_resource.position[0] - new_position[0]) ** 2
                + (closest_resource.position[1] - new_position[1]) ** 2
            )
            reward += 0.3 if new_distance < old_distance else -0.2

    # Store experience and train
    if agent.move_module.last_state is not None:
        next_state = agent.get_state()
        if not isinstance(next_state, torch.Tensor):
            next_state = next_state.to_tensor(agent.move_module.device)

        agent.move_module.store_experience(
            state=agent.move_module.last_state,
            action=agent.move_module.last_action,
            reward=reward,
            next_state=next_state,
            done=False,
        )

        # Train if enough samples
        if len(agent.move_module.memory) >= 2:
            agent.move_module.train(
                random.sample(
                    agent.move_module.memory, min(32, len(agent.move_module.memory))
                )
            )

    logger.debug(
        f"Agent {id(agent)} moved from {initial_position} to {new_position}. "
        f"Reward: {reward:.3f}, Epsilon: {agent.move_module.epsilon:.3f}"
    )
