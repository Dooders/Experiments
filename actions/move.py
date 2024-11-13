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

import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


class MoveQNetwork(nn.Module):
    """Neural network architecture for Q-value approximation in movement learning.

    This network maps state observations to Q-values for each possible movement action.
    Uses a fully connected architecture with ReLU activations for deep Q-learning.

    Architecture:
        Input Layer: state_dimension neurons
        Hidden Layer 1: hidden_size neurons (default 64) with ReLU
        Hidden Layer 2: hidden_size neurons (default 64) with ReLU
        Output Layer: 4 neurons (one for each movement action)

    Args:
        input_dim (int): Dimension of the input state vector
        hidden_size (int, optional): Number of neurons in hidden layers. Defaults to 64.

    Forward Pass:
        Input: State tensor of shape (batch_size, input_dim)
        Output: Q-values tensor of shape (batch_size, 4)
    """

    def __init__(self, input_dim, hidden_size=64):
        super(MoveQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4),  # 4 actions: right, left, up, down
        )

    def forward(self, x):
        return self.network(x)


class MoveModule:
    """Movement learning and execution module using Deep Q-Learning.

    This module implements Deep Q-Learning with experience replay and target networks
    to learn optimal movement policies in a 2D environment.

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
    """

    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = MoveQNetwork(input_dim=4).to(self.device)
        self.target_network = MoveQNetwork(input_dim=4).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=config.learning_rate
        )
        self.criterion = nn.MSELoss()
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
            0: (1, 0),  # Right
            1: (-1, 0),  # Left
            2: (0, 1),  # Up
            3: (0, -1),  # Down
        }

    def select_action(self, state, epsilon=None):
        """Select movement action using epsilon-greedy policy.

        Chooses between random exploration and learned policy based on epsilon value.
        Handles both numpy arrays and torch tensors as input states.

        Args:
            state (Union[np.ndarray, torch.Tensor]): Current state observation
            epsilon (float, optional): Override default epsilon value. Defaults to None.

        Returns:
            int: Selected action index:
                0: Move Right
                1: Move Left
                2: Move Up
                3: Move Down
        """
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            return random.randint(0, 3)

        with torch.no_grad():
            # Ensure state is a tensor and on the correct device
            if not isinstance(state, torch.Tensor):
                state_tensor = torch.FloatTensor(state).to(self.device)
            else:
                state_tensor = state.to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().argmax().item()

    def train(self, batch):
        """Train Q-network using a batch of experiences.

        Implements one step of DQN training:
        1. Compute current Q-values for batch states/actions
        2. Compute target Q-values using target network
        3. Calculate TD error loss
        4. Update network weights
        5. Update target network if needed
        6. Decay exploration rate

        Args:
            batch (List[Tuple]): List of (state, action, reward, next_state, done) tuples
                - state: Current state observation
                - action: Action taken
                - reward: Reward received
                - next_state: Resulting state
                - done: Whether episode ended

        Returns:
            float: Training loss value, or None if batch size < 2
        """
        if len(batch) < 2:  # Minimum batch size check
            return None

        # Convert batch elements to tensors, ensuring they're on CPU first
        states = torch.FloatTensor([x[0] for x in batch]).to(self.device)
        actions = torch.LongTensor([[x[1]] for x in batch]).to(self.device)
        rewards = torch.FloatTensor([x[2] for x in batch]).to(self.device)
        next_states = torch.FloatTensor([x[3] for x in batch]).to(self.device)
        dones = torch.FloatTensor([x[4] for x in batch]).to(self.device)

        # Get current Q values
        current_q_values = self.q_network(states).gather(1, actions)

        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and update
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.cpu().item()

    def get_movement(self, agent, state):
        """Determine next movement position using learned policy.

        Selects action using current policy and converts it to valid coordinates,
        ensuring movement stays within environment bounds.

        Args:
            agent: Agent object with attributes:
                - position (Tuple[float, float]): Current (x,y) position
                - max_movement (float): Maximum movement distance
                - environment: Environment with width/height bounds
            state (Union[np.ndarray, torch.Tensor]): Current state observation

        Returns:
            Tuple[float, float]: New (x,y) position coordinates
        """
        action = self.select_action(state)
        dx, dy = self.action_space[action]

        # Scale movement by agent's max_movement
        dx *= agent.max_movement
        dy *= agent.max_movement

        # Calculate new position within bounds
        new_x = max(0, min(agent.environment.width, agent.position[0] + dx))
        new_y = max(0, min(agent.environment.height, agent.position[1] + dy))

        # Store state and action for learning
        # Convert state tensor to CPU if it's on CUDA
        if isinstance(state, torch.Tensor):
            self.last_state = state.cpu().numpy()
        else:
            self.last_state = state
        self.last_action = action

        return (new_x, new_y)

    def store_experience(self, state, action, reward, next_state, done=False):
        """Store a transition in experience replay memory.

        Handles conversion between tensor and numpy formats before storing.
        Manages memory buffer size automatically.

        Args:
            state (Union[np.ndarray, torch.Tensor]): State observation at t
            action (int): Action taken at t
            reward (float): Reward received at t+1
            next_state (Union[np.ndarray, torch.Tensor]): State observation at t+1
            done (bool, optional): Whether episode ended. Defaults to False.
        """
        # Convert any tensors to numpy arrays before storing
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()

        self.memory.append((state, action, reward, next_state, done))
