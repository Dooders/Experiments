# Deep Q-Network (DQN) Implementation Guide

## Overview
This implementation uses Deep Q-Networks (DQN) to teach agents optimal movement strategies in a 2D environment. DQN combines Q-learning with deep neural networks to handle complex state spaces and learn effective policies.

## How DQN Works

### Core Concepts
1. **Q-Learning**: A method that learns to predict the quality (Q-value) of taking an action in a given state
2. **Neural Network**: Approximates the Q-function, mapping states to action values
3. **Experience Replay**: Stores and randomly samples past experiences for stable learning
4. **Target Network**: A separate network that provides stable Q-value targets

### Architecture
Our implementation uses:
- Input layer: 4 neurons (state dimensions)
- Hidden layers: 2 layers with 64 neurons each
- Output layer: 4 neurons (one for each movement action)

```python
self.network = nn.Sequential(
    nn.Linear(input_dim, hidden_size),  # 4 → 64
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size), # 64 → 64
    nn.ReLU(),
    nn.Linear(hidden_size, 4),  # 64 → 4 actions
)
```

## Key Components

### 1. Action Space
Four possible movements:
- Right (0): (1, 0)
- Left (1): (-1, 0)
- Up (2): (0, 1)
- Down (3): (0, -1)

### 2. Exploration Strategy
Uses ε-greedy exploration:
- With probability ε: Choose random action (exploration)
- With probability 1-ε: Choose best action (exploitation)
- ε decays over time from `epsilon_start` to `epsilon_min`

### 3. Experience Replay
```python
self.memory = deque(maxlen=config.memory_size)
```
Stores transitions as (state, action, reward, next_state, done) tuples for batch learning.

### 4. Training Process

#### a. Computing Q-Values
1. Current Q-values from main network
2. Target Q-values from target network
3. Loss calculated using Mean Squared Error

#### b. Update Steps
```python
# Get current Q values
current_q_values = self.q_network(states).gather(1, actions)

# Compute target Q values
next_q_values = self.target_network(next_states).max(1)[0]
target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

# Update network
loss = self.criterion(current_q_values.squeeze(), target_q_values)
```

## Configuration Parameters

- `learning_rate`: Rate at which the network updates weights
- `memory_size`: Maximum number of experiences to store
- `gamma`: Discount factor for future rewards (0 to 1)
- `epsilon_start`: Initial exploration rate
- `epsilon_min`: Minimum exploration rate
- `epsilon_decay`: Rate at which exploration decreases
- `target_update_freq`: How often to update target network

## Usage Example

```python
config = MovementConfig(
    learning_rate=0.001,
    memory_size=10000,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    target_update_freq=100
)

# Initialize module
move_module = MoveModule(config)

# Get movement decision
next_position = move_module.get_movement(agent, current_state)

# Store experience
move_module.store_experience(state, action, reward, next_state)

# Train on batch
loss = move_module.train(batch)
```

## Performance Considerations

1. **GPU Acceleration**
   - Automatically uses CUDA if available
   - Falls back to CPU if CUDA is not available

2. **Memory Management**
   - Handles tensor-to-numpy conversions automatically
   - Ensures proper device placement (CPU/GPU)

3. **Bounded Movement**
   - Enforces environment boundaries
   - Scales movement by agent's max_movement parameter

## Best Practices

1. **Training**
   - Use sufficiently large batch sizes (>2)
   - Allow enough episodes for ε to decay
   - Monitor loss values for convergence

2. **Hyperparameter Tuning**
   - Start with high exploration (ε ≈ 1.0)
   - Use moderate decay rate (0.995-0.999)
   - Adjust learning rate based on stability

3. **State Design**
   - Ensure state representations are normalized
   - Include relevant information for decision-making
   - Keep dimensions consistent