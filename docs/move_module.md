# Move Module Documentation

The Move Module is a movement learning and execution system that leverages Deep Q-Learning (DQN) to enable agents to learn optimal movement policies in a 2D environment. This module is designed with advanced features to ensure efficient learning and stable performance, making it suitable for complex reinforcement learning tasks.

---

## Overview

The Move Module implements a Deep Q-Learning approach with several enhancements:

- **Double Q-Learning**: Reduces overestimation bias by decoupling action selection and evaluation.
- **Soft Target Updates**: Gradually updates the target network for stability.
- **Adaptive Exploration**: Dynamically adjusts the exploration rate based on learning progress.
- **Temperature-based Action Selection**: Utilizes softmax with adaptive temperature for smarter exploration.
- **Experience Replay**: Stores and samples past experiences to stabilize learning.
- **Hardware Acceleration**: Automatically uses GPU acceleration when available.

---

## Key Components

### 1. `MoveQNetwork`

A neural network architecture for Q-value approximation. It maps state observations to Q-values for each possible movement action using an enhanced architecture with:

- **Layer Normalization**: For training stability.
- **Dropout Layers**: To prevent overfitting.
- **Xavier/Glorot Initialization**: For better gradient flow.
- **ReLU Activation**: Non-linear activation function for hidden layers.

**Architecture:**

- **Input Layer**: Receives a state vector of dimension 4.
- **Hidden Layers**: Two hidden layers with configurable size (default 64 neurons each).
- **Output Layer**: Produces Q-values for the 4 discrete actions (right, left, up, down).

### 2. `MoveModule`

The main class that handles training, action selection, and movement execution. It incorporates advanced features like:

- **Double Q-Learning**: Uses two networks to reduce overestimation.
- **Soft Target Network Updates**: Smoothly updates the target network parameters.
- **Adaptive Exploration Strategy**: Adjusts the exploration rate (`epsilon`) based on the agent's learning progress.
- **Gradient Clipping**: Prevents exploding gradients during training.
- **Huber Loss Function**: More robust to outliers compared to Mean Squared Error (MSE).

**Key Methods:**

- `get_movement(agent, state)`: Determines the next movement action based on the current state.
- `store_experience(state, action, reward, next_state, done)`: Stores experiences in the replay buffer.
- `train(batch)`: Trains the Q-network using a batch of experiences.
- `select_action(state_tensor, epsilon)`: Selects an action using an epsilon-greedy strategy with temperature-based softmax.

### 3. Experience Replay

Utilizes a deque (double-ended queue) to store a finite number of past experiences, defined by `move_memory_size`. This allows the agent to learn from a diverse set of experiences and helps in breaking correlations between sequential data.

### 4. Target Network

A separate neural network (`target_network`) used to compute stable Q-value targets. It is periodically or softly updated with the weights of the main Q-network (`q_network`) to stabilize training.

---

## Technical Details

- **State Space**: A 4-dimensional vector representing the agent's current state.
- **Action Space**: 4 discrete actions corresponding to moving right, left, up, or down.
- **Learning Algorithm**: Enhanced Deep Q-Learning with Double Q-Learning and adaptive exploration.
- **Exploration Strategy**: Epsilon-greedy strategy with decay and adaptive adjustment based on reward improvement.
- **Soft Update Parameter (`tau`)**: Controls the rate of the soft update for the target network.

---

## Configuration Parameters (`MoveConfig`)

- **`move_learning_rate`**: Learning rate for the optimizer (default: `0.001`).
- **`move_memory_size`**: Maximum size of the experience replay buffer (default: `10000`).
- **`move_gamma`**: Discount factor for future rewards (default: `0.99`).
- **`move_epsilon_start`**: Initial exploration rate (default: `1.0`).
- **`move_epsilon_min`**: Minimum exploration rate (default: `0.01`).
- **`move_epsilon_decay`**: Base rate for epsilon decay (default: `0.995`).
- **`move_target_update_freq`**: Frequency of target network updates (default: `100` steps).
- **`move_dqn_hidden_size`**: Number of neurons in hidden layers (default: `64`).
- **`move_batch_size`**: Batch size for training (default: `32`).
- **`move_reward_history_size`**: Size of the reward history for adaptive exploration (default: `100`).
- **`move_epsilon_adapt_threshold`**: Threshold for adapting epsilon (default: `0.1`).
- **`move_epsilon_adapt_factor`**: Factor to adjust the epsilon decay rate (default: `1.5`).
- **`move_min_reward_samples`**: Minimum number of reward samples before adapting epsilon (default: `10`).
- **`move_tau`**: Soft update parameter for the target network (default: `0.005`).

---

## Usage Example

```python
# Initialize configuration
config = MoveConfig(
    move_learning_rate=0.001,
    move_memory_size=10000,
    move_gamma=0.99,
    move_epsilon_start=1.0,
    move_epsilon_min=0.01,
    move_epsilon_decay=0.995,
    move_target_update_freq=100,
    move_dqn_hidden_size=64,
    move_batch_size=32,
    move_reward_history_size=100,
    move_epsilon_adapt_threshold=0.1,
    move_epsilon_adapt_factor=1.5,
    move_min_reward_samples=10,
    move_tau=0.005
)

# Create and use the MoveModule
move_module = MoveModule(config)
current_state = torch.FloatTensor([0.5, 0.2, -0.1, 0.3])  # Example state vector
next_position = move_module.get_movement(agent, current_state)

# After performing the action and receiving a reward
reward = 1.0  # Example reward
next_state = torch.FloatTensor([0.6, 0.1, -0.05, 0.25])  # Example next state
done = False  # Whether the episode has ended

# Store the experience
move_module.store_experience(current_state, action, reward, next_state, done)

# Train the network
if len(move_module.memory) >= move_module.config.move_batch_size:
    batch = random.sample(move_module.memory, move_module.config.move_batch_size)
    loss = move_module.train(batch)
```

---

## How It Works

### Action Selection

The `select_action` method uses an epsilon-greedy strategy with temperature-based softmax to select actions. The temperature parameter is derived from the current epsilon value, creating smoother exploration behavior compared to uniform random selection.

- **Exploration**: With probability epsilon, the agent explores by selecting an action based on the softmax probabilities of the Q-values.
- **Exploitation**: With probability (1 - epsilon), the agent selects the action with the highest Q-value.

### Training Process

1. **Experience Storage**: After each action, the experience `(state, action, reward, next_state, done)` is stored in the replay buffer.
2. **Batch Sampling**: A random batch of experiences is sampled from the replay buffer.
3. **Q-Value Calculation**:
   - **Current Q-Values**: Predicted by the main network for the current states.
   - **Target Q-Values**: Calculated using the target network for the next states, following the Double Q-Learning approach.
4. **Loss Computation**: The loss between current Q-values and target Q-values is computed using the Huber Loss function.
5. **Backpropagation**: The gradients are computed, and the network parameters are updated using the optimizer.
6. **Target Network Update**: The target network is softly updated using the `tau` parameter.

### Adaptive Exploration

The epsilon value, which governs the exploration rate, is adaptively adjusted based on the agent's learning progress:

- **Reward Improvement**: The module tracks recent rewards and adjusts epsilon decay if improvement falls below a threshold.
- **Epsilon Decay Adjustment**: Slows down the decay if learning has plateaued to encourage exploration.

---

## Integration with Agents

To integrate the Move Module with an agent:

1. **Initialization**: Instantiate the `MoveModule` and assign it to the agent.
2. **State Representation**: Ensure the agent can provide its current state as a tensor.
3. **Action Execution**: Use `get_movement` to obtain the next action and update the agent's position.
4. **Experience Handling**: After executing the action and receiving a reward, store the experience using `store_experience`.
5. **Training**: Periodically call `train` to update the Q-network based on experiences.

---

## Helper Functions

### `move_action(agent)`

Executes movement for the agent using the optimized policy:

- Retrieves the current state.
- Determines the new position using `get_movement`.
- Calculates the reward based on movement.
- Stores the experience.
- Trains the network if enough experiences are available.

### `_calculate_movement_reward`

Calculates the reward for movement based on:

- **Base Movement Cost**: A small negative reward for making a move.
- **Distance Moved**: Adjusts reward based on the distance moved towards the closest resource.
- **Resource Proximity**: Provides positive reward if the agent moves closer to a resource.

### `_find_closest_resource`

Finds the closest non-depleted resource in the environment to the agent's current position.

### `_calculate_distance`

Computes the Euclidean distance between two positions.

### `_ensure_tensor`

Ensures that the state is converted to a tensor and moved to the correct device (CPU or GPU).

### `_store_and_train`

Handles experience storage and triggers training if enough experiences are available in the replay buffer.

---

## Dependencies

- **PyTorch (`torch`)**: For neural network implementation and tensor operations.
- **NumPy (`numpy`)**: For numerical computations.
- **Random**: For random sampling in the replay buffer.
- **Collections (`deque`)**: For managing the experience replay buffer.

---

## Best Practices

- **Batch Size**: Ensure that the batch size for training is appropriate relative to the replay buffer size.
- **Exploration vs. Exploitation**: Monitor the epsilon value to balance exploration and exploitation.
- **Reward Shaping**: Carefully design the reward function to guide the agent effectively.
- **Gradient Clipping**: Use gradient clipping to prevent exploding gradients during training.
- **Device Management**: Leverage GPU acceleration if available by ensuring tensors and networks are moved to the correct device.

---

## Troubleshooting

- **Convergence Issues**: If the agent is not learning, consider adjusting the learning rate, exploration parameters, or reward function.
- **Overfitting**: If the agent performs well on the training environment but poorly elsewhere, consider increasing exploration or adding regularization.
- **Performance**: Ensure that the replay buffer is sufficiently large and that experiences are sampled randomly.

---

## Extensibility

The Move Module is designed to be extensible:

- **State and Action Spaces**: Can be modified to accommodate different environments.
- **Network Architecture**: The `MoveQNetwork` can be customized with additional layers or different activation functions.
- **Learning Algorithms**: While it currently uses Double Q-Learning, other algorithms like Dueling DQN or Prioritized Experience Replay can be integrated.

---

## Conclusion

The Move Module provides a robust and flexible framework for implementing movement learning in agents using advanced Deep Q-Learning techniques. By integrating this module, developers can create agents capable of learning complex movement policies in dynamic environments.

---

## References

- **Deep Q-Networks (DQN)**: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- **Double Q-Learning**: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- **Adaptive Exploration**: Techniques for adjusting exploration rates based on learning progress.
