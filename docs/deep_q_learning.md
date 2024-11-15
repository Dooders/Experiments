# Deep Q-Network (DQN) Implementation Guide

## Overview
This implementation uses a modular Deep Q-Network (DQN) architecture with a base class that can be extended for different action types. The system combines Q-learning with deep neural networks to handle complex state spaces and learn effective policies.

## Base DQN Architecture

### BaseDQNConfig
Common configuration parameters for all DQN modules:
```python
class BaseDQNConfig:
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
```

### BaseQNetwork
Base neural network architecture that can be extended:
```python
class BaseQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 64):
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
```

### BaseDQNModule
Core DQN functionality that specific modules inherit from:
- Experience replay management
- Network training
- Target network updates
- Epsilon-greedy exploration

## Specialized Implementations

### 1. Movement Module
Extends base DQN for movement learning:
```python
class MoveModule(BaseDQNModule):
    def __init__(self, config: MoveConfig = DEFAULT_MOVE_CONFIG):
        super().__init__(input_dim=4, output_dim=4, config=config)
```

### 2. Attack Module
Extends base DQN for combat learning:
```python
class AttackModule(BaseDQNModule):
    def __init__(self, config: AttackConfig = DEFAULT_ATTACK_CONFIG):
        super().__init__(input_dim=6, output_dim=5, config=config)
```

## Creating New DQN Modules

To create a new DQN-based module:

1. Create a config class extending BaseDQNConfig:
```python
class NewActionConfig(BaseDQNConfig):
    new_action_specific_param: float = 1.0
```

2. Create a module class extending BaseDQNModule:
```python
class NewActionModule(BaseDQNModule):
    def __init__(self, config: NewActionConfig):
        super().__init__(
            input_dim=your_input_dim,
            output_dim=your_output_dim,
            config=config
        )
```

## Best Practices

1. **Configuration**
   - Inherit from BaseDQNConfig for common parameters
   - Add action-specific parameters in derived config classes

2. **Network Architecture**
   - Use BaseQNetwork as starting point
   - Customize architecture if needed by extending BaseQNetwork

3. **Module Implementation**
   - Inherit core functionality from BaseDQNModule
   - Override methods only when necessary
   - Implement action-specific logic in derived classes

## Performance Considerations

1. **Shared Optimizations**
   - GPU acceleration handled by base module
   - Common memory management
   - Gradient clipping
   - Experience replay

2. **Module-Specific Tuning**
   - Configure input/output dimensions
   - Adjust hyperparameters via config
   - Implement custom reward functions

## Error Handling

The base module includes built-in error handling for:
- Tensor device management
- Batch size validation
- Network initialization
- Experience storage

## Example Usage

```python
# Create config
config = MoveConfig(
    learning_rate=0.001,
    memory_size=10000,
    gamma=0.99
)

# Initialize module
move_module = MoveModule(config)

# Use module
next_position = move_module.get_movement(agent, state)
move_module.store_experience(state, action, reward, next_state)
loss = move_module.train(batch)
```