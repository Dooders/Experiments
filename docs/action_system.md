# Multi-Agent Action System Documentation

## Overview
The action system implements a flexible framework for agent behaviors in a multi-agent environment, combining both rule-based actions and learned movement policies using Deep Q-Learning.

## Core Components

### Action Class (`action.py`)
The base class for all agent behaviors with:
- Named actions with weighted selection probabilities
- Flexible execution function support
- Support for additional parameters

### Action Types

#### 1. Movement (`move_action`)
- Uses Deep Q-Learning for intelligent navigation
- Reward structure:
  - Base cost: -0.1
  - Moving toward resources: +0.3
  - Moving away from resources: -0.2
- Automatic state conversion between numpy/torch tensors

#### 2. Resource Gathering (`gather_action`)
- Range-based resource collection
- Configurable gathering parameters:
  - Gathering range from config
  - Maximum gather amount
- Vectorized distance calculations for efficiency

#### 3. Resource Sharing (`share_action`)
- Cooperative behavior mechanism
- Requirements:
  - 30-unit interaction radius
  - Sharer must have >1 resource
  - Valid recipient in range
- Transfers 1 resource unit per action

#### 4. Combat (`attack_action`)
- Competitive resource acquisition
- Requirements:
  - 20-unit attack radius
  - Attacker must have >2 resources
  - Valid target in range
- Effects:
  - Costs 1 resource to attack
  - Deals 1-2 damage to target

## Movement Learning System (`move.py`)

### Architecture

#### MoveQNetwork
- Fully connected neural network
- Architecture:
  - Input: 4D state vector
  - Hidden layers: 2x64 neurons with ReLU
  - Output: 4 actions (right, left, up, down)

#### MoveModule
- Deep Q-Learning implementation
- Features:
  - Experience replay buffer
  - Target network for stable learning
  - Epsilon-greedy exploration
  - Automatic GPU acceleration

### Learning Process
1. State observation
2. Action selection (ε-greedy)
3. Environment interaction
4. Experience storage
5. Batch training
6. Target network updates

### Configuration Parameters
```python
MovementConfig(
    learning_rate=0.001,    # Optimizer learning rate
    memory_size=10000,      # Experience replay size
    gamma=0.99,             # Reward discount factor
    epsilon_start=1.0,      # Initial exploration rate
    epsilon_min=0.01,       # Minimum exploration rate
    epsilon_decay=0.995,    # Exploration decay rate
    target_update_freq=100  # Target network update frequency
)
```

## Integration
- Actions are executed through the Action class wrapper
- Movement learning occurs continuously during agent interactions
- All actions include detailed logging for monitoring
- Automatic resource management and bounds checking

## Performance Considerations
- Vectorized distance calculations
- Batch processing for learning
- Automatic tensor device management
- Efficient memory usage with deque

## Usage Example
```python
# Create action instances
move = Action("move", 0.4, move_action)
gather = Action("gather", 0.3, gather_action)
share = Action("share", 0.2, share_action)
attack = Action("attack", 0.1, attack_action)

# Execute action
move.execute(agent)
```