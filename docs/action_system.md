# Multi-Agent Action System Documentation

## Table of Contents
- [Multi-Agent Action System Documentation](#multi-agent-action-system-documentation)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Core Components](#core-components)
    - [Base DQN System](#base-dqn-system)
      - [BaseDQNConfig](#base-dqn-config)
      - [BaseQNetwork](#base-q-network)
      - [BaseDQNModule](#base-d-qn-module)
    - [Action Types](#action-types)
      - [1. Movement (`move_action`)](#1-movement-move_action)
      - [2. Combat (`attack_action`)](#2-combat-attack_action)
  - [Action Selection Mechanism](#action-selection-mechanism)
    - [Overview](#overview-1)
    - [Base Action Weights](#base-action-weights)
    - [State-Based Adjustments](#state-based-adjustments)
      - [Configurable Multipliers](#configurable-multipliers)
    - [Epsilon-Greedy Exploration](#epsilon-greedy-exploration)
    - [Selection Process](#selection-process)
    - [Configuration](#configuration)
  - [Movement Learning System](#movement-learning-system)
    - [Architecture](#architecture)
      - [MoveQNetwork](#moveqnetwork)
      - [MoveModule](#movemodule)
    - [Learning Process](#learning-process)
    - [Configuration Parameters](#configuration-parameters)
  - [Integration](#integration)
  - [Performance Considerations](#performance-considerations)
  - [Usage Example](#usage-example)
  - [Further Reading and Resources](#further-reading-and-resources)

---

## Overview

The action system provides a flexible framework for agent behaviors in a multi-agent environment, combining both rule-based actions and learned movement policies using Deep Q-Learning (DQN). The system enables agents to interact with their environment through actions such as movement, gathering resources, sharing, and combat, with rewards and conditions guiding their behaviors.

---

## Core Components

### Base DQN System
The action system now uses a modular DQN architecture with:
- **BaseDQNConfig**: Common configuration parameters
- **BaseQNetwork**: Base neural network architecture
- **BaseDQNModule**: Core DQN functionality

### Action Types
Each action type extends the base DQN system:

#### 1. Movement (`move_action`)
- Inherits from BaseDQNModule
- Input dimension: 4
- Output dimension: 4 (directions)
- Custom reward structure:
  - Base movement cost: -0.1
  - Resource approach reward: +0.3
  - Resource retreat penalty: -0.2

#### 2. Combat (`attack_action`)
- Inherits from BaseDQNModule
- Input dimension: 6
- Output dimension: 5 (4 directions + defend)
- Custom features:
  - Health-based defense boost
  - Adaptive combat rewards

---

## Action Selection Mechanism

### Overview

The system’s action selection mechanism balances exploration and exploitation. It uses a weighted probability system, adjusts action likelihoods based on agent states and environmental conditions, and incorporates an epsilon-greedy exploration strategy.

### Base Action Weights

Each action is assigned a base weight representing its initial likelihood of being chosen. These weights are normalized to create a probability distribution. The default weights are:

- **Move**: 0.4
- **Gather**: 0.3
- **Share**: 0.2
- **Attack**: 0.1

### State-Based Adjustments

The system adjusts action probabilities dynamically based on current state factors:

- **Environmental Factors**:
  - Presence of nearby resources.
  - Proximity of other agents.
  - Agent’s current resource levels.
  - Starvation risk.

#### Configurable Multipliers

The following multipliers adjust action probabilities based on state:

```python
# Movement
move_mult_no_resources = 1.5    # Increased likelihood to move when no resources are nearby

# Gathering
gather_mult_low_resources = 1.5  # Increased likelihood to gather when agent's resources are low

# Sharing
share_mult_wealthy = 1.3        # Increased likelihood to share when agent has excess resources
share_mult_poor = 0.5          # Decreased likelihood to share when resources are needed

# Attack
attack_mult_desperate = 1.4     # Increased likelihood to attack when agent is starving
attack_mult_stable = 0.6       # Decreased likelihood to attack when agent's resources are stable
```

### Epsilon-Greedy Exploration

To promote exploration, the system uses an epsilon-greedy approach:

- **Epsilon (ε)**: Represents the exploration rate, which decays from `start_epsilon` to `min_epsilon`.
- **Action Selection**:
  - With probability ε, a random action is selected.
  - With probability (1 - ε), actions are chosen based on weighted selection.

### Selection Process

1. **Calculate Base Probabilities**: Begin with normalized base weights.
2. **Apply State-Based Adjustments**: Multiply weights by relevant state-based multipliers.
3. **Incorporate Epsilon-Greedy Exploration**:
   - Generate a random number between 0 and 1.
   - If the number is less than ε, select a random action.
   - Otherwise, proceed to the next step.
4. **Normalize Final Probabilities**: Ensure probabilities sum to 1.
5. **Select Action**: Choose an action based on final adjusted probabilities.

### Configuration

The action selection mechanism is configurable via `SimulationConfig`:

```python
config = SimulationConfig(
    social_range=30,               # Social interaction range
    move_mult_no_resources=1.5,    # Movement multiplier
    gather_mult_low_resources=1.5, # Gathering multiplier
    share_mult_wealthy=1.3,        # Sharing multiplier (wealthy)
    share_mult_poor=0.5,           # Sharing multiplier (poor)
    attack_starvation_threshold=0.5,
    attack_mult_desperate=1.4,     # Attack multiplier (desperate)
    attack_mult_stable=0.6,        # Attack multiplier (stable)
    start_epsilon=1.0,
    min_epsilon=0.01,
    epsilon_decay=0.995
)
```

---

## Movement Learning System

### Architecture

#### MoveQNetwork

The `MoveQNetwork` class defines a fully connected neural network for movement learning:

- **Input**: 4-dimensional state vector.
- **Hidden Layers**: 2 layers with 64 neurons and ReLU activations.
- **Output**: 4 actions (right, left, up, down).

#### MoveModule

The `MoveModule` handles the main Deep Q-Learning (DQN) operations, including experience replay, target network updates, and epsilon-greedy exploration.

### Learning Process

1. **State Observation**: Obtain current state representation.
2. **Action Selection**: Choose an action using ε-greedy.
3. **Environment Interaction**: Execute action and observe results.
4. **Experience Storage**: Store the experience in the replay buffer.
5. **Batch Training**: Sample and train on a batch of experiences.
6. **Target Network Updates**: Update target network periodically for stable learning.

### Configuration Parameters

The learning system is configured through `MovementConfig`:

| Parameter            | Description                         | Default Value |
| -------------------- | ----------------------------------- | ------------- |
| `learning_rate`      | Optimizer learning rate             | 0.001         |
| `memory_size`        | Experience replay buffer size       | 10,000        |
| `gamma`              | Reward discount factor              | 0.99          |
| `epsilon_start`      | Initial exploration rate            | 1.0           |
| `epsilon_min`        | Minimum exploration rate            | 0.01          |
| `epsilon_decay`      | Exploration decay rate              | 0.995         |
| `target_update_freq` | Frequency of target network updates | 100           |

---

## Integration

- **Action Execution**: Actions are managed through the `Action` class.
- **Continuous Movement Learning**: Learning updates occur during agent interactions.
- **Logging**: Each action includes detailed logging for monitoring behavior.
- **Resource Management**: The system handles resources automatically, including bounds checking.

---

## Performance Considerations

- **Vectorized Distance Calculations**: For efficient resource and agent proximity checks.
- **Batch Processing**: Experience replay uses batch processing to stabilize learning.
- **Automatic Device Management**: Tensor operations are managed to ensure efficient use of CPU/GPU resources.
- **Efficient Memory Usage**: Experience replay is managed using `deque` to limit memory usage.

---

## Usage Example

```python
# Define actions
move = Action("move", 0.4, move_action)
gather = Action("gather

", 0.3, gather_action)
share = Action("share", 0.2, share_action)
attack = Action("attack", 0.1, attack_action)

# Select and execute an action
selected_action = select_action(agent)
selected_action.execute(agent)
```

---

## Further Reading and Resources

- **Reinforcement Learning**: For more on DQN and action selection strategies, see "Deep Reinforcement Learning Hands-On" by Maxim Lapan.
- **Multi-Agent Systems**: For further background on multi-agent coordination, refer to "Multiagent Systems" by Gerhard Weiss.
- **Epsilon-Greedy Exploration**: Details on ε-greedy and other exploration strategies are available in the OpenAI Spinning Up documentation ([link](https://spinningup.openai.com)).