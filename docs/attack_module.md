# Attack Module Documentation

The Attack Module is a combat learning and execution system that leverages Deep Q-Learning (DQN) to enable agents to learn optimal attack and defense policies in a multi-agent environment. This module is designed with advanced features to ensure efficient learning and adaptive combat behavior.

---

## Overview

The Attack Module implements a Deep Q-Learning approach with several enhancements:

- **Double Q-Learning**: Reduces overestimation bias in action value estimation
- **Adaptive Defense**: Dynamically adjusts defense probability based on health status
- **Experience Replay**: Stores and samples past combat experiences
- **Target Network**: Maintains stable learning targets
- **Hardware Acceleration**: Automatically uses GPU when available

---

## Key Components

### 1. `AttackQNetwork`

Neural network architecture for Q-value approximation in combat scenarios:

- **Input Layer**: 6-dimensional state vector
- **Hidden Layers**: Two layers (64 neurons each) with:
  - Layer Normalization
  - ReLU activation
  - 10% Dropout
  - Xavier/Glorot initialization
- **Output Layer**: 5 actions (4 attack directions + defend)

### 2. `AttackModule`

Main class handling combat training and execution:

- **Combat Actions**: 
  - Four directional attacks (up, down, left, right)
  - Defensive stance
- **Training Features**:
  - Double Q-Learning implementation
  - Soft target network updates
  - Gradient clipping (max norm: 1.0)
  - SmoothL1Loss criterion
  - Experience replay with fixed memory size
- **Metrics Tracking**:
  - Loss history
  - Episode rewards

---

## Configuration Parameters

The module uses `AttackConfig` for customization:

- **Network Parameters**:
  - `attack_dqn_hidden_size`: 64
  - `attack_batch_size`: 32
  - `attack_tau`: 0.005 (soft update parameter)
  - `attack_target_update_freq`: 100

- **Memory and Learning**:
  - `attack_memory_size`: 10000
  - `attack_learning_rate`: 0.001
  - `attack_gamma`: 0.99 (discount factor)
  - `attack_epsilon_start`: 1.0
  - `attack_epsilon_min`: 0.01
  - `attack_epsilon_decay`: 0.995

- **Combat Parameters**:
  - `attack_base_cost`: -0.2
  - `attack_success_reward`: 1.0
  - `attack_failure_penalty`: -0.3
  - `attack_defense_threshold`: 0.3
  - `attack_defense_boost`: 2.0 