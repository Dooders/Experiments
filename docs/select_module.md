# Selection Module Documentation

## Overview

The Selection Module provides an intelligent action selection system that combines rule-based heuristics with deep Q-learning to help agents make optimal decisions about which actions to take during their turns.

## Key Components

### SelectConfig

Configuration class for the selection module with the following parameters:

```python
class SelectConfig:
    # Base action weights
    move_weight: float = 0.4
    gather_weight: float = 0.3
    share_weight: float = 0.2
    attack_weight: float = 0.1

    # State-based multipliers
    move_mult_no_resources: float = 1.5
    gather_mult_low_resources: float = 1.5
    share_mult_wealthy: float = 1.3
    share_mult_poor: float = 0.5
    attack_mult_desperate: float = 1.4
    attack_mult_stable: float = 0.6

    # Thresholds
    attack_starvation_threshold: float = 0.5
    attack_defense_threshold: float = 0.3
```

### SelectQNetwork

Neural network architecture for action selection decisions:
- Input dimension: 8 (state features)
- Output dimension: Number of possible actions
- Hidden layers: Configurable size (default 64 neurons)

### SelectModule

Main class handling action selection logic and learning:

```python
class SelectModule:
    def select_action(self, agent, actions, state) -> Action:
        """Select an action using both predefined weights and learned preferences."""
```

## State Representation

The selection module uses an 8-dimensional state vector:

1. `resource_ratio`: Agent's current resources normalized by max resources
2. `health_ratio`: Current health / max health
3. `starvation_ratio`: Current starvation threshold / max starvation time
4. `nearby_resources_ratio`: Number of nearby resources / total resources
5. `nearby_agents_ratio`: Number of nearby agents / total agents
6. `time_indicator`: Binary indicator if not first step
7. `is_defending`: Binary indicator if agent is in defense stance
8. `is_alive`: Binary indicator of agent's alive status

## Action Selection Process

### 1. Base Probabilities

Initial action probabilities are determined by predefined weights:
- Move: 0.4
- Gather: 0.3
- Share: 0.2
- Attack: 0.1

### 2. State-Based Adjustments

Probabilities are adjusted based on current state:

#### Movement
- Increased when no resources nearby (`move_mult_no_resources`)

#### Gathering
- Increased when agent has low resources (`gather_mult_low_resources`)

#### Sharing
- Increased when wealthy and agents nearby (`share_mult_wealthy`)
- Decreased when poor (`share_mult_poor`)

#### Attack
- Increased when desperate (`attack_mult_desperate`)
- Decreased when stable (`attack_mult_stable`)
- Modified based on health ratio

### 3. Q-Learning Integration

The module combines rule-based probabilities with learned Q-values:
- 70% weight to adjusted probabilities
- 30% weight to normalized Q-values

### 4. Exploration vs Exploitation

Uses epsilon-greedy strategy:
- Random action with probability ε
- Best action with probability 1-ε
- Epsilon decays over time

## Usage Example

```python
# Initialize module
select_module = SelectModule(
    num_actions=len(actions),
    config=SelectConfig(),
    device=device
)

# Get state
state = create_selection_state(agent)

# Select action
action = select_module.select_action(
    agent=agent,
    actions=actions,
    state=state
)
```

## Integration with BaseAgent

The selection module is automatically initialized in BaseAgent:

```python
def __init__(self, ...):
    # Initialize selection module
    self.select_module = SelectModule(
        num_actions=len(self.actions),
        config=SelectConfig(),
        device=self.device
    )
```

## Learning Process

1. **Experience Collection**
   - State observation
   - Action selection
   - Reward calculation
   - Next state observation

2. **Training**
   - Batch sampling from experience replay
   - Q-value computation
   - Loss calculation
   - Network update

3. **Target Network**
   - Separate network for stable Q-value targets
   - Soft updates using tau parameter

## Performance Considerations

- State calculations are vectorized for efficiency
- Tensor operations utilize GPU when available
- Experience replay provides stable learning
- Soft target updates prevent oscillations

## Best Practices

1. **Configuration**
   - Adjust base weights based on desired agent behavior
   - Tune multipliers for specific environments
   - Set appropriate thresholds for state-based decisions

2. **State Design**
   - Normalize all state components to [0,1] range
   - Include relevant environmental information
   - Consider adding custom state features for specific scenarios

3. **Learning**
   - Monitor epsilon decay rate
   - Adjust learning rate if needed
   - Consider reward shaping for desired behaviors

## Troubleshooting

Common issues and solutions:

1. **Unstable Learning**
   - Reduce learning rate
   - Increase replay buffer size
   - Adjust target network update frequency

2. **Poor Action Selection**
   - Check state normalization
   - Verify multiplier values
   - Ensure appropriate exploration rate

3. **Performance Issues**
   - Batch state calculations
   - Use appropriate device (CPU/GPU)
   - Profile state creation bottlenecks

## Future Improvements

Potential enhancements:

1. **Advanced Architectures**
   - Dueling DQN
   - Prioritized Experience Replay
   - Multi-head attention for state processing

2. **Dynamic Adjustment**
   - Adaptive multipliers based on environment
   - Dynamic epsilon decay
   - Meta-learning for parameter optimization

3. **Enhanced State**
   - Historical information
   - Social network features
   - Resource distribution patterns

## References

- Deep Q-Learning: [Original DQN Paper](https://www.nature.com/articles/nature14236)
- Epsilon-Greedy: [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- Neural Networks: [Deep Learning Book](https://www.deeplearningbook.org/) 