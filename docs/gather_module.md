# Gather Module Documentation

The Gather Module is a resource gathering optimization system that leverages Deep Q-Learning (DQN) to enable agents to learn optimal gathering strategies. This module is designed to make intelligent decisions about when and where to gather resources based on environmental conditions, resource availability, and agent needs.

---

## Overview

The Gather Module implements a Deep Q-Learning approach with several enhancements:

- **Intelligent Resource Selection**: Evaluates and ranks potential resource nodes based on multiple factors
- **Temporal Decision Making**: Can choose to wait for better opportunities
- **Efficiency-Based Rewards**: Rewards efficient gathering with minimal movement costs
- **Adaptive Behavior**: Learns from past gathering experiences
- **Resource-Aware**: Considers resource regeneration rates and density

---

## Key Components

### 1. `GatherQNetwork`

Neural network architecture for Q-value approximation of gathering decisions:

- **Input Layer**: 6-dimensional state vector representing:
  - Distance to nearest resource
  - Resource amount
  - Agent's current resources
  - Resource density in area
  - Steps since last gather
  - Resource regeneration rate

- **Hidden Layers**: Two layers with configurable size (default 64 neurons each)
  - Layer Normalization
  - ReLU activation
  - Dropout (10%)
  - Xavier/Glorot initialization

- **Output Layer**: 3 actions
  - GATHER: Attempt to gather resources
  - WAIT: Wait for better opportunity
  - SKIP: Skip gathering this step

### 2. `GatherModule`

Main class handling gathering decisions and learning:

**Key Methods:**
- `get_gather_decision(agent, state)`: Determines whether to gather and from which resource
- `_process_gather_state(agent)`: Creates state representation for gathering decisions
- `_find_best_resource(agent)`: Identifies optimal resource node for gathering
- `calculate_gather_reward(agent, initial_resources, target_resource)`: Computes gathering rewards

**Features:**
- Tracks consecutive failed attempts
- Manages waiting periods
- Calculates gathering efficiency
- Handles experience storage and training

### 3. Experience Replay

Stores gathering experiences for stable learning:
- State: 6-dimensional vector
- Action: GATHER, WAIT, or SKIP
- Reward: Based on gathering success and efficiency
- Next State: Resulting environment state

### 4. Reward System

Complex reward structure considering multiple factors:
- Base success/failure rewards
- Efficiency multipliers
- Movement cost penalties
- Consecutive failure penalties

---

## Technical Details

### State Space (6 dimensions)
1. **Distance to Resource**: Normalized distance to nearest viable resource
2. **Resource Amount**: Available amount at nearest resource
3. **Agent Resources**: Current agent resource level
4. **Resource Density**: Local resource concentration
5. **Steps Since Gather**: Time since last successful gathering
6. **Regeneration Rate**: Resource replenishment rate

### Action Space (3 discrete actions)
- **GATHER (0)**: Attempt to gather from best available resource
- **WAIT (1)**: Wait for better gathering opportunity
- **SKIP (2)**: Skip gathering this step

### Configuration Parameters (`GatherConfig`)

**Reward Parameters:**
```python
gather_success_reward: float = 1.0
gather_fail_penalty: float = -0.1
gather_efficiency_multiplier: float = 0.5
gather_cost_multiplier: float = 0.3
```

**Gathering Parameters:**
```python
min_resource_threshold: float = 0.1
max_wait_steps: int = 5
```

**Learning Parameters:**
```python
learning_rate: float = 0.001
memory_size: int = 10000
gamma: float = 0.99
epsilon_start: float = 1.0
epsilon_min: float = 0.01
epsilon_decay: float = 0.995
```

---

## Usage Example

```python
# Initialize configuration
config = GatherConfig(
    gather_efficiency_multiplier=0.5,
    gather_cost_multiplier=0.3,
    min_resource_threshold=0.1,
    max_wait_steps=5
)

# Create gather module
gather_module = GatherModule(config)

# Get gathering decision
should_gather, target_resource = gather_module.get_gather_decision(agent, state)

# Execute gathering if appropriate
if should_gather and target_resource:
    initial_resources = agent.resource_level
    gather_amount = min(agent.config.max_gather_amount, target_resource.amount)
    target_resource.consume(gather_amount)
    agent.resource_level += gather_amount
    
    # Calculate and store reward
    reward = gather_module.calculate_gather_reward(
        agent, initial_resources, target_resource
    )
    
    # Store experience
    gather_module.store_experience(state, action, reward, next_state, done=False)
```

---

## Integration with Agents

### System Agent Configuration
```python
# Sustainable gathering strategy
gather_module.config.gather_efficiency_multiplier = 0.4
gather_module.config.gather_cost_multiplier = 0.4
gather_module.config.min_resource_threshold = 0.2
```

### Independent Agent Configuration
```python
# Aggressive gathering strategy
gather_module.config.gather_efficiency_multiplier = 0.7
gather_module.config.gather_cost_multiplier = 0.2
gather_module.config.min_resource_threshold = 0.05
```

---

## Best Practices

1. **Resource Evaluation**
   - Consider both quantity and accessibility
   - Account for regeneration rates
   - Factor in competition from other agents

2. **Efficiency Optimization**
   - Minimize movement costs
   - Maximize gathering amounts
   - Balance waiting vs. gathering

3. **Learning Management**
   - Maintain sufficient exploration
   - Regular training updates
   - Monitor reward trends

4. **State Representation**
   - Normalize input values
   - Include temporal information
   - Consider local resource density

---

## Troubleshooting

### Common Issues

1. **Inefficient Gathering**
   - Check resource thresholds
   - Adjust efficiency multipliers
   - Review movement cost penalties

2. **Excessive Waiting**
   - Reduce max_wait_steps
   - Adjust wait rewards
   - Check resource regeneration rates

3. **Poor Learning**
   - Verify reward scaling
   - Check state normalization
   - Adjust learning rate

---

## Performance Considerations

1. **Computational Efficiency**
   - Vectorized distance calculations
   - Efficient resource scoring
   - Batch processing for learning

2. **Memory Management**
   - Limited experience replay buffer
   - Efficient state representation
   - Regular memory cleanup

3. **Learning Stability**
   - Gradual epsilon decay
   - Regular target network updates
   - Balanced reward structure

---

## Future Extensions

1. **Enhanced Decision Making**
   - Multi-resource gathering strategies
   - Cooperative gathering behaviors
   - Dynamic threshold adjustment

2. **Advanced Learning**
   - Prioritized experience replay
   - Dueling network architecture
   - Multi-step learning

3. **Environmental Adaptation**
   - Dynamic resource prediction
   - Competition awareness
   - Territory-based gathering

---

## References

1. **Deep Q-Learning**: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
2. **Resource Management**: [Optimal foraging theory](https://en.wikipedia.org/wiki/Optimal_foraging_theory)
3. **Multi-Agent Systems**: [Multiagent Systems](https://mitpress.mit.edu/books/multiagent-systems) 