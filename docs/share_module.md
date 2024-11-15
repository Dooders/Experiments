# Share Module Documentation

The Share Module is a cooperative behavior learning system that uses Deep Q-Learning (DQN) to enable agents to develop optimal resource sharing strategies. This module implements intelligent sharing policies that consider resource levels, agent relationships, and environmental conditions.

---

## Overview

The Share Module implements a Deep Q-Learning approach with several enhancements:

- **Intelligent Target Selection**: Uses weighted selection based on need and cooperation history
- **Multi-Level Sharing**: Supports different sharing amounts (low/medium/high)
- **Cooperation Tracking**: Maintains history of sharing interactions
- **Experience Replay**: Stores past sharing experiences for stable learning
- **Reward Shaping**: Encourages beneficial sharing through structured rewards

---

## Key Components

### 1. `ShareQNetwork`

Neural network architecture for share decision making:

- **Input Layer**: 6-dimensional state vector representing:
  - Agent's normalized resource level
  - Proportion of nearby agents
  - Average neighbor resources
  - Minimum neighbor resources
  - Maximum neighbor resources
  - Cooperation score

- **Hidden Layers**: Two layers (64 neurons each) with:
  - Layer Normalization
  - ReLU activation
  - Dropout (10%)
  - Xavier/Glorot initialization

- **Output Layer**: 4 actions:
  - NO_SHARE
  - SHARE_LOW
  - SHARE_MEDIUM
  - SHARE_HIGH

### 2. `ShareModule`

Main class handling sharing decisions and learning:

- **Cooperation Tracking**: 
  - Maintains history of sharing interactions
  - Calculates cooperation scores
  - Adapts sharing behavior based on past interactions

- **Target Selection**:
  - Identifies nearby agents within sharing range
  - Weights selection based on need and cooperation history
  - Considers resource levels and starvation risk

- **Share Amount Calculation**:
  - Determines appropriate share amounts
  - Ensures minimum resource retention
  - Scales sharing based on action level

---

## Configuration Parameters (`ShareConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `share_range` | 30.0 | Maximum distance for sharing interactions |
| `min_share_amount` | 1 | Minimum resources that can be shared |
| `share_success_reward` | 0.3 | Base reward for successful sharing |
| `share_failure_penalty` | -0.1 | Penalty for failed sharing attempts |
| `altruism_bonus` | 0.2 | Extra reward for helping needy agents |
| `cooperation_memory` | 100 | Number of past interactions to remember |
| `max_resources` | 30 | Maximum agent resources (for normalization) |

---

## State Representation

The share module uses a 6-dimensional state vector:

1. **Normalized Agent Resources**: `agent.resource_level / max_resources`
2. **Nearby Agents Ratio**: `len(nearby_agents) / len(total_agents)`
3. **Average Neighbor Resources**: Mean resources of nearby agents
4. **Minimum Neighbor Resources**: Lowest resources among neighbors
5. **Maximum Neighbor Resources**: Highest resources among neighbors
6. **Cooperation Score**: Historical cooperation rating

---

## Action Space

The module supports four discrete actions:

```python
class ShareActionSpace:
    NO_SHARE: int = 0    # Don't share resources
    SHARE_LOW: int = 1   # Share minimum amount
    SHARE_MEDIUM: int = 2  # Share moderate amount
    SHARE_HIGH: int = 3   # Share larger amount
```

---

## Reward System

The reward calculation considers multiple factors:

1. **Base Rewards**:
   - Successful sharing: +0.3
   - Failed attempt: -0.1

2. **Bonuses**:
   - Altruism bonus (helping needy): +0.2
   - Scaled by share amount

3. **Cooperation Impact**:
   - Successful sharing increases cooperation score
   - Affects future interaction probabilities

---

## Usage Example

```python
# Initialize module with config
config = ShareConfig(
    share_range=30.0,
    min_share_amount=1,
    share_success_reward=0.3,
    altruism_bonus=0.2
)
share_module = ShareModule(config)

# Get sharing decision
state = _get_share_state(agent)
action, target, amount = share_module.get_share_decision(agent, state)

# Execute sharing if applicable
if target and amount > 0:
    agent.resource_level -= amount
    target.resource_level += amount
    
    # Update cooperation history
    share_module.update_cooperation(target.agent_id, True)
```

---

## Technical Details

### Cooperation Tracking

The module maintains a cooperation history for each agent:

```python
def update_cooperation(self, agent_id: int, cooperative: bool):
    """Update cooperation history for an agent."""
    if agent_id not in self.cooperation_history:
        self.cooperation_history[agent_id] = []
    self.cooperation_history[agent_id].append(
        1.0 if cooperative else -1.0
    )
```

### Target Selection

Agents are selected based on weighted probabilities:

```python
def _select_target(self, agent, nearby_agents):
    """Select target based on need and cooperation history."""
    weights = []
    for target in nearby_agents:
        weight = 1.0
        # Increase weight for needy agents
        if target.resource_level < target.config.starvation_threshold:
            weight *= 2.0
        # Consider past cooperation
        weight *= (1.0 + self._get_cooperation_score(target.agent_id))
        weights.append(weight)
    return np.random.choice(nearby_agents, p=weights/sum(weights))
```

---

## Integration

### With Base Agent

```python
class BaseAgent:
    def __init__(self, ...):
        # Initialize share module
        self.share_module = ShareModule(DEFAULT_SHARE_CONFIG)
```

### Share Action Function

```python
def share_action(agent):
    """Execute sharing action using learned policy."""
    state = _get_share_state(agent)
    action, target, amount = agent.share_module.get_share_decision(
        agent, state
    )
    
    if target and amount > 0:
        # Execute sharing
        agent.resource_level -= amount
        target.resource_level += amount
        
        # Calculate and store reward
        reward = _calculate_share_reward(agent, target, amount)
        agent.share_module.store_experience(
            state, action, reward, _get_share_state(agent), False
        )
```

---

## Performance Considerations

1. **Memory Management**:
   - Fixed-size cooperation history
   - Efficient state calculations
   - Vectorized distance computations

2. **Computational Efficiency**:
   - Cached cooperation scores
   - Optimized nearby agent detection
   - Batch processing for learning updates

3. **Learning Stability**:
   - Experience replay buffer
   - Soft target network updates
   - Gradient clipping

---

## Best Practices

1. **Configuration**:
   - Adjust share range based on environment size
   - Balance rewards to encourage cooperation
   - Set appropriate memory size for cooperation history

2. **Integration**:
   - Initialize module early in agent lifecycle
   - Update cooperation scores regularly
   - Monitor sharing patterns

3. **Monitoring**:
   - Track cooperation scores
   - Monitor resource distribution
   - Analyze sharing patterns

---

## Troubleshooting

Common issues and solutions:

1. **Low Sharing Rate**:
   - Check share range
   - Adjust rewards
   - Verify resource availability

2. **Unstable Learning**:
   - Adjust learning rate
   - Modify batch size
   - Check reward scaling

3. **Poor Cooperation**:
   - Increase altruism bonus
   - Adjust cooperation memory
   - Review target selection weights

---

## Future Enhancements

Potential improvements:

1. **Advanced Cooperation**:
   - Reciprocal sharing tracking
   - Group-based cooperation
   - Dynamic reward adjustment

2. **Learning Enhancements**:
   - Prioritized experience replay
   - Multi-agent learning
   - Meta-learning for adaptation

3. **State Representation**:
   - Additional environmental features
   - Social network metrics
   - Historical sharing patterns

---

## References

- **Deep Q-Learning**: [Original DQN Paper](https://www.nature.com/articles/nature14236)
- **Cooperation in Multi-Agent Systems**: [Survey Paper](https://www.jair.org/index.php/jair/article/view/10166)
- **Resource Sharing in MAS**: [Overview](https://www.sciencedirect.com/science/article/abs/pii/S0004370215001277) 