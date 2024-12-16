# ActionStatsAnalyzer

The `ActionStatsAnalyzer` class processes and analyzes statistics and patterns of agent actions in a simulation environment. It provides comprehensive metrics about action frequency, rewards, interaction rates, and various behavioral patterns.

## Overview

This analyzer aggregates and processes action data to generate detailed metrics about agent behavior, including:
- Action frequency and counts
- Reward statistics
- Agent interaction rates
- Performance metrics for solo vs. interactive actions
- Temporal patterns
- Resource impacts
- Decision-making patterns

## Usage

```python
from database.repositories.agent_action_repository import AgentActionRepository
from analysis.action_stats_analyzer import ActionStatsAnalyzer

# Initialize
repository = AgentActionRepository()
analyzer = ActionStatsAnalyzer(repository)

# Analyze all simulation actions
metrics = analyzer.analyze()

# Analyze specific agent's actions
agent_metrics = analyzer.analyze(agent_id=1)

# Analyze specific time period
step_metrics = analyzer.analyze(step_range=(100, 200))
```

## Methods

### `analyze(scope=AnalysisScope.SIMULATION, agent_id=None, step=None, step_range=None)`

Analyzes action statistics based on specified scope and filters.

#### Parameters:
- `scope` (Union[str, AnalysisScope]): Scope of analysis (default: AnalysisScope.SIMULATION)
- `agent_id` (Optional[int]): Specific agent to analyze
- `step` (Optional[int]): Specific step to analyze
- `step_range` (Optional[Tuple[int, int]]): Range of steps to analyze

#### Returns:
List[ActionMetrics] containing metrics for each action type:
- `action_type`: Type of the analyzed action
- `count`: Total occurrences
- `frequency`: Relative frequency (0-1)
- `avg_reward`: Mean reward received
- `min_reward`: Minimum reward received
- `max_reward`: Maximum reward received
- `interaction_rate`: Proportion of interactive actions
- `solo_performance`: Average reward for non-interactive actions
- `interaction_performance`: Average reward for interactive actions
- `temporal_patterns`: Timing and sequence patterns
- `resource_impacts`: Resource utilization effects
- `decision_patterns`: Decision-making patterns
- `variance_reward`: Variance of rewards
- `std_dev_reward`: Standard deviation of rewards
- `median_reward`: Median value of rewards
- `quartiles_reward`: First and third quartiles (25th and 75th percentiles) of rewards
- `confidence_interval`: 95% confidence interval for the average reward

## Example Output

```python
[
    ActionMetrics(
        action_type="gather",
        count=100,
        frequency=0.4,                # 40% of all actions
        avg_reward=2.5,               # Average reward of +2.5
        min_reward=0.0,
        max_reward=5.0,
        interaction_rate=0.1,         # 10% involved other agents
        solo_performance=2.7,         # Average reward when alone
        interaction_performance=1.2,  # Average reward with others
        temporal_patterns=[...],      
        resource_impacts=[...],       
        decision_patterns=[...],       
        variance_reward=0.2,          # Variance of rewards
        std_dev_reward=0.447,         # Standard deviation of rewards
        median_reward=2.5,            # Median value of rewards
        quartiles_reward=[0.0, 5.0],   # First and third quartiles of rewards
        confidence_interval=[2.0, 3.0] # 95% confidence interval for the average reward
    ),
    # Additional action types...
]
```

## Notes

- All frequency and rate values are expressed as decimals between 0 and 1
- Performance metrics are only calculated for actions with valid rewards
- Pattern analysis includes detailed examination of behavior sequences and context
- The analyzer integrates with TemporalPatternAnalyzer, ResourceImpactAnalyzer, and DecisionPatternAnalyzer for comprehensive analysis

## Dependencies

- `AgentActionRepository`: For accessing agent action data
- `TemporalPatternAnalyzer`: For analyzing timing patterns
- `ResourceImpactAnalyzer`: For analyzing resource effects
- `DecisionPatternAnalyzer`: For analyzing decision patterns
