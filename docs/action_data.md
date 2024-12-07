
# **Action Retriever Data Definitions**

This document defines the data structures and types used in the **Actions Retriever** module, incorporating advanced analytical capabilities for deep insights into agent behavior, decision-making, and simulation dynamics.

---

## **Core Data Types**

### **ActionStats**
Comprehensive statistics for a specific action type.
```python
ActionStats(
    action_type: str,          # Type of action performed
    count: int,                # Total occurrences
    frequency: float,          # Proportion of total actions
    avg_reward: float,         # Average reward received
    min_reward: float,         # Minimum reward received
    max_reward: float          # Maximum reward received
)
```

### **TimePattern**
Temporal analysis of action patterns.
```python
TimePattern(
    time_distribution: List[int],      # Action counts per time period
    reward_progression: List[float]    # Average rewards per time period
)
```

### **ResourceImpact**
Analysis of how actions affect agent resources.
```python
ResourceImpact(
    avg_resources_before: float,       # Mean resources before action execution
    avg_resource_change: float,        # Average change in resources per action
    resource_efficiency: float         # Resource change per action execution
)
```

### **InteractionStats**
Statistics about agent interactions for a specific action type.
```python
InteractionStats(
    interaction_rate: float,           # Proportion of actions involving other agents
    solo_performance: float,           # Average reward for actions without targets
    interaction_performance: float     # Average reward for actions with targets
)
```

### **ActionAnalysis**
Consolidated analysis for a specific action type.
```python
ActionAnalysis(
    stats: ActionStats,                # Basic action metrics
    time_pattern: TimePattern,         # Temporal patterns
    resource_impact: ResourceImpact,   # Resource effects
    interaction_stats: InteractionStats, # Interaction performance
    sequence_patterns: Dict[str, SequencePattern]  # Action sequence analysis
)
```

---

## **Advanced Analysis Types**

### **CausalAnalysis**
Measures the causal impact of actions on outcomes.
```python
CausalAnalysis(
    action_type: str,                  # Type of action analyzed
    causal_impact: float,              # Average causal effect of the action
    delayed_rewards: List[float],      # Long-term rewards associated with the action
    state_transition_probs: Dict[str, float]  # Probabilities of transitioning to states
)
```

### **BehaviorClustering**
Clustering agents based on behavior.
```python
BehaviorClustering(
    clusters: Dict[int, List[int]],    # Cluster ID mapped to agent IDs
    cluster_centroids: Dict[int, List[float]],  # Centroids of behavioral clusters
    cluster_trends: Dict[int, Dict[str, float]] # Trends within each cluster
)
```

### **ExplorationExploitation**
Analysis of exploration versus exploitation in decision-making.
```python
ExplorationExploitation(
    exploration_rate: float,           # Proportion of unique or novel actions
    exploitation_rate: float,          # Proportion of repeated successful actions
    reward_comparison: Dict[str, float] # Comparison of rewards between new and known actions
)
```

### **AdversarialInteractionAnalysis**
Performance in competitive scenarios.
```python
AdversarialInteractionAnalysis(
    win_rate: float,                   # Proportion of successful adversarial actions
    damage_efficiency: float,          # Reward or resources gained per adversarial action
    counter_strategies: Dict[str, float] # Actions used to counter adversarial behavior
)
```

### **CollaborativeInteractionAnalysis**
Performance in cooperative scenarios.
```python
CollaborativeInteractionAnalysis(
    collaboration_rate: float,         # Proportion of actions involving cooperation
    group_reward_impact: float,        # Rewards gained from collaborative actions
    synergy_metrics: float             # Efficiency of combined actions by agents
)
```

### **LearningCurveAnalysis**
Tracking agent improvement over time.
```python
LearningCurveAnalysis(
    action_success_over_time: List[float], # Improvement in success rates over time
    reward_progression: List[float],       # Changes in average rewards over time
    mistake_reduction: float               # Reduction in suboptimal actions over time
)
```

### **EnvironmentalImpactAnalysis**
Impact of the environment on actions.
```python
EnvironmentalImpactAnalysis(
    environmental_state_impact: Dict[str, float], # Correlation between environment and actions
    adaptive_behavior: Dict[str, float],         # Changes in actions due to environment shifts
    spatial_analysis: Dict[str, float]           # Impact of spatial factors on actions
)
```

### **ConflictAnalysis**
Insights into conflict dynamics.
```python
ConflictAnalysis(
    conflict_trigger_actions: Dict[str, float],  # Actions most likely to lead to conflicts
    conflict_resolution_actions: Dict[str, float], # Actions resolving conflicts
    conflict_outcome_metrics: Dict[str, float]   # Success or resource metrics post-conflict
)
```

### **RiskRewardAnalysis**
Understanding risk-taking behavior.
```python
RiskRewardAnalysis(
    high_risk_actions: Dict[str, float], # Actions with high variability in outcomes
    low_risk_actions: Dict[str, float],  # Actions with consistent rewards
    risk_appetite: float                 # Proportion of risky actions
)
```

### **CounterfactualAnalysis**
Simulating alternative outcomes for actions.
```python
CounterfactualAnalysis(
    counterfactual_rewards: Dict[str, float], # Estimated rewards for alternative actions
    missed_opportunities: Dict[str, float],  # Gains from unused or underused actions
    strategy_comparison: Dict[str, float]    # Performance of alternative strategies
)
```

### **ResilienceAnalysis**
Agent adaptability post-failure.
```python
ResilienceAnalysis(
    recovery_rate: float,           # Time to recover from failure
    adaptation_rate: float,         # Speed of behavioral adjustment
    failure_impact: float           # Resources or rewards lost due to failures
)
```

---

## **Step Analysis Types**

### **StepActionData**
Comprehensive data about actions in a specific simulation step.
```python
StepActionData(
    step_summary: StepSummary,                    # Overall step statistics
    action_statistics: Dict[str, ActionStats],    # Per-action statistics for the step
    resource_metrics: ResourceMetricsStep,        # Resource change analysis
    interaction_network: InteractionNetwork,      # Agent interaction network
    performance_metrics: PerformanceMetrics,      # Success and efficiency metrics
    detailed_actions: List[Dict]                 # Detailed list of actions in raw format
)
```

### **StepSummary**
Overall statistics for a simulation step.
```python
StepSummary(
    total_actions: int,           # Total actions in the step
    unique_agents: int,           # Number of active agents
    action_types: int,            # Number of different action types
    total_interactions: int,      # Number of agent interactions
    total_reward: float           # Total reward accumulated
)
```

### **ResourceMetricsStep**
Resource changes during a simulation step.
```python
ResourceMetricsStep(
    net_resource_change: float,       # Total resource change
    average_resource_change: float,   # Average resource change per action
    resource_transactions: int        # Number of resource changes
)
```

### **InteractionNetwork**
Network of agent interactions during a step.
```python
InteractionNetwork(
    interactions: List[Dict],         # List of interaction events
    unique_interacting_agents: int    # Number of unique agents involved in interactions
)
```

### **PerformanceMetrics**
Success and efficiency metrics for a step.
```python
PerformanceMetrics(
    success_rate: float,         # Proportion of successful actions
    average_reward: float,       # Mean reward per action
    action_efficiency: float     # Proportion of state-changing actions
)
```