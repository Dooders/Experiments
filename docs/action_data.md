# **Action Retriever Data Definitions**

This document defines the data structures and types used in the **Actions Retriever** module for analyzing agent behavior and simulation dynamics.

---

## **Core Data Types**

### **ActionMetrics**
Basic statistics for a specific action type.
```python
ActionMetrics(
    action_type: str,          # Type of action performed
    count: int,                # Total occurrences
    frequency: float,          # Proportion of total actions (0.0 to 1.0)
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
    avg_resource_change: float,        # Average change in resources from action
    resource_efficiency: float         # Resource change per action execution
)
```

### **InteractionStats**
Statistics about agent interactions.
```python
InteractionStats(
    action_type: str,                  # Type of action being analyzed
    interaction_rate: float,           # Proportion of actions involving other agents
    solo_performance: float,           # Average reward for actions without targets
    interaction_performance: float     # Average reward for actions with targets
)
```

### **DecisionPatterns**
Comprehensive analysis of decision-making patterns.
```python
DecisionPatterns(
    decision_patterns: Dict[str, DecisionPatternStats],  # Statistics per action type
    sequence_analysis: Dict[str, SequencePattern],       # Action sequence analysis
    resource_impact: Dict[str, ResourceImpact],         # Resource effects
    temporal_patterns: Dict[str, TimePattern],          # Time-based patterns
    interaction_analysis: Dict[str, InteractionStats],  # Interaction statistics
    decision_summary: DecisionSummary                   # Overall metrics
)
```

### **DecisionPatternStats**
Detailed statistics for each action type.
```python
DecisionPatternStats(
    count: int,                        # Total number of times action was taken
    frequency: float,                  # Proportion of times action was chosen
    reward_stats: Dict[str, float]     # Average/min/max rewards
)
```

### **SequencePattern**
Analysis of action sequences.
```python
SequencePattern(
    count: int,                        # Number of times sequence occurred
    probability: float                 # Likelihood of sequence occurring
)
```

---

## **Step Analysis Types**

### **StepActionData**
Comprehensive data about actions in a specific step.
```python
StepActionData(
    step_summary: StepSummary,                    # Overall step statistics
    action_statistics: Dict[str, Dict],           # Per-action statistics
    resource_metrics: ResourceMetricsStep,        # Resource changes
    interaction_network: InteractionNetwork,      # Agent interactions
    performance_metrics: PerformanceMetrics,      # Success metrics
    detailed_actions: List[Dict]                  # Raw action details
)
```

### **StepSummary**
Overall statistics for a simulation step.
```python
StepSummary(
    total_actions: int,           # Total actions taken
    unique_agents: int,           # Number of active agents
    action_types: int,            # Unique action types used
    total_interactions: int,      # Number of interactions
    total_reward: float          # Total reward accumulated
)
```

### **ResourceMetricsStep**
Resource changes during a step.
```python
ResourceMetricsStep(
    net_resource_change: float,       # Total resource change
    average_resource_change: float,   # Average change per action
    resource_transactions: int        # Number of transactions
)
```

### **InteractionNetwork**
Network of agent interactions.
```python
InteractionNetwork(
    interactions: List[Dict],         # List of interaction events
    unique_interacting_agents: int    # Number of unique interacting agents
)
```

### **PerformanceMetrics**
Success and efficiency metrics.
```python
PerformanceMetrics(
    success_rate: float,         # Proportion of successful actions
    average_reward: float,       # Mean reward per action
    action_efficiency: float     # Proportion of effective actions
)
```

---

## **Analysis Types**

### **ExplorationExploitation**
Analysis of exploration vs exploitation.
```python
ExplorationExploitation(
    exploration_rate: float,           # Rate of new action attempts
    exploitation_rate: float,          # Rate of repeated actions
    reward_comparison: Dict[str, float] # New vs known action rewards
)
```

### **ResilienceAnalysis**
Analysis of recovery from failures.
```python
ResilienceAnalysis(
    recovery_rate: float,           # Steps needed to recover
    adaptation_rate: float,         # Rate of strategy modification
    failure_impact: float           # Performance impact of failures
)
```

### **CollaborativeInteractionAnalysis**
Analysis of cooperative behaviors.
```python
CollaborativeInteractionAnalysis(
    collaboration_rate: float,         # Rate of cooperative actions
    group_reward_impact: float,        # Benefit from collaboration
    synergy_metrics: float            # Collaborative vs solo performance
)
```

### **AdversarialInteractionAnalysis**
Analysis of competitive behaviors.
```python
AdversarialInteractionAnalysis(
    win_rate: float,                   # Success rate in conflicts
    damage_efficiency: float,          # Effectiveness of actions
    counter_strategies: Dict[str, float] # Response patterns
)
```