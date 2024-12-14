# Actions Analysis Module

The actions module provides comprehensive analysis of agent actions, interactions, and decision-making patterns within the simulation database. It enables deep insights into agent behaviors, strategies, and performance.

## Key Features

- Action pattern analysis and metrics calculation
- Decision-making behavior tracking and analysis
- Agent interaction statistics and network analysis  
- Temporal pattern recognition and trend analysis
- Resource impact assessment and efficiency metrics
- Step-by-step action monitoring and analysis
- Advanced behavioral analytics including:
  - Causal analysis
  - Exploration/exploitation patterns
  - Collaborative and adversarial interactions
  - Learning curves and adaptation
  - Risk/reward analysis
  - Resilience and recovery patterns

## Core Classes

### ActionsRetriever

Main class for retrieving and analyzing action data. Provides methods for:

- Basic action retrieval and filtering
- Statistical analysis and metrics
- Pattern recognition and clustering
- Resource impact assessment
- Decision-making analysis
- Behavioral profiling

Key methods:

```python
def actions(scope="simulation", agent_id=None, step=None, step_range=None) -> List[AgentActionData]
"""Retrieve filtered action data with complete metadata"""

def action_stats(scope="simulation", agent_id=None, step=None, step_range=None) -> List[ActionMetrics]
"""Get comprehensive statistics for each action type"""

def temporal_patterns(scope="simulation", agent_id=None, step_range=None) -> List[TimePattern]
"""Analyze action patterns over time"""

def resource_impacts(scope="simulation", agent_id=None, step=None, step_range=None) -> List[ResourceImpact]
"""Analyze resource impacts of different actions"""

def decision_patterns(scope="simulation", agent_id=None, step=None, step_range=None) -> DecisionPatterns
"""Analyze comprehensive decision-making patterns"""

def sequence_patterns(scope="simulation", agent_id=None, step=None, step_range=None) -> List[SequencePattern]
"""Analyze sequential action patterns and transitions"""

def causal_analysis(action_type: str) -> CausalAnalysis
"""Analyze cause-effect relationships for actions"""

def behavior_clustering() -> BehaviorClustering
"""Group agents by behavioral patterns and strategies"""
```

### AnalysisScope

Enum defining valid analysis scope levels:

- SIMULATION: All data without filtering
- STEP: Single step analysis  
- STEP_RANGE: Analysis over step range
- AGENT: Single agent analysis

## Data Types

- **AgentActionData**: Structured representation of individual actions
- **ActionMetrics**: Statistical metrics for action types
- **TimePattern**: Temporal evolution patterns
- **ResourceImpact**: Resource consumption/generation metrics
- **DecisionPatterns**: Decision-making analysis results
- **SequencePattern**: Action sequence statistics
- **CausalAnalysis**: Cause-effect relationship data
- **BehaviorClustering**: Agent behavioral groupings

## Usage Examples

```python
from database.actions import ActionsRetriever

# Initialize retriever
retriever = ActionsRetriever(session)

# Get action statistics
stats = retriever.action_stats()
for metric in stats:
    print(f"{metric.action_type}: {metric.avg_reward:.2f}")

# Analyze temporal patterns
patterns = retriever.temporal_patterns()
for pattern in patterns:
    print(f"{pattern.action_type} trend:")
    print(pattern.time_distribution)

# Cluster agent behaviors
clusters = retriever.behavior_clustering()
for strategy, agents in clusters.clusters.items():
    print(f"{strategy}: {len(agents)} agents")
```

## Analysis Scopes

All analysis methods support multiple scoping options:

- "simulation": Analyze all data (no filters)
- "step": Analyze specific step
- "step_range": Analyze range of steps  
- "agent": Analyze specific agent

## Dependencies

- sqlalchemy: Database ORM and query building
- numpy: Numerical computations and analysis
- pandas: Data manipulation and analysis
- scipy: Statistical analysis and clustering

## Notes

- All analysis methods support flexible scope filtering
- Heavy computations are optimized through database queries
- Results are returned as structured data types
- Analysis methods handle missing or incomplete data