# Experiments
Repository for all specific experiments and tests

## Refactored Directory Structure

The `ActionsRetriever` module has been refactored into smaller, more manageable components. The new directory structure is as follows:

```
database/
    models.py
    session_manager.py
    scope_utils.py
    repositories/
        agent_action_repository.py
analysis/
    __init__.py
    action_stats_analyzer.py
    temporal_pattern_analyzer.py
    resource_impact_analyzer.py
    decision_pattern_analyzer.py
    sequence_pattern_analyzer.py
    causal_analyzer.py
    behavior_clustering_analyzer.py
data_types.py
enums.py
actions_retriever.py
```

## ActionsRetriever Usage Example

The `ActionsRetriever` class has been refactored to use the new structure. Here is an example of how to use it:

```python
from database.actions import ActionsRetriever
from database.session_manager import SessionManager

# Initialize the session manager
session_manager = SessionManager()

# Create a new session
session = session_manager.create_session()

# Initialize the ActionsRetriever
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

## Summary of Changes

- The `ActionsRetriever` module has been refactored into smaller, more manageable components.
- A dedicated data access layer (`AgentActionRepository`) has been introduced for database interactions.
- Analysis logic has been separated into distinct analyzers under the `analysis/` directory.
- Scope filtering logic has been moved to a new `database/scope_utils.py` file.
- Data transfer objects and enums have been standardized and moved to `data_types.py` and `enums.py`.

## Benefits

- Improved maintainability and readability by separating concerns (data retrieval, analysis, data models).
- Easier testing with clearer boundaries and dependencies.
- Enhanced extensibility for adding new analysis methods and data structures.
- Standardized scope filtering and data queries.
