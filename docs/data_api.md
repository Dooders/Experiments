# Data API Documentation

## Introduction

This document provides an overview of the Data API being developed, which is designed to support simulation state persistence, analysis, and data retrieval functionalities. The API is structured into several key components, including the Database module, Analysis module, Repositories, and Services. Each component plays a vital role in managing data flow and processing within the simulation environment.

## Overview of the Data API Components

### Database Module (`database`)

The Database module is responsible for handling all interactions with the underlying database using SQLAlchemy ORM. It provides classes and methods to persist simulation data, manage configurations, and ensure data integrity. Key components of the Database module include:

- **`SimulationDatabase`**: The main database interface class that manages connections, sessions, and provides high-level methods for data operations.

- **Models** (`database/models.py`): Defines the ORM models representing different entities in the simulation, such as `Agent`, `AgentState`, `ResourceState`, `SimulationStep`, and others.

- **Data Retrieval** (`database/data_retrieval.py`): Contains classes like `DataRetriever` that facilitate querying and fetching data from the database for analysis purposes.

- **Utilities** (`database/utilities.py`): Provides helper functions for database schema creation, execution with retry mechanisms, data formatting, and validation.

#### Example: `SimulationDatabase` Class

```python
class SimulationDatabase:
    """
    Database interface for simulation state persistence and analysis.

    This class provides a high-level interface for storing and retrieving simulation
    data using SQLAlchemy ORM. It handles all database operations including state
    logging, configuration management, and data analysis with transaction safety
    and efficient batch operations.
    """

    def __init__(self, db_path: str) -> None:
        # Initialization code...
```

### Analysis Module (`analysis`)

The Analysis module comprises a set of analyzers that process simulation data to extract insights, patterns, and metrics. These analyzers use data retrieved from the database to perform various forms of analysis, such as statistical summaries, pattern recognition, and performance evaluation. Key analyzers include:

- **`ActionStatsAnalyzer`** (`analysis/action_stats_analyzer.py`): Analyzes agent actions to generate metrics like frequency, rewards, and interaction rates.

- **`BehaviorClusteringAnalyzer`** (`analysis/behavior_clustering_analyzer.py`): Clusters agents based on their behavioral patterns derived from action sequences.

- **`CausalAnalyzer`** (`analysis/causal_analyzer.py`): Examines causal relationships between agent actions and their outcomes.

- **`DecisionPatternAnalyzer`** (`analysis/decision_pattern_analyzer.py`): Identifies decision-making patterns among agents.

- **`ResourceImpactAnalyzer`** (`analysis/resource_impact_analyzer.py`): Calculates resource-related metrics resulting from agent actions.

- **`SequencePatternAnalyzer`** (`analysis/sequence_pattern_analyzer.py`): Analyzes sequences of actions to find common patterns.

- **`TemporalPatternAnalyzer`** (`analysis/temporal_pattern_analyzer.py`): Identifies temporal trends in agent behaviors over time.

#### Example: `ActionStatsAnalyzer` Class

```python
class ActionStatsAnalyzer:
    """
    Analyzes statistics and patterns of agent actions in a simulation.

    This class processes action data to generate metrics including frequency, rewards,
    interaction rates, and various patterns of agent behavior.
    """

    def __init__(self, repository: AgentActionRepository):
        # Initialization code...
```

### Repositories (`database/repositories`)

Repositories act as data access layers that encapsulate the logic required to access data sources. They provide methods to query the database and retrieve data in a structured way, helping to decouple data access from business logic. Key repositories include:

- **`AgentActionRepository`** (`database/repositories/agent_action_repository.py`): Provides methods to query agent actions based on various filters and scopes.

#### Example: `AgentActionRepository` Class

```python
class AgentActionRepository:
    """
    Repository class for managing agent action records in the database.

    This class provides methods to query and retrieve agent actions based on various
    filtering criteria such as scope, agent ID, and step numbers.
    """

    def __init__(self, session_manager: SessionManager):
        # Initialization code...

    def get_actions_by_scope(
        self,
        scope: str,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[AgentActionData]:
        # Method implementation...
```

### Services (`services`)

Services are high-level classes that orchestrate complex operations by coordinating between multiple components of the system. They act as an abstraction layer between the application logic and the underlying implementation details, providing a unified interface for performing various analyses.

- **`ActionsService`** (`services/actions_service.py`): Orchestrates analysis of agent actions using various analyzers. Provides methods to perform comprehensive analysis and to obtain action summaries.

#### Example: `ActionsService` Class

```python
class ActionsService:
    """
    High-level service for analyzing agent actions using various analyzers.

    This service orchestrates different types of analysis on agent actions including:
    - Basic action statistics and metrics
    - Behavioral patterns and clustering
    - Causal relationships
    - Decision patterns
    - Resource impacts
    - Action sequences
    - Temporal patterns
    """

    def __init__(self, action_repository: AgentActionRepository):
        # Initialization code...

    def analyze_actions(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
        analysis_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        # Method implementation...
```

### Services Module (`services/__init__.py`)

The `__init__.py` file in the `services` module provides an easy way to import key services and includes module-level documentation.

```python
"""
Services Module

This module contains high-level service classes that orchestrate complex operations by coordinating
between multiple components of the system. Services act as an abstraction layer between the
application logic and the underlying implementation details.

Key Services:
-------------
ActionsService
    Orchestrates analysis of agent actions using various analyzers. Provides a unified interface
    for analyzing action patterns, behaviors, resource impacts, and other metrics.
"""

from services.actions_service import ActionsService

__all__ = ['ActionsService']
```

## Interaction Between Components

The Data API is designed with a modular architecture where each component interacts with others to perform its functions:

- The **Services** use **Repositories** to access data from the **Database**.

- **Analyzers** in the **Analysis Module** rely on **Repositories** to retrieve data required for analysis.

- **Services** coordinate **Analyzers** to perform complex operations and provide high-level interfaces for application logic.

- The **Database Module** provides the foundational data models and data access mechanisms that the other components build upon.

## Usage Example

Here's an example of how the components might be used together to perform an analysis:

```python
from database.session_manager import SessionManager
from database.repositories.agent_action_repository import AgentActionRepository
from services.actions_service import ActionsService

# Initialize session manager and repositories
session_manager = SessionManager('sqlite:///simulation_results.db')
action_repository = AgentActionRepository(session_manager)

# Initialize services
actions_service = ActionsService(action_repository)

# Perform comprehensive analysis on agent actions
results = actions_service.analyze_actions(
    scope='SIMULATION',
    agent_id=123,
    analysis_types=['stats', 'behavior', 'causal']
)

# Access analysis results
action_stats = results.get('action_stats')
behavior_clusters = results.get('behavior_clusters')
causal_analysis = results.get('causal_analysis')
```

## Conclusion

The Data API provides a structured and extensible framework for managing simulation data and performing analyses. By separating concerns across the Database, Analysis, Repositories, and Services modules, the API facilitates maintainability and scalability. Users can leverage the high-level Services to perform complex analyses without needing to delve into the underlying data access and processing logic.