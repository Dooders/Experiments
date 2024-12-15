"""Actions retrieval and analysis module for simulation database.
#! This module is deprecated and will be removed in the future.
#! Use the services module instead. Which is a higher level of abstraction.
#! Takes the repository and provides a high level interface for analysis.
This module provides comprehensive analysis of agent actions, interactions, and decision-making
patterns within the simulation. The ActionsRetriever class offers specialized queries and 
analysis methods for extracting insights from action-related data.

Key Features
-----------
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

Classes
-------
ActionsRetriever
    Main class handling action data retrieval and analysis. Provides methods for:
    - Basic action retrieval and filtering
    - Statistical analysis and metrics
    - Pattern recognition and clustering
    - Resource impact assessment
    - Decision-making analysis
    - Behavioral profiling

AnalysisScope
    Enum class defining valid analysis scope levels:
    - SIMULATION: All data without filtering
    - STEP: Single step analysis
    - STEP_RANGE: Analysis over step range
    - AGENT: Single agent analysis

Data Types
----------
AgentActionData
    Structured representation of individual actions

ActionMetrics
    Statistical metrics for action types

TimePattern
    Temporal evolution patterns

ResourceImpact
    Resource consumption/generation metrics

DecisionPatterns
    Decision-making analysis results

SequencePattern
    Action sequence statistics

CausalAnalysis
    Cause-effect relationship data

BehaviorClustering
    Agent behavioral groupings

Examples
--------
>>> from database.actions import ActionsRetriever
>>> retriever = ActionsRetriever(session)

>>> # Get action statistics
>>> stats = retriever.action_stats()
>>> for metric in stats:
...     print(f"{metric.action_type}: {metric.avg_reward:.2f}")

>>> # Analyze temporal patterns
>>> patterns = retriever.temporal_patterns()
>>> for pattern in patterns:
...     print(f"{pattern.action_type} trend:")
...     print(pattern.time_distribution)

>>> # Cluster agent behaviors
>>> clusters = retriever.behavior_clustering()
>>> for strategy, agents in clusters.clusters.items():
...     print(f"{strategy}: {len(agents)} agents")

Dependencies
-----------
- sqlalchemy: Database ORM and query building
- numpy: Numerical computations and analysis
- pandas: Data manipulation and analysis
- scipy: Statistical analysis and clustering

Notes
-----
- All analysis methods support flexible scope filtering
- Heavy computations are optimized through database queries
- Results are returned as structured data types
- Analysis methods handle missing or incomplete data
- Documentation includes type hints and examples

See Also
--------
database.models : Database model definitions
database.retrievers : Base retriever functionality
database.data_types : Data structure definitions
"""

from typing import List, Optional, Tuple, Union

from analysis.action_stats_analyzer import ActionStatsAnalyzer
from analysis.behavior_clustering_analyzer import BehaviorClusteringAnalyzer
from analysis.causal_analyzer import CausalAnalyzer
from analysis.decision_pattern_analyzer import DecisionPatternAnalyzer
from analysis.resource_impact_analyzer import ResourceImpactAnalyzer
from analysis.sequence_pattern_analyzer import SequencePatternAnalyzer
from analysis.temporal_pattern_analyzer import TemporalPatternAnalyzer
from database.data_types import (
    ActionMetrics,
    AgentActionData,
    BehaviorClustering,
    CausalAnalysis,
    DecisionPatterns,
    ResourceImpact,
    SequencePattern,
    TimePattern,
)
from database.enums import AnalysisScope
from database.repositories.agent_action_repository import AgentActionRepository
from database.retrievers import BaseRetriever
from database.session_manager import SessionManager


class ActionsRetriever(BaseRetriever):
    """Comprehensive retriever and analyzer for action-related simulation data.

    This class provides extensive methods for analyzing agent behaviors, including action
    patterns, decision-making, interactions, resource management, and advanced behavioral
    metrics.

    Key Capabilities
    ---------------
    - Basic action metrics and statistics
    - Decision pattern analysis
    - Interaction network analysis
    - Resource impact assessment
    - Temporal pattern recognition
    - Advanced behavioral analytics
    - Step-by-step action monitoring

    Methods
    -------
    actions(scope: str = "simulation", agent_id: Optional[int] = None, ...) -> List[AgentActionData]
        Retrieve filtered action data with complete metadata

    action_stats(scope: str = "simulation", agent_id: Optional[int] = None, ...) -> List[ActionMetrics]
        Get comprehensive statistics for each action type

    temporal_patterns(scope: str = "simulation", agent_id: Optional[int] = None, ...) -> List[TimePattern]
        Analyze action patterns over time

    resource_impacts(scope: str = "simulation", agent_id: Optional[int] = None, ...) -> List[ResourceImpact]
        Analyze resource impacts of different actions

    decision_patterns(scope: str = "simulation", agent_id: Optional[int] = None, ...) -> DecisionPatterns
        Analyze comprehensive decision-making patterns

    sequence_patterns(scope: str = "simulation", agent_id: Optional[int] = None, ...) -> List[SequencePattern]
        Analyze sequential action patterns and transitions

    causal_analysis(action_type: str) -> CausalAnalysis
        Analyze cause-effect relationships for actions

    behavior_clustering() -> BehaviorClustering
        Group agents by behavioral patterns and strategies

    Analysis Scopes
    --------------
    All analysis methods support multiple scoping options:
    - "simulation": Analyze all data (no filters)
    - "step": Analyze specific step
    - "step_range": Analyze range of steps
    - "agent": Analyze specific agent

    Return Types
    -----------
    AgentActionData
        Complete action metadata including resources and rewards

    ActionMetrics
        Comprehensive statistics for action types

    TimePattern
        Temporal evolution of action patterns

    ResourceImpact
        Resource consumption and generation metrics

    DecisionPatterns
        Decision-making analysis and patterns

    SequencePattern
        Action sequence statistics and probabilities

    CausalAnalysis
        Cause-effect relationships and impacts

    BehaviorClustering
        Agent groupings and characteristics

    Examples
    --------
    >>> retriever = ActionsRetriever(session)

    >>> # Get all actions for a step
    >>> step_actions = retriever.actions(scope="step", step=5)
    >>> for action in step_actions:
    ...     print(f"Agent {action.agent_id}: {action.action_type}")

    >>> # Analyze decision patterns
    >>> patterns = retriever.decision_patterns()
    >>> print(f"Most common action: {patterns.decision_summary.most_frequent}")
    >>> print(f"Action diversity: {patterns.decision_summary.action_diversity:.2f}")

    >>> # Analyze resource impacts
    >>> impacts = retriever.resource_impacts()
    >>> for impact in impacts:
    ...     print(f"{impact.action_type}: {impact.resource_efficiency:.2f}")

    >>> # Cluster agent behaviors
    >>> clusters = retriever.behavior_clustering()
    >>> for name, agents in clusters.clusters.items():
    ...     print(f"{name} strategy: {len(agents)} agents")

    Notes
    -----
    - All analysis methods support flexible scoping options
    - Methods return structured data types for consistent analysis
    - Resource tracking includes both consumption and generation
    - Temporal analysis uses binned time periods
    - Behavioral analysis considers multiple metrics
    - Clustering identifies emergent strategies

    See Also
    --------
    BaseRetriever : Parent class providing core database functionality
    AnalysisScope : Enum defining valid analysis scopes
    """

    def __init__(self, database_url: str = "sqlite:///results/simulation.db") -> None:
        """Initialize the actions retriever.

        Parameters
        ----------
        database_url : str, optional
            SQLAlchemy database URL, by default "sqlite:///results/simulation.db"
        """
        self.session_manager = SessionManager(database_url)
        self.repository = AgentActionRepository(self.session_manager)

        # Initialize analyzers with repository
        self.action_stats_analyzer = ActionStatsAnalyzer(self.repository)
        self.temporal_pattern_analyzer = TemporalPatternAnalyzer(self.repository)
        self.resource_impact_analyzer = ResourceImpactAnalyzer(self.repository)
        self.decision_pattern_analyzer = DecisionPatternAnalyzer(self.repository)
        self.sequence_pattern_analyzer = SequencePatternAnalyzer(self.repository)
        self.causal_analyzer = CausalAnalyzer(self.repository)
        self.behavior_clustering_analyzer = BehaviorClusteringAnalyzer(self.repository)

    def close(self) -> None:
        """Close database connections and clean up resources."""
        if hasattr(self, "session_manager"):
            self.session_manager.remove_session()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()

    def actions(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[AgentActionData]:
        """Retrieve filtered action data with complete metadata.

        Parameters
        ----------
        scope : Union[str, AnalysisScope], default=AnalysisScope.SIMULATION
            Analysis scope level to filter data
        agent_id : Optional[int], default=None
            Specific agent ID to filter by
        step : Optional[int], default=None
            Specific simulation step to analyze
        step_range : Optional[Tuple[int, int]], default=None
            Range of simulation steps to analyze

        Returns
        -------
        List[AgentActionData]
            List of action data entries matching the specified filters
        """
        return self.repository.get_actions_by_scope(scope, agent_id, step, step_range)

    def action_stats(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[ActionMetrics]:
        """Calculate comprehensive statistics for each action type.

        Parameters
        ----------
        scope : Union[str, AnalysisScope], default=AnalysisScope.SIMULATION
            Analysis scope level to filter data
        agent_id : Optional[int], default=None
            Specific agent ID to filter by
        step : Optional[int], default=None
            Specific simulation step to analyze
        step_range : Optional[Tuple[int, int]], default=None
            Range of simulation steps to analyze

        Returns
        -------
        List[ActionMetrics]
            Statistical metrics for each action type
        """
        return self.action_stats_analyzer.analyze(scope, agent_id, step, step_range)

    def temporal_patterns(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[TimePattern]:
        """Analyze action patterns over time.

        Parameters
        ----------
        scope : Union[str, AnalysisScope], default=AnalysisScope.SIMULATION
            Analysis scope level to filter data
        agent_id : Optional[int], default=None
            Specific agent ID to filter by
        step_range : Optional[Tuple[int, int]], default=None
            Range of simulation steps to analyze

        Returns
        -------
        List[TimePattern]
            Temporal evolution patterns for actions
        """
        return self.temporal_pattern_analyzer.analyze(scope, agent_id, step_range)

    def resource_impacts(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[ResourceImpact]:
        """Analyze resource impacts of different actions.

        Parameters
        ----------
        scope : Union[str, AnalysisScope], default=AnalysisScope.SIMULATION
            Analysis scope level to filter data
        agent_id : Optional[int], default=None
            Specific agent ID to filter by
        step : Optional[int], default=None
            Specific simulation step to analyze
        step_range : Optional[Tuple[int, int]], default=None
            Range of simulation steps to analyze

        Returns
        -------
        List[ResourceImpact]
            Resource consumption and generation metrics for actions
        """
        return self.resource_impact_analyzer.analyze(scope, agent_id, step, step_range)

    def decision_patterns(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> DecisionPatterns:
        """Analyze comprehensive decision-making patterns.

        Parameters
        ----------
        scope : Union[str, AnalysisScope], default=AnalysisScope.SIMULATION
            Analysis scope level to filter data
        agent_id : Optional[int], default=None
            Specific agent ID to filter by
        step : Optional[int], default=None
            Specific simulation step to analyze
        step_range : Optional[Tuple[int, int]], default=None
            Range of simulation steps to analyze

        Returns
        -------
        DecisionPatterns
            Analysis of decision-making patterns and behaviors
        """
        return self.decision_pattern_analyzer.analyze(scope, agent_id, step, step_range)

    def sequence_patterns(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[SequencePattern]:
        """Analyze sequential action patterns and transitions.

        Parameters
        ----------
        scope : Union[str, AnalysisScope], default=AnalysisScope.SIMULATION
            Analysis scope level to filter data
        agent_id : Optional[int], default=None
            Specific agent ID to filter by
        step : Optional[int], default=None
            Specific simulation step to analyze
        step_range : Optional[Tuple[int, int]], default=None
            Range of simulation steps to analyze

        Returns
        -------
        List[SequencePattern]
            Action sequence statistics and transition patterns
        """
        return self.sequence_pattern_analyzer.analyze(scope, agent_id, step, step_range)

    def causal_analysis(self, action_type: str) -> CausalAnalysis:
        """Analyze cause-effect relationships for specific action type.

        Parameters
        ----------
        action_type : str
            Type of action to analyze causal relationships for

        Returns
        -------
        CausalAnalysis
            Cause-effect relationships and impact analysis
        """
        return self.causal_analyzer.analyze(action_type)

    def behavior_clustering(self) -> BehaviorClustering:
        """Group agents by behavioral patterns and strategies.

        Returns
        -------
        BehaviorClustering
            Clustered agent behaviors and strategy classifications
        """
        return self.behavior_clustering_analyzer.analyze()

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.close()
