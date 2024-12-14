"""Actions retrieval and analysis module for simulation database.

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

from enum import Enum
from json import loads
from typing import List, Optional, Tuple, Union

from sqlalchemy import case, func

from database.data_types import (
    ActionMetrics,
    AgentActionData,
    BehaviorClustering,
    CausalAnalysis,
    DecisionPatterns,
    DecisionPatternStats,
    DecisionSummary,
    ResourceImpact,
    SequencePattern,
    TimePattern,
)
from database.models import AgentAction
from database.retrievers import BaseRetriever
from database.utilities import execute_query


class AnalysisScope(str, Enum):
    """Scope levels for analysis queries.

    SIMULATION: All data (no filters)
    STEP: Single step
    STEP_RANGE: Range of steps
    AGENT: Single agent
    """

    SIMULATION = "simulation"
    STEP = "step"
    STEP_RANGE = "step_range"
    AGENT = "agent"

    @classmethod
    def from_string(cls, scope_str: str) -> "AnalysisScope":
        """Convert string to AnalysisScope, case-insensitive."""
        try:
            return cls(scope_str.lower())
        except ValueError:
            valid_scopes = [s.value for s in cls]
            raise ValueError(
                f"Invalid scope '{scope_str}'. Must be one of: {valid_scopes}"
            )


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

    @execute_query
    def actions(
        self,
        session,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[AgentActionData]:
        """Retrieve filtered action data from the simulation database.

        Provides chronological action data based on specified scope and filters. Returns complete
        metadata for each action including resources, rewards, and custom details.

        Parameters
        ----------
        session : Session
            Database session for executing queries
        scope : Union[str, AnalysisScope], default=AnalysisScope.SIMULATION
            Analysis scope level:
            - "simulation": All data (no filters)
            - "step": Single step
            - "step_range": Range of steps
            - "agent": Single agent
        agent_id : Optional[int], default=None
            Specific agent ID to analyze. Required when scope is "agent".
            If None and scope is "agent", a random agent is selected.
        step : Optional[int], default=None
            Specific step to analyze. Required when scope is "step".
        step_range : Optional[Tuple[int, int]], default=None
            (start_step, end_step) range to analyze. Required when scope is "step_range".

        Returns
        -------
        List[AgentActionData]
            List of actions matching criteria, ordered by step_number and agent_id.
            Each AgentActionData contains:
            - agent_id : int
                ID of acting agent
            - action_type : str
                Type of action performed
            - step_number : int
                Simulation step when action occurred
            - action_target_id : Optional[int]
                Target agent ID if action involved interaction
            - resources_before : float
                Agent's resources before action
            - resources_after : float
                Agent's resources after action
            - state_before_id : int
                Agent state ID before action
            - state_after_id : int
                Agent state ID after action
            - reward : float
                Action outcome reward value
            - details : Optional[Dict]
                Additional action metadata

        Examples
        --------
        >>> # Get all actions for a specific step
        >>> actions = retriever.actions(scope="step", step=5)
        >>> for action in actions:
        ...     print(f"Agent {action.agent_id}: {action.action_type}")

        >>> # Get actions for specific agent
        >>> agent_actions = retriever.actions(scope="agent", agent_id=1)

        >>> # Get actions within step range
        >>> range_actions = retriever.actions(
        ...     scope="step_range",
        ...     step_range=(100, 200)
        ... )

        See Also
        --------
        action_stats : Get aggregated statistics about actions
        temporal_patterns : Analyze action patterns over time
        decision_patterns : Analyze decision-making patterns

        Notes
        -----
        - Actions are always returned in chronological order
        - For agent scope, if no agent_id provided, randomly selects an agent
        - The details field contains action-specific metadata as a dictionary
        """
        # Build base query
        query = session.query(AgentAction).order_by(
            AgentAction.step_number, AgentAction.agent_id
        )

        # Apply scope filters
        query = self._validate_and_filter_scope(
            session, query, scope, agent_id, step, step_range
        )

        actions = query.all()
        return [
            AgentActionData(
                agent_id=action.agent_id,
                action_type=action.action_type,
                step_number=action.step_number,
                action_target_id=action.action_target_id,
                resources_before=action.resources_before,
                resources_after=action.resources_after,
                state_before_id=action.state_before_id,
                state_after_id=action.state_after_id,
                reward=action.reward,
                details=action.details if action.details else None,
            )
            for action in actions
        ]

    @execute_query
    def action_stats(
        self,
        session,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[ActionMetrics]:
        """Get comprehensive statistics for each action type including interaction data.

        Analyzes action frequencies, rewards, and interaction patterns to provide detailed
        performance metrics for each action type within the specified scope.

        Parameters
        ----------
        session : Session
            Database session for executing queries
        scope : Union[str, AnalysisScope], default=AnalysisScope.SIMULATION
            Analysis scope level:
            - "simulation": All data (no filters)
            - "step": Single step
            - "step_range": Range of steps
            - "agent": Single agent
        agent_id : Optional[int], default=None
            Specific agent ID to analyze. Required when scope is "agent".
            If None and scope is "agent", a random agent is selected.
        step : Optional[int], default=None
            Specific step to analyze. Required when scope is "step".
        step_range : Optional[Tuple[int, int]], default=None
            (start_step, end_step) range to analyze. Required when scope is "step_range".

        Returns
        -------
        List[ActionMetrics]
            List of statistics for each action type, containing:
            - action_type: str
                Type of action performed
            - count: int
                Number of times this action was taken
            - frequency: float
                Proportion of total actions (0.0 to 1.0)
            - avg_reward: float
                Mean reward received for this action
            - min_reward: float
                Minimum reward received
            - max_reward: float
                Maximum reward received
            - interaction_rate: float
                Proportion of actions involving other agents
            - solo_performance: float
                Average reward for actions without targets
            - interaction_performance: float
                Average reward for actions with targets
            - temporal_patterns: TimePattern
                Patterns of action usage over time
            - resource_impacts: ResourceImpact
                Resource consumption and generation metrics
            - decision_patterns: DecisionPatternStats
                Decision-making statistics and trends

        Examples
        --------
        >>> # Get global action statistics
        >>> stats = retriever.action_stats()
        >>> for metric in stats:
        ...     print(f"{metric.action_type}:")
        ...     print(f"  Count: {metric.count}")
        ...     print(f"  Success rate: {metric.avg_reward:.2f}")

        >>> # Get stats for specific agent
        >>> agent_stats = retriever.action_stats(
        ...     scope="agent",
        ...     agent_id=1
        ... )

        See Also
        --------
        temporal_patterns : Analyze patterns over time
        resource_impacts : Analyze resource effects
        decision_patterns : Analyze decision-making

        Notes
        -----
        - Statistics are calculated using all available data within scope
        - Interaction metrics only consider actions with explicit targets
        - Temporal patterns are grouped into periods for trend analysis
        """
        # Get basic action metrics
        query = session.query(
            AgentAction.action_type,
            func.count().label("count"),
            func.avg(AgentAction.reward).label("avg_reward"),
            func.min(AgentAction.reward).label("min_reward"),
            func.max(AgentAction.reward).label("max_reward"),
            func.avg(
                case(
                    (AgentAction.action_target_id.isnot(None), AgentAction.reward),
                    else_=None,
                )
            ).label("interaction_reward"),
            func.avg(
                case(
                    (AgentAction.action_target_id.is_(None), AgentAction.reward),
                    else_=None,
                )
            ).label("solo_reward"),
            func.count(AgentAction.action_target_id).label("interaction_count"),
        )

        query = self._validate_and_filter_scope(
            session, query, scope, agent_id, step, step_range
        )
        results = query.group_by(AgentAction.action_type).all()

        total_actions = sum(r.count for r in results)

        # Get additional patterns and organize by action type
        temporal_patterns = {
            p.action_type: p
            for p in self.temporal_patterns(scope, agent_id, step_range)
        }
        resource_impacts = {
            p.action_type: p
            for p in self.resource_impacts(scope, agent_id, step, step_range)
        }
        decision_patterns = {
            p.action_type: p
            for p in self.decision_patterns(
                scope, agent_id, step, step_range
            ).decision_patterns
        }

        return [
            ActionMetrics(
                action_type=r.action_type,
                count=r.count,
                frequency=r.count / total_actions if total_actions > 0 else 0,
                avg_reward=float(r.avg_reward or 0),
                min_reward=float(r.min_reward or 0),
                max_reward=float(r.max_reward or 0),
                interaction_rate=float(
                    r.interaction_count / r.count if r.count > 0 else 0
                ),
                solo_performance=float(r.solo_reward or 0),
                interaction_performance=float(r.interaction_reward or 0),
                temporal_patterns=temporal_patterns.get(r.action_type),
                resource_impacts=resource_impacts.get(r.action_type),
                decision_patterns=decision_patterns.get(r.action_type),
            )
            for r in results
        ]

    @execute_query
    def temporal_patterns(
        self,
        session,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[TimePattern]:
        """Analyze how action patterns evolve over time.

        Examines the temporal evolution of action choices and their effectiveness,
        grouping data into time periods to identify trends and patterns.

        Parameters
        ----------
        session : Session
            Database session for executing queries
        scope : Union[str, AnalysisScope], default=AnalysisScope.SIMULATION
            Analysis scope level:
            - "simulation": All data (no filters)
            - "step_range": Range of steps
            - "agent": Single agent
        agent_id : Optional[int], default=None
            Specific agent ID to analyze. Required when scope is "agent".
            If None and scope is "agent", a random agent is selected.
        step_range : Optional[Tuple[int, int]], default=None
            (start_step, end_step) range to analyze. Required when scope is "step_range".

        Returns
        -------
        List[TimePattern]
            List of temporal patterns for each action type, containing:
            - action_type : str
                Type of action being analyzed
            - time_distribution : List[int]
                Action counts per time period showing usage patterns
            - reward_progression : List[float]
                Average rewards per time period showing effectiveness trends

        Examples
        --------
        >>> # Get global temporal patterns
        >>> patterns = retriever.temporal_patterns()
        >>> for pattern in patterns:
        ...     print(f"{pattern.action_type} trends:")
        ...     print("  Usage:", pattern.time_distribution)
        ...     print("  Rewards:", pattern.reward_progression)

        >>> # Get patterns for specific agent
        >>> agent_patterns = retriever.temporal_patterns(
        ...     scope="agent",
        ...     agent_id=1
        ... )

        See Also
        --------
        action_stats : Get overall action statistics
        decision_patterns : Analyze decision-making patterns

        Notes
        -----
        - Time periods are determined by grouping steps into bins of 100 steps each
        - Empty periods will have 0 counts and rewards
        - Trends can reveal learning, adaptation, and strategy evolution
        """
        # Build base query
        query = session.query(
            AgentAction.action_type,
            func.count().label("count"),
            func.avg(AgentAction.reward).label("avg_reward"),
        )

        # Apply scope filters
        query = self._validate_and_filter_scope(
            session, query, scope, agent_id, step=None, step_range=step_range
        )

        # Group by action type and time period
        patterns = query.group_by(
            AgentAction.action_type,
            func.round(AgentAction.step_number / 100),  # Group by time periods
        ).all()

        # Process results into a list
        temporal_patterns = []
        current_pattern = None
        current_action_type = None

        for action_type, count, avg_reward in patterns:
            if action_type != current_action_type:
                if current_pattern is not None:
                    temporal_patterns.append(current_pattern)
                current_pattern = TimePattern(
                    action_type=action_type,
                    time_distribution=[],
                    reward_progression=[],
                )
                current_action_type = action_type

            current_pattern.time_distribution.append(int(count))
            current_pattern.reward_progression.append(float(avg_reward or 0))

        # Add the last pattern if it exists
        if current_pattern is not None:
            temporal_patterns.append(current_pattern)

        return temporal_patterns

    @execute_query
    def resource_impacts(
        self,
        session,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[ResourceImpact]:
        """Analyze how different actions affect agent resources.

        Evaluates the resource efficiency and impact of different actions, including
        consumption patterns, generation rates, and overall resource management metrics.

        Parameters
        ----------
        session : Session
            Database session for executing queries
        scope : Union[str, AnalysisScope], default=AnalysisScope.SIMULATION
            Analysis scope level:
            - "simulation": All data (no filters)
            - "step": Single step
            - "step_range": Range of steps
            - "agent": Single agent
        agent_id : Optional[int], default=None
            Specific agent ID to analyze. Required when scope is "agent".
            If None and scope is "agent", a random agent is selected.
        step : Optional[int], default=None
            Specific step to analyze. Required when scope is "step".
        step_range : Optional[Tuple[int, int]], default=None
            (start_step, end_step) range to analyze. Required when scope is "step_range".

        Returns
        -------
        List[ResourceImpact]
            List of resource impact statistics for each action type, containing:
            - action_type : str
                Type of action being analyzed
            - avg_resources_before : float
                Mean resources available before action execution
            - avg_resource_change : float
                Average change in resources from action execution
            - resource_efficiency : float
                Resource change per action execution (change/count)

        Examples
        --------
        >>> # Get global resource impacts
        >>> impacts = retriever.resource_impacts()
        >>> for impact in impacts:
        ...     print(f"{impact.action_type}:")
        ...     print(f"  Average change: {impact.avg_resource_change:+.2f}")
        ...     print(f"  Efficiency: {impact.resource_efficiency:.3f}")

        >>> # Get impacts for specific agent
        >>> agent_impacts = retriever.resource_impacts(
        ...     scope="agent",
        ...     agent_id=1
        ... )

        See Also
        --------
        action_stats : Get overall action statistics
        temporal_patterns : Analyze patterns over time

        Notes
        -----
        - Positive resource changes indicate generation/acquisition
        - Negative changes indicate consumption/loss
        - Efficiency metrics help identify optimal resource strategies
        """
        # Build base query
        query = session.query(
            AgentAction.action_type,
            func.avg(AgentAction.resources_before).label("avg_before"),
            func.avg(AgentAction.resources_after - AgentAction.resources_before).label(
                "avg_change"
            ),
            func.count().label("count"),
        ).filter(
            AgentAction.resources_before.isnot(None),
            AgentAction.resources_after.isnot(None),
        )

        # Apply scope filters
        query = self._validate_and_filter_scope(
            session, query, scope, agent_id, step, step_range
        )

        # Group by action type and execute query
        impacts = query.group_by(AgentAction.action_type).all()

        return [
            ResourceImpact(
                action_type=action_type,
                avg_resources_before=float(avg_before or 0),
                avg_resource_change=float(avg_change or 0),
                resource_efficiency=float(avg_change or 0) / count if count > 0 else 0,
            )
            for action_type, avg_before, avg_change, count in impacts
        ]

    @execute_query
    def decision_patterns(
        self,
        session,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> DecisionPatterns:
        """Analyze comprehensive decision-making patterns.

        Performs a detailed analysis of agent decision-making patterns, including action
        frequencies, rewards, sequences, resource impacts, and temporal trends.

        Parameters
        ----------
        session : Session
            Database session for executing queries
        scope : Union[str, AnalysisScope], default=AnalysisScope.SIMULATION
            Analysis scope level:
            - "simulation": All data (no filters)
            - "step": Single step
            - "step_range": Range of steps
            - "agent": Single agent
        agent_id : Optional[int], default=None
            Specific agent ID to analyze. Required when scope is "agent".
            If None and scope is "agent", a random agent is selected.
        step : Optional[int], default=None
            Specific step to analyze. Required when scope is "step".
        step_range : Optional[Tuple[int, int]], default=None
            (start_step, end_step) range to analyze. Required when scope is "step_range".

        Returns
        -------
        DecisionPatterns
            Comprehensive decision pattern analysis with:
            decision_patterns : List[DecisionPatternStats]
                Statistics for each action type:
                - action_type: Type of action
                - count: Total times action taken
                - frequency: Proportion of choices
                - reward_stats: Dict with avg/min/max rewards

            decision_summary : DecisionSummary
                Overall decision-making metrics:
                - total_decisions: Total decisions made
                - unique_actions: Different action types used
                - most_frequent: Most common action
                - most_rewarding: Highest avg reward action
                - action_diversity: Shannon entropy measure

        Examples
        --------
        >>> # Get global decision patterns
        >>> patterns = retriever.decision_patterns()
        >>> print(f"Total decisions: {patterns.decision_summary.total_decisions}")
        >>> print(f"Most frequent: {patterns.decision_summary.most_frequent}")

        >>> # Get patterns for specific agent
        >>> agent_patterns = retriever.decision_patterns(
        ...     scope="agent",
        ...     agent_id=1
        ... )

        See Also
        --------
        action_stats : Get action statistics
        temporal_patterns : Analyze time patterns
        sequence_patterns : Analyze action sequences

        Notes
        -----
        - Action diversity uses Shannon entropy to measure decision variety
        - Higher diversity indicates more varied decision-making
        - Lower diversity suggests specialized strategies
        """
        # Get basic action metrics with scope
        query = session.query(
            AgentAction.action_type,
            func.count().label("decision_count"),
            func.avg(AgentAction.reward).label("avg_reward"),
            func.min(AgentAction.reward).label("min_reward"),
            func.max(AgentAction.reward).label("max_reward"),
        )

        # Apply scope filters
        query = self._validate_and_filter_scope(
            session, query, scope, agent_id, step, step_range
        )

        # Group by action type and execute
        metrics = query.group_by(AgentAction.action_type).all()

        # Calculate total decisions
        total_decisions = sum(m.decision_count for m in metrics)

        # Format decision patterns
        patterns = [
            DecisionPatternStats(
                action_type=m.action_type,  # Add action_type to the stats object
                count=m.decision_count,
                frequency=(
                    m.decision_count / total_decisions if total_decisions > 0 else 0
                ),
                reward_stats={
                    "average": float(m.avg_reward or 0),
                    "min": float(m.min_reward or 0),
                    "max": float(m.max_reward or 0),
                },
            )
            for m in metrics
        ]

        # Create decision summary
        summary = DecisionSummary(
            total_decisions=total_decisions,
            unique_actions=len(metrics),
            most_frequent=(
                max(patterns, key=lambda x: x.count).action_type if patterns else None
            ),
            most_rewarding=(
                max(patterns, key=lambda x: x.reward_stats["average"]).action_type
                if patterns
                else None
            ),
            action_diversity=self._calculate_diversity(patterns),
        )

        return DecisionPatterns(
            decision_patterns=patterns,
            decision_summary=summary,
        )

    @execute_query
    def sequence_patterns(
        self,
        session,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[SequencePattern]:
        """Calculate action sequence patterns within the specified scope.

        Analyzes sequential patterns in agent decision-making by examining pairs of consecutive
        actions and their transition probabilities. Helps identify common action chains and
        decision strategies.

        Parameters
        ----------
        session : Session
            Database session for executing queries
        scope : Union[str, AnalysisScope], default=AnalysisScope.SIMULATION
            Analysis scope level:
            - "simulation": All data (no filters)
            - "step": Single step
            - "step_range": Range of steps
            - "agent": Single agent
        agent_id : Optional[int], default=None
            Specific agent ID to analyze. Required when scope is "agent".
            If None and scope is "agent", a random agent is selected.
        step : Optional[int], default=None
            Specific step to analyze. Required when scope is "step".
        step_range : Optional[Tuple[int, int]], default=None
            (start_step, end_step) range to analyze. Required when scope is "step_range".

        Returns
        -------
        List[SequencePattern]
            List of action sequence statistics, where each contains:
            - sequence : str
                Action sequence in format "action1->action2"
            - count : int
                Number of times this sequence occurred
            - probability : float
                Likelihood of action2 following action1 (0.0 to 1.0)

        Examples
        --------
        >>> patterns = retriever.sequence_patterns()
        >>> for pattern in patterns:
        ...     print(f"{pattern.sequence}: {pattern.probability:.2%}")
        attack->defend: 35.20%
        defend->gather: 28.50%

        See Also
        --------
        decision_patterns : Get broader decision-making analysis
        temporal_patterns : Analyze patterns over time

        Notes
        -----
        - Only considers consecutive actions by the same agent
        - Probabilities are normalized within each agent's action set
        - Can reveal strategic patterns and decision chains
        """
        # First get ordered actions
        query = session.query(
            AgentAction.agent_id, AgentAction.action_type, AgentAction.step_number
        ).order_by(AgentAction.step_number)

        # Apply scope filters
        query = self._validate_and_filter_scope(
            session, query, scope, agent_id, step, step_range
        )

        actions = query.all()

        # Calculate sequences and counts manually
        sequences = {}
        action_counts = {}  # For calculating probabilities

        # Process action pairs
        for i in range(len(actions) - 1):
            current = actions[i]
            next_action = actions[i + 1]

            # Only count sequences within same agent
            if current.agent_id == next_action.agent_id:
                sequence_key = f"{current.action_type}->{next_action.action_type}"

                # Count sequences
                sequences[sequence_key] = sequences.get(sequence_key, 0) + 1

                # Count total occurrences of first action (for probability calculation)
                action_counts[current.action_type] = (
                    action_counts.get(current.action_type, 0) + 1
                )

        return [
            SequencePattern(
                sequence=sequence,
                count=count,
                probability=(
                    count / action_counts[sequence.split("->")[0]]
                    if sequence.split("->")[0] in action_counts
                    else 0
                ),
            )
            for sequence, count in sequences.items()
        ]

    def _calculate_diversity(self, patterns: List[DecisionPatternStats]) -> float:
        """Calculate Shannon entropy for action diversity.

        Computes the Shannon entropy of the action distribution to measure
        the diversity of decision-making patterns. Higher values indicate
        more varied strategies.

        Parameters
        ----------
        patterns : List[DecisionPatternStats]
            List of decision pattern statistics containing:
            - frequency : float
                Proportion of times each action was chosen

        Returns
        -------
        float
            Shannon entropy diversity measure:
            - Higher values = more diverse action selection
            - Lower values = more focused/specialized behavior
            - 0.0 = only one action used
            - ln(n) = perfectly uniform distribution among n actions

        Notes
        -----
        Uses the standard Shannon entropy formula: -Σ(p * log(p))
        where p is the frequency of each action type.
        """
        import math

        return -sum(
            p.frequency * math.log(p.frequency) if p.frequency > 0 else 0
            for p in patterns
        )

    def _execute(self) -> DecisionPatterns:
        #! is this still needed?
        """Execute comprehensive action analysis.

        Performs a complete analysis of decision-making patterns by calling
        decision_patterns() with default parameters.

        Returns
        -------
        DecisionPatterns
            Complete action and decision-making analysis containing:
            - Decision patterns
            - Sequence analysis
            - Resource impact
            - Temporal patterns
            - Interaction analysis
            - Decision summary
        """
        return self.decision_patterns()

    @execute_query
    def causal_analysis(self, session, action_type: str) -> CausalAnalysis:
        """Analyze causal relationships between actions
        Examines cause-effect relationships between actions and their consequences,
        including direct impacts and downstream effects. Uses temporal sequence and
        correlation patterns to suggest causal relationships.

        Parameters
        ----------
        session : Session
            Database session for executing queries
        action_type : str
            Specific action type to analyze causal patterns for

        Returns
        -------
        CausalAnalysis
            Analysis results containing:
            - action_type : str
                Type of action analyzed
            - causal_impact : float
                Estimated direct effect strength
            - state_transition_probs : Dict[str, float]
                Probability distribution of subsequent states/outcomes

        Examples
        --------
        >>> analysis = retriever.causal_analysis("attack")
        >>> print(f"Impact strength: {analysis.causal_impact:.2f}")
        >>> for outcome, prob in analysis.state_transition_probs.items():
        ...     print(f"{outcome}: {prob:.2%}")

        See Also
        --------
        decision_patterns : Analyze decision-making patterns
        sequence_patterns : Analyze action sequences

        Notes
        -----
        - Correlation doesn't guarantee causation
        - Analysis considers context and temporal ordering
        - Transition probabilities account for environmental factors
        """
        # Get all actions of the specified type
        actions = (
            session.query(AgentAction)
            .filter(AgentAction.action_type == action_type)
            .order_by(AgentAction.step_number)
            .all()
        )

        if not actions:
            return CausalAnalysis(
                action_type=action_type,
                causal_impact=0.0,
                state_transition_probs={},
            )

        # Calculate immediate causal impact (average reward)
        causal_impact = sum(a.reward or 0 for a in actions) / len(actions)

        # Analyze state transitions
        state_transitions = {}
        for i in range(len(actions) - 1):
            current = actions[i]
            next_action = actions[i + 1]

            # Get state changes from action details
            current_details = loads(current.details) if current.details else {}
            next_details = loads(next_action.details) if next_action.details else {}

            # Build transition key with context
            context_parts = []

            # Add success/failure context for both actions
            success = current_details.get("success", False)
            next_success = next_details.get("success", False)
            context_parts.append(f"success_{success}")
            context_parts.append(f"next_success_{next_success}")

            # Add resource context if available
            if (
                "resource_before" in current_details
                and "resource_after" in current_details
            ):
                resource_change = (
                    current_details["resource_after"]
                    - current_details["resource_before"]
                )
                if abs(resource_change) > 0:
                    context_parts.append(f"resource_change_{resource_change:+.1f}")

            # Add target context if relevant
            if current.action_target_id:
                context_parts.append("targeted")

            # Add specific outcome contexts based on action type
            if current.action_type == "gather":
                if "amount_gathered" in current_details:
                    context_parts.append(
                        f"gathered_{current_details['amount_gathered']}"
                    )
            elif current.action_type == "share":
                if "amount_shared" in current_details:
                    context_parts.append(f"shared_{current_details['amount_shared']}")

            # Add next action's context
            if next_action.action_type == "gather":
                if "amount_gathered" in next_details:
                    context_parts.append(
                        f"next_gathered_{next_details['amount_gathered']}"
                    )
            elif next_action.action_type == "share":
                if "amount_shared" in next_details:
                    context_parts.append(f"next_shared_{next_details['amount_shared']}")

            # Combine context into transition key
            transition_key = f"{next_action.action_type}|{','.join(context_parts)}"

            if transition_key not in state_transitions:
                state_transitions[transition_key] = 0
            state_transitions[transition_key] += 1

        # Convert transitions to probabilities
        total_transitions = sum(state_transitions.values())
        state_transition_probs = (
            {k: v / total_transitions for k, v in state_transitions.items()}
            if total_transitions > 0
            else {}
        )

        return CausalAnalysis(
            action_type=action_type,
            causal_impact=causal_impact,
            state_transition_probs=state_transition_probs,
        )

    @execute_query
    def behavior_clustering(self, session) -> BehaviorClustering:
        """Cluster agents based on behavioral patterns.

        Groups agents with similar decision-making patterns and analyzes the
        characteristics of each behavioral cluster. Helps identify distinct
        strategies and agent archetypes.

        Parameters
        ----------
        session : Session
            Database session for executing queries

        Returns
        -------
        BehaviorClustering
            Clustering analysis results containing:
            - clusters : Dict[str, List[int]]
                Groups of agent IDs with similar behaviors:
                - "aggressive": Combat-focused agents
                - "cooperative": Sharing/interaction-focused
                - "efficient": Resource-optimization focused
                - "balanced": Mixed strategy agents
            - cluster_characteristics : Dict[str, Dict[str, float]]
                Key behavioral metrics for each cluster:
                - attack_rate: Combat action frequency
                - cooperation: Sharing/interaction rate
                - risk_taking: High-risk action rate
                - success_rate: Action success frequency
                - resource_efficiency: Resource management score
            - cluster_performance : Dict[str, float]
                Average performance metrics per cluster

        Examples
        --------
        >>> clusters = retriever.behavior_clustering()
        >>> for name, agents in clusters.clusters.items():
        ...     print(f"{name}: {len(agents)} agents")
        ...     chars = clusters.cluster_characteristics[name]
        ...     print(f"  Success rate: {chars['success_rate']:.2%}")

        See Also
        --------
        decision_patterns : Analyze individual decision patterns
        action_stats : Get action-specific statistics

        Notes
        -----
        - Clustering uses behavioral metrics, not just action counts
        - Agents may show characteristics of multiple clusters
        - Performance metrics help identify successful strategies
        """
        # Get all agent actions with their details
        actions = session.query(
            AgentAction.agent_id,
            AgentAction.action_type,
            AgentAction.action_target_id,
            AgentAction.reward,
            func.json_extract(AgentAction.details, "$.success").label("success"),
            AgentAction.resources_before,
            AgentAction.resources_after,
        ).all()

        # Calculate behavioral metrics per agent
        agent_metrics = {}
        for action in actions:
            agent_id = action.agent_id
            if agent_id not in agent_metrics:
                agent_metrics[agent_id] = {
                    "action_counts": {
                        "attack": 0,
                        "defend": 0,
                        "gather": 0,
                        "share": 0,
                        "move": 0,
                    },
                    "total_actions": 0,
                    "interaction_rate": 0,
                    "success_rate": 0,
                    "risk_taking": 0,
                    "resource_efficiency": 0,
                    "total_reward": 0,
                    "interactions": 0,
                    "successful_actions": 0,
                }

            metrics = agent_metrics[agent_id]
            metrics["action_counts"][action.action_type] += 1
            metrics["total_actions"] += 1
            metrics["total_reward"] += action.reward or 0

            if action.action_target_id:
                metrics["interactions"] += 1

            if action.success:
                metrics["successful_actions"] += 1

            # Calculate resource efficiency
            if (
                action.resources_before is not None
                and action.resources_after is not None
            ):
                resource_change = action.resources_after - action.resources_before
                if resource_change > 0:
                    metrics["resource_efficiency"] += resource_change

        # Calculate final metrics and cluster agents
        clusters = {
            "aggressive": [],
            "cooperative": [],
            "efficient": [],
            "balanced": [],
        }

        for agent_id, metrics in agent_metrics.items():
            if metrics["total_actions"] == 0:
                continue

            # Calculate normalized metrics
            attack_rate = metrics["action_counts"]["attack"] / metrics["total_actions"]
            share_rate = metrics["action_counts"]["share"] / metrics["total_actions"]
            interaction_rate = metrics["interactions"] / metrics["total_actions"]
            success_rate = metrics["successful_actions"] / metrics["total_actions"]
            avg_reward = metrics["total_reward"] / metrics["total_actions"]

            # Classify agent based on behavioral patterns
            if attack_rate > 0.3:
                clusters["aggressive"].append(agent_id)
            elif share_rate > 0.3 or interaction_rate > 0.4:
                clusters["cooperative"].append(agent_id)
            elif success_rate > 0.7 and avg_reward > 1.0:
                clusters["efficient"].append(agent_id)
            else:
                clusters["balanced"].append(agent_id)

        # Calculate cluster characteristics
        characteristics = {}
        performance = {}

        for cluster_name, agent_ids in clusters.items():
            if not agent_ids:
                continue

            cluster_metrics = {
                "attack_rate": 0,
                "cooperation": 0,
                "risk_taking": 0,
                "success_rate": 0,
                "resource_efficiency": 0,
            }

            total_reward = 0

            for agent_id in agent_ids:
                metrics = agent_metrics[agent_id]
                total_actions = metrics["total_actions"]

                cluster_metrics["attack_rate"] += (
                    metrics["action_counts"]["attack"] / total_actions
                )
                cluster_metrics["cooperation"] += (
                    metrics["action_counts"]["share"] + metrics["interactions"]
                ) / total_actions
                cluster_metrics["success_rate"] += (
                    metrics["successful_actions"] / total_actions
                )
                cluster_metrics["resource_efficiency"] += (
                    metrics["resource_efficiency"] / total_actions
                )
                total_reward += metrics["total_reward"] / total_actions

            # Average the metrics
            n_agents = len(agent_ids)
            for metric in cluster_metrics:
                cluster_metrics[metric] /= n_agents

            characteristics[cluster_name] = cluster_metrics
            performance[cluster_name] = total_reward / n_agents

        return BehaviorClustering(
            clusters=clusters,
            cluster_characteristics=characteristics,
            cluster_performance=performance,
        )

    def _validate_and_filter_scope(
        self,
        session,
        base_query,
        scope: Union[str, AnalysisScope],
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ):
        """Validate scope parameters and apply appropriate query filters.

        Helper method that handles common validation and filtering logic for queries
        that support different analysis scopes. Ensures required parameters are
        provided and applies appropriate filters to the base query.

        Parameters
        ----------
        session : Session
            SQLAlchemy database session for executing queries
        base_query : Query
            Initial SQLAlchemy query to be filtered. Should include AgentAction table.
        scope : Union[str, AnalysisScope]
            Level at which to perform the analysis:
            - "simulation": Analyze all data without filters
            - "step": Analyze a specific step
            - "step_range": Analyze a range of steps
            - "agent": Analyze a specific agent
            Can be provided as string or AnalysisScope enum.
        agent_id : Optional[int], default=None
            ID of agent to analyze. Required when scope is "agent".
            Must be a valid agent ID in the database.
        step : Optional[int], default=None
            Specific step number to analyze. Required when scope is "step".
            Must be >= 0.
        step_range : Optional[Tuple[int, int]], default=None
            Range of steps to analyze as (start_step, end_step). Required when
            scope is "step_range". Both values must be >= 0 and start <= end.

        Returns
        -------
        Query
            SQLAlchemy query with appropriate scope filters applied

        Raises
        ------
        ValueError
            If required parameters are missing for the specified scope:
            - step missing when scope is "step"
            - step_range missing when scope is "step_range"
        TypeError
            If parameters are of incorrect type

        Examples
        --------
        >>> # Filter for specific agent
        >>> query = session.query(AgentAction)
        >>> query = self._validate_and_filter_scope(
        ...     session, query, "agent", agent_id=1
        ... )

        >>> # Filter for step range
        >>> query = session.query(AgentAction)
        >>> query = self._validate_and_filter_scope(
        ...     session, query, "step_range", step_range=(100, 200)
        ... )

        Notes
        -----
        - The simulation scope returns the unmodified query without filters
        - String scopes are converted to AnalysisScope enum values
        - This method is typically used internally by analysis methods that
          support different scoping options

        See Also
        --------
        AnalysisScope : Enum defining valid analysis scopes
        get_action_stats : Example method using this scope validation
        interactions : Another method using this scope validation
        """
        # Convert string scope to enum if needed
        if isinstance(scope, str):
            scope = AnalysisScope.from_string(scope)

        # For AGENT scope, randomly select an agent_id if none provided
        if scope == AnalysisScope.AGENT and agent_id is None:
            # Get a random agent_id from the database
            random_agent = (
                session.query(AgentAction.agent_id).order_by(func.random()).first()
            )
            if random_agent is None:
                raise ValueError("No agents found in database")
            agent_id = random_agent[0]

        # Validate remaining parameters based on scope
        if scope == AnalysisScope.STEP and step is None:
            raise ValueError("step is required when scope is STEP")
        if scope == AnalysisScope.STEP_RANGE and step_range is None:
            raise ValueError("step_range is required when scope is STEP_RANGE")

        # Apply filters based on scope
        if scope == AnalysisScope.AGENT:
            base_query = base_query.filter(AgentAction.agent_id == agent_id)
        elif scope == AnalysisScope.STEP:
            base_query = base_query.filter(AgentAction.step_number == step)
        elif scope == AnalysisScope.STEP_RANGE:
            start_step, end_step = step_range
            base_query = base_query.filter(
                AgentAction.step_number >= start_step,
                AgentAction.step_number <= end_step,
            )
        # SIMULATION scope requires no filters

        return base_query
