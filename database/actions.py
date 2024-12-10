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
    Main class handling action data retrieval and analysis
"""

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import case, func

from database.data_types import (
    ActionAnalysis,
    ActionMetrics,
    AdversarialInteractionAnalysis,
    BehaviorClustering,
    CausalAnalysis,
    CollaborativeInteractionAnalysis,
    ConflictAnalysis,
    CounterfactualAnalysis,
    DecisionPatterns,
    DecisionPatternStats,
    DecisionSummary,
    EnvironmentalImpactAnalysis,
    ExplorationExploitation,
    InteractionNetwork,
    InteractionStats,
    LearningCurveAnalysis,
    PerformanceMetrics,
    ResilienceAnalysis,
    ResourceImpact,
    ResourceMetricsStep,
    RiskRewardAnalysis,
    SequencePattern,
    StepActionData,
    StepSummary,
    TimePattern,
)
from database.models import AgentAction, AgentState
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
    get_action_stats(scope: str = "simulation", agent_id: Optional[int] = None, ...) -> List[ActionMetrics]
        Get comprehensive statistics for each action type
    get_interactions(scope: str = "simulation", agent_id: Optional[int] = None, ...) -> List[InteractionStats]
        Analyze agent interaction patterns and outcomes
    temporal_patterns(scope: str = "simulation", agent_id: Optional[int] = None, ...) -> Dict[str, TimePattern]
        Analyze action patterns over time
    resource_impacts(scope: str = "simulation", agent_id: Optional[int] = None, ...) -> List[ResourceImpact]
        Analyze resource impacts of different actions
    decision_patterns(scope: str = "simulation", agent_id: Optional[int] = None, ...) -> DecisionPatterns
        Analyze comprehensive decision-making patterns
    step(step_number: int) -> StepActionData
        Get detailed analysis of actions in a specific step
    agent_actions(agent_id: int, start_step: Optional[int], end_step: Optional[int]) -> Dict
        Get detailed action history for a specific agent
    get_causal_analysis(action_type: str) -> CausalAnalysis
        Analyze causal relationships between actions and outcomes
    get_exploration_exploitation(agent_id: Optional[int] = None) -> ExplorationExploitation
        Analyze exploration vs exploitation patterns
    get_learning_curve(agent_id: Optional[int] = None) -> LearningCurveAnalysis
        Analyze agent learning progress over time
    get_resilience_analysis(agent_id: Optional[int] = None) -> ResilienceAnalysis
        Analyze agent recovery patterns from failures
    get_action_analysis(action_type: str) -> ActionAnalysis
        Get comprehensive analysis for a specific action type
    get_step_data(step_number: int) -> StepActionData
        Get detailed analysis of actions in a specific step
    get_behavior_clustering() -> BehaviorClustering
        Cluster agents based on behavioral patterns
    get_adversarial_analysis(agent_id: Optional[int] = None) -> AdversarialInteractionAnalysis
        Analyze performance in competitive scenarios
    get_collaborative_analysis(agent_id: Optional[int] = None) -> CollaborativeInteractionAnalysis
        Analyze patterns and outcomes of cooperative behaviors
    get_environmental_impact(agent_id: Optional[int] = None) -> EnvironmentalImpactAnalysis
        Analyze how environment affects agent action outcomes
    get_conflict_analysis(agent_id: Optional[int] = None) -> ConflictAnalysis
        Analyze patterns of conflict and resolution strategies
    get_risk_reward_analysis(agent_id: Optional[int] = None) -> RiskRewardAnalysis
        Analyze risk-taking behavior and associated outcomes
    get_counterfactual_analysis(agent_id: Optional[int] = None) -> CounterfactualAnalysis
        Analyze potential alternative outcomes and missed opportunities

    Examples
    --------
    >>> retriever = ActionsRetriever(session)
    >>> metrics = retriever.get_action_stats()
    >>> patterns = retriever.decision_patterns()
    >>> learning = retriever.get_learning_curve()
    """

    @execute_query
    def get_action_stats(
        self,
        session,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[ActionMetrics]:
        """Get comprehensive statistics for each action type.

        Retrieves and analyzes statistics for all action types within the specified scope,
        including frequency of use, reward metrics, and performance indicators.

        Parameters
        ----------
        session : Session
            SQLAlchemy database session
        scope : Union[str, AnalysisScope], default=AnalysisScope.SIMULATION
            Analysis scope level:
            - "simulation": All data (no filters)
            - "step": Single step
            - "step_range": Range of steps
            - "agent": Single agent
        agent_id : Optional[int], default=None
            Specific agent ID to analyze. Required when scope is "agent".
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

        Examples
        --------
        >>> # Get global statistics
        >>> stats = retriever.get_action_stats()
        >>> for metric in stats:
        ...     print(f"{metric.action_type}:")
        ...     print(f"  Count: {metric.count}")
        ...     print(f"  Frequency: {metric.frequency:.2%}")
        ...     print(f"  Avg Reward: {metric.avg_reward:.2f}")
        attack:
          Count: 150
          Frequency: 25.00%
          Avg Reward: 1.75

        >>> # Get stats for specific agent
        >>> agent_stats = retriever.get_action_stats(scope="agent", agent_id=1)

        Notes
        -----
        - All reward metrics default to 0.0 if no rewards are recorded
        - Frequency calculations use the total actions within the scope
        - The method automatically validates scope parameters
        """
        query = session.query(
            AgentAction.action_type,
            func.count().label("count"),
            func.avg(AgentAction.reward).label("avg_reward"),
            func.min(AgentAction.reward).label("min_reward"),
            func.max(AgentAction.reward).label("max_reward"),
        )

        query = self._validate_and_filter_scope(
            session, query, scope, agent_id, step, step_range
        )
        results = query.group_by(AgentAction.action_type).all()

        total_actions = sum(r[1] for r in results)

        return [
            ActionMetrics(
                action_type=r[0],
                count=r[1],
                frequency=r[1] / total_actions if total_actions > 0 else 0,
                avg_reward=float(r[2] or 0),
                min_reward=float(r[3] or 0),
                max_reward=float(r[4] or 0),
            )
            for r in results
        ]

    @execute_query
    def get_interactions(
        self,
        session,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[InteractionStats]:
        """Analyze patterns and outcomes of agent interactions.

        Examines how agents interact with each other, analyzing the frequency and effectiveness
        of interactive vs. solo actions within the specified scope.

        Parameters
        ----------
        session : Session
            SQLAlchemy database session
        scope : Union[str, AnalysisScope], default=AnalysisScope.SIMULATION
            Analysis scope level:
            - "simulation": All data (no filters)
            - "step": Single step
            - "step_range": Range of steps
            - "agent": Single agent
        agent_id : Optional[int], default=None
            Specific agent ID to analyze. Required when scope is "agent".
        step : Optional[int], default=None
            Specific step to analyze. Required when scope is "step".
        step_range : Optional[Tuple[int, int]], default=None
            (start_step, end_step) range to analyze. Required when scope is "step_range".

        Returns
        -------
        List[InteractionStats]
            List of interaction statistics for each action type, containing:
            - action_type: str
                Type of action being analyzed
            - interaction_rate: float
                Proportion of actions involving other agents (0.0 to 1.0)
            - solo_performance: float
                Average reward for actions without targets
            - interaction_performance: float
                Average reward for actions with targets

        Examples
        --------
        >>> # Get global interaction patterns
        >>> interactions = retriever.interactions()
        >>> for stat in interactions:
        ...     print(f"{stat.action_type}:")
        ...     print(f"  Interaction rate: {stat.interaction_rate:.2%}")
        ...     print(f"  Solo reward: {stat.solo_performance:.2f}")
        ...     print(f"  Interactive reward: {stat.interaction_performance:.2f}")
        trade:
          Interaction rate: 95.00%
          Solo reward: 0.50
          Interactive reward: 2.25

        >>> # Get interactions for specific step range
        >>> step_interactions = retriever.interactions(
        ...     scope="step_range",
        ...     step_range=(100, 200)
        ... )

        Notes
        -----
        - Interactive actions are identified by non-null action_target_id
        - Performance metrics default to 0.0 if no rewards are recorded
        - Interaction rates are calculated per action type
        - The method handles both cooperative and competitive interactions

        See Also
        --------
        get_collaborative_analysis : Detailed analysis of cooperative behaviors
        get_adversarial_analysis : Analysis of competitive interactions
        """
        query = session.query(
            AgentAction.action_type,
            AgentAction.action_target_id.isnot(None).label("is_interaction"),
            func.count().label("count"),
            func.avg(AgentAction.reward).label("avg_reward"),
        )

        query = self._validate_and_filter_scope(
            session, query, scope, agent_id, step, step_range
        )
        stats = query.group_by(
            AgentAction.action_type,
            AgentAction.action_target_id.isnot(None),
        ).all()

        # Process results
        interaction_stats = {}
        for action_type, is_interaction, count, avg_reward in stats:
            if action_type not in interaction_stats:
                interaction_stats[action_type] = InteractionStats(
                    action_type=action_type,
                    interaction_rate=0.0,
                    solo_performance=0.0,
                    interaction_performance=0.0,
                )

            total = sum(s[2] for s in stats if s[0] == action_type)
            if is_interaction:
                interaction_stats[action_type].interaction_rate = (
                    count / total if total > 0 else 0
                )
                interaction_stats[action_type].interaction_performance = float(
                    avg_reward or 0
                )
            else:
                interaction_stats[action_type].solo_performance = float(avg_reward or 0)

        return list(interaction_stats.values())

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
        scope : Union[str, AnalysisScope], default=AnalysisScope.SIMULATION
            Analysis scope level:
            - "simulation": All data (no filters)
            - "step_range": Range of steps
            - "agent": Single agent
        agent_id : Optional[int], default=None
            Specific agent ID to analyze. Required when scope is "agent".
        step_range : Optional[Tuple[int, int]], default=None
            (start_step, end_step) range to analyze. Required when scope is "step_range".

        Returns
        -------
        List[TimePattern]
            List of temporal patterns for each action type, containing:
            - time_distribution : List[int]
                Action counts per time period showing usage patterns
            - reward_progression : List[float]
                Average rewards per time period showing effectiveness trends

        Notes
        -----
        Time periods are determined by grouping steps into bins of 100 steps each.
        Empty periods will have 0 counts and rewards.
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
        scope : Union[str, AnalysisScope], default=AnalysisScope.SIMULATION
            Analysis scope level:
            - "simulation": All data (no filters)
            - "step": Single step
            - "step_range": Range of steps
            - "agent": Single agent
        agent_id : Optional[int], default=None
            Specific agent ID to analyze. Required when scope is "agent".
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

        Notes
        -----
        Positive resource changes indicate generation/acquisition, while negative
        changes indicate consumption/loss.

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

        >>> # Get impacts for step range
        >>> range_impacts = retriever.resource_impacts(
        ...     scope="step_range",
        ...     step_range=(100, 200)
        ... )
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
        scope : Union[str, AnalysisScope], default=AnalysisScope.SIMULATION
            Analysis scope level:
            - "simulation": All data (no filters)
            - "step": Single step
            - "step_range": Range of steps
            - "agent": Single agent
        agent_id : Optional[int], default=None
            Specific agent ID to analyze. Required when scope is "agent".
        step : Optional[int], default=None
            Specific step to analyze. Required when scope is "step".
        step_range : Optional[Tuple[int, int]], default=None
            (start_step, end_step) range to analyze. Required when scope is "step_range".

        Returns
        -------
        DecisionPatterns
            Comprehensive decision pattern analysis with the following components:

            decision_patterns : List[DecisionPatternStats]
                Statistics for each action type, containing:
                - action_type: str
                    Type of action performed
                - count: Total number of times the action was taken
                - frequency: Proportion of times this action was chosen
                - reward_stats: Dict containing average/min/max rewards

            decision_summary : DecisionSummary
                Overall decision-making metrics, containing:
                - total_decisions: Total number of decisions made
                - unique_actions: Number of different action types used
                - most_frequent: Most commonly chosen action
                - most_rewarding: Action with highest average reward
                - action_diversity: Shannon entropy of action distribution

        Examples
        --------
        >>> # Get global decision patterns
        >>> patterns = retriever.decision_patterns()
        >>> print(f"Total decisions: {patterns.decision_summary.total_decisions}")
        >>> print(f"Most frequent action: {patterns.decision_summary.most_frequent}")
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
    def step(self, session, step_number: int) -> StepActionData:
        #! is this same as actions at step scope???
        #! should this be action_summary? Allowing scoping
        """Get detailed analysis of actions in a specific simulation step.

        Retrieves and analyzes all actions performed during a given simulation step,
        including action statistics, resource changes, interaction networks, and
        performance metrics.

        Parameters
        ----------
        step_number : int
            The simulation step number to analyze (must be >= 0)

        Returns
        -------
        StepActionData
            Comprehensive data about the step's actions, containing:

            step_summary : StepSummary
                Overall statistics including total actions, unique agents, etc.

            action_statistics : Dict[str, Dict]
                Statistics for each action type, including counts and rewards

            resource_metrics : ResourceMetricsStep
                Analysis of resource changes during the step

            interaction_network : InteractionNetwork
                Network of agent interactions and their outcomes

            performance_metrics : PerformanceMetrics
                Success rates and efficiency metrics

            detailed_actions : List[Dict]
                Detailed list of all actions with complete metadata

        Examples
        --------
        >>> step_data = retriever.step(step_number=5)
        >>> print("Step Summary:")
        Step Summary:
        >>> print(f"  Total actions: {step_data.step_summary.total_actions}")
        Total actions: 48
        >>> print(f"  Unique agents: {step_data.step_summary.unique_agents}")
        Unique agents: 12
        >>> print(f"  Total reward: {step_data.step_summary.total_reward:.2f}")
        Total reward: 85.50
        >>> print("\nPerformance Metrics:")
        Performance Metrics:
        >>> print(f"  Success rate: {step_data.performance_metrics.success_rate:.2%}")
        Success rate: 72.50%
        >>> print(f"  Average reward: {step_data.performance_metrics.average_reward:.2f}")
        Average reward: 1.78
        >>> print("\nResource Changes:")
        Resource Changes:
        >>> print(f"  Net change: {step_data.resource_metrics.net_resource_change:+.2f}")
        Net change: +125.50
        >>> print(f"  Average change: {step_data.resource_metrics.average_resource_change:+.2f}")
        Average change: +2.61
        >>> print("\nInteractions:")
        Interactions:
        >>> print(f"  Total: {step_data.step_summary.total_interactions}")
        Total: 18
        >>> print("  Network sample:")
        Network sample:
        >>> for i in step_data.interaction_network.interactions[:3]:
        ...     print(f"    {i['source']} -> {i['target']}: {i['action_type']} ({i['reward']:+.2f})")
        1 -> 4: trade (+2.50)
        2 -> 5: attack (-1.25)
        3 -> 1: help (+1.75)

        Notes
        -----
        This method provides a comprehensive snapshot of simulation activity at a specific
        step, useful for detailed analysis of agent behavior and system dynamics at
        particular points in time.

        See Also
        --------
        summary : Get basic metrics for all actions
        interactions : Get interaction patterns
        resource_impacts : Get resource change analysis
        """
        # Get all component data
        actions = self._get_step_actions(session, step_number)
        if not actions:
            return {}

        action_stats = self._get_action_statistics(session, step_number)
        resource_changes = self._get_resource_changes(session, step_number)

        # Build interaction network
        interactions = [
            action for action in actions if action.action_target_id is not None
        ]

        # Format detailed action list with state references
        action_list = [
            {
                "agent_id": action.agent_id,
                "action_type": action.action_type,
                "action_target_id": action.action_target_id,
                "state_before_id": action.state_before_id,
                "state_after_id": action.state_after_id,
                "resources_before": action.resources_before,
                "resources_after": action.resources_after,
                "reward": action.reward,
                "details": json.loads(action.details) if action.details else None,
            }
            for action in actions
        ]

        return StepActionData(
            step_summary=StepSummary(
                total_actions=len(actions),
                unique_agents=len(set(a.agent_id for a in actions)),
                action_types=len(set(a.action_type for a in actions)),
                total_interactions=len(interactions),
                total_reward=sum(a.reward for a in actions if a.reward is not None),
            ),
            action_statistics={
                action_type: {
                    "count": count,
                    "frequency": count / len(actions),
                    "avg_reward": float(avg_reward or 0),
                    "total_reward": float(total_reward or 0),
                }
                for action_type, count, avg_reward, total_reward in action_stats
            },
            resource_metrics=ResourceMetricsStep(
                net_resource_change=float(resource_changes[0] or 0),
                average_resource_change=float(resource_changes[1] or 0),
                resource_transactions=len(
                    [a for a in actions if a.resources_before != a.resources_after]
                ),
            ),
            interaction_network=InteractionNetwork(
                interactions=[
                    {
                        "source": action.agent_id,
                        "target": action.action_target_id,
                        "action_type": action.action_type,
                        "reward": action.reward,
                    }
                    for action in interactions
                ],
                unique_interacting_agents=len(
                    set(
                        [a.agent_id for a in interactions]
                        + [a.action_target_id for a in interactions]
                    )
                ),
            ),
            performance_metrics=PerformanceMetrics(
                success_rate=len([a for a in actions if a.reward and a.reward > 0]),
                average_reward=sum(a.reward for a in actions if a.reward is not None),
                action_efficiency=len(
                    [a for a in actions if a.state_before_id != a.state_after_id]
                )
                / len(actions),
            ),
            detailed_actions=action_list,
        )

    def _get_step_actions(self, session, step_number: int) -> List[AgentAction]:
        """Get all actions for a specific step.

        Retrieves a chronological list of all actions performed during the specified
        simulation step, including complete metadata for each action.

        Parameters
        ----------
        session : Session
            Database session for executing queries
        step_number : int
            The simulation step number to query (must be >= 0)

        Returns
        -------
        List[AgentAction]
            List of actions performed during the step, ordered by agent_id.
            Each action contains:
            - agent_id: ID of the acting agent
            - action_type: Type of action performed
            - action_target_id: Target agent ID (if any)
            - resources_before/after: Resource states
            - reward: Action outcome
            - details: Additional metadata

        Examples
        --------
        >>> actions = retriever._get_step_actions(session, step_number=5)
        >>> for action in actions:
        ...     print(f"Agent {action.agent_id}: {action.action_type}")
        """
        return (
            session.query(AgentAction)
            .filter(AgentAction.step_number == step_number)
            .order_by(AgentAction.agent_id)
            .all()
        )

    def _get_action_statistics(self, session, step_number: int) -> List[Tuple]:
        """Get action type statistics for a specific step.

        Analyzes the frequency and outcomes of different action types within a single
        simulation step, including counts and reward metrics.

        Parameters
        ----------
        session : Session
            Database session for executing queries
        step_number : int
            The simulation step number to query (must be >= 0)

        Returns
        -------
        List[Tuple]
            List of tuples containing:
            - action_type (str): The type of action performed
            - count (int): Number of times this action was taken
            - avg_reward (float): Average reward for this action type
            - total_reward (float): Total reward accumulated for this action type

        Notes
        -----
        Rewards may be None if no reward was recorded for an action.
        """
        return (
            session.query(
                AgentAction.action_type,
                func.count().label("count"),
                func.avg(AgentAction.reward).label("avg_reward"),
                func.sum(AgentAction.reward).label("total_reward"),
            )
            .filter(AgentAction.step_number == step_number)
            .group_by(AgentAction.action_type)
            .all()
        )

    def _get_resource_changes(self, session, step_number: int) -> Tuple:
        """Get resource change statistics for a specific step.

        Calculates aggregate resource changes across all agents during a single
        simulation step, providing both net and average changes.

        Parameters
        ----------
        session : Session
            Database session for executing queries
        step_number : int
            The simulation step number to query (must be >= 0)

        Returns
        -------
        Tuple
            Two-element tuple containing:
            - net_change (float): Total resource change across all agents
            - avg_change (float): Average resource change per agent

        Notes
        -----
        Only considers actions where both resources_before and resources_after
        are not None. Changes are calculated as (resources_after - resources_before).
        """
        return (
            session.query(
                func.sum(
                    AgentAction.resources_after - AgentAction.resources_before
                ).label("net_change"),
                func.avg(
                    AgentAction.resources_after - AgentAction.resources_before
                ).label("avg_change"),
            )
            .filter(
                AgentAction.step_number == step_number,
                AgentAction.resources_before.isnot(None),
                AgentAction.resources_after.isnot(None),
            )
            .first()
        )

    def _calculate_sequence_patterns(
        self,
        session,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, SequencePattern]:
        """Calculate action sequence patterns within the specified scope.

        Analyzes the sequential patterns in agent decision-making, identifying
        common action pairs and their transition probabilities.

        Parameters
        ----------
        session : Session
            Database session
        scope : Union[str, AnalysisScope], default=AnalysisScope.SIMULATION
            Analysis scope level:
            - "simulation": All data (no filters)
            - "step": Single step
            - "step_range": Range of steps
            - "agent": Single agent
        agent_id : Optional[int], default=None
            Specific agent ID to analyze. Required when scope is "agent".
        step : Optional[int], default=None
            Specific step to analyze. Required when scope is "step".
        step_range : Optional[Tuple[int, int]], default=None
            (start_step, end_step) range to analyze. Required when scope is "step_range".

        Returns
        -------
        Dict[str, SequencePattern]
            Statistics about action sequences, where keys are "action1->action2" and
            values contain:
            - count: Number of times the sequence occurred
            - probability: Likelihood of action2 following action1

        Examples
        --------
        >>> patterns = retriever._calculate_sequence_patterns(
        ...     session,
        ...     scope="agent",
        ...     agent_id=1
        ... )
        >>> for seq, stats in patterns.items():
        ...     print(f"{seq}: {stats.probability:.2%} probability")
        attack->defend: 35.20% probability
        defend->gather: 28.50% probability
        """
        # First get ordered actions
        query = session.query(
            AgentAction.agent_id, AgentAction.action_type, AgentAction.step_number
        ).order_by(AgentAction.step_number)

        # Apply scope filters
        if agent_id:
            query = query.filter(AgentAction.agent_id == agent_id)
        if step:
            query = query.filter(AgentAction.step_number == step)
        if step_range:
            query = query.filter(
                AgentAction.step_number >= step_range[0],
                AgentAction.step_number <= step_range[1],
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

        # Convert to SequencePattern objects with probabilities
        return {
            sequence: SequencePattern(
                count=count,
                probability=(
                    count / action_counts[sequence.split("->")[0]]
                    if sequence.split("->")[0] in action_counts
                    else 0
                ),
            )
            for sequence, count in sequences.items()
        }

    def _calculate_diversity(self, patterns: List[DecisionPatternStats]) -> float:
        """Calculate Shannon entropy for action diversity.

        Computes the Shannon entropy of the action distribution to measure
        the diversity of decision-making patterns.

        Parameters
        ----------
        patterns : List[DecisionPatternStats]
            List of decision pattern statistics containing frequency information

        Returns
        -------
        float
            Shannon entropy diversity measure, where:
            - Higher values indicate more diverse action selection
            - Lower values indicate more focused/specialized behavior

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
    def agent_actions(
        self,
        session,
        agent_id: int,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get detailed action history for a specific agent.

        Retrieves and analyzes the complete action history for a single agent,
        including chronological actions, statistics, interaction patterns,
        and resource management behavior.

        Parameters
        ----------
        agent_id : int
            ID of the agent to analyze
        start_step : Optional[int]
            Starting step for analysis window (inclusive)
        end_step : Optional[int]
            Ending step for analysis window (inclusive)

        Returns
        -------
        Dict[str, Any]
            Comprehensive analysis containing:
            action_history : Dict[str, Any]
                Chronological record of agent actions:
                - chronological_actions: List of actions with full details
                - total_actions: Total number of actions taken
                - unique_action_types: Number of different actions used
                - time_range: First and last action steps

            action_statistics : Dict[str, Dict[str, float]]
                Statistics per action type:
                - count: Times action was taken
                - frequency: Proportion of total actions
                - avg_reward: Average reward received
                - total_reward: Total reward accumulated

            interaction_patterns : Dict[str, InteractionPattern]
                Analysis of agent interactions:
                - interactions: List of interaction events
                - unique_interacting_agents: Number of unique interactions

            reward_analysis : RewardStats
                Reward statistics:
                - total_reward: Total accumulated reward
                - average_reward: Mean reward per action
                - reward_distribution: Distribution analysis
                - best_performing_actions: Top actions by reward

            resource_impact : ResourceBehavior
                Resource management analysis:
                - net_resource_change: Total resource change
                - resource_efficiency: Resources per action
                - resource_patterns: Usage patterns
                - resource_strategy: Management strategy

        Examples
        --------
        >>> history = retriever.agent_actions(agent_id=1, start_step=0, end_step=100)
        >>> print(f"Total actions: {history['action_history']['total_actions']}")
        >>> print(f"Best action: {history['reward_analysis']['best_performing_actions'][0]}")
        """
        # Use ActionsRetriever for analysis
        actions_retriever = self._retrievers["actions"]

        # Get decision patterns with time range filter
        patterns = actions_retriever.decision_patterns(session, agent_id)

        # Get chronological action list
        actions = session.query(AgentAction).filter(AgentAction.agent_id == agent_id)

        if start_step is not None:
            actions = actions.filter(AgentAction.step_number >= start_step)
        if end_step is not None:
            actions = actions.filter(AgentAction.step_number <= end_step)

        actions = actions.order_by(AgentAction.step_number).all()

        # Format chronological action list
        action_list = [
            {
                "step_number": action.step_number,
                "action_type": action.action_type,
                "action_target_id": action.action_target_id,
                "position_before": action.position_before,
                "position_after": action.position_after,
                "resources_before": action.resources_before,
                "resources_after": action.resources_after,
                "reward": action.reward,
                "details": json.loads(action.details) if action.details else None,
            }
            for action in actions
        ]

        return {
            "action_history": {
                "chronological_actions": action_list,
                "total_actions": len(actions),
                "unique_action_types": len(set(a.action_type for a in actions)),
                "time_range": {
                    "first_action": (
                        action_list[0]["step_number"] if action_list else None
                    ),
                    "last_action": (
                        action_list[-1]["step_number"] if action_list else None
                    ),
                },
            },
            "action_statistics": patterns.decision_patterns,
            "interaction_patterns": patterns.interaction_analysis,
            "reward_analysis": patterns.decision_summary,
            "resource_impact": patterns.resource_impact,
        }

    @execute_query
    def get_action_analysis(self, session, action_type: str) -> ActionAnalysis:
        """Get comprehensive analysis for a specific action type.

        Performs detailed analysis of a single action type, examining its usage patterns,
        effectiveness, resource impacts, and interaction characteristics throughout the
        simulation.

        Parameters
        ----------
        action_type : str
            The specific type of action to analyze (e.g., "attack", "defend", "trade")

        Returns
        -------
        ActionAnalysis
            Comprehensive analysis results containing:
            - stats : ActionStats
                Basic statistics including counts, frequencies, and reward metrics
            - time_pattern : TimePattern
                Temporal evolution of action usage and effectiveness
            - resource_impact : ResourceImpact
                Analysis of resource consumption and generation
            - interaction_stats : InteractionStats
                Patterns of agent interactions and their outcomes
            - sequence_patterns : Dict[str, SequencePattern]
                Common action sequences and their probabilities

        Examples
        --------
        >>> analysis = retriever.get_action_analysis("attack")
        >>> print(f"Usage frequency: {analysis.stats.frequency:.2%}")
        Usage frequency: 25.30%
        >>> print(f"Average reward: {analysis.stats.avg_reward:.2f}")
        Average reward: 1.85
        >>> print("Common sequences:", list(analysis.sequence_patterns.keys())[:3])
        Common sequences: ['attack->defend', 'attack->retreat', 'attack->attack']

        Notes
        -----
        This method combines multiple analysis perspectives to provide a complete
        understanding of how the action type is used and its effectiveness in
        different contexts.

        See Also
        --------
        summary : Get basic metrics for all actions
        temporal_patterns : Analyze patterns over time
        resource_impacts : Analyze resource effects
        """
        return ActionAnalysis(
            stats=self._get_action_stats_single(session, action_type),
            time_pattern=self._get_time_pattern(session, action_type),
            resource_impact=self._get_resource_impact(session, action_type),
            interaction_stats=self._get_interaction_stats(session, action_type),
            sequence_patterns=self._get_sequence_patterns(session, action_type),
        )

    @execute_query
    def get_step_data(self, session, step_number: int) -> StepActionData:
        """Get detailed analysis of actions in a specific simulation step.

        Retrieves and analyzes all actions performed during a given simulation step,
        including action statistics, resource changes, interaction networks, and
        performance metrics.

        Parameters
        ----------
        step_number : int
            The simulation step number to analyze (must be >= 0)

        Returns
        -------
        StepActionData
            Comprehensive data about the step's actions, containing:

            step_summary : StepSummary
                Overall statistics including total actions, unique agents, etc.

            action_statistics : Dict[str, Dict]
                Statistics for each action type, including counts and rewards

            resource_metrics : ResourceMetricsStep
                Analysis of resource changes during the step

            interaction_network : InteractionNetwork
                Network of agent interactions and their outcomes

            performance_metrics : PerformanceMetrics
                Success rates and efficiency metrics

            detailed_actions : List[Dict]
                Detailed list of all actions with complete metadata

        Examples
        --------
        >>> step_data = retriever.get_step_data(step_number=5)
        >>> print("Step Summary:")
        Step Summary:
        >>> print(f"  Total actions: {step_data.step_summary.total_actions}")
        Total actions: 48
        >>> print(f"  Unique agents: {step_data.step_summary.unique_agents}")
        Unique agents: 12
        >>> print(f"  Total reward: {step_data.step_summary.total_reward:.2f}")
        Total reward: 85.50
        >>> print("\nPerformance Metrics:")
        Performance Metrics:
        >>> print(f"  Success rate: {step_data.performance_metrics.success_rate:.2%}")
        Success rate: 72.50%
        >>> print(f"  Average reward: {step_data.performance_metrics.average_reward:.2f}")
        Average reward: 1.78
        >>> print("\nResource Changes:")
        Resource Changes:
        >>> print(f"  Net change: {step_data.resource_metrics.net_resource_change:+.2f}")
        Net change: +125.50
        >>> print(f"  Average change: {step_data.resource_metrics.average_resource_change:+.2f}")
        Average change: +2.61
        >>> print("\nInteractions:")
        Interactions:
        >>> print(f"  Total: {step_data.step_summary.total_interactions}")
        Total: 18
        >>> print("  Network sample:")
        Network sample:
        >>> for i in step_data.interaction_network.interactions[:3]:
        ...     print(f"    {i['source']} -> {i['target']}: {i['action_type']} ({i['reward']:+.2f})")
        1 -> 4: trade (+2.50)
        2 -> 5: attack (-1.25)
        3 -> 1: help (+1.75)

        Notes
        -----
        This method provides a comprehensive snapshot of simulation activity at a specific
        step, useful for detailed analysis of agent behavior and system dynamics at
        particular points in time.

        See Also
        --------
        summary : Get basic metrics for all actions
        interactions : Get interaction patterns
        resource_impacts : Get resource change analysis
        """
        # Get all actions for this step
        actions = (
            session.query(AgentAction)
            .filter(AgentAction.step_number == step_number)
            .all()
        )

        if not actions:
            return StepActionData(
                step_summary=StepSummary(
                    total_actions=0,
                    unique_agents=0,
                    action_types=0,
                    total_interactions=0,
                    total_reward=0,
                ),
                action_statistics={},
                resource_metrics=ResourceMetricsStep(
                    net_resource_change=0,
                    average_resource_change=0,
                    resource_transactions=0,
                ),
                interaction_network=InteractionNetwork(
                    interactions=[],
                    unique_interacting_agents=0,
                ),
                performance_metrics=PerformanceMetrics(
                    success_rate=0,
                    average_reward=0,
                    action_efficiency=0,
                ),
                detailed_actions=[],
            )

        # Get interactions (actions with targets)
        interactions = [
            action for action in actions if action.action_target_id is not None
        ]

        # Calculate action statistics
        action_stats = self._get_action_statistics(session, step_number)
        resource_changes = self._get_resource_changes(session, step_number)

        # Format detailed action list
        action_list = [
            {
                "agent_id": action.agent_id,
                "action_type": action.action_type,
                "action_target_id": action.action_target_id,
                "state_before_id": action.state_before_id,
                "state_after_id": action.state_after_id,
                "resources_before": action.resources_before,
                "resources_after": action.resources_after,
                "reward": action.reward,
                "details": json.loads(action.details) if action.details else None,
            }
            for action in actions
        ]

        return StepActionData(
            step_summary=StepSummary(
                total_actions=len(actions),
                unique_agents=len(set(a.agent_id for a in actions)),
                action_types=len(set(a.action_type for a in actions)),
                total_interactions=len(interactions),
                total_reward=sum(a.reward for a in actions if a.reward is not None),
            ),
            action_statistics={
                action_type: {
                    "count": count,
                    "frequency": count / len(actions),
                    "avg_reward": float(avg_reward or 0),
                    "total_reward": float(total_reward or 0),
                }
                for action_type, count, avg_reward, total_reward in action_stats
            },
            resource_metrics=ResourceMetricsStep(
                net_resource_change=float(resource_changes[0] or 0),
                average_resource_change=float(resource_changes[1] or 0),
                resource_transactions=len(
                    [a for a in actions if a.resources_before != a.resources_after]
                ),
            ),
            interaction_network=InteractionNetwork(
                interactions=[
                    {
                        "source": action.agent_id,
                        "target": action.action_target_id,
                        "action_type": action.action_type,
                        "reward": action.reward,
                    }
                    for action in interactions
                ],
                unique_interacting_agents=len(
                    set(
                        [a.agent_id for a in interactions]
                        + [a.action_target_id for a in interactions]
                    )
                ),
            ),
            performance_metrics=PerformanceMetrics(
                success_rate=len([a for a in actions if a.reward and a.reward > 0])
                / len(actions),
                average_reward=sum(a.reward for a in actions if a.reward is not None)
                / len(actions),
                action_efficiency=len(
                    [a for a in actions if a.state_before_id != a.state_after_id]
                )
                / len(actions),
            ),
            detailed_actions=action_list,
        )

    @execute_query
    def get_causal_analysis(self, session, action_type: str) -> CausalAnalysis:
        """Analyze causal relationships between actions and outcomes.

        Examines the cause-effect relationships between actions and their
        consequences, including direct impacts and downstream effects.

        Parameters
        ----------
        action_type : str
            Specific action type to analyze causal patterns for

        Returns
        -------
        CausalAnalysis
            Analysis results containing:
            - direct_effects : Dict[str, float]
                Immediate outcomes of the action
            - indirect_effects : Dict[str, float]
                Secondary effects observed in subsequent steps
            - context_dependency : Dict[str, float]
                How outcomes vary based on environmental conditions

        Notes
        -----
        Analysis uses temporal sequence and correlation patterns to suggest
        causal relationships, but correlation doesn't guarantee causation.

        Examples
        --------
        >>> causal = retriever.get_causal_analysis("attack")
        >>> print("Direct effects:", causal.direct_effects)
        Direct effects: {'damage': -2.5, 'resource_loss': -1.2, 'position_change': 0.8}
        >>> print("Context factors:", causal.context_dependency)
        Context factors: {'resource_level': 0.65, 'agent_density': 0.45, 'time_of_day': 0.12}
        >>> print("Indirect effects:", causal.indirect_effects)
        Indirect effects: {'reputation': -0.8, 'future_interactions': -0.35}
        """
        # Implementation details...
        pass

    @execute_query
    def get_behavior_clustering(self, session) -> BehaviorClustering:
        """Cluster agents based on behavioral patterns.

        Groups agents with similar decision-making patterns and analyzes
        characteristics of each behavioral cluster.

        Returns
        -------
        BehaviorClustering
            Clustering analysis results containing:
            - clusters : Dict[str, List[int]]
                Groups of agent IDs with similar behaviors
            - cluster_characteristics : Dict[str, Dict[str, float]]
                Key behavioral metrics for each cluster
            - cluster_performance : Dict[str, float]
                Average performance metrics per cluster

        Notes
        -----
        Clustering is based on action frequencies, interaction patterns,
        and performance metrics.

        Examples
        --------
        >>> clusters = retriever.get_behavior_clustering()
        >>> for name, agents in clusters.clusters.items():
        ...     print(f"{name}: {len(agents)} agents")
        ...     print(f"Performance: {clusters.cluster_performance[name]:.2f}")
        ...     print("Characteristics:", clusters.cluster_characteristics[name])
        Aggressive: 12 agents
        Performance: 1.85
        Characteristics: {'attack_rate': 0.75, 'cooperation': 0.15, 'risk_taking': 0.85}
        Cooperative: 8 agents
        Performance: 2.15
        Characteristics: {'attack_rate': 0.25, 'cooperation': 0.85, 'risk_taking': 0.35}
        Balanced: 15 agents
        Performance: 1.95
        Characteristics: {'attack_rate': 0.45, 'cooperation': 0.55, 'risk_taking': 0.50}
        """
        # Implementation details...
        pass

    @execute_query
    def get_exploration_exploitation(
        self, session, agent_id: Optional[int] = None
    ) -> ExplorationExploitation:
        """Analyze how agents balance exploration and exploitation.

        Examines the trade-off between trying new actions (exploration) and
        repeating successful actions (exploitation), including reward comparisons
        and strategy evolution.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.

        Returns
        -------
        ExplorationExploitation
            Analysis results containing:
            - exploration_rate : float
                Proportion of actions that are first-time attempts
            - exploitation_rate : float
                Proportion of actions that repeat previous choices
            - reward_comparison : Dict[str, float]
                Comparison of rewards between new and known actions:
                - new_actions_avg: Mean reward for first attempts
                - known_actions_avg: Mean reward for repeated actions

        Notes
        -----
        Rates sum to 1.0. Higher exploration rates indicate more experimental
        behavior, while higher exploitation rates suggest more conservative
        strategies.

        Examples
        --------
        >>> analysis = retriever.get_exploration_exploitation(agent_id=1)
        >>> print(f"Exploration rate: {analysis.exploration_rate:.2%}")
        >>> print(f"New vs Known rewards: {analysis.reward_comparison}")
        """
        query = session.query(AgentAction)
        if agent_id:
            query = query.filter(AgentAction.agent_id == agent_id)

        actions = query.order_by(AgentAction.step_number).all()

        # Track unique and repeated actions
        action_history = {}
        exploration_count = 0
        exploitation_count = 0
        new_action_rewards = []
        known_action_rewards = []

        for action in actions:
            action_key = (action.agent_id, action.action_type)

            if action_key not in action_history:
                exploration_count += 1
                if action.reward is not None:
                    new_action_rewards.append(action.reward)
                action_history[action_key] = action.reward
            else:
                exploitation_count += 1
                if action.reward is not None:
                    known_action_rewards.append(action.reward)

        total_actions = exploration_count + exploitation_count

        return ExplorationExploitation(
            exploration_rate=(
                exploration_count / total_actions if total_actions > 0 else 0
            ),
            exploitation_rate=(
                exploitation_count / total_actions if total_actions > 0 else 0
            ),
            reward_comparison={
                "new_actions_avg": (
                    sum(new_action_rewards) / len(new_action_rewards)
                    if new_action_rewards
                    else 0
                ),
                "known_actions_avg": (
                    sum(known_action_rewards) / len(known_action_rewards)
                    if known_action_rewards
                    else 0
                ),
            },
        )

    @execute_query
    def get_adversarial_analysis(
        self, session, agent_id: Optional[int] = None
    ) -> AdversarialInteractionAnalysis:
        """Analyze performance in competitive scenarios.

        Examines the relationship between action risk levels and rewards,
        including risk appetite assessment and outcome analysis.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.

        Returns
        -------
        AdversarialInteractionAnalysis
            Analysis results containing:
            - win_rate : float
                Proportion of high-risk actions taken (0.0 to 1.0)
            - damage_efficiency : float
                Average reward for high-risk actions
            - counter_strategies : Dict[str, float]
                Counter-strategies used against high-risk actions

        Notes
        -----
        Risk level is determined by reward variance relative to mean reward.
        High risk: variance > mean/2, Low risk: variance <= mean/2

        Examples
        --------
        >>> adversarial = retriever.get_adversarial_analysis(agent_id=1)
        >>> print(f"Win rate: {adversarial.win_rate:.2%}")
        Win rate: 65.30%
        >>> print(f"Damage efficiency: {adversarial.damage_efficiency:.2f}")
        Damage efficiency: 2.45
        >>> print("Counter strategies:")
        Counter strategies:
        >>> for action, freq in adversarial.counter_strategies.items():
        ...     print(f"  {action}: {freq:.2%}")
        defend: 45.20%
        retreat: 30.50%
        counter_attack: 24.30%
        """
        query = session.query(AgentAction).filter(
            AgentAction.action_type.in_(["attack", "defend", "compete"])
        )

        if agent_id:
            query = query.filter(AgentAction.agent_id == agent_id)

        actions = query.all()

        successful = [a for a in actions if a.reward and a.reward > 0]

        # Calculate counter-strategies
        counter_actions = {}
        for action in actions:
            if action.action_target_id:
                # Get target's next action
                target_response = (
                    session.query(AgentAction)
                    .filter(
                        AgentAction.agent_id == action.action_target_id,
                        AgentAction.step_number > action.step_number,
                    )
                    .order_by(AgentAction.step_number)
                    .first()
                )
                if target_response:
                    if target_response.action_type not in counter_actions:
                        counter_actions[target_response.action_type] = 0
                    counter_actions[target_response.action_type] += 1

        total_counters = sum(counter_actions.values())
        counter_frequencies = (
            {
                action: count / total_counters
                for action, count in counter_actions.items()
            }
            if total_counters > 0
            else {}
        )

        return AdversarialInteractionAnalysis(
            win_rate=len(successful) / len(actions) if actions else 0,
            damage_efficiency=(
                sum(a.reward for a in successful) / len(successful) if successful else 0
            ),
            counter_strategies=counter_frequencies,
        )

    @execute_query
    def get_collaborative_analysis(
        self, session, agent_id: Optional[int] = None
    ) -> CollaborativeInteractionAnalysis:
        """Analyze patterns and outcomes of cooperative behaviors.

        Examines collaborative interactions between agents, including resource sharing,
        mutual aid, and the benefits of cooperation versus individual action.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.

        Returns
        -------
        CollaborativeInteractionAnalysis
            Analysis results containing:
            - collaboration_rate : float
                Proportion of actions involving cooperation
            - group_reward_impact : float
                Average reward benefit from collaborative actions
            - synergy_metrics : float
                Performance difference between collaborative and solo actions

        Notes
        -----
        Positive synergy metrics indicate that collaboration is more effective
        than individual action.

        Examples
        --------
        >>> collab = retriever.get_collaborative_analysis(agent_id=1)
        >>> print(f"Collaboration rate: {collab.collaboration_rate:.2%}")
        >>> print(f"Group reward impact: {collab.group_reward_impact:.2f}")
        >>> print(f"Synergy benefit: {collab.synergy_metrics:+.2f}")
        """
        query = session.query(AgentAction).filter(
            AgentAction.action_type.in_(["share", "help", "cooperate"])
        )

        if agent_id:
            query = query.filter(AgentAction.agent_id == agent_id)

        actions = query.all()
        total_actions = session.query(AgentAction).count()

        collaborative_rewards = [a.reward for a in actions if a.reward is not None]
        solo_actions = (
            session.query(AgentAction)
            .filter(AgentAction.action_target_id.is_(None))
            .all()
        )
        solo_rewards = [a.reward for a in solo_actions if a.reward is not None]

        return CollaborativeInteractionAnalysis(
            collaboration_rate=len(actions) / total_actions if total_actions > 0 else 0,
            group_reward_impact=(
                sum(collaborative_rewards) / len(collaborative_rewards)
                if collaborative_rewards
                else 0
            ),
            synergy_metrics=(
                (
                    sum(collaborative_rewards) / len(collaborative_rewards)
                    - sum(solo_rewards) / len(solo_rewards)
                )
                if collaborative_rewards and solo_rewards
                else 0
            ),
        )

    @execute_query
    def get_learning_curve(
        self, session, agent_id: Optional[int] = None
    ) -> LearningCurveAnalysis:
        """Analyze agent learning progress and performance improvements over time.

        Examines how agent performance evolves throughout the simulation by tracking success
        rates, reward progression, and mistake reduction patterns.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.

        Returns
        -------
        LearningCurveAnalysis
            Comprehensive learning analysis containing:
            - action_success_over_time : List[float]
                Success rates across time periods
            - reward_progression : List[float]
                Average rewards across time periods
            - mistake_reduction : float
                Measure of improvement in avoiding mistakes

        Examples
        --------
        >>> curve = retriever.get_learning_curve(agent_id=1)
        >>> print(f"Mistake reduction: {curve.mistake_reduction:.2%}")
        Mistake reduction: 23.50%
        >>> print(f"Final success rate: {curve.action_success_over_time[-1]:.2%}")
        Final success rate: 78.25%
        >>> print("Reward progression:", [f"{r:.2f}" for r in curve.reward_progression[:5]])
        Reward progression: ['0.12', '0.45', '0.67', '0.89', '1.23']
        """
        query = session.query(
            AgentAction.step_number,
            func.avg(AgentAction.reward).label("avg_reward"),
            func.count(case([(AgentAction.reward > 0, 1)])).label("successes"),
            func.count().label("total"),
        ).group_by(
            func.floor(AgentAction.step_number / 100)  # Group by time periods
        )

        if agent_id:
            query = query.filter(AgentAction.agent_id == agent_id)

        results = query.all()

        return LearningCurveAnalysis(
            action_success_over_time=[
                r.successes / r.total if r.total > 0 else 0 for r in results
            ],
            reward_progression=[float(r.avg_reward or 0) for r in results],
            mistake_reduction=self._calculate_mistake_reduction(results),
        )

    @execute_query
    def get_environmental_impact(
        self, session, agent_id: Optional[int] = None
    ) -> EnvironmentalImpactAnalysis:
        """Analyze how environment affects agent action outcomes.

        Examines the relationship between environmental states (like resource levels)
        and action outcomes, including adaptation patterns and spatial effects.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.

        Returns
        -------
        EnvironmentalImpactAnalysis
            Analysis results containing:
            - environmental_state_impact : Dict[str, float]
                Correlation between resource levels and action outcomes
            - adaptive_behavior : Dict[str, float]
                Measures of agent adaptation to environmental changes
            - spatial_analysis : Dict[str, Any]
                Analysis of location-based patterns and effects

        Examples
        --------
        >>> impact = retriever.get_environmental_impact(agent_id=1)
        >>> print("Resource correlations:")
        Resource correlations:
        >>> for action, corr in impact.environmental_state_impact.items():
        ...     print(f"  {action}: {corr:.2f}")
        attack: -0.45
        defend: +0.25
        gather: +0.85
        >>> print(f"Adaptation rate: {impact.adaptive_behavior['adaptation_rate']:.2%}")
        Adaptation rate: 65.30%
        >>> print("Spatial patterns:", impact.spatial_analysis['movement_patterns'])
        Spatial patterns: ['resource_following', 'threat_avoidance', 'group_formation']
        """
        # Get actions with their states
        query = session.query(AgentAction, AgentState).join(
            AgentState, AgentAction.state_before_id == AgentState.id
        )

        if agent_id:
            query = query.filter(AgentAction.agent_id == agent_id)

        results = query.all()

        # Analyze resource levels impact
        resource_impacts = {}
        for action, state in results:
            resource_level = state.resource_level
            if action.action_type not in resource_impacts:
                resource_impacts[action.action_type] = []
            if action.reward is not None:
                resource_impacts[action.action_type].append(
                    (resource_level, action.reward)
                )

        return EnvironmentalImpactAnalysis(
            environmental_state_impact={
                action: self._calculate_correlation(
                    [r[0] for r in rewards], [r[1] for r in rewards]
                )
                for action, rewards in resource_impacts.items()
            },
            adaptive_behavior=self._analyze_adaptation(results),
            spatial_analysis=self._analyze_spatial_patterns(results),
        )

    @execute_query
    def get_conflict_analysis(
        self, session, agent_id: Optional[int] = None
    ) -> ConflictAnalysis:
        """Analyze patterns of conflict and conflict resolution strategies.

        Examines sequences of actions leading to and resolving conflicts, including
        trigger patterns, resolution strategies, and outcome analysis.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.

        Returns
        -------
        ConflictAnalysis
            Analysis results containing:
            - conflict_trigger_actions : Dict[str, float]
                Actions that commonly lead to conflicts
            - conflict_resolution_actions : Dict[str, float]
                Actions used to resolve conflicts
            - conflict_outcome_metrics : Dict[str, float]
                Success rates of different resolution strategies

        Examples
        --------
        >>> conflicts = retriever.get_conflict_analysis(agent_id=1)
        >>> print("Common triggers:", conflicts.conflict_trigger_actions)
        >>> print("Best resolution:", max(conflicts.conflict_outcome_metrics.items(),
        ...       key=lambda x: x[1])[0])
        """
        # Get sequences of actions
        query = session.query(AgentAction).order_by(
            AgentAction.agent_id, AgentAction.step_number
        )

        if agent_id:
            query = query.filter(AgentAction.agent_id == agent_id)

        actions = query.all()

        # Analyze action sequences leading to conflicts
        conflict_triggers = {}
        conflict_resolutions = {}
        conflict_outcomes = {}

        for i in range(len(actions) - 1):
            current = actions[i]
            next_action = actions[i + 1]

            if next_action.action_type in ["attack", "defend"]:
                if current.action_type not in conflict_triggers:
                    conflict_triggers[current.action_type] = 0
                conflict_triggers[current.action_type] += 1

            if current.action_type in ["attack", "defend"]:
                if next_action.action_type not in conflict_resolutions:
                    conflict_resolutions[next_action.action_type] = 0
                    conflict_outcomes[next_action.action_type] = []
                conflict_resolutions[next_action.action_type] += 1
                if next_action.reward is not None:
                    conflict_outcomes[next_action.action_type].append(
                        next_action.reward
                    )

        return ConflictAnalysis(
            conflict_trigger_actions=self._normalize_dict(conflict_triggers),
            conflict_resolution_actions=self._normalize_dict(conflict_resolutions),
            conflict_outcome_metrics={
                action: sum(outcomes) / len(outcomes) if outcomes else 0
                for action, outcomes in conflict_outcomes.items()
            },
        )

    @execute_query
    def get_risk_reward_analysis(
        self, session, agent_id: Optional[int] = None
    ) -> RiskRewardAnalysis:
        """Analyze risk-taking behavior and associated outcomes.

        Examines the relationship between action risk levels and rewards,
        including risk appetite assessment and outcome analysis.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.

        Returns
        -------
        RiskRewardAnalysis
            Analysis results containing:
            - high_risk_actions : Dict[str, float]
                Actions with high reward variance and their average returns
            - low_risk_actions : Dict[str, float]
                Actions with low reward variance and their average returns
            - risk_appetite : float
                Proportion of high-risk actions taken (0.0 to 1.0)

        Notes
        -----
        Risk level is determined by reward variance relative to mean reward.
        High risk: variance > mean/2, Low risk: variance <= mean/2

        Examples
        --------
        >>> risk = retriever.get_risk_reward_analysis(agent_id=1)
        >>> print(f"Risk appetite: {risk.risk_appetite:.2%}")
        Risk appetite: 35.80%
        >>> print("High-risk actions:")
        High-risk actions:
        >>> for action, reward in risk.high_risk_actions.items():
        ...     print(f"  {action}: {reward:+.2f}")
        attack: +2.45
        explore: +1.78
        trade: -0.92
        >>> print("Best high-risk action:", max(risk.high_risk_actions.items(),
        ...       key=lambda x: x[1])[0])
        Best high-risk action: attack
        """
        query = session.query(
            AgentAction.action_type,
            func.stddev(AgentAction.reward).label("reward_std"),
            func.avg(AgentAction.reward).label("reward_avg"),
            func.count().label("count"),
        ).group_by(AgentAction.action_type)

        if agent_id:
            query = query.filter(AgentAction.agent_id == agent_id)

        results = query.all()

        # Classify actions by risk level
        high_risk = {}
        low_risk = {}
        for result in results:
            if result.reward_std > result.reward_avg / 2:  # High variability
                high_risk[result.action_type] = float(result.reward_avg or 0)
            else:
                low_risk[result.action_type] = float(result.reward_avg or 0)

        # Calculate risk appetite
        total_actions = sum(r.count for r in results)
        high_risk_actions = sum(r.count for r in results if r.action_type in high_risk)

        return RiskRewardAnalysis(
            high_risk_actions=high_risk,
            low_risk_actions=low_risk,
            risk_appetite=high_risk_actions / total_actions if total_actions > 0 else 0,
        )

    @execute_query
    def get_counterfactual_analysis(
        self, session, agent_id: Optional[int] = None
    ) -> CounterfactualAnalysis:
        """Analyze potential alternative outcomes and missed opportunities.

        Examines what-if scenarios by analyzing unused or underutilized actions
        and their potential impacts based on observed outcomes.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.

        Returns
        -------
        CounterfactualAnalysis
            Analysis results containing:
            - counterfactual_rewards : Dict[str, float]
                Potential rewards from alternative actions
            - missed_opportunities : Dict[str, float]
                High-value actions that were underutilized
            - strategy_comparison : Dict[str, float]
                Performance delta between actual and optimal strategies

        Examples
        --------
        >>> analysis = retriever.get_counterfactual_analysis(agent_id=1)
        >>> print("Missed opportunities:")
        Missed opportunities:
        >>> for action, value in analysis.missed_opportunities.items():
        ...     print(f"  {action}: {value:+.2f}")
        trade: +2.45
        cooperate: +1.85
        explore: +1.25
        >>> print("Strategy comparison:")
        Strategy comparison:
        >>> for strategy, delta in analysis.strategy_comparison.items():
        ...     print(f"  {strategy}: {delta:+.2f} vs optimal")
        aggressive: -0.85 vs optimal
        defensive: -0.45 vs optimal
        cooperative: +0.35 vs optimal
        """
        # Get actual action history
        query = session.query(AgentAction)
        if agent_id:
            query = query.filter(AgentAction.agent_id == agent_id)
        actions = query.all()

        # Calculate average rewards for each action type
        action_rewards = {}
        for action in actions:
            if action.action_type not in action_rewards:
                action_rewards[action.action_type] = []
            if action.reward is not None:
                action_rewards[action.action_type].append(action.reward)

        avg_rewards = {
            action: sum(rewards) / len(rewards)
            for action, rewards in action_rewards.items()
            if rewards
        }

        # Find unused or underused actions
        all_action_types = set(a.action_type for a in actions)
        action_counts = {
            action: len(rewards) for action, rewards in action_rewards.items()
        }
        median_usage = sorted(action_counts.values())[len(action_counts) // 2]
        underused = {
            action: count
            for action, count in action_counts.items()
            if count < median_usage / 2
        }

        return CounterfactualAnalysis(
            counterfactual_rewards=avg_rewards,
            missed_opportunities={
                action: avg_rewards.get(action, 0) for action in underused
            },
            strategy_comparison={
                action: reward - sum(avg_rewards.values()) / len(avg_rewards)
                for action, reward in avg_rewards.items()
            },
        )

    @execute_query
    def get_resilience_analysis(
        self, session, agent_id: Optional[int] = None
    ) -> ResilienceAnalysis:
        """Analyze agent recovery patterns and adaptation to failures.

        Examines how agents respond to and recover from negative outcomes, including recovery
        speed, adaptation strategies, and impact assessment.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.

        Returns
        -------
        ResilienceAnalysis
            Comprehensive resilience metrics containing:
            - recovery_rate : float
                Average time steps needed to recover from failures
            - adaptation_rate : float
                Rate at which agents modify strategies after failures
            - failure_impact : float
                Average performance impact of failures

        Examples
        --------
        >>> resilience = retriever.get_resilience_analysis(agent_id=1)
        >>> print(f"Recovery rate: {resilience.recovery_rate:.2f} steps")
        >>> print(f"Adaptation rate: {resilience.adaptation_rate:.2%}")
        """
        query = session.query(AgentAction).order_by(
            AgentAction.agent_id, AgentAction.step_number
        )

        if agent_id:
            query = query.filter(AgentAction.agent_id == agent_id)

        actions = query.all()

        # Track failure sequences
        recovery_times = []
        adaptation_speeds = []
        failure_impacts = []

        current_failure = False
        failure_start = 0
        pre_failure_reward = 0

        for i, action in enumerate(actions):
            if action.reward is not None and action.reward < 0:
                if not current_failure:
                    current_failure = True
                    failure_start = i
                    pre_failure_reward = (
                        sum(
                            a.reward
                            for a in actions[max(0, i - 5) : i]
                            if a.reward is not None
                        )
                        / 5
                    )
            elif current_failure and action.reward is not None and action.reward > 0:
                # Recovery detected
                recovery_times.append(i - failure_start)

                # Calculate adaptation speed
                post_failure_actions = actions[failure_start:i]
                if post_failure_actions:
                    adaptation = sum(
                        1
                        for a in post_failure_actions
                        if a.action_type != actions[failure_start].action_type
                    ) / len(post_failure_actions)
                    adaptation_speeds.append(adaptation)

                # Calculate impact
                failure_impacts.append(
                    pre_failure_reward
                    - min(
                        a.reward for a in post_failure_actions if a.reward is not None
                    )
                )

                current_failure = False

        return ResilienceAnalysis(
            recovery_rate=(
                sum(recovery_times) / len(recovery_times) if recovery_times else 0
            ),
            adaptation_rate=(
                sum(adaptation_speeds) / len(adaptation_speeds)
                if adaptation_speeds
                else 0
            ),
            failure_impact=(
                sum(failure_impacts) / len(failure_impacts) if failure_impacts else 0
            ),
        )

    def _calculate_mistake_reduction(self, results) -> float:
        """Calculate the reduction in mistake rate over time.

        Compares early-stage mistake rates with late-stage rates to measure
        improvement in decision-making accuracy.

        Parameters
        ----------
        results : List[Row]
            Query results containing success/failure counts per time period

        Returns
        -------
        float
            Reduction in mistake rate, where:
            - Positive values indicate fewer mistakes over time
            - Zero indicates no improvement
            - Values range from 0.0 to 1.0

        Notes
        -----
        Mistake rate is calculated as (1 - success_rate) for each period.
        The reduction is the difference between early and late mistake rates.
        """
        if not results:
            return 0
        early_mistakes = (
            1 - results[0].successes / results[0].total if results[0].total > 0 else 0
        )
        late_mistakes = (
            1 - results[-1].successes / results[-1].total
            if results[-1].total > 0
            else 0
        )
        return max(0, early_mistakes - late_mistakes)

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient between two variables.

        Measures the linear correlation between two sets of values, typically
        used for analyzing relationships between actions and outcomes.

        Parameters
        ----------
        x : List[float]
            First variable's values
        y : List[float]
            Second variable's values (must be same length as x)

        Returns
        -------
        float
            Correlation coefficient between -1.0 and 1.0, where:
            - 1.0 indicates perfect positive correlation
            - -1.0 indicates perfect negative correlation
            - 0.0 indicates no linear correlation

        Notes
        -----
        Returns 0.0 if either list is empty or lengths don't match.
        """
        if not x or not y or len(x) != len(y):
            return 0
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(i * j for i, j in zip(x, y))
        sum_x2 = sum(i * i for i in x)
        sum_y2 = sum(i * i for i in y)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = (
            (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)
        ) ** 0.5

        return numerator / denominator if denominator != 0 else 0

    def _normalize_dict(self, d: Dict[str, int]) -> Dict[str, float]:
        """Normalize dictionary values to proportions.

        Converts raw counts to proportions by dividing each value by the sum
        of all values.

        Parameters
        ----------
        d : Dict[str, int]
            Dictionary of raw counts

        Returns
        -------
        Dict[str, float]
            Dictionary with normalized values (proportions), where:
            - Each value is between 0.0 and 1.0
            - Sum of all values equals 1.0
            - Empty input returns empty dictionary

        Examples
        --------
        >>> d = {'attack': 20, 'defend': 30, 'explore': 50}
        >>> normalized = self._normalize_dict(d)
        >>> print(normalized)
        {'attack': 0.2, 'defend': 0.3, 'explore': 0.5}
        >>> print(f"Total: {sum(normalized.values())}")
        Total: 1.0
        """
        total = sum(d.values())
        return {k: v / total if total > 0 else 0 for k, v in d.items()}

    def _analyze_adaptation(
        self, results: List[Tuple[AgentAction, AgentState]]
    ) -> Dict[str, float]:
        """Analyze how agents adapt to changing conditions.

        Examines patterns of behavioral changes in response to environmental
        conditions and action outcomes.

        Parameters
        ----------
        results : List[Tuple[AgentAction, AgentState]]
            List of action-state pairs to analyze

        Returns
        -------
        Dict[str, float]
            Adaptation metrics including:
            - adaptation_rate: How quickly agents modify behavior
            - success_rate: Success rate of adapted behaviors
            - stability: Consistency of adapted behaviors

        Notes
        -----
        Adaptation is measured by comparing action choices before and after
        significant changes in state or rewards.
        """
        # Implementation details...
        pass

    def _analyze_spatial_patterns(
        self, results: List[Tuple[AgentAction, AgentState]]
    ) -> Dict[str, Any]:
        """Analyze spatial patterns in agent behavior.

        Examines how agent actions and outcomes vary based on position
        and spatial relationships with other agents.

        Parameters
        ----------
        results : List[Tuple[AgentAction, AgentState]]
            List of action-state pairs to analyze

        Returns
        -------
        Dict[str, Any]
            Spatial analysis results including:
            - position_effects: Impact of location on outcomes
            - clustering: Patterns of agent grouping
            - movement_patterns: Common movement strategies

        Notes
        -----
        Requires valid position data in AgentState records.
        Returns empty patterns if position data is missing.
        """
        # Implementation details...
        pass

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

        This helper method handles common validation and filtering logic for queries that
        support different analysis scopes. It ensures required parameters are provided
        and applies the appropriate filters to the base query.

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
        agent_id : Optional[int]
            ID of the agent to analyze. Required when scope is "agent".
            Must be a valid agent ID in the database.
        step : Optional[int]
            Specific step number to analyze. Required when scope is "step".
            Must be >= 0.
        step_range : Optional[Tuple[int, int]]
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
            - agent_id missing when scope is "agent"
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

        # Validate parameters based on scope
        if scope == AnalysisScope.AGENT and agent_id is None:
            raise ValueError("agent_id is required when scope is AGENT")
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
