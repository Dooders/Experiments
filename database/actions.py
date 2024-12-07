"""Actions retrieval module for simulation database.

This module provides specialized queries and analysis methods for action-related
data, including action patterns, interaction statistics, and decision-making metrics.

The ActionsRetriever class handles action-specific database operations with
optimized queries and efficient data aggregation methods.
"""

import json
from typing import Dict, List, Optional, Tuple

from sqlalchemy import case, distinct, func

from database.data_types import (
    ActionMetrics,
    DecisionPatterns,
    DecisionPatternStats,
    DecisionSummary,
    InteractionStats,
    ResourceImpact,
    SequencePattern,
    StepActionData,
    TimePattern,
)
from database.models import Agent, AgentAction
from database.retrievers import BaseRetriever
from database.utilities import execute_query


class ActionsRetriever(BaseRetriever):
    """Handles retrieval and analysis of action-related data.

    This class provides methods for analyzing action patterns, decision-making
    behaviors, and interaction statistics throughout the simulation.

    Methods
    -------
    action_metrics(agent_id: Optional[int] = None) -> List[ActionMetrics]
        Calculate basic metrics for agent actions
    decision_patterns(agent_id: Optional[int] = None) -> DecisionPatterns
        Analyze comprehensive decision-making patterns
    interaction_statistics() -> Dict[str, InteractionStats]
        Calculate statistics about agent interactions
    temporal_patterns() -> Dict[str, TimePattern]
        Analyze action patterns over time
    resource_impacts() -> Dict[str, ResourceImpact]
        Analyze resource impacts of different actions
    execute() -> DecisionPatterns
        Generate comprehensive action analysis
    """

    @execute_query
    def action_metrics(
        self, session, agent_id: Optional[int] = None
    ) -> List[ActionMetrics]:
        """Calculate basic metrics for agent actions.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.

        Returns
        -------
        List[ActionMetrics]
            List of metrics for each action type
        """
        query = session.query(
            AgentAction.action_type,
            func.count().label("decision_count"),
            func.avg(AgentAction.reward).label("avg_reward"),
            func.min(AgentAction.reward).label("min_reward"),
            func.max(AgentAction.reward).label("max_reward"),
        )

        if agent_id is not None:
            query = query.filter(AgentAction.agent_id == agent_id)

        results = query.group_by(AgentAction.action_type).all()

        return [
            ActionMetrics(
                action_type=result[0],
                decision_count=int(result[1]),
                avg_reward=float(result[2] or 0),
                min_reward=float(result[3] or 0),
                max_reward=float(result[4] or 0),
            )
            for result in results
        ]

    @execute_query
    def interaction_statistics(self, session) -> Dict[str, InteractionStats]:
        """Calculate statistics about agent interactions.

        Returns
        -------
        Dict[str, InteractionStats]
            Statistics for each action type's interactions
        """
        stats = (
            session.query(
                AgentAction.action_type,
                AgentAction.action_target_id.isnot(None).label("is_interaction"),
                func.count().label("count"),
                func.avg(AgentAction.reward).label("avg_reward"),
            )
            .group_by(
                AgentAction.action_type,
                AgentAction.action_target_id.isnot(None),
            )
            .all()
        )

        interaction_stats = {}
        for action_type, is_interaction, count, avg_reward in stats:
            if action_type not in interaction_stats:
                interaction_stats[action_type] = InteractionStats(
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

        return interaction_stats

    @execute_query
    def temporal_patterns(self, session) -> Dict[str, TimePattern]:
        """Analyze action patterns over time.

        Returns
        -------
        Dict[str, TimePattern]
            Temporal patterns for each action type
        """
        patterns = (
            session.query(
                AgentAction.action_type,
                func.count().label("count"),
                func.avg(AgentAction.reward).label("avg_reward"),
            )
            .group_by(
                AgentAction.action_type,
                func.round(AgentAction.step_number / 100),  # Group by time periods
            )
            .all()
        )

        temporal_patterns = {}
        for action_type, count, avg_reward in patterns:
            if action_type not in temporal_patterns:
                temporal_patterns[action_type] = TimePattern(
                    time_distribution=[],
                    reward_progression=[],
                )
            temporal_patterns[action_type].time_distribution.append(int(count))
            temporal_patterns[action_type].reward_progression.append(
                float(avg_reward or 0)
            )

        return temporal_patterns

    @execute_query
    def resource_impacts(self, session) -> Dict[str, ResourceImpact]:
        """Analyze resource impacts of different actions.

        Returns
        -------
        Dict[str, ResourceImpact]
            Resource impact statistics for each action type
        """
        impacts = (
            session.query(
                AgentAction.action_type,
                func.avg(AgentAction.resources_before).label("avg_before"),
                func.avg(
                    AgentAction.resources_after - AgentAction.resources_before
                ).label("avg_change"),
                func.count().label("count"),
            )
            .filter(
                AgentAction.resources_before.isnot(None),
                AgentAction.resources_after.isnot(None),
            )
            .group_by(AgentAction.action_type)
            .all()
        )

        return {
            action_type: ResourceImpact(
                avg_resources_before=float(avg_before or 0),
                avg_resource_change=float(avg_change or 0),
                resource_efficiency=float(avg_change or 0) / count if count > 0 else 0,
            )
            for action_type, avg_before, avg_change, count in impacts
        }

    @execute_query
    def decision_patterns(
        self, session, agent_id: Optional[int] = None
    ) -> DecisionPatterns:
        """Analyze comprehensive decision-making patterns.

        Performs a detailed analysis of agent decision-making patterns, including action
        frequencies, rewards, sequences, resource impacts, and temporal trends.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.

        Returns
        -------
        DecisionPatterns
            Comprehensive decision pattern analysis with the following components:
            
            decision_patterns : Dict[str, DecisionPatternStats]
                Statistics for each action type, containing:
                - count: Total number of times the action was taken
                - frequency: Proportion of times this action was chosen
                - reward_stats: Dict containing average/min/max rewards
            
            sequence_analysis : Dict[str, SequencePattern]
                Analysis of action sequences (e.g., "action1->action2"), containing:
                - count: Number of times this sequence occurred
                - probability: Likelihood of the second action following the first
            
            resource_impact : Dict[str, ResourceImpact]
                Resource effects for each action type, containing:
                - avg_resources_before: Average resources before action
                - avg_resource_change: Average change in resources
                - resource_efficiency: Resource change per action
            
            temporal_patterns : Dict[str, TimePattern]
                Time-based analysis for each action, containing:
                - time_distribution: List of action counts per time period
                - reward_progression: List of average rewards per time period
            
            interaction_analysis : Dict[str, InteractionStats]
                Statistics about agent interactions, containing:
                - interaction_rate: Proportion of actions involving other agents
                - solo_performance: Average reward for solo actions
                - interaction_performance: Average reward for interactive actions
            
            decision_summary : DecisionSummary
                Overall decision-making metrics, containing:
                - total_decisions: Total number of decisions made
                - unique_actions: Number of different action types used
                - most_frequent: Most commonly chosen action
                - most_rewarding: Action with highest average reward
                - action_diversity: Shannon entropy of action distribution
        """
        # Get basic action metrics
        metrics = self.action_metrics(agent_id)

        # Calculate total decisions
        total_decisions = sum(m.decision_count for m in metrics)

        # Format decision patterns
        patterns = {
            m.action_type: DecisionPatternStats(
                count=m.decision_count,
                frequency=(
                    m.decision_count / total_decisions if total_decisions > 0 else 0
                ),
                reward_stats={
                    "average": m.avg_reward,
                    "min": m.min_reward,
                    "max": m.max_reward,
                },
            )
            for m in metrics
        }

        # Get interaction statistics
        interaction_stats = self.interaction_statistics()

        # Get temporal patterns
        temporal_patterns = self.temporal_patterns()

        # Get resource impacts
        resource_impacts = self.resource_impacts()

        # Calculate sequence patterns
        sequence_patterns = self._calculate_sequence_patterns(session, agent_id)

        # Create decision summary
        summary = DecisionSummary(
            total_decisions=total_decisions,
            unique_actions=len(metrics),
            most_frequent=(
                max(patterns.items(), key=lambda x: x[1].count)[0] if patterns else None
            ),
            most_rewarding=(
                max(patterns.items(), key=lambda x: x[1].reward_stats["average"])[0]
                if patterns
                else None
            ),
            action_diversity=self._calculate_diversity(patterns),
        )

        return DecisionPatterns(
            decision_patterns=patterns,
            sequence_analysis=sequence_patterns,
            resource_impact=resource_impacts,
            temporal_patterns=temporal_patterns,
            interaction_analysis=interaction_stats,
            decision_summary=summary,
        )

    @execute_query
    def step_actions(self, session, step_number: int) -> StepActionData:
        """Get all actions performed during a specific simulation step.

        Parameters
        ----------
        step_number : int
            The simulation step number to query

        Returns
        -------
        StepActionData
            Data containing:
            - step_summary: Dict[str, int]
                - total_actions: Total number of actions
                - unique_agents: Number of unique agents
                - action_types: Number of unique action types
                - total_interactions: Total number of interactions
                - total_reward: Total reward received
            - action_statistics: Dict[str, Dict[str, float]]
                Per action type statistics including:
                - count: Number of occurrences
                - frequency: Frequency of the action type
                - avg_reward: Average reward for the action
                - total_reward: Total reward for the action
            - resource_metrics: Dict[str, float]
                - net_resource_change: Net resource change
                - average_resource_change: Average resource change
                - resource_transactions: Number of resource transactions
            - interaction_network: Dict[str, Any]
                - interactions: List[Dict[str, Any]]
                    List of interactions, each containing:
                    - source: Source agent ID
                    - target: Target agent ID
                    - action_type: Action type
                    - reward: Reward received
                - unique_interacting_agents: Number of unique interacting agents
            - performance_metrics: Dict[str, float]
                - success_rate: Success rate of actions
                - average_reward: Average reward per action
                - action_efficiency: Efficiency of actions
            - detailed_actions: List[Dict[str, Any]]
                Detailed list of all actions
        """
        # Get all actions for the step
        actions = (
            session.query(AgentAction)
            .filter(AgentAction.step_number == step_number)
            .order_by(AgentAction.agent_id)
            .all()
        )

        if not actions:
            return {}

        # Get action type statistics
        action_stats = (
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

        # Calculate resource changes
        resource_changes = (
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
                "state_before_id": action.state_before_id,  # Reference to state table
                "state_after_id": action.state_after_id,    # Reference to state table
                "resources_before": action.resources_before,
                "resources_after": action.resources_after,
                "reward": action.reward,
                "details": json.loads(action.details) if action.details else None,
            }
            for action in actions
        ]

        return StepActionData(
            step_summary={
                "total_actions": len(actions),
                "unique_agents": len(set(a.agent_id for a in actions)),
                "action_types": len(set(a.action_type for a in actions)),
                "total_interactions": len(interactions),
                "total_reward": sum(
                    a.reward for a in actions if a.reward is not None
                ),
            },
            action_statistics={
                action_type: {
                    "count": count,
                    "frequency": count / len(actions),
                    "avg_reward": float(avg_reward or 0),
                    "total_reward": float(total_reward or 0),
                }
                for action_type, count, avg_reward, total_reward in action_stats
            },
            resource_metrics={
                "net_resource_change": float(resource_changes[0] or 0),
                "average_resource_change": float(resource_changes[1] or 0),
                "resource_transactions": len(
                    [a for a in actions if a.resources_before != a.resources_after]
                ),
            },
            interaction_network={
                "interactions": [
                    {
                        "source": action.agent_id,
                        "target": action.action_target_id,
                        "action_type": action.action_type,
                        "reward": action.reward,
                    }
                    for action in interactions
                ],
                "unique_interacting_agents": len(
                    set(
                        [a.agent_id for a in interactions]
                        + [a.action_target_id for a in interactions]
                    )
                ),
            },
            performance_metrics={
                "success_rate": len(
                    [a for a in actions if a.reward and a.reward > 0]
                )
                / len(actions),
                "average_reward": sum(
                    a.reward for a in actions if a.reward is not None
                )
                / len(actions),
                "action_efficiency": len(
                    [a for a in actions if a.position_before != a.position_after]
                )
                / len(actions),
            },
            detailed_actions=action_list,
        )

    def _calculate_sequence_patterns(
        self, session, agent_id: Optional[int] = None
    ) -> Dict[str, SequencePattern]:
        """Calculate action sequence patterns.

        Parameters
        ----------
        session : Session
            Database session
        agent_id : Optional[int]
            Specific agent ID to analyze

        Returns
        -------
        Dict[str, SequencePattern]
            Statistics about action sequences
        """
        # Base query for actions
        query = session.query(AgentAction)
        if agent_id is not None:
            query = query.filter(AgentAction.agent_id == agent_id)

        # Get action pairs and their frequencies
        sequences = (
            query.with_entities(
                AgentAction.action_type,
                func.lead(AgentAction.action_type)
                .over(order_by=AgentAction.step_number)
                .label("next_action"),
                func.count().label("sequence_count"),
            )
            .group_by(AgentAction.action_type, "next_action")
            .all()
        )

        # Calculate total occurrences for each initial action
        totals = {}
        for action, next_action, count in sequences:
            if action not in totals:
                totals[action] = 0
            totals[action] += count

        # Format sequence patterns
        return {
            f"{action}->{next_action}": SequencePattern(
                count=count,
                probability=count / totals[action] if action in totals else 0,
            )
            for action, next_action, count in sequences
            if action is not None and next_action is not None
        }

    def _calculate_diversity(self, patterns: Dict[str, DecisionPatternStats]) -> float:
        """Calculate Shannon entropy for action diversity.

        Parameters
        ----------
        patterns : Dict[str, DecisionPatternStats]
            Decision pattern statistics

        Returns
        -------
        float
            Shannon entropy diversity measure
        """
        import math

        return -sum(
            p.frequency * math.log(p.frequency) if p.frequency > 0 else 0
            for p in patterns.values()
        )

    def _execute(self) -> DecisionPatterns:
        """Execute comprehensive action analysis.

        Returns
        -------
        DecisionPatterns
            Complete action and decision-making analysis
        """
        return self.decision_patterns()
