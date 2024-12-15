from typing import List, Optional, Tuple, Union

from analysis.decision_pattern_analyzer import DecisionPatternAnalyzer
from analysis.resource_impact_analyzer import ResourceImpactAnalyzer
from analysis.temporal_pattern_analyzer import TemporalPatternAnalyzer
from database.data_types import ActionMetrics
from database.enums import AnalysisScope
from database.repositories.agent_action_repository import AgentActionRepository


class ActionStatsAnalyzer:
    """
    Analyzes statistics and patterns of agent actions in a simulation.

    This class processes action data to generate metrics including frequency, rewards,
    interaction rates, and various patterns of agent behavior.

    Attributes:
        repository (AgentActionRepository): Repository for accessing agent action data.
    """

    def __init__(self, repository: AgentActionRepository):
        """
        Initialize the ActionStatsAnalyzer.

        Args:
            repository (AgentActionRepository): Repository instance for accessing agent action data.
        """
        self.repository = repository

    def analyze(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[ActionMetrics]:
        """
        Analyze action statistics based on specified scope and filters.

        Processes action data to generate comprehensive metrics including action frequencies,
        rewards, interaction rates, and various behavioral patterns.

        Args:
            scope (Union[str, AnalysisScope]): The scope of analysis (e.g., SIMULATION, EPISODE).
                Defaults to AnalysisScope.SIMULATION.
            agent_id (Optional[int]): ID of the specific agent to analyze. If None, analyzes all agents.
            step (Optional[int]): Specific simulation step to analyze. If None, analyzes all steps.
            step_range (Optional[Tuple[int, int]]): Range of steps to analyze (inclusive).
                If None, analyzes all steps.

        Returns:
            List[ActionMetrics]: List of action metrics objects containing:
                - action_type: Type of the action
                - count: Total number of occurrences
                - frequency: Relative frequency of the action
                - avg_reward: Average reward received
                - min_reward: Minimum reward received
                - max_reward: Maximum reward received
                - interaction_rate: Rate of interactions with other agents
                - solo_performance: Average reward for solo actions
                - interaction_performance: Average reward for interactive actions
                - temporal_patterns: Patterns in timing of actions
                - resource_impacts: Effects on resource utilization
                - decision_patterns: Patterns in decision-making
        """
        actions = self.repository.get_actions_by_scope(
            scope, agent_id, step, step_range
        )
        total_actions = len(actions)

        action_metrics = {}
        for action in actions:
            if action.action_type not in action_metrics:
                action_metrics[action.action_type] = {
                    "count": 0,
                    "total_reward": 0,
                    "min_reward": float("inf"),
                    "max_reward": float("-inf"),
                    "interaction_count": 0,
                    "interaction_reward": 0,
                    "solo_reward": 0,
                }
            metrics = action_metrics[action.action_type]
            metrics["count"] += 1
            metrics["total_reward"] += action.reward or 0
            metrics["min_reward"] = min(metrics["min_reward"], action.reward or 0)
            metrics["max_reward"] = max(metrics["max_reward"], action.reward or 0)
            if action.action_target_id:
                metrics["interaction_count"] += 1
                metrics["interaction_reward"] += action.reward or 0
            else:
                metrics["solo_reward"] += action.reward or 0

        temporal_patterns = TemporalPatternAnalyzer(self.repository).analyze(
            scope, agent_id, step, step_range
        )
        resource_impacts = ResourceImpactAnalyzer(self.repository).analyze(
            scope, agent_id, step, step_range
        )
        decision_patterns = DecisionPatternAnalyzer(self.repository).analyze(
            scope, agent_id, step, step_range
        )

        return [
            ActionMetrics(
                action_type=action_type,
                count=metrics["count"],
                frequency=metrics["count"] / total_actions if total_actions > 0 else 0,
                avg_reward=(
                    metrics["total_reward"] / metrics["count"]
                    if metrics["count"] > 0
                    else 0
                ),
                min_reward=metrics["min_reward"],
                max_reward=metrics["max_reward"],
                interaction_rate=(
                    metrics["interaction_count"] / metrics["count"]
                    if metrics["count"] > 0
                    else 0
                ),
                solo_performance=(
                    metrics["solo_reward"]
                    / (metrics["count"] - metrics["interaction_count"])
                    if (metrics["count"] - metrics["interaction_count"]) > 0
                    else 0
                ),
                interaction_performance=(
                    metrics["interaction_reward"] / metrics["interaction_count"]
                    if metrics["interaction_count"] > 0
                    else 0
                ),
                temporal_patterns=[
                    p for p in temporal_patterns if p.action_type == action_type
                ],
                resource_impacts=[
                    r for r in resource_impacts if r.action_type == action_type
                ],
                decision_patterns=[
                    d for d in decision_patterns if d.action_type == action_type
                ],
            )
            for action_type, metrics in action_metrics.items()
        ]
