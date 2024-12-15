from typing import List, Optional, Tuple, Union

from database.data_types import DecisionPatterns, DecisionPatternStats, DecisionSummary
from database.enums import AnalysisScope
from database.repositories.agent_action_repository import AgentActionRepository


class DecisionPatternAnalyzer:
    """
    Analyzes decision patterns from agent actions to identify behavioral trends and statistics.

    This class processes agent actions to extract meaningful patterns, frequencies, and reward statistics,
    providing insights into agent decision-making behavior.
    """

    def __init__(self, repository: AgentActionRepository):
        """
        Initialize the DecisionPatternAnalyzer.

        Args:
            repository (AgentActionRepository): Repository for accessing agent action data.
        """
        self.repository = repository

    def analyze(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> DecisionPatterns:
        """
        Analyze decision patterns within the specified scope and parameters.

        Args:
            scope (Union[str, AnalysisScope]): The scope of analysis (e.g., SIMULATION, EPISODE).
                Defaults to AnalysisScope.SIMULATION.
            agent_id (Optional[int]): Specific agent ID to analyze. Defaults to None.
            step (Optional[int]): Specific step to analyze. Defaults to None.
            step_range (Optional[Tuple[int, int]]): Range of steps to analyze. Defaults to None.

        Returns:
            DecisionPatterns: Object containing detailed pattern statistics and summary information.
                Includes frequencies, reward statistics, and diversity metrics for different action types.
        """
        actions = self.repository.get_actions_by_scope(
            scope, agent_id, step, step_range
        )
        total_decisions = len(actions)

        decision_metrics = {}
        for action in actions:
            if action.action_type not in decision_metrics:
                decision_metrics[action.action_type] = {
                    "count": 0,
                    "total_reward": 0,
                    "min_reward": float("inf"),
                    "max_reward": float("-inf"),
                }
            metrics = decision_metrics[action.action_type]
            metrics["count"] += 1
            metrics["total_reward"] += action.reward or 0
            metrics["min_reward"] = min(metrics["min_reward"], action.reward or 0)
            metrics["max_reward"] = max(metrics["max_reward"], action.reward or 0)

        patterns = [
            DecisionPatternStats(
                action_type=action_type,
                count=metrics["count"],
                frequency=(
                    metrics["count"] / total_decisions if total_decisions > 0 else 0
                ),
                reward_stats={
                    "average": (
                        metrics["total_reward"] / metrics["count"]
                        if metrics["count"] > 0
                        else 0
                    ),
                    "min": metrics["min_reward"],
                    "max": metrics["max_reward"],
                },
            )
            for action_type, metrics in decision_metrics.items()
        ]

        summary = DecisionSummary(
            total_decisions=total_decisions,
            unique_actions=len(decision_metrics),
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

    def _calculate_diversity(self, patterns: List[DecisionPatternStats]) -> float:
        """
        Calculate the diversity of decision patterns using Shannon entropy.

        A higher value indicates more diverse decision-making patterns.

        Args:
            patterns (List[DecisionPatternStats]): List of decision pattern statistics.

        Returns:
            float: Shannon entropy value representing decision diversity.
        """
        import math

        return -sum(
            p.frequency * math.log(p.frequency) if p.frequency > 0 else 0
            for p in patterns
        )
