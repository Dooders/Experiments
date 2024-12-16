from typing import List, Optional, Tuple, Union, Dict
import numpy as np  # Add numpy import for statistical calculations
from collections import defaultdict

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

        # Track action sequences for co-occurrence analysis
        action_sequences = []
        current_sequence = []
        current_step = None

        decision_metrics = {}
        for action in actions:
            # Handle action sequences
            if current_step != action.step_number:
                if current_sequence:
                    action_sequences.append(current_sequence)
                current_sequence = []
                current_step = action.step_number
            current_sequence.append(action.action_type)

            # Existing metrics tracking
            if action.action_type not in decision_metrics:
                decision_metrics[action.action_type] = {
                    "count": 0,
                    "rewards": [],
                }
            metrics = decision_metrics[action.action_type]
            metrics["count"] += 1
            metrics["rewards"].append(action.reward or 0)

        # Add final sequence
        if current_sequence:
            action_sequences.append(current_sequence)

        patterns = [
            DecisionPatternStats(
                action_type=action_type,
                count=metrics["count"],
                frequency=(
                    metrics["count"] / total_decisions if total_decisions > 0 else 0
                ),
                reward_stats=self._calculate_reward_stats(metrics["rewards"]),
                contribution_metrics=self._calculate_contribution_metrics(
                    metrics["rewards"],
                    sum(m["count"] for m in decision_metrics.values()),
                    sum(sum(m["rewards"]) for m in decision_metrics.values()),
                ),
            )
            for action_type, metrics in decision_metrics.items()
        ]

        # Calculate co-occurrence matrix
        co_occurrence = self._calculate_co_occurrence(action_sequences)

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
            normalized_diversity=self._calculate_normalized_diversity(patterns),
            co_occurrence_patterns=co_occurrence,
        )

        return DecisionPatterns(
            decision_patterns=patterns,
            decision_summary=summary,
        )

    def _calculate_reward_stats(self, rewards: List[float]) -> dict:
        """
        Calculate comprehensive reward statistics.

        Args:
            rewards (List[float]): List of reward values for a specific action type.

        Returns:
            dict: Dictionary containing various reward statistics.
        """
        if not rewards:
            return {
                "average": 0,
                "median": 0,
                "min": 0,
                "max": 0,
                "variance": 0,
                "std_dev": 0,
                "percentile_25": 0,
                "percentile_50": 0,
                "percentile_75": 0,
            }

        rewards_array = np.array(rewards)
        return {
            "average": float(np.mean(rewards_array)),
            "median": float(np.median(rewards_array)),
            "min": float(np.min(rewards_array)),
            "max": float(np.max(rewards_array)),
            "variance": float(np.var(rewards_array)),
            "std_dev": float(np.std(rewards_array)),
            "percentile_25": float(np.percentile(rewards_array, 25)),
            "percentile_50": float(np.percentile(rewards_array, 50)),
            "percentile_75": float(np.percentile(rewards_array, 75)),
        }

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

    def _calculate_contribution_metrics(
        self, rewards: List[float], total_actions: int, total_rewards: float
    ) -> dict:
        """
        Calculate metrics showing how this action contributes to overall diversity and rewards.

        Args:
            rewards (List[float]): List of rewards for this action type
            total_actions (int): Total number of actions across all types
            total_rewards (float): Sum of all rewards across all action types

        Returns:
            dict: Dictionary containing contribution metrics
        """
        if not rewards or total_actions == 0 or total_rewards == 0:
            return {"action_share": 0.0, "reward_share": 0.0, "reward_efficiency": 0.0}

        action_share = len(rewards) / total_actions
        reward_share = sum(rewards) / total_rewards
        reward_efficiency = reward_share / action_share if action_share > 0 else 0

        return {
            "action_share": action_share,
            "reward_share": reward_share,
            "reward_efficiency": reward_efficiency,
        }

    def _calculate_normalized_diversity(
        self, patterns: List[DecisionPatternStats]
    ) -> float:
        """
        Calculate normalized Shannon diversity index (0-1 scale).

        Args:
            patterns (List[DecisionPatternStats]): List of decision pattern statistics

        Returns:
            float: Normalized diversity index between 0 and 1
        """
        if not patterns:
            return 0.0

        raw_diversity = self._calculate_diversity(patterns)
        max_diversity = np.log(len(patterns)) if patterns else 0

        return raw_diversity / max_diversity if max_diversity > 0 else 0

    def _calculate_co_occurrence(
        self, action_sequences: List[List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate the co-occurrence matrix for action types.

        Args:
            action_sequences (List[List[str]]): List of action sequences by step

        Returns:
            Dict[str, Dict[str, float]]: Co-occurrence frequencies and correlations
        """
        if not action_sequences:
            return {}

        # Count co-occurrences
        co_occurrences = defaultdict(lambda: defaultdict(int))
        action_counts = defaultdict(int)

        for sequence in action_sequences:
            # Count individual actions
            for action in sequence:
                action_counts[action] += 1

            # Count co-occurrences
            for i, action1 in enumerate(sequence):
                for action2 in sequence[i + 1 :]:
                    co_occurrences[action1][action2] += 1
                    co_occurrences[action2][action1] += 1

        # Calculate correlation coefficients
        total_steps = len(action_sequences)
        correlations = {}

        for action1 in action_counts:
            correlations[action1] = {}
            for action2 in action_counts:
                if action1 == action2:
                    continue

                # Calculate correlation coefficient
                p_a1 = action_counts[action1] / total_steps
                p_a2 = action_counts[action2] / total_steps
                p_both = co_occurrences[action1][action2] / total_steps

                # Calculate correlation (phi coefficient)
                numerator = p_both - (p_a1 * p_a2)
                denominator = np.sqrt(p_a1 * p_a2 * (1 - p_a1) * (1 - p_a2))
                correlation = numerator / denominator if denominator != 0 else 0

                correlations[action1][action2] = {
                    "count": co_occurrences[action1][action2],
                    "frequency": p_both,
                    "correlation": correlation,
                }

        return correlations
