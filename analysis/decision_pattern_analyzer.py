from typing import List, Optional, Tuple, Union
from database.data_types import DecisionPatterns, DecisionPatternStats, DecisionSummary
from database.repositories.agent_action_repository import AgentActionRepository
from database.enums import AnalysisScope

class DecisionPatternAnalyzer:
    def __init__(self, repository: AgentActionRepository):
        self.repository = repository

    def analyze(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> DecisionPatterns:
        actions = self.repository.get_actions_by_scope(scope, agent_id, step, step_range)
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
                frequency=metrics["count"] / total_decisions if total_decisions > 0 else 0,
                reward_stats={
                    "average": metrics["total_reward"] / metrics["count"] if metrics["count"] > 0 else 0,
                    "min": metrics["min_reward"],
                    "max": metrics["max_reward"],
                },
            )
            for action_type, metrics in decision_metrics.items()
        ]

        summary = DecisionSummary(
            total_decisions=total_decisions,
            unique_actions=len(decision_metrics),
            most_frequent=max(patterns, key=lambda x: x.count).action_type if patterns else None,
            most_rewarding=max(patterns, key=lambda x: x.reward_stats["average"]).action_type if patterns else None,
            action_diversity=self._calculate_diversity(patterns),
        )

        return DecisionPatterns(
            decision_patterns=patterns,
            decision_summary=summary,
        )

    def _calculate_diversity(self, patterns: List[DecisionPatternStats]) -> float:
        import math
        return -sum(
            p.frequency * math.log(p.frequency) if p.frequency > 0 else 0
            for p in patterns
        )
