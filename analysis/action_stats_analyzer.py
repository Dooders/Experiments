from typing import List, Optional, Tuple, Union
from database.data_types import ActionMetrics, AgentActionData, TimePattern, ResourceImpact, DecisionPatternStats
from database.repositories.agent_action_repository import AgentActionRepository

class ActionStatsAnalyzer:
    def __init__(self, repository: AgentActionRepository):
        self.repository = repository

    def analyze(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[ActionMetrics]:
        actions = self.repository.get_actions_by_scope(scope, agent_id, step, step_range)
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

        return [
            ActionMetrics(
                action_type=action_type,
                count=metrics["count"],
                frequency=metrics["count"] / total_actions if total_actions > 0 else 0,
                avg_reward=metrics["total_reward"] / metrics["count"] if metrics["count"] > 0 else 0,
                min_reward=metrics["min_reward"],
                max_reward=metrics["max_reward"],
                interaction_rate=metrics["interaction_count"] / metrics["count"] if metrics["count"] > 0 else 0,
                solo_performance=metrics["solo_reward"] / (metrics["count"] - metrics["interaction_count"]) if (metrics["count"] - metrics["interaction_count"]) > 0 else 0,
                interaction_performance=metrics["interaction_reward"] / metrics["interaction_count"] if metrics["interaction_count"] > 0 else 0,
                temporal_patterns=[],  # Placeholder for temporal patterns
                resource_impacts=[],  # Placeholder for resource impacts
                decision_patterns=[],  # Placeholder for decision patterns
            )
            for action_type, metrics in action_metrics.items()
        ]
