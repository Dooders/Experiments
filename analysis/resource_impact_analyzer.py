from typing import List, Optional, Tuple, Union
from database.data_types import ResourceImpact, AgentActionData
from database.repositories.agent_action_repository import AgentActionRepository
from database.enums import AnalysisScope

class ResourceImpactAnalyzer:
    def __init__(self, repository: AgentActionRepository):
        self.repository = repository

    def analyze(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[ResourceImpact]:
        actions = self.repository.get_actions_by_scope(scope, agent_id, step, step_range)
        impacts = {}

        for action in actions:
            if action.action_type not in impacts:
                impacts[action.action_type] = {
                    "total_resources_before": 0,
                    "total_resource_change": 0,
                    "count": 0,
                }

            impact = impacts[action.action_type]
            impact["total_resources_before"] += action.resources_before or 0
            impact["total_resource_change"] += (action.resources_after or 0) - (action.resources_before or 0)
            impact["count"] += 1

        return [
            ResourceImpact(
                action_type=action_type,
                avg_resources_before=impact["total_resources_before"] / impact["count"] if impact["count"] > 0 else 0,
                avg_resource_change=impact["total_resource_change"] / impact["count"] if impact["count"] > 0 else 0,
                resource_efficiency=impact["total_resource_change"] / impact["count"] if impact["count"] > 0 else 0,
            )
            for action_type, impact in impacts.items()
        ]
