from typing import List, Optional, Tuple, Union
from database.data_types import TimePattern, AgentActionData
from database.repositories.agent_action_repository import AgentActionRepository
from database.enums import AnalysisScope

class TemporalPatternAnalyzer:
    def __init__(self, repository: AgentActionRepository):
        self.repository = repository

    def analyze(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[TimePattern]:
        actions = self.repository.get_actions_by_scope(scope, agent_id, step_range=step_range)
        patterns = {}

        for action in actions:
            if action.action_type not in patterns:
                patterns[action.action_type] = {
                    "time_distribution": [],
                    "reward_progression": [],
                }

            time_period = action.step_number // 100
            while len(patterns[action.action_type]["time_distribution"]) <= time_period:
                patterns[action.action_type]["time_distribution"].append(0)
                patterns[action.action_type]["reward_progression"].append(0)

            patterns[action.action_type]["time_distribution"][time_period] += 1
            patterns[action.action_type]["reward_progression"][time_period] += action.reward or 0

        for action_type, data in patterns.items():
            for i in range(len(data["reward_progression"])):
                if data["time_distribution"][i] > 0:
                    data["reward_progression"][i] /= data["time_distribution"][i]

        return [
            TimePattern(
                action_type=action_type,
                time_distribution=data["time_distribution"],
                reward_progression=data["reward_progression"],
            )
            for action_type, data in patterns.items()
        ]
