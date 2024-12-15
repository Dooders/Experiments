from typing import List, Optional, Tuple, Union

from database.data_types import TimePattern
from database.enums import AnalysisScope
from database.repositories.agent_action_repository import AgentActionRepository


class TemporalPatternAnalyzer:
    """
    Analyzes temporal patterns in agent actions over time.

    This class processes agent actions to identify patterns in their occurrence
    and associated rewards across different time periods.

    Attributes:
        repository (AgentActionRepository): Repository for accessing agent action data.
    """

    def __init__(self, repository: AgentActionRepository):
        """
        Initialize the TemporalPatternAnalyzer.

        Args:
            repository (AgentActionRepository): Repository for accessing agent action data.
        """
        self.repository = repository

    def analyze(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[TimePattern]:
        """
        Analyze temporal patterns in agent actions.

        This method processes agent actions to identify patterns in their occurrence
        and rewards over time. It divides the timeline into periods of 100 steps
        and aggregates action counts and average rewards for each period.

        Args:
            scope (Union[str, AnalysisScope], optional): The scope of analysis.
                Defaults to AnalysisScope.SIMULATION.
            agent_id (Optional[int], optional): ID of the specific agent to analyze.
                If None, analyzes all agents. Defaults to None.
            step_range (Optional[Tuple[int, int]], optional): Range of steps to analyze,
                as (start_step, end_step). If None, analyzes all steps. Defaults to None.

        Returns:
            List[TimePattern]: A list of TimePattern objects, each containing:
                - action_type: The type of action
                - time_distribution: List of action counts per time period
                - reward_progression: List of average rewards per time period
        """
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
