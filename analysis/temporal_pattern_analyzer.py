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
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
        time_period_size: int = 100,
    ) -> List[TimePattern]:
        """
        Analyze temporal patterns in agent actions.

        This method processes agent actions to:
        1. Identify patterns in action occurrence over time
        2. Calculate reward progression across time periods
        3. Generate time-based distribution metrics

        Args:
            scope (Union[str, AnalysisScope], optional): Scope of analysis. Defaults to AnalysisScope.SIMULATION.
            agent_id (Optional[int], optional): Specific agent to analyze. Defaults to None.
            step (Optional[int], optional): Specific step to analyze. Defaults to None.
            step_range (Optional[Tuple[int, int]], optional): Range of steps to analyze. Defaults to None.
            time_period_size (int, optional): Size of time period in steps. Defaults to 100.

        Returns:
            List[TimePattern]: List of temporal patterns for each action type containing:
                - action_type: Type of the analyzed action
                - time_distribution: List of action counts per time period (N-step intervals)
                - reward_progression: List of average rewards per time period

        Example:
            For a "gather" action analysis over 300 steps, the result might look like:

            TimePattern(
                action_type="gather",
                time_distribution=[25, 30, 15],  # Action counts in each N-step period
                reward_progression=[2.0, 2.5, 1.8]  # Average rewards in each period
            )

        Note:
            - Time periods are fixed N-step intervals
            - Time distribution shows frequency of actions in each period
            - Reward progression tracks performance changes over time
            - Empty periods are represented with zero counts and rewards
            - Rewards are averaged per period to account for varying action counts
        """
        actions = self.repository.get_actions_by_scope(
            scope, agent_id, step_range=step_range
        )
        patterns = {}

        for action in actions:
            if action.action_type not in patterns:
                patterns[action.action_type] = {
                    "time_distribution": [],
                    "reward_progression": [],
                }

            time_period = action.step_number // time_period_size
            while len(patterns[action.action_type]["time_distribution"]) <= time_period:
                patterns[action.action_type]["time_distribution"].append(0)
                patterns[action.action_type]["reward_progression"].append(0)

            patterns[action.action_type]["time_distribution"][time_period] += 1
            patterns[action.action_type]["reward_progression"][time_period] += (
                action.reward or 0
            )

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
