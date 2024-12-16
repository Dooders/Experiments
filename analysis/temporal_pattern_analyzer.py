from typing import List, Optional, Tuple, Union

import numpy as np

from database.data_types import EventSegment, TimePattern
from database.enums import AnalysisScope
from database.repositories.agent_action_repository import AgentActionRepository


class TemporalPatternAnalyzer:
    """
    Analyzes temporal patterns in agent actions over time, including rolling averages and event segmentation.

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
        rolling_window_size: int = 10,
    ) -> List[TimePattern]:
        """
        Analyze temporal patterns in agent actions, including rolling averages.

        This method processes agent actions to:
        1. Identify patterns in action occurrence over time
        2. Calculate reward progression across time periods
        3. Generate rolling averages over specified window sizes

        Args:
            scope (Union[str, AnalysisScope], optional): Scope of analysis. Defaults to AnalysisScope.SIMULATION.
            agent_id (Optional[int], optional): Specific agent to analyze. Defaults to None.
            step (Optional[int], optional): Specific step to analyze. Defaults to None.
            step_range (Optional[Tuple[int, int]], optional): Range of steps to analyze. Defaults to None.
            time_period_size (int, optional): Size of time period in steps. Defaults to 100.
            rolling_window_size (int, optional): Window size for calculating rolling averages. Defaults to 10.

        Returns:
            List[TimePattern]: List of temporal patterns for each action type.
        """
        actions = self.repository.get_actions_by_scope(
            scope, agent_id, step_range=step_range
        )
        patterns = {}
        max_time_period = 0

        for action in actions:
            if action.action_type not in patterns:
                patterns[action.action_type] = {
                    "time_distribution": [],
                    "reward_progression": [],
                    "steps": [],
                }

            time_period = action.step_number // time_period_size
            while len(patterns[action.action_type]["time_distribution"]) <= time_period:
                patterns[action.action_type]["time_distribution"].append(0)
                patterns[action.action_type]["reward_progression"].append(0)
                max_time_period = max(max_time_period, time_period)

            patterns[action.action_type]["time_distribution"][time_period] += 1
            patterns[action.action_type]["reward_progression"][time_period] += (
                action.reward or 0
            )
            patterns[action.action_type]["steps"].append(action.step_number)

        # Calculate average rewards per time period
        for action_type, data in patterns.items():
            for i in range(len(data["reward_progression"])):
                if data["time_distribution"][i] > 0:
                    data["reward_progression"][i] /= data["time_distribution"][i]

        # Calculate rolling averages
        for action_type, data in patterns.items():
            rewards = data["reward_progression"]
            counts = data["time_distribution"]
            # Extend counts and rewards to match max_time_period
            counts += [0] * (max_time_period + 1 - len(counts))
            rewards += [0] * (max_time_period + 1 - len(rewards))
            # Use numpy for rolling average calculation
            rewards_array = np.array(rewards)
            counts_array = np.array(counts)
            rolling_rewards = (
                np.convolve(rewards_array, np.ones(rolling_window_size), "valid")
                / rolling_window_size
            )
            rolling_counts = (
                np.convolve(counts_array, np.ones(rolling_window_size), "valid")
                / rolling_window_size
            )
            data["rolling_average_rewards"] = rolling_rewards.tolist()
            data["rolling_average_counts"] = rolling_counts.tolist()

        return [
            TimePattern(
                action_type=action_type,
                time_distribution=data["time_distribution"],
                reward_progression=data["reward_progression"],
                rolling_average_rewards=data.get("rolling_average_rewards", []),
                rolling_average_counts=data.get("rolling_average_counts", []),
            )
            for action_type, data in patterns.items()
        ]

    def segment_events(
        self,
        event_steps: List[int],
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[EventSegment]:
        """
        Segment metrics before and after key events.

        Args:
            event_steps (List[int]): List of step numbers where events occur.
            scope (Union[str, AnalysisScope], optional): Scope of analysis. Defaults to AnalysisScope.SIMULATION.
            agent_id (Optional[int], optional): Specific agent to analyze. Defaults to None.
            step_range (Optional[Tuple[int, int]], optional): Range of steps to analyze. Defaults to None.

        Returns:
            List[EventSegment]: List of metrics segmented by events.
        """
        actions = self.repository.get_actions_by_scope(
            scope, agent_id, step_range=step_range
        )
        # Sort events to ensure correct segmentation
        event_steps = sorted(event_steps)
        segments = []
        last_event_step = 0
        for event_step in event_steps + [None]:  # Add None to capture last segment
            segment_actions = [
                a
                for a in actions
                if last_event_step <= a.step_number < (event_step or float("inf"))
            ]
            # Calculate metrics for this segment
            action_counts = {}
            total_rewards = {}
            for action in segment_actions:
                action_type = action.action_type
                action_counts[action_type] = action_counts.get(action_type, 0) + 1
                total_rewards[action_type] = total_rewards.get(action_type, 0) + (
                    action.reward or 0
                )
            average_rewards = {
                action_type: total_rewards[action_type] / action_counts[action_type]
                for action_type in action_counts
            }
            segments.append(
                EventSegment(
                    start_step=last_event_step,
                    end_step=event_step,
                    action_counts=action_counts,
                    average_rewards=average_rewards,
                )
            )
            last_event_step = event_step

        return segments
