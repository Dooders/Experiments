"""Learning retrieval module for simulation database.

This module provides specialized queries and analysis methods for learning-related
data, including learning experiences, module performance, and adaptation metrics.

The LearningRetriever class handles learning-specific database operations with
optimized queries and efficient data aggregation methods.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import distinct, func

from database.data_types import (
    AgentLearningStats,
    LearningEfficiencyMetrics,
    LearningProgress,
    LearningStatistics,
    ModulePerformance,
)
from database.models import LearningExperience
from database.utilities import execute_query


class LearningRetriever:
    """Handles learning-related data retrieval and analysis.

    This class provides methods for analyzing learning experiences, module performance,
    and adaptation patterns throughout the simulation. It interfaces with the database
    to retrieve and aggregate learning metrics.

    Methods
    -------
    learning_progress() -> LearningProgress
        Retrieves time-series data of learning progress
    module_performance() -> Dict[str, ModulePerformance]
        Calculates performance metrics per learning module
    agent_learning_stats(agent_id: Optional[int]) -> Dict[str, AgentLearningStats]
        Analyzes learning statistics for specific or all agents
    learning_efficiency() -> LearningEfficiencyMetrics
        Computes efficiency metrics across learning experiences
    execute() -> LearningStatistics
        Generates comprehensive learning statistics report
    """

    def __init__(self, database):
        """Initialize with database connection.

        Parameters
        ----------
        database : SimulationDatabase
            Database instance to use for queries
        """
        self.db = database

    @execute_query
    def learning_progress(self, session) -> List[LearningProgress]:
        """Calculate aggregated learning progress metrics over time.

        Retrieves and aggregates learning metrics for each simulation step, including
        rewards earned and action patterns. Each step's data is returned as a separate
        LearningProgress object.

        Parameters
        ----------
        session : Session
            SQLAlchemy session object (automatically injected by execute_query decorator)

        Returns
        -------
        List[LearningProgress]
            List of learning progress metrics per step, where each object contains:
            - step : int
                Step number in the simulation
            - reward : float
                Average reward achieved in this step
            - action_count : int
                Total number of actions taken in this step
            - unique_actions : int
                Number of distinct actions used in this step
        """
        progress_data = (
            session.query(
                LearningExperience.step_number,
                func.avg(LearningExperience.reward).label("avg_reward"),
                func.count(LearningExperience.action_taken).label("action_count"),
                func.count(distinct(LearningExperience.action_taken_mapped)).label(
                    "unique_actions"
                ),
            )
            .group_by(LearningExperience.step_number)
            .order_by(LearningExperience.step_number)
            .all()
        )

        return [
            LearningProgress(
                step=step,
                reward=float(reward or 0),
                action_count=int(count or 0),
                unique_actions=int(unique or 0),
            )
            for step, reward, count, unique in progress_data
        ]

    @execute_query
    def module_performance(self, session) -> Dict[str, ModulePerformance]:
        """Calculate performance metrics for each learning module type.

        Aggregates and analyzes performance data for each unique learning module,
        including rewards, action counts, and action diversity metrics.

        Parameters
        ----------
        session : Session
            SQLAlchemy session object (automatically injected by execute_query decorator)

        Returns
        -------
        Dict[str, ModulePerformance]
            Dictionary mapping module identifiers to their performance metrics, where each
            ModulePerformance contains:
            - module_type : str
                Type of learning module
            - module_id : str
                Unique identifier for the module
            - avg_reward : float
                Average reward achieved by the module
            - total_actions : int
                Total number of actions taken by the module
            - unique_actions : int
                Number of distinct actions used by the module
        """
        module_stats = (
            session.query(
                LearningExperience.module_type,
                LearningExperience.module_id,
                func.avg(LearningExperience.reward).label("avg_reward"),
                func.count(LearningExperience.action_taken).label("total_actions"),
                func.count(distinct(LearningExperience.action_taken_mapped)).label(
                    "unique_actions"
                ),
            )
            .group_by(LearningExperience.module_type, LearningExperience.module_id)
            .all()
        )

        return {
            f"{module_type}": ModulePerformance(
                module_type=module_type,
                module_id=module_id,
                avg_reward=float(avg_reward or 0),
                total_actions=int(total_actions or 0),
                unique_actions=int(unique_actions or 0),
            )
            for module_type, module_id, avg_reward, total_actions, unique_actions in module_stats
        }

    @execute_query
    def agent_learning_stats(
        self, session, agent_id: Optional[int] = None
    ) -> Dict[str, AgentLearningStats]:
        """Get learning statistics for specific agent or all agents.

        Retrieves and analyzes learning performance metrics either for a specific
        agent or aggregated across all agents.

        Parameters
        ----------
        agent_id : Optional[int]
            If provided, limits analysis to specific agent. If None, includes all agents.

        Returns
        -------
        Dict[str, AgentLearningStats]
            Dictionary mapping agent/module combinations to their statistics:
            - reward_mean: Average reward achieved
            - total_actions: Total number of actions taken
            - actions_used: List of unique actions performed
        """
        query = session.query(
            LearningExperience.agent_id,
            LearningExperience.module_type,
            func.avg(LearningExperience.reward).label("reward_mean"),
            func.count(LearningExperience.action_taken).label("total_actions"),
            func.group_concat(distinct(LearningExperience.action_taken_mapped)).label(
                "actions_used"
            ),
        )

        if agent_id is not None:
            query = query.filter(LearningExperience.agent_id == agent_id)

        results = query.group_by(
            LearningExperience.agent_id, LearningExperience.module_type
        ).all()

        return {
            f"{module_type}": AgentLearningStats(
                agent_id=agent_id,
                reward_mean=float(reward_mean or 0),
                total_actions=int(total_actions or 0),
                actions_used=actions_used.split(",") if actions_used else [],
            )
            for agent_id, module_type, reward_mean, total_actions, actions_used in results
        }

    @execute_query
    def learning_efficiency(self, session) -> LearningEfficiencyMetrics:
        """Calculate learning efficiency metrics.

        Computes various efficiency metrics to evaluate the overall learning
        performance and stability of the system.

        Parameters
        ----------
        session : Session
            SQLAlchemy session object (automatically injected by execute_query decorator)

        Returns
        -------
        LearningEfficiencyMetrics
            Object containing efficiency metrics:
            - reward_efficiency : float
                Average reward across all experiences (0.0 to 1.0)
            - action_diversity : float
                Ratio of unique actions to total actions (0.0 to 1.0)
            - learning_stability : float
                Measure of consistency in learning performance (0.0 to 1.0)
        """
        experiences = pd.read_sql(
            session.query(
                LearningExperience.step_number,
                LearningExperience.module_type,
                LearningExperience.reward,
                LearningExperience.action_taken_mapped,
            )
            .order_by(LearningExperience.step_number)
            .statement,
            session.bind,
        )

        if experiences.empty:
            return LearningEfficiencyMetrics(
                reward_efficiency=0.0,
                action_diversity=0.0,
                learning_stability=0.0,
            )

        # Calculate metrics
        reward_efficiency = experiences["reward"].mean()

        # Calculate action diversity (unique actions / total actions)
        total_actions = len(experiences)
        unique_actions = experiences["action_taken_mapped"].nunique()
        action_diversity = unique_actions / total_actions if total_actions > 0 else 0

        # Calculate learning stability (inverse of reward variance)
        reward_variance = experiences.groupby("module_type")["reward"].var().mean()
        learning_stability = 1 / (1 + reward_variance) if reward_variance > 0 else 1.0

        return LearningEfficiencyMetrics(
            reward_efficiency=float(reward_efficiency or 0),
            action_diversity=float(action_diversity or 0),
            learning_stability=float(learning_stability or 0),
        )

    @execute_query
    def execute(self, session) -> LearningStatistics:
        """Generate a comprehensive learning statistics report.

        Combines multiple analysis methods to create a complete picture of
        learning performance, including progress over time, module-specific
        metrics, and efficiency measures.

        Returns
        -------
        LearningStatistics
            Complete learning statistics including:
            - learning_progress: Time series of rewards and losses
            - module_performance: Per-module performance metrics
            - agent_learning_stats: Per-agent learning statistics
            - learning_efficiency: Overall efficiency metrics
        """
        return LearningStatistics(
            learning_progress=self.learning_progress(),
            module_performance=self.module_performance(),
            agent_learning_stats=self.agent_learning_stats(),
            learning_efficiency=self.learning_efficiency(),
        )
