"""Learning retrieval module for simulation database.

This module provides specialized queries and analysis methods for learning-related
data, including learning experiences, module performance, and adaptation metrics.

The LearningRetriever class handles learning-specific database operations with
optimized queries and efficient data aggregation methods.
"""

from typing import Dict, Optional

import pandas as pd
from sqlalchemy import func

from database.data_retrieval import execute_query
from database.data_types import LearningProgress, LearningStatistics, ModulePerformance
from database.models import LearningExperience


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
    agent_learning_stats(agent_id: Optional[int]) -> Dict[str, Dict[str, float]]
        Analyzes learning statistics for specific or all agents
    learning_efficiency() -> Dict[str, float]
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
    def learning_progress(self, session) -> LearningProgress:
        """Calculate aggregated learning progress metrics over time.

        Retrieves and aggregates reward and loss data for each simulation step,
        providing insight into the overall learning trajectory.

        Returns
        -------
        LearningProgress
            Time series data containing:
            - average_reward: List[float]
                Mean reward values for each step
            - average_loss: List[float]
                Mean loss values for each step
        """
        progress_data = (
            session.query(
                LearningExperience.step_number,
                func.avg(LearningExperience.reward).label("avg_reward"),
                func.avg(LearningExperience.loss).label("avg_loss"),
            )
            .group_by(LearningExperience.step_number)
            .order_by(LearningExperience.step_number)
            .all()
        )

        df = pd.DataFrame(progress_data, columns=["step", "reward", "loss"])

        return LearningProgress(
            average_reward=df["reward"].tolist(), average_loss=df["loss"].tolist()
        )

    @execute_query
    def module_performance(self, session) -> Dict[str, ModulePerformance]:
        """Calculate aggregated performance metrics for each learning module type.

        Analyzes the effectiveness of different learning modules by computing
        their average reward and loss metrics.

        Returns
        -------
        Dict[str, ModulePerformance]
            Dictionary mapping module types to their performance metrics:
            - avg_reward: float
                Average reward achieved by the module
            - avg_loss: float
                Average loss experienced by the module
        """
        module_stats = (
            session.query(
                LearningExperience.module_type,
                func.avg(LearningExperience.reward).label("avg_reward"),
                func.avg(LearningExperience.loss).label("avg_loss"),
            )
            .group_by(LearningExperience.module_type)
            .all()
        )

        return {
            module_type: ModulePerformance(
                avg_reward=float(avg_reward or 0), avg_loss=float(avg_loss or 0)
            )
            for module_type, avg_reward, avg_loss in module_stats
        }

    @execute_query
    def agent_learning_stats(
        self, session, agent_id: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """Get learning statistics for specific agent or all agents.

        Parameters
        ----------
        agent_id : Optional[int]
            ID of specific agent to analyze, if None analyzes all agents

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary containing per-module statistics:
            - reward_stats: Dict[str, float]
                - mean: Average reward
                - std: Standard deviation
                - min: Minimum reward
                - max: Maximum reward
            - loss_stats: Dict[str, float]
                - mean: Average loss
                - std: Standard deviation
                - min: Minimum loss
                - max: Maximum loss
        """
        query = session.query(
            LearningExperience.module_type,
            func.avg(LearningExperience.reward).label("reward_mean"),
            func.stddev(LearningExperience.reward).label("reward_std"),
            func.min(LearningExperience.reward).label("reward_min"),
            func.max(LearningExperience.reward).label("reward_max"),
            func.avg(LearningExperience.loss).label("loss_mean"),
            func.stddev(LearningExperience.loss).label("loss_std"),
            func.min(LearningExperience.loss).label("loss_min"),
            func.max(LearningExperience.loss).label("loss_max"),
        )

        if agent_id is not None:
            query = query.filter(LearningExperience.agent_id == agent_id)

        results = query.group_by(LearningExperience.module_type).all()

        return {
            module_type: {
                "reward_stats": {
                    "mean": float(reward_mean or 0),
                    "std": float(reward_std or 0),
                    "min": float(reward_min or 0),
                    "max": float(reward_max or 0),
                },
                "loss_stats": {
                    "mean": float(loss_mean or 0),
                    "std": float(loss_std or 0),
                    "min": float(loss_min or 0),
                    "max": float(loss_max or 0),
                },
            }
            for (
                module_type,
                reward_mean,
                reward_std,
                reward_min,
                reward_max,
                loss_mean,
                loss_std,
                loss_min,
                loss_max,
            ) in results
        }

    @execute_query
    def learning_efficiency(self, session) -> Dict[str, float]:
        """Calculate learning efficiency metrics.

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - reward_efficiency: Average reward per learning experience
            - loss_improvement: Rate of loss decrease
            - adaptation_rate: Rate of successful adaptations
            - learning_stability: Measure of learning stability
        """
        # Get sequential learning experiences to calculate improvements
        experiences = pd.read_sql(
            session.query(
                LearningExperience.step_number,
                LearningExperience.module_type,
                LearningExperience.reward,
                LearningExperience.loss,
            )
            .order_by(LearningExperience.step_number)
            .statement,
            session.bind,
        )

        if experiences.empty:
            return {
                "reward_efficiency": 0.0,
                "loss_improvement": 0.0,
                "adaptation_rate": 0.0,
                "learning_stability": 0.0,
            }

        # Calculate metrics
        reward_efficiency = experiences["reward"].mean()

        # Calculate loss improvement rate
        loss_changes = experiences.groupby("module_type")["loss"].diff()
        loss_improvement = (
            -loss_changes.mean()
        )  # Negative because decrease is improvement

        # Calculate adaptation rate (percentage of positive rewards)
        adaptation_rate = (experiences["reward"] > 0).mean()

        # Calculate learning stability (inverse of reward variance)
        reward_variance = experiences.groupby("module_type")["reward"].var().mean()
        learning_stability = 1 / (1 + reward_variance) if reward_variance > 0 else 1.0

        return {
            "reward_efficiency": float(reward_efficiency or 0),
            "loss_improvement": float(loss_improvement or 0),
            "adaptation_rate": float(adaptation_rate or 0),
            "learning_stability": float(learning_stability or 0),
        }

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
