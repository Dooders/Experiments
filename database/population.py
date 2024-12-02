from typing import Dict

import pandas as pd
from sqlalchemy import func

from database.utilities import execute_query
from database.data_types import PopulationStatistics
from database.models import AgentState, SimulationStep


class PopulationStatisticsRetriever:
    """Handles retrieval and analysis of population statistics.

    This class encapsulates methods for analyzing population dynamics, resource utilization,
    and agent distributions across the simulation.
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
    def population_data(self, session) -> pd.DataFrame:
        """Get base population and resource data for each step.

        Returns
        -------
        pd.DataFrame
            DataFrame containing step-wise population data:
            - step_number: Step number
            - total_agents: Total number of agents
            - total_resources: Total resources
            - resources_consumed: Resources consumed
        """
        pop_data = (
            session.query(
                SimulationStep.step_number,
                SimulationStep.total_agents,
                SimulationStep.total_resources,
                func.sum(AgentState.resource_level).label("resources_consumed"),
            )
            .outerjoin(AgentState, AgentState.step_number == SimulationStep.step_number)
            .filter(SimulationStep.total_agents > 0)
            .group_by(SimulationStep.step_number)
            .subquery()
        )
        return pop_data

    @execute_query
    def basic_population_statistics(self, session, pop_data) -> Dict[str, float]:
        """Calculate basic population statistics.

        Parameters
        ----------
        pop_data : Subquery
            Population data subquery

        Returns
        -------
        Dict[str, float]
            Dictionary containing basic statistics:
            - avg_population: Average population
            - death_step: Final step
            - peak_population: Maximum population
            - resources_consumed: Total resources consumed
            - resources_available: Total resources available
            - sum_squared: Sum of squared population counts
            - step_count: Total number of steps
        """
        stats = session.query(
            func.avg(pop_data.c.total_agents).label("avg_population"),
            func.max(pop_data.c.step_number).label("death_step"),
            func.max(pop_data.c.total_agents).label("peak_population"),
            func.sum(pop_data.c.resources_consumed).label("total_resources_consumed"),
            func.sum(pop_data.c.total_resources).label("total_resources_available"),
            func.sum(pop_data.c.total_agents * pop_data.c.total_agents).label(
                "sum_squared"
            ),
            func.count().label("step_count"),
        ).first()

        if not stats:
            return {}

        return {
            "avg_population": float(stats[0] or 0),
            "death_step": int(stats[1] or 0),
            "peak_population": int(stats[2] or 0),
            "resources_consumed": float(stats[3] or 0),
            "resources_available": float(stats[4] or 0),
            "sum_squared": float(stats[5] or 0),
            "step_count": int(stats[6] or 1),
        }

    @execute_query
    def agent_type_distribution(self, session) -> Dict[str, float]:
        """Get distribution of agent types.

        Returns
        -------
        Dict[str, float]
            Dictionary containing average counts for each agent type:
            - system_agents: Average number of system agents
            - independent_agents: Average number of independent agents
            - control_agents: Average number of control agents
        """
        type_stats = session.query(
            func.avg(SimulationStep.system_agents).label("avg_system"),
            func.avg(SimulationStep.independent_agents).label("avg_independent"),
            func.avg(SimulationStep.control_agents).label("avg_control"),
        ).first()

        return {
            "system_agents": float(type_stats[0] or 0),
            "independent_agents": float(type_stats[1] or 0),
            "control_agents": float(type_stats[2] or 0),
        }

    @execute_query
    def execute(self, session) -> PopulationStatistics:
        """Calculate comprehensive population statistics.

        Returns
        -------
        PopulationStatistics
            Complete population statistics including:
            - basic_stats: Basic population metrics
            - resource_metrics: Resource utilization metrics
            - population_variance: Population variance metrics
            - agent_distribution: Agent type distribution
            - survival_metrics: Survival and lifespan metrics
        """
        # Get base population data
        pop_data = self.population_data(session)

        # Get basic statistics
        stats = self.basic_population_statistics(session, pop_data)
        if not stats:
            return {}

        # Calculate derived statistics
        variance = (stats["sum_squared"] / stats["step_count"]) - (
            stats["avg_population"] ** 2
        )
        std_dev = variance**0.5
        resource_utilization = (
            stats["resources_consumed"] / stats["resources_available"]
            if stats["resources_available"] > 0
            else 0
        )
        cv = std_dev / stats["avg_population"] if stats["avg_population"] > 0 else 0

        # Get agent type distribution
        type_stats = self.agent_type_distribution(session)

        return {
            "basic_stats": {
                "average_population": stats["avg_population"],
                "peak_population": stats["peak_population"],
                "death_step": stats["death_step"],
                "total_steps": stats["step_count"],
            },
            "resource_metrics": {
                "resource_utilization": resource_utilization,
                "resources_consumed": stats["resources_consumed"],
                "resources_available": stats["resources_available"],
                "utilization_per_agent": (
                    stats["resources_consumed"]
                    / (stats["avg_population"] * stats["death_step"])
                    if stats["avg_population"] * stats["death_step"] > 0
                    else 0
                ),
            },
            "population_variance": {
                "variance": variance,
                "standard_deviation": std_dev,
                "coefficient_variation": cv,
            },
            "agent_distribution": type_stats,
            "survival_metrics": {
                "survival_rate": (
                    stats["avg_population"] / stats["peak_population"]
                    if stats["peak_population"] > 0
                    else 0
                ),
                "average_lifespan": (
                    stats["death_step"] / 2 if stats["death_step"] > 0 else 0
                ),
            },
        }
