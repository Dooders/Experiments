from typing import List, Optional

from sqlalchemy import func

from database.data_types import (
    AgentDistribution,
    BasicPopulationStatistics,
    Population,
    PopulationMetrics,
    PopulationStatistics,
    PopulationVariance,
)
from database.models import AgentState, SimulationStep
from database.retrievers import BaseRetriever
from database.utilities import execute_query


class PopulationStatisticsRetriever(BaseRetriever):
    """Retrieves and analyzes population statistics from simulation data.

    Provides comprehensive analysis of population dynamics, resource utilization,
    and agent distributions across simulation steps. Calculates statistics about
    agent populations, resource consumption, and survival metrics.

    Methods
    -------
    population_data() -> List[Population]
        Retrieves population and resource data for each step
    basic_population_statistics(pop_data: Optional[List[Population]]) -> BasicPopulationStatistics
        Calculates fundamental population statistics
    agent_type_distribution() -> AgentDistribution
        Analyzes distribution of agent types across steps
    population_momentum() -> float
        Calculates population sustainability metric
    population_metrics() -> PopulationMetrics
        Calculates total agents and type distribution
    population_variance(basic_stats: Optional[BasicPopulationStatistics]) -> PopulationVariance
        Calculates statistical variance measures
    execute() -> PopulationStatistics
        Generates comprehensive population analysis
    """

    def _execute(self) -> PopulationStatistics:
        """Execute comprehensive population statistics calculation.

        Returns
        -------
        PopulationStatistics
            Complete population analysis including metrics, variance,
            momentum, and agent distribution.
        """
        return self.execute()  # Calls the existing execute method

    @execute_query
    def population_data(self, session) -> List[Population]:
        """Retrieve base population and resource data for each simulation step.

        Queries the database to get step-wise population metrics including total agents,
        resources, and consumption data.

        Returns
        -------
        List[Population]
            List of Population objects containing:
            - step_number : int
                The simulation step number
            - total_agents : int
                Total number of agents in that step
            - total_resources : float
                Total available resources in that step
            - resources_consumed : float
                Total resources consumed by agents in that step
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
            .all()
        )

        return [
            Population(
                step_number=row[0],
                total_agents=row[1],
                total_resources=row[2],
                resources_consumed=row[3],
            )
            for row in pop_data
        ]

    @execute_query
    def basic_population_statistics(
        self, session, pop_data: Optional[List[Population]] = None
    ) -> BasicPopulationStatistics:
        """Calculate basic population statistics from step data.

        Processes raw population data to compute fundamental statistics about
        the population and resource usage.

        Parameters
        ----------
        pop_data : List[Population]
            List of Population objects containing step-wise simulation data

        Returns
        -------
        BasicPopulationStatistics
            Object containing:
            - avg_population : float
                Average population across all steps
            - death_step : int
                Final step number where agents existed
            - peak_population : int
                Maximum population reached
            - resources_consumed : float
                Total resources consumed across all steps
            - resources_available : float
                Total resources available across all steps
            - sum_squared : float
                Sum of squared population counts (for variance calculation)
            - step_count : int
                Total number of steps with active agents
        """
        if not pop_data:
            pop_data = self.population_data()

        # Calculate statistics directly from Population objects
        stats = {
            "avg_population": sum(p.total_agents for p in pop_data) / len(pop_data),
            "death_step": max(p.step_number for p in pop_data),
            "peak_population": max(p.total_agents for p in pop_data),
            "lowest_population": min(p.total_agents for p in pop_data),
            "resources_consumed": sum(p.resources_consumed for p in pop_data),
            "resources_available": sum(p.total_resources for p in pop_data),
            "sum_squared": sum(p.total_agents * p.total_agents for p in pop_data),
            "initial_population": pop_data[0].total_agents,
            "final_population": pop_data[-1].total_agents,
            "step_count": len(pop_data),
        }

        return BasicPopulationStatistics(
            avg_population=float(stats["avg_population"] or 0),
            death_step=int(stats["death_step"] or 0),
            peak_population=int(stats["peak_population"] or 0),
            lowest_population=int(stats["lowest_population"] or 0),
            resources_consumed=float(stats["resources_consumed"] or 0),
            resources_available=float(stats["resources_available"] or 0),
            sum_squared=float(stats["sum_squared"] or 0),
            initial_population=int(stats["initial_population"] or 0),
            final_population=int(stats["final_population"] or 0),
            step_count=int(stats["step_count"] or 1),
        )

    @execute_query
    def agent_type_distribution(self, session) -> AgentDistribution:
        """Analyze the distribution of different agent types across the simulation.

        Calculates the average number of each agent type (system, independent, and control)
        across all simulation steps.

        Returns
        -------
        AgentDistribution
            Distribution metrics containing:
            - system_agents : float
                Average number of system-controlled agents
            - independent_agents : float
                Average number of independently operating agents
            - control_agents : float
                Average number of control group agents
        """
        type_stats = session.query(
            func.avg(SimulationStep.system_agents).label("avg_system"),
            func.avg(SimulationStep.independent_agents).label("avg_independent"),
            func.avg(SimulationStep.control_agents).label("avg_control"),
        ).first()

        return AgentDistribution(
            system_agents=float(type_stats[0] or 0),
            independent_agents=float(type_stats[1] or 0),
            control_agents=float(type_stats[2] or 0),
        )

    @execute_query
    def population_momentum(self, session) -> float:
        """Population momentum is a metric that captures the relationship between
        population growth and simulation duration, calculated as:
        (final_step * max_population) / initial_population

        Returns
        -------
        float
            Population momentum metric. Returns 0.0 if initial population is 0.

        Notes
        -----
        This metric helps understand how well the population sustained and grew
        over time. Higher values indicate better population sustainability.
        """
        # Get initial population
        initial = (
            session.query(SimulationStep.total_agents)
            .order_by(SimulationStep.step_number)
            .first()
        )

        # Get max population and final step
        stats = session.query(
            func.max(SimulationStep.total_agents).label("max_count"),
            func.max(SimulationStep.step_number).label("final_step"),
        ).first()

        if initial and stats and initial[0] > 0:
            return (float(stats[1]) * float(stats[0])) / float(initial[0])

        return 0.0

    @execute_query
    def population_metrics(self, session) -> PopulationMetrics:
        """Calculate population metrics including total agents and distribution by type.

        Returns
        -------
        PopulationMetrics
            Object containing:
            - total_agents : int
                Peak population reached during simulation
            - system_agents : int
                Number of system-controlled agents
            - independent_agents : int
                Number of independently operating agents
            - control_agents : int
                Number of control group agents

        Notes
        -----
        This method combines data from basic population statistics and agent type
        distribution to provide a complete picture of population composition.
        """
        # Get basic statistics for peak population
        pop_data = self.population_data()
        basic_stats = self.basic_population_statistics(pop_data)

        # Get agent type distribution
        type_stats = self.agent_type_distribution()

        return PopulationMetrics(
            total_agents=basic_stats.peak_population,
            system_agents=int(type_stats.system_agents),
            independent_agents=int(type_stats.independent_agents),
            control_agents=int(type_stats.control_agents),
        )

    def population_variance(
        self, basic_stats: Optional[BasicPopulationStatistics] = None
    ) -> PopulationVariance:
        """Calculate statistical measures of population variation.

        Parameters
        ----------
        basic_stats : Optional[BasicPopulationStatistics]
            Pre-calculated basic statistics. If None, will be calculated.

        Returns
        -------
        PopulationVariance
            Object containing:
            - variance : float
                Statistical variance of population size
            - standard_deviation : float
                Standard deviation of population size
            - coefficient_variation : float
                Coefficient of variation (std_dev / mean)

        Notes
        -----
        Variance calculations use the formula:
        variance = (sum_squared / n) - (mean)^2
        where sum_squared is the sum of squared population counts
        and n is the number of steps.
        """
        if not basic_stats:
            pop_data = self.population_data()
            basic_stats = self.basic_population_statistics(pop_data)

        # Calculate variance using the sum of squares method
        variance = (basic_stats.sum_squared / basic_stats.step_count) - (
            basic_stats.avg_population**2
        )

        # Calculate standard deviation
        std_dev = variance**0.5

        # Calculate coefficient of variation
        cv = (
            std_dev / basic_stats.avg_population
            if basic_stats.avg_population > 0
            else 0
        )

        return PopulationVariance(
            variance=variance, standard_deviation=std_dev, coefficient_variation=cv
        )

    @execute_query
    def execute(self, session) -> PopulationStatistics:
        """Execute comprehensive population statistics calculation.

        Analyzes population data to generate complete statistical overview
        including population metrics, variance measures, momentum, and
        agent type distribution.

        Returns
        -------
        PopulationStatistics
            Complete analysis containing:
            - population_metrics: Total agents and distribution by type
            - population_variance: Statistical variation measures
            - population_momentum: Population sustainability metric
            - agent_distribution: Distribution of agent types
        """
        # Get base population data
        pop_data = self.population_data()

        # Get basic statistics
        basic_stats = self.basic_population_statistics(pop_data)

        # Return PopulationStatistics with the correct structure
        return PopulationStatistics(
            population_metrics=self.population_metrics(),
            population_variance=self.population_variance(basic_stats),
            population_momentum=self.population_momentum(),
            agent_distribution=self.agent_type_distribution(),
        )
