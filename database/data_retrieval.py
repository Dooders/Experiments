"""Data retrieval module for simulation database.

This module provides a comprehensive interface for querying and analyzing simulation data,
including agent states, resource distributions, learning patterns, and system metrics.

The DataRetriever class handles all database operations with optimized queries and
efficient data aggregation methods.

Features
--------
- Agent statistics and lifecycle analysis
- Population dynamics and demographics
- Resource distribution and consumption patterns
- Learning and adaptation metrics
- Behavioral pattern analysis
- Historical trend analysis
- Performance and efficiency metrics

Classes
-------
DataRetriever
    Main class handling all data retrieval and analysis operations

Dependencies
-----------
- sqlalchemy: Database ORM and query building
- pandas: Data processing and statistical analysis
- json: JSON data handling
- logging: Error and operation logging
- typing: Type hints and annotations

Notes
-----
All database operations are executed within transactions to ensure data consistency.
Query optimization is implemented for handling large datasets efficiently.
"""

import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sqlalchemy import and_, case, exists, func, not_
from sqlalchemy.orm import aliased

from .data_types import (
    ActionMetrics,
    AdvancedStatistics,
    AgentBehaviorMetrics,
    AgentDecisionMetrics,
    AgentDistribution,
    AgentLifespanStats,
    AgentStateData,
    AgentStates,
    DecisionPattern,
    DecisionPatterns,
    DecisionPatternStats,
    DecisionSummary,
    EfficiencyMetrics,
    HistoricalMetrics,
    InteractionMetrics,
    InteractionPattern,
    InteractionStats,
    LearningProgress,
    LearningStatistics,
    ModulePerformance,
    PopulationMetrics,
    PopulationStatistics,
    PopulationVariance,
    ResourceBehavior,
    ResourceDistribution,
    ResourceImpact,
    ResourceMetrics,
    ResourceStates,
    RewardStats,
    SequencePattern,
    SimulationResults,
    SimulationState,
    StepActionData,
    SurvivalMetrics,
    TimePattern,
)
from .models import (
    Agent,
    AgentAction,
    AgentState,
    HealthIncident,
    LearningExperience,
    ResourceState,
    SimulationStep,
)

logger = logging.getLogger(__name__)


def execute_query(func):
    """Decorator to execute database queries within a transaction.

    Wraps methods that contain database query logic, executing them within
    the database transaction context.

    Parameters
    ----------
    func : callable
        The method containing the database query logic

    Returns
    -------
    callable
        Wrapped method that executes within a transaction
    """

    def wrapper(self, *args, **kwargs):
        def query(session):
            return func(self, session, *args, **kwargs)

        return self.db._execute_in_transaction(query)

    return wrapper


class SimulationStateRetriever:
    """Handles retrieval of simulation state data.

    This class encapsulates methods for retrieving agent states, resource states,
    and overall simulation state data for specific simulation steps.
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
    def agent_states(
        self, session, step_number: Optional[int] = None
    ) -> List[AgentStates]:
        """Retrieve agent states for a specific step or all steps.

        Gets the state data for all agents at either a specific simulation step or
        across all steps. When no step is specified, returns the complete history
        ordered by step number and agent ID.

        Parameters
        ----------
        session : Session
            SQLAlchemy database session (injected by decorator)
        step_number : Optional[int], default=None
            The simulation step number to retrieve data for.
            If None, returns data for all steps.

        Returns
        -------
        List[AgentStates]
            List of agent states containing:
            - step_number: int
                Simulation step number
            - agent_id: int
                Unique identifier for the agent
            - agent_type: str
                Type/category of the agent
            - position_x: float
                X coordinate of agent position
            - position_y: float
                Y coordinate of agent position
            - resource_level: float
                Current resource level of the agent
            - current_health: float
                Current health level of the agent
            - is_defending: bool
                Whether the agent is in defensive stance

        Notes
        -----
        The returned data is ordered by step_number and agent_id when retrieving
        multiple steps. For single step queries, only agent_id ordering is applied.
        """
        agent_states = session.query(
            AgentState.step_number,
            AgentState.agent_id,
            Agent.agent_type,
            AgentState.position_x,
            AgentState.position_y,
            AgentState.resource_level,
            AgentState.current_health,
            AgentState.is_defending,
        ).join(Agent)

        if step_number is not None:
            agent_states = agent_states.filter(AgentState.step_number == step_number)
        else:
            agent_states = agent_states.order_by(
                AgentState.step_number, AgentState.agent_id
            )

        return agent_states.all()

    @execute_query
    def resource_states(self, session, step_number: int) -> List[ResourceStates]:
        """Retrieve resource states for a specific step.

        Gets the state of all resources in the simulation at the specified step number,
        including their positions and amounts.

        Parameters
        ----------
        step_number : int
            The simulation step number to retrieve data for

        Returns
        -------
        ResourceStates
            List of ResourceState objects containing:
            - resource_id: int
                Unique identifier for the resource
            - amount: float
                Current amount of the resource
            - position_x: float
                X coordinate of resource position
            - position_y: float
                Y coordinate of resource position

        Notes
        -----
        Resources that were depleted or removed will not be included in the results.
        Positions are in simulation grid coordinates.
        """
        results = (
            session.query(
                ResourceState.resource_id,
                ResourceState.amount,
                ResourceState.position_x,
                ResourceState.position_y,
            )
            .filter(ResourceState.step_number == step_number)
            .all()
        )
        return results

    @execute_query
    def simulation_state(self, session, step_number: int) -> SimulationState:
        """Retrieve simulation state for a specific step.

        Gets the overall simulation state metrics and configuration at the specified
        step number, including population counts, resource totals, and system metrics.

        Parameters
        ----------
        step_number : int
            The simulation step number to retrieve data for

        Returns
        -------
        SimulationState
            Dictionary containing:
            - step_number: int
                Current step number
            - total_agents: int
                Total number of agents alive
            - total_resources: float
                Total resources available
            - average_agent_health: float
                Mean health across all agents
            - average_agent_resources: float
                Mean resources per agent
            - births: int
                Number of births this step
            - deaths: int
                Number of deaths this step
            - system_metrics: Dict[str, float]
                Additional system performance metrics

        Notes
        -----
        Returns None if the step number is not found in the database.
        The SimulationState type is defined in data_types.py.
        """
        simulation_step = (
            session.query(SimulationStep)
            .filter(SimulationStep.step_number == step_number)
            .first()
        )
        return simulation_step.as_dict()

    def get_results(self, step_number: int) -> SimulationResults:
        """Retrieve complete simulation state for a specific step.

        Parameters
        ----------
        step_number : int
            The simulation step number to retrieve data for

        Returns
        -------
        SimulationResults
            Data containing agent states, resource states, and simulation metrics
        """
        return {
            "agent_states": self.agent_states(step_number),
            "resource_states": self.resource_states(step_number),
            "simulation_state": self.simulation_state(step_number),
        }


class AgentLifespanRetriever:
    """Handles retrieval of agent lifespan statistics.
    
    This class encapsulates methods for analyzing agent lifespans, survival rates,
    and generational statistics across the simulation.
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
    def get_lifespans(self, session) -> Dict[str, Dict[str, float]]:
        """Calculate lifespan statistics by agent type and generation.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary containing:
            - average_lifespan: float
                Mean lifespan across all agents
            - lifespan_by_type: Dict[str, float]
                Mean lifespan per agent type
            - lifespan_by_generation: Dict[int, float]
                Mean lifespan per generation
        """
        lifespans = (
            session.query(
                Agent.agent_type,
                Agent.generation,
                (Agent.death_time - Agent.birth_time).label("lifespan"),
            )
            .filter(Agent.death_time.isnot(None))
            .all()
        )

        lifespan_data = pd.DataFrame(
            lifespans, columns=["agent_type", "generation", "lifespan"]
        )

        return {
            "average_lifespan": lifespan_data["lifespan"].mean(),
            "lifespan_by_type": lifespan_data.groupby("agent_type")["lifespan"]
            .mean()
            .to_dict(),
            "lifespan_by_generation": lifespan_data.groupby("generation")[
                "lifespan"
            ]
            .mean()
            .to_dict(),
        }

    @execute_query
    def get_survival_rates(self, session) -> Dict[int, float]:
        """Calculate survival rates by generation.

        Returns
        -------
        Dict[int, float]
            Dictionary mapping generation numbers to their survival rates (0-100).
            Survival rate is the percentage of agents still alive in each generation.
        """
        survival_rates = (
            session.query(
                Agent.generation,
                func.count(case([(Agent.death_time.is_(None), 1)]))
                * 100.0
                / func.count(),
            )
            .group_by(Agent.generation)
            .all()
        )

        survival_data = pd.DataFrame(
            survival_rates, columns=["generation", "survival_rate"]
        )
        
        return survival_data.set_index("generation")["survival_rate"].to_dict()

    @execute_query
    def get_statistics(self, session) -> AgentLifespanStats:
        """Calculate comprehensive statistics about agent lifespans.

        Returns
        -------
        AgentLifespanStats
            Data containing:
            - average_lifespan: float
                Mean lifespan across all agents
            - lifespan_by_type: Dict[str, float]
                Mean lifespan per agent type
            - lifespan_by_generation: Dict[int, float]
                Mean lifespan per generation
            - survival_rates: Dict[int, float]
                Survival rate per generation
        """
        # Get lifespan statistics
        lifespan_stats = self.get_lifespans()
        
        # Get survival rates
        survival_rates = self.get_survival_rates()

        return {
            **lifespan_stats,
            "survival_rates": survival_rates,
        }


class DataRetriever:
    """Handles data retrieval operations for the simulation database."""

    def __init__(self, database):
        """Initialize data retriever with database connection.

        Parameters
        ----------
        database : SimulationDatabase
            Database instance to use for queries
        """
        self.db = database
        self._retrievers = {
            "simulation_state": SimulationStateRetriever(database),
            "agent_lifespan": AgentLifespanRetriever(database),
        }

    def __getattr__(self, name):
        """Dynamically search for methods in sub-retrievers.

        Parameters
        ----------
        name : str
            Name of the attribute/method being accessed

        Returns
        -------
        Any
            The found method/attribute from a sub-retriever

        Raises
        ------
        AttributeError
            If the attribute/method is not found in any sub-retriever
        """
        # Search through all sub-retrievers for the requested method
        for retriever in self._retrievers.values():
            if hasattr(retriever, name):
                return getattr(retriever, name)

        # If method not found in any retriever, raise AttributeError
        raise AttributeError(
            f"'{self.__class__.__name__}' object and its sub-retrievers "
            f"have no attribute '{name}'"
        )

    def simulation_results(self, step_number: int) -> SimulationResults:
        """Retrieve complete simulation state for a specific step."""
        return self._retrievers["simulation_state"].get_results(step_number)

    def agent_lifespan_statistics(self) -> AgentLifespanStats:
        """Calculate comprehensive statistics about agent lifespans.

        Returns
        -------
        AgentLifespanStats
            Data containing:
            - average_lifespan: float, mean lifespan across all agents
            - lifespan_by_type: Dict[str, float], mean lifespan per agent type
            - lifespan_by_generation: Dict[int, float], mean lifespan per generation
            - survival_rates: Dict[int, float], survival rate per generation
        """
        return self._retrievers["agent_lifespan"].get_statistics()

    def get_population_statistics(self) -> PopulationStatistics:
        """Calculate comprehensive population statistics using SQLAlchemy.

        Returns
        -------
        PopulationStatistics
            Data containing:
            - basic_stats: Dict[str, float]
                - average_population: Average number of agents
                - peak_population: Maximum population reached
                - death_step: Step when population died out
                - total_steps: Total simulation steps
            - resource_metrics: Dict[str, float]
                - resource_utilization: Resource usage efficiency
                - resources_consumed: Total resources consumed
                - resources_available: Total resources available
                - utilization_per_agent: Average resource usage per agent
            - population_variance: Dict[str, float]
                - variance: Population size variance
                - standard_deviation: Population size standard deviation
                - coefficient_variation: Coefficient of variation
            - agent_distribution: Dict[str, float]
                - system_agents: Average number of system agents
                - independent_agents: Average number of independent agents
                - control_agents: Average number of control agents
            - survival_metrics: Dict[str, float]
                - survival_rate: Population survival rate
                - average_lifespan: Average agent lifespan
        """

        def _query(session):
            # Subquery for population data
            pop_data = (
                session.query(
                    SimulationStep.step_number,
                    SimulationStep.total_agents,
                    SimulationStep.total_resources,
                    func.sum(AgentState.resource_level).label("resources_consumed"),
                )
                .outerjoin(
                    AgentState, AgentState.step_number == SimulationStep.step_number
                )
                .filter(SimulationStep.total_agents > 0)
                .group_by(SimulationStep.step_number)
                .subquery()
            )

            # Calculate basic statistics
            stats = session.query(
                func.avg(pop_data.c.total_agents).label("avg_population"),
                func.max(pop_data.c.step_number).label("death_step"),
                func.max(pop_data.c.total_agents).label("peak_population"),
                func.sum(pop_data.c.resources_consumed).label(
                    "total_resources_consumed"
                ),
                func.sum(pop_data.c.total_resources).label("total_resources_available"),
                func.sum(pop_data.c.total_agents * pop_data.c.total_agents).label(
                    "sum_squared"
                ),
                func.count().label("step_count"),
            ).first()

            if not stats:
                return {}

            avg_pop = float(stats[0] or 0)
            death_step = int(stats[1] or 0)
            peak_pop = int(stats[2] or 0)
            resources_consumed = float(stats[3] or 0)
            resources_available = float(stats[4] or 0)
            sum_squared = float(stats[5] or 0)
            step_count = int(stats[6] or 1)

            # Calculate derived statistics
            variance = (sum_squared / step_count) - (avg_pop * avg_pop)
            std_dev = variance**0.5
            resource_utilization = (
                resources_consumed / resources_available
                if resources_available > 0
                else 0
            )
            cv = std_dev / avg_pop if avg_pop > 0 else 0

            # Get agent type distribution
            type_stats = session.query(
                func.avg(SimulationStep.system_agents).label("avg_system"),
                func.avg(SimulationStep.independent_agents).label("avg_independent"),
                func.avg(SimulationStep.control_agents).label("avg_control"),
            ).first()

            return {
                "basic_stats": {
                    "average_population": avg_pop,
                    "peak_population": peak_pop,
                    "death_step": death_step,
                    "total_steps": step_count,
                },
                "resource_metrics": {
                    "resource_utilization": resource_utilization,
                    "resources_consumed": resources_consumed,
                    "resources_available": resources_available,
                    "utilization_per_agent": (
                        resources_consumed / (avg_pop * death_step)
                        if avg_pop * death_step > 0
                        else 0
                    ),
                },
                "population_variance": {
                    "variance": variance,
                    "standard_deviation": std_dev,
                    "coefficient_variation": cv,
                },
                "agent_distribution": {
                    "system_agents": float(type_stats[0] or 0),
                    "independent_agents": float(type_stats[1] or 0),
                    "control_agents": float(type_stats[2] or 0),
                },
                "survival_metrics": {
                    "survival_rate": (avg_pop / peak_pop if peak_pop > 0 else 0),
                    "average_lifespan": (death_step / 2 if death_step > 0 else 0),
                },
            }

        return self.db._execute_in_transaction(_query)

    def get_resource_statistics(self) -> Dict[str, Any]:
        """Get statistics about resource distribution and consumption.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - resource_distribution: ResourceDistribution
            - efficiency_metrics: EfficiencyMetrics
        """

        def _query(session):
            # Get resource states over time
            resource_data = (
                session.query(
                    SimulationStep.step_number,
                    SimulationStep.total_resources,
                    SimulationStep.average_agent_resources,
                    SimulationStep.resource_efficiency,
                    SimulationStep.resource_distribution_entropy,
                )
                .order_by(SimulationStep.step_number)
                .all()
            )

            df = pd.DataFrame(
                resource_data,
                columns=[
                    "step",
                    "total_resources",
                    "avg_agent_resources",
                    "efficiency",
                    "entropy",
                ],
            )

            return {
                "resource_distribution": {
                    "steps": df["step"].tolist(),
                    "total_resources": df["total_resources"].tolist(),
                    "average_per_agent": df["avg_agent_resources"].tolist(),
                },
                "efficiency_metrics": {
                    "average_efficiency": df["efficiency"].mean(),
                    "efficiency_trend": df["efficiency"].tolist(),
                    "distribution_entropy": df["entropy"].tolist(),
                },
            }

        return self.db._execute_in_transaction(_query)

    def get_learning_statistics(self) -> LearningStatistics:
        """Get statistics about agent learning and adaptation.

        Returns
        -------
        LearningStatistics
            Data containing:
            - learning_progress: LearningProgress
            - module_performance: Dict[str, ModulePerformance]
        """

        def _query(session):
            # Get learning experiences
            learning_data = (
                session.query(
                    LearningExperience.step_number,
                    LearningExperience.module_type,
                    LearningExperience.reward,
                    LearningExperience.loss,
                )
                .order_by(LearningExperience.step_number)
                .all()
            )

            df = pd.DataFrame(
                learning_data, columns=["step", "module_type", "reward", "loss"]
            )

            return {
                "learning_progress": {
                    "average_reward": df.groupby("step")["reward"].mean().to_dict(),
                    "average_loss": df.groupby("step")["loss"].mean().to_dict(),
                },
                "module_performance": {
                    module: {
                        "avg_reward": group["reward"].mean(),
                        "avg_loss": group["loss"].mean(),
                    }
                    for module, group in df.groupby("module_type")
                },
            }

        return self.db._execute_in_transaction(_query)

    def get_historical_data(self) -> HistoricalMetrics:
        """Retrieve historical metrics for the entire simulation."""

        def _query(session):
            steps = (
                session.query(SimulationStep).order_by(SimulationStep.step_number).all()
            )

            return {
                "steps": [step.step_number for step in steps],
                "metrics": {
                    "total_agents": [step.total_agents for step in steps],
                    "system_agents": [step.system_agents for step in steps],
                    "independent_agents": [step.independent_agents for step in steps],
                    "control_agents": [step.control_agents for step in steps],
                    "total_resources": [step.total_resources for step in steps],
                    "average_agent_resources": [
                        step.average_agent_resources for step in steps
                    ],
                    "births": [step.births for step in steps],
                    "deaths": [step.deaths for step in steps],
                },
            }

        return self.db._execute_in_transaction(_query)

    def population_momentum(self) -> float:
        """Calculate population momentum using simpler SQL queries.

        Returns
        -------
        float
            Population momentum metric, calculated as:
            (final_step * max_population) / initial_population
            Returns 0.0 if initial population is 0.
        """

        def _query(session):
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

        return self.db._execute_in_transaction(_query)

    def population_stats(self, session) -> Dict[str, Any]:
        """Get basic population statistics.

        Parameters
        ----------
        session : Session
            SQLAlchemy database session

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - average_population: Average number of agents
            - peak_population: Maximum population reached
            - minimum_population: Minimum population reached
            - total_steps: Total simulation steps
            - average_health: Average agent health
        """
        stats = session.query(
            func.avg(SimulationStep.total_agents).label("avg_pop"),
            func.max(SimulationStep.total_agents).label("peak_pop"),
            func.min(SimulationStep.total_agents).label("min_pop"),
            func.count(SimulationStep.step_number).label("total_steps"),
            func.avg(SimulationStep.average_agent_health).label("avg_health"),
        ).first()

        return {
            "average_population": float(stats[0] or 0),
            "peak_population": int(stats[1] or 0),
            "minimum_population": int(stats[2] or 0),
            "total_steps": int(stats[3] or 0),
            "average_health": float(stats[4] or 0),
        }

    def agent_type_ratios(self, session) -> Dict[str, float]:
        """Get distribution of agent types.

        Parameters
        ----------
        session : Session
            SQLAlchemy database session

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - system_ratio: Proportion of system agents
            - independent_ratio: Proportion of independent agents
            - control_ratio: Proportion of control agents
        """
        stats = session.query(
            func.avg(SimulationStep.system_agents).label("avg_system"),
            func.avg(SimulationStep.independent_agents).label("avg_independent"),
            func.avg(SimulationStep.control_agents).label("avg_control"),
            func.avg(SimulationStep.total_agents).label("avg_total"),
        ).first()

        total = float(stats[3] or 1)  # Avoid division by zero
        ratios = [float(count or 0) / total for count in stats[:3]]

        return {
            "system_ratio": ratios[0],
            "independent_ratio": ratios[1],
            "control_ratio": ratios[2],
        }

    def basic_interaction_metrics(self, session) -> Dict[str, float]:
        """Get basic interaction statistics.

        Parameters
        ----------
        session : Session
            SQLAlchemy database session

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - total_actions: Total number of actions
        """
        basic_stats = session.query(
            func.count(AgentAction.action_id).label("total_actions"),
            func.sum(
                case([(AgentAction.action_type.in_(["attack", "defend"]), 1)], else_=0)
            ).label("conflicts"),
            func.sum(
                case([(AgentAction.action_type.in_(["share", "help"]), 1)], else_=0)
            ).label("cooperation"),
            func.sum(
                case([(AgentAction.action_type == "reproduce", 1)], else_=0)
            ).label("reproductions"),
        ).first()

        return {"total_actions": int(basic_stats[0] or 0)}

    def reward_metrics(self, session) -> Dict[str, float]:
        # Get reward statistics for different interaction types
        reward_stats = session.query(
            func.avg(
                case(
                    [
                        (
                            AgentAction.action_type.in_(["attack", "defend"]),
                            AgentAction.reward,
                        )
                    ],
                    else_=None,
                )
            ).label("conflict_reward"),
            func.avg(
                case(
                    [
                        (
                            AgentAction.action_type.in_(["share", "help"]),
                            AgentAction.reward,
                        )
                    ],
                    else_=None,
                )
            ).label("coop_reward"),
            func.count(case([(AgentAction.reward > 0, 1)])).label(
                "successful_interactions"
            ),
            func.count(case([(AgentAction.action_target_id.isnot(None), 1)])).label(
                "total_interactions"
            ),
        ).first()

    def interaction_metrics(self, session) -> InteractionMetrics:
        """Get interaction statistics.

        Parameters
        ----------
        session : Session
            SQLAlchemy database session

        Returns
        -------
        InteractionMetrics
            - total_actions: int
                Total number of actions
            - conflict_rate: float
                Rate of conflict actions (attack/defend)
            - cooperation_rate: float
                Rate of cooperative actions (share/help)
            - reproduction_rate: float
                Rate of reproduction actions
            - interaction_density: float
                Ratio of interactive to total actions
            - avg_reward_conflict: float
                Average reward for conflict actions
            - avg_reward_coop: float
                Average reward for cooperative actions
            - interaction_success: float
                Rate of successful interactions (positive reward)
        """
        # Get basic interaction counts and rates
        basic_stats = self.basic_interaction_metrics(session)

        # Get reward statistics for different interaction types
        reward_stats = self.reward_metrics(session)

        total = float(basic_stats[0] or 1)  # Avoid division by zero
        total_interactions = float(reward_stats[3] or 1)  # Avoid division by zero

        return {
            "total_actions": int(basic_stats[0] or 0),
            "conflict_rate": float(basic_stats[1] or 0) / total,
            "cooperation_rate": float(basic_stats[2] or 0) / total,
            "reproduction_rate": float(basic_stats[3] or 0) / total,
            "interaction_density": float(reward_stats[3] or 0) / total,
            "avg_reward_conflict": float(reward_stats[0] or 0),
            "avg_reward_coop": float(reward_stats[1] or 0),
            "interaction_success": float(reward_stats[2] or 0) / total_interactions,
        }

    def resource_metrics(self, session) -> Dict[str, float]:
        """Get resource efficiency metrics.

        Parameters
        ----------
        session : Session
            SQLAlchemy database session

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - average_efficiency: Mean resource utilization efficiency
            - average_total_resources: Mean total resources available
            - average_agent_resources: Mean resources per agent
            - resource_utilization: Resource usage efficiency ratio
        """
        stats = session.query(
            func.avg(SimulationStep.resource_efficiency).label("avg_efficiency"),
            func.avg(SimulationStep.total_resources).label("avg_resources"),
            func.avg(SimulationStep.average_agent_resources).label(
                "avg_agent_resources"
            ),
        ).first()

        return {
            "average_efficiency": float(stats[0] or 0),
            "average_total_resources": float(stats[1] or 0),
            "average_agent_resources": float(stats[2] or 0),
            "resource_utilization": float(stats[2] or 0) / float(stats[1] or 1),
        }

    def _calculate_diversity_index(self, ratios: List[float]) -> float:
        """Calculate Shannon entropy for agent type diversity.

        Parameters
        ----------
        ratios : List[float]
            List of agent type proportions

        Returns
        -------
        float
            Shannon entropy diversity index
        """
        import math

        return sum(-ratio * math.log(ratio) if ratio > 0 else 0 for ratio in ratios)

    def advanced_statistics(self) -> AdvancedStatistics:
        """Calculate advanced simulation statistics using optimized queries.

        Returns
        -------
        AdvancedStatistics: Dict[str, Any]
            Data containing:
            - Population metrics (peak, average, diversity)
            - Interaction statistics (conflict/cooperation ratios)
            - Resource efficiency metrics
            - Agent type distribution
            - Survival and adaptation metrics
        """

        def _query(session):
            # Get component statistics using helper methods
            pop_stats = self.population_stats(session)
            type_ratios = self.agent_type_ratios(session)
            interaction_metrics = self.interaction_metrics(session)
            resource_metrics = self.resource_metrics(session)

            # Calculate diversity index
            diversity = self._calculate_diversity_index(list(type_ratios.values()))

            return AdvancedStatistics(
                population_metrics=PopulationStatistics(
                    population_metrics=PopulationMetrics(**pop_stats),
                    population_variance=PopulationVariance(
                        total_agents=pop_stats["total_agents"],
                        system_agents=pop_stats["system_agents"],
                        independent_agents=pop_stats["independent_agents"],
                        control_agents=pop_stats["control_agents"],
                    ),
                ),
                interaction_metrics=InteractionPattern(
                    **interaction_metrics,
                    conflict_cooperation_ratio=(
                        interaction_metrics["conflict_rate"]
                        / interaction_metrics["cooperation_rate"]
                        if interaction_metrics["cooperation_rate"] > 0
                        else float("inf")
                    ),
                ),
                resource_metrics=ResourceMetrics(**resource_metrics),
                agent_distribution=AgentDistribution(
                    **type_ratios,
                    type_entropy=diversity,
                ),
                survival_metrics=SurvivalMetrics(
                    population_stability=(
                        pop_stats["minimum_population"] / pop_stats["peak_population"]
                    ),
                    health_maintenance=(pop_stats["average_health"] / 100.0),
                    interaction_rate=(
                        interaction_metrics["total_actions"]
                        / pop_stats["total_steps"]
                        / pop_stats["average_population"]
                    ),
                ),
            )

        return self.db._execute_in_transaction(_query)

    def get_agent_data(self, agent_id: int) -> AgentStateData:
        """Get comprehensive data for a specific agent.

        Parameters
        ----------
        agent_id : int
            The unique identifier of the agent

        Returns
        -------
        AgentStateData
            Data containing:
            - basic_info: Dict[str, Any]
                - agent_id: Agent ID
                - agent_type: Agent type
                - birth_time: Birth step
                - death_time: Death step (None if alive)
                - lifespan: Lifespan in steps (None if alive)
                - initial_resources: Initial resource level
                - max_health: Maximum health
                - starvation_threshold: Starvation threshold
            - genetic_info: Dict[str, Any]
                - genome_id: Genome ID
                - parent_id: Parent ID (None if no parent)
                - generation: Generation number
            - current_state: Dict[str, Any]
                - current_health: Current health
                - resource_level: Current resource level
                - total_reward: Total reward received
                - age: Age in steps
                - is_defending: Whether the agent is defending
                - position: (x, y) position
                - step_number: Current step number
            - historical_metrics: Dict[str, float]
                - average_health: Average health over lifetime
                - average_resources: Average resource level over lifetime
                - total_steps: Total steps lived
                - total_reward: Total reward received
            - action_history: Dict[str, Dict[str, float]]
                Per action type statistics including:
                - count: Number of occurrences
                - average_reward: Average reward for the action
            - health_incidents: List[Dict[str, Any]]
                List of health incidents including:
                - step: Step number
                - health_before: Health before the incident
                - health_after: Health after the incident
                - cause: Cause of the incident
                - details: Additional details
        """

        def _query(session):
            # Get basic agent information
            agent = session.query(Agent).filter(Agent.agent_id == agent_id).first()

            if not agent:
                return {}

            # Get latest state
            latest_state = (
                session.query(AgentState)
                .filter(AgentState.agent_id == agent_id)
                .order_by(AgentState.step_number.desc())
                .first()
            )

            # Get historical metrics
            historical_metrics = (
                session.query(
                    func.avg(AgentState.current_health).label("avg_health"),
                    func.avg(AgentState.resource_level).label("avg_resources"),
                    func.count(AgentState.step_number).label("total_steps"),
                    func.max(AgentState.total_reward).label("total_reward"),
                )
                .filter(AgentState.agent_id == agent_id)
                .first()
            )

            # Get action statistics
            action_stats = (
                session.query(
                    AgentAction.action_type,
                    func.count().label("count"),
                    func.avg(AgentAction.reward).label("avg_reward"),
                )
                .filter(AgentAction.agent_id == agent_id)
                .group_by(AgentAction.action_type)
                .all()
            )

            # Get health incidents
            health_incidents = (
                session.query(HealthIncident)
                .filter(HealthIncident.agent_id == agent_id)
                .order_by(HealthIncident.step_number)
                .all()
            )

            # Format the response
            response = {
                "basic_info": {
                    "agent_id": agent.agent_id,
                    "agent_type": agent.agent_type,
                    "birth_time": agent.birth_time,
                    "death_time": agent.death_time,
                    "lifespan": (
                        agent.death_time - agent.birth_time
                        if agent.death_time
                        else None
                    ),
                    "initial_resources": agent.initial_resources,
                    "max_health": agent.max_health,
                    "starvation_threshold": agent.starvation_threshold,
                },
                "genetic_info": {
                    "genome_id": agent.genome_id,
                    "parent_id": agent.parent_id,
                    "generation": agent.generation,
                },
                "current_state": (
                    {
                        "current_health": latest_state.current_health,
                        "resource_level": latest_state.resource_level,
                        "total_reward": latest_state.total_reward,
                        "age": latest_state.age,
                        "is_defending": latest_state.is_defending,
                        "position": (latest_state.position_x, latest_state.position_y),
                        "step_number": latest_state.step_number,
                    }
                    if latest_state
                    else None
                ),
                "historical_metrics": {
                    "average_health": float(historical_metrics[0] or 0),
                    "average_resources": float(historical_metrics[1] or 0),
                    "total_steps": int(historical_metrics[2] or 0),
                    "total_reward": float(historical_metrics[3] or 0),
                },
                "action_history": {
                    action_type: {
                        "count": count,
                        "average_reward": float(avg_reward or 0),
                    }
                    for action_type, count, avg_reward in action_stats
                },
                "health_incidents": [
                    {
                        "step": incident.step_number,
                        "health_before": incident.health_before,
                        "health_after": incident.health_after,
                        "cause": incident.cause,
                        "details": incident.details,
                    }
                    for incident in health_incidents
                ],
            }

            return response

        return self.db._execute_in_transaction(_query)

    def get_agent_actions(
        self,
        agent_id: int,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get detailed action history for a specific agent.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - action_history: Dict[str, Any]
            - action_statistics: Dict[str, Dict[str, float]]
            - interaction_patterns: Dict[str, InteractionPattern]
            - reward_analysis: RewardStats
            - resource_impact: ResourceBehavior
        """

        def _query(session):
            # Build base query
            base_query = session.query(AgentAction).filter(
                AgentAction.agent_id == agent_id
            )

            # Apply time range filters if provided
            if start_step is not None:
                base_query = base_query.filter(AgentAction.step_number >= start_step)
            if end_step is not None:
                base_query = base_query.filter(AgentAction.step_number <= end_step)

            # Get chronological action list
            actions = base_query.order_by(AgentAction.step_number).all()

            # Get action type statistics
            action_stats = (
                base_query.with_entities(
                    AgentAction.action_type,
                    func.count().label("count"),
                    func.avg(AgentAction.reward).label("avg_reward"),
                    func.sum(AgentAction.reward).label("total_reward"),
                    func.avg(
                        AgentAction.resources_after - AgentAction.resources_before
                    ).label("avg_resource_change"),
                )
                .group_by(AgentAction.action_type)
                .all()
            )

            # Get interaction patterns
            interaction_stats = (
                base_query.filter(AgentAction.action_target_id.isnot(None))
                .with_entities(
                    AgentAction.action_target_id,
                    func.count().label("interaction_count"),
                    func.avg(AgentAction.reward).label("avg_interaction_reward"),
                )
                .group_by(AgentAction.action_target_id)
                .all()
            )

            # Format chronological action list
            action_list = [
                {
                    "step_number": action.step_number,
                    "action_type": action.action_type,
                    "action_target_id": action.action_target_id,
                    "position_before": action.position_before,
                    "position_after": action.position_after,
                    "resources_before": action.resources_before,
                    "resources_after": action.resources_after,
                    "reward": action.reward,
                    "details": json.loads(action.details) if action.details else None,
                }
                for action in actions
            ]

            # Calculate reward statistics
            rewards = [action.reward for action in actions if action.reward is not None]
            reward_stats = {
                "total_reward": sum(rewards),
                "average_reward": sum(rewards) / len(rewards) if rewards else 0,
                "max_reward": max(rewards) if rewards else 0,
                "min_reward": min(rewards) if rewards else 0,
                "reward_variance": (
                    sum((r - (sum(rewards) / len(rewards))) ** 2 for r in rewards)
                    / len(rewards)
                    if rewards
                    else 0
                ),
            }

            return {
                "action_history": {
                    "chronological_actions": action_list,
                    "total_actions": len(actions),
                    "unique_action_types": len(set(a.action_type for a in actions)),
                    "time_range": {
                        "first_action": (
                            action_list[0]["step_number"] if action_list else None
                        ),
                        "last_action": (
                            action_list[-1]["step_number"] if action_list else None
                        ),
                    },
                },
                "action_statistics": {
                    action_type: {
                        "count": count,
                        "frequency": count / len(actions) if actions else 0,
                        "avg_reward": float(avg_reward or 0),
                        "total_reward": float(total_reward or 0),
                        "avg_resource_change": float(avg_resource_change or 0),
                    }
                    for action_type, count, avg_reward, total_reward, avg_resource_change in action_stats
                },
                "interaction_patterns": {
                    str(target_id): {
                        "interaction_count": count,
                        "avg_reward": float(avg_reward or 0),
                    }
                    for target_id, count, avg_reward in interaction_stats
                },
                "reward_analysis": reward_stats,
                "resource_impact": {
                    "total_resource_gain": sum(
                        max(0, a.resources_after - a.resources_before)
                        for a in actions
                        if a.resources_after is not None
                        and a.resources_before is not None
                    ),
                    "total_resource_loss": abs(
                        sum(
                            min(0, a.resources_after - a.resources_before)
                            for a in actions
                            if a.resources_after is not None
                            and a.resources_before is not None
                        )
                    ),
                },
            }

        return self.db._execute_in_transaction(_query)

    def get_step_actions(self, step_number: int) -> StepActionData:
        """Get all actions performed during a specific simulation step.

        Parameters
        ----------
        step_number : int
            The simulation step number to query

        Returns
        -------
        StepActionData
            Data containing:
            - step_summary: Dict[str, int]
                - total_actions: Total number of actions
                - unique_agents: Number of unique agents
                - action_types: Number of unique action types
                - total_interactions: Total number of interactions
                - total_reward: Total reward received
            - action_statistics: Dict[str, Dict[str, float]]
                Per action type statistics including:
                - count: Number of occurrences
                - frequency: Frequency of the action type
                - avg_reward: Average reward for the action
                - total_reward: Total reward for the action
            - resource_metrics: Dict[str, float]
                - net_resource_change: Net resource change
                - average_resource_change: Average resource change
                - resource_transactions: Number of resource transactions
            - interaction_network: Dict[str, Any]
                - interactions: List[Dict[str, Any]]
                    List of interactions, each containing:
                    - source: Source agent ID
                    - target: Target agent ID
                    - action_type: Action type
                    - reward: Reward received
                - unique_interacting_agents: Number of unique interacting agents
            - performance_metrics: Dict[str, float]
                - success_rate: Success rate of actions
                - average_reward: Average reward per action
                - action_efficiency: Efficiency of actions
            - detailed_actions: List[Dict[str, Any]]
                Detailed list of all actions
        """

        def _query(session):
            # Get all actions for the step
            actions = (
                session.query(AgentAction)
                .filter(AgentAction.step_number == step_number)
                .order_by(AgentAction.agent_id)
                .all()
            )

            if not actions:
                return {}

            # Get action type statistics
            action_stats = (
                session.query(
                    AgentAction.action_type,
                    func.count().label("count"),
                    func.avg(AgentAction.reward).label("avg_reward"),
                    func.sum(AgentAction.reward).label("total_reward"),
                )
                .filter(AgentAction.step_number == step_number)
                .group_by(AgentAction.action_type)
                .all()
            )

            # Calculate resource changes
            resource_changes = (
                session.query(
                    func.sum(
                        AgentAction.resources_after - AgentAction.resources_before
                    ).label("net_change"),
                    func.avg(
                        AgentAction.resources_after - AgentAction.resources_before
                    ).label("avg_change"),
                )
                .filter(
                    AgentAction.step_number == step_number,
                    AgentAction.resources_before.isnot(None),
                    AgentAction.resources_after.isnot(None),
                )
                .first()
            )

            # Build interaction network
            interactions = [
                action for action in actions if action.action_target_id is not None
            ]

            # Format detailed action list
            action_list = [
                {
                    "agent_id": action.agent_id,
                    "action_type": action.action_type,
                    "action_target_id": action.action_target_id,
                    "position_before": action.position_before,
                    "position_after": action.position_after,
                    "resources_before": action.resources_before,
                    "resources_after": action.resources_after,
                    "reward": action.reward,
                    "details": json.loads(action.details) if action.details else None,
                }
                for action in actions
            ]

            return StepActionData(
                step_summary={
                    "total_actions": len(actions),
                    "unique_agents": len(set(a.agent_id for a in actions)),
                    "action_types": len(set(a.action_type for a in actions)),
                    "total_interactions": len(interactions),
                    "total_reward": sum(
                        a.reward for a in actions if a.reward is not None
                    ),
                },
                action_statistics={
                    action_type: {
                        "count": count,
                        "frequency": count / len(actions),
                        "avg_reward": float(avg_reward or 0),
                        "total_reward": float(total_reward or 0),
                    }
                    for action_type, count, avg_reward, total_reward in action_stats
                },
                resource_metrics={
                    "net_resource_change": float(resource_changes[0] or 0),
                    "average_resource_change": float(resource_changes[1] or 0),
                    "resource_transactions": len(
                        [a for a in actions if a.resources_before != a.resources_after]
                    ),
                },
                interaction_network={
                    "interactions": [
                        {
                            "source": action.agent_id,
                            "target": action.action_target_id,
                            "action_type": action.action_type,
                            "reward": action.reward,
                        }
                        for action in interactions
                    ],
                    "unique_interacting_agents": len(
                        set(
                            [a.agent_id for a in interactions]
                            + [a.action_target_id for a in interactions]
                        )
                    ),
                },
                performance_metrics={
                    "success_rate": len(
                        [a for a in actions if a.reward and a.reward > 0]
                    )
                    / len(actions),
                    "average_reward": sum(
                        a.reward for a in actions if a.reward is not None
                    )
                    / len(actions),
                    "action_efficiency": len(
                        [a for a in actions if a.position_before != a.position_after]
                    )
                    / len(actions),
                },
                detailed_actions=action_list,
            )

        return self.db._execute_in_transaction(_query)

    def get_agent_types(self) -> List[str]:
        """Get list of all unique agent types in the simulation.

        Returns
        -------
        List[str]
            List of unique agent type names
        """

        def _query(session):
            types = session.query(Agent.agent_type).distinct().all()
            return [t[0] for t in types]

        return self.db._execute_in_transaction(_query)

    def get_agent_behaviors(
        self, start_step: Optional[int] = None, end_step: Optional[int] = None
    ) -> AgentBehaviorMetrics:
        """Get comprehensive analysis of agent behaviors across the simulation.

        Parameters
        ----------
        start_step : int, optional
            Starting step number for analysis window
        end_step : int, optional
            Ending step number for analysis window

        Returns
        -------
        AgentBehaviorMetrics
            Data containing:
            - temporal_patterns: Dict[str, Any]
                - step_data: Dict[int, Dict[str, Dict[str, float]]]
                    Per step action type statistics including:
                    - count: Number of occurrences
                    - average_reward: Average reward for the action
                - time_range: Dict[str, int]
                    - start: Starting step number
                    - end: Ending step number
            - type_behaviors: Dict[str, Dict[str, Any]]
                Per agent type behavior statistics including:
                - actions: Dict[str, Dict[str, float]]
                    Per action type statistics including:
                    - count: Number of occurrences
                    - average_reward: Average reward for the action
                    - reward_stddev: Standard deviation of rewards
                - most_common_action: Most common action type
                - most_rewarding_action: Most rewarding action type
            - interaction_patterns: Dict[str, Dict[str, float]]
                Per action type interaction statistics including:
                - interaction_count: Number of interactions
                - average_reward: Average reward for the interactions
            - resource_behaviors: Dict[str, Dict[str, float]]
                Per action type resource impact statistics including:
                - average_resource_change: Average resource change
                - action_count: Number of occurrences
            - behavior_summary: Dict[str, Any]
                - total_actions: Total number of actions
                - unique_action_types: Number of unique action types
                - most_common_behaviors: List of most common action types
        """

        def _query(session):
            # Build base query with time window
            base_query = session.query(AgentAction)
            if start_step is not None:
                base_query = base_query.filter(AgentAction.step_number >= start_step)
            if end_step is not None:
                base_query = base_query.filter(AgentAction.step_number <= end_step)

            # Get temporal behavior patterns
            temporal_patterns = (
                base_query.with_entities(
                    AgentAction.step_number,
                    AgentAction.action_type,
                    func.count().label("action_count"),
                    func.avg(AgentAction.reward).label("avg_reward"),
                )
                .group_by(AgentAction.step_number, AgentAction.action_type)
                .order_by(AgentAction.step_number)
                .all()
            )

            # Get behavior patterns by agent type
            type_patterns = (
                base_query.join(Agent)
                .with_entities(
                    Agent.agent_type,
                    AgentAction.action_type,
                    func.count().label("action_count"),
                    func.avg(AgentAction.reward).label("avg_reward"),
                    func.stddev(AgentAction.reward).label("reward_stddev"),
                )
                .group_by(Agent.agent_type, AgentAction.action_type)
                .all()
            )

            # Get interaction patterns
            interaction_patterns = (
                base_query.filter(AgentAction.action_target_id.isnot(None))
                .with_entities(
                    AgentAction.action_type,
                    func.count().label("interaction_count"),
                    func.avg(AgentAction.reward).label("avg_interaction_reward"),
                )
                .group_by(AgentAction.action_type)
                .all()
            )

            # Get resource-related behaviors
            resource_behaviors = (
                base_query.filter(
                    AgentAction.resources_before.isnot(None),
                    AgentAction.resources_after.isnot(None),
                )
                .with_entities(
                    AgentAction.action_type,
                    func.avg(
                        AgentAction.resources_after - AgentAction.resources_before
                    ).label("avg_resource_change"),
                    func.count().label("resource_action_count"),
                )
                .group_by(AgentAction.action_type)
                .all()
            )

            # Format temporal patterns
            temporal_data = {}
            for step, action_type, count, avg_reward in temporal_patterns:
                if step not in temporal_data:
                    temporal_data[step] = {}
                temporal_data[step][action_type] = {
                    "count": int(count),
                    "average_reward": float(avg_reward or 0),
                }

            # Format type-specific patterns
            type_data = {}
            for agent_type, action_type, count, avg_reward, stddev in type_patterns:
                if agent_type not in type_data:
                    type_data[agent_type] = {}
                type_data[agent_type][action_type] = {
                    "count": int(count),
                    "average_reward": float(avg_reward or 0),
                    "reward_stddev": float(stddev or 0),
                }

            return {
                "temporal_patterns": {
                    "step_data": temporal_data,
                    "time_range": {
                        "start": min(temporal_data.keys()) if temporal_data else None,
                        "end": max(temporal_data.keys()) if temporal_data else None,
                    },
                },
                "type_behaviors": {
                    agent_type: {
                        "actions": actions,
                        "most_common_action": (
                            max(actions.items(), key=lambda x: x[1]["count"])[0]
                            if actions
                            else None
                        ),
                        "most_rewarding_action": (
                            max(actions.items(), key=lambda x: x[1]["average_reward"])[
                                0
                            ]
                            if actions
                            else None
                        ),
                    }
                    for agent_type, actions in type_data.items()
                },
                "interaction_patterns": {
                    action_type: {
                        "interaction_count": int(count),
                        "average_reward": float(avg_reward or 0),
                    }
                    for action_type, count, avg_reward in interaction_patterns
                },
                "resource_behaviors": {
                    action_type: {
                        "average_resource_change": float(avg_change or 0),
                        "action_count": int(count),
                    }
                    for action_type, avg_change, count in resource_behaviors
                },
                "behavior_summary": {
                    "total_actions": sum(pattern[2] for pattern in type_patterns),
                    "unique_action_types": len(
                        set(pattern[1] for pattern in type_patterns)
                    ),
                    "most_common_behaviors": sorted(
                        [
                            (
                                action_type,
                                sum(1 for p in type_patterns if p[1] == action_type),
                            )
                            for action_type in set(p[1] for p in type_patterns)
                        ],
                        key=lambda x: x[1],
                        reverse=True,
                    )[:5],
                },
            }

        return self.db._execute_in_transaction(_query)

    def action_rewards(
        self,
        session,
        action_type: str,
        agent_id: Optional[int] = None,
    ) -> List[float]:
        """Get all rewards for a specific action type.

        Parameters
        ----------
        session : Session
            SQLAlchemy database session
        action_type : str
            Type of action to get rewards for
        agent_id : Optional[int]
            Specific agent ID to filter by

        Returns
        -------
        List[float]
            List of reward values for the action type
        """
        rewards = session.query(AgentAction.reward).filter(
            AgentAction.action_type == action_type
        )
        if agent_id is not None:
            rewards = rewards.filter(AgentAction.agent_id == agent_id)
        return [r[0] for r in rewards.all() if r[0] is not None]

    def rewards_by_type(self, session, stats, agent_id=None):
        """Get rewards for each action type.

        Parameters
        ----------
        session : Session
            SQLAlchemy database session
        stats : List[Tuple]
            List of action type statistics tuples
        agent_id : Optional[int]
            Specific agent ID to filter by

        Returns
        -------
        Dict[str, List[float]]
            Dictionary mapping action types to lists of reward values
        """
        rewards_by_type = {}
        for action_type in set(s[0] for s in stats):
            rewards_by_type[action_type] = self.action_rewards(
                session, action_type, agent_id
            )
        return rewards_by_type

    def action_metrics(
        self,
        session,
        agent_id: Optional[int] = None,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
    ) -> List[ActionMetrics]:
        """Get basic metrics for agent actions with optional filters.

        Parameters
        ----------
        session : Session
            SQLAlchemy database session
        agent_id : Optional[int]
            Specific agent ID to filter by
        start_step : Optional[int]
            Starting step number for analysis window
        end_step : Optional[int]
            Ending step number for analysis window

        Returns
        -------
        List[ActionMetrics]
            List of action metrics containing counts and reward statistics
        """
        base_query = session.query(
            AgentAction.action_type,
            func.count().label("decision_count"),
            func.avg(AgentAction.reward).label("avg_reward"),
            func.min(AgentAction.reward).label("min_reward"),
            func.max(AgentAction.reward).label("max_reward"),
        )

        if agent_id is not None:
            base_query = base_query.filter(AgentAction.agent_id == agent_id)
        if start_step is not None:
            base_query = base_query.filter(AgentAction.step_number >= start_step)
        if end_step is not None:
            base_query = base_query.filter(AgentAction.step_number <= end_step)

        # Get basic stats grouped by action type and convert to ActionMetrics objects
        results = base_query.group_by(AgentAction.action_type).all()
        return [
            ActionMetrics(
                action_type=result[0],
                decision_count=int(result[1]),
                avg_reward=float(result[2] or 0),
                min_reward=float(result[3] or 0),
                max_reward=float(result[4] or 0),
            )
            for result in results
        ]

    def _format_interaction_patterns(
        self, interaction_patterns
    ) -> Dict[str, Dict[str, float]]:
        """Format raw interaction pattern data into analysis dictionary.

        Parameters
        ----------
        interaction_patterns : List[Tuple]
            List of tuples containing (action_type, is_interaction, count, avg_reward)

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping action types to their interaction statistics:
            - interaction_rate: Rate of interactive vs solo actions
            - solo_performance: Average reward for solo actions
            - interaction_performance: Average reward for interactive actions
        """
        interaction_analysis = {}

        for action_type, is_interaction, count, avg_reward in interaction_patterns:
            if action_type not in interaction_analysis:
                interaction_analysis[action_type] = {
                    "interaction_rate": 0,
                    "solo_performance": 0,
                    "interaction_performance": 0,
                }

            total = sum(i[2] for i in interaction_patterns if i[0] == action_type)
            if is_interaction:
                interaction_analysis[action_type]["interaction_rate"] = (
                    count / total if total > 0 else 0
                )
                interaction_analysis[action_type]["interaction_performance"] = float(
                    avg_reward or 0
                )
            else:
                interaction_analysis[action_type]["solo_performance"] = float(
                    avg_reward or 0
                )

        return interaction_analysis

    def _format_temporal_patterns(self, time_patterns) -> Dict[str, Dict[str, List]]:
        """Format raw time-based pattern data into analysis dictionary.

        Parameters
        ----------
        time_patterns : List[Tuple]
            List of tuples containing (action_type, count, avg_reward)

        Returns
        -------
        Dict[str, Dict[str, List]]
            Dictionary mapping action types to their temporal statistics:
            - time_distribution: List of action counts over time
            - reward_progression: List of average rewards over time
        """
        temporal_patterns = {}
        for action_type, count, avg_reward in time_patterns:
            if action_type not in temporal_patterns:
                temporal_patterns[action_type] = {
                    "time_distribution": [],
                    "reward_progression": [],
                }
            temporal_patterns[action_type]["time_distribution"].append(int(count))
            temporal_patterns[action_type]["reward_progression"].append(
                float(avg_reward or 0)
            )
        return temporal_patterns

    def _format_resource_patterns(
        self, resource_patterns
    ) -> Dict[str, Dict[str, float]]:
        """Format raw resource pattern data into analysis dictionary.

        Parameters
        ----------
        resource_patterns : List[Tuple]
            List of tuples containing (action_type, avg_before, avg_change, count)

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping action types to their resource statistics:
            - avg_resources_before: Average resources before action
            - avg_resource_change: Average change in resources
            - resource_efficiency: Resource change per action
        """
        return {
            action_type: {
                "avg_resources_before": float(avg_before or 0),
                "avg_resource_change": float(avg_change or 0),
                "resource_efficiency": (
                    float(avg_change or 0) / count if count > 0 else 0
                ),
            }
            for action_type, avg_before, avg_change, count in resource_patterns
        }

    def _format_sequence_analysis(
        self, sequential_patterns: List[Tuple], decision_patterns: Dict[str, Any]
    ) -> Dict[str, Dict[str, Union[int, float]]]:
        """Format raw sequential pattern data into analysis dictionary.

        Parameters
        ----------
        sequential_patterns : List[Tuple]
            List of tuples containing (action, next_action, count)
        decision_patterns : Dict[str, Any]
            Dictionary containing decision pattern statistics

        Returns
        -------
        Dict[str, Dict[str, Union[int, float]]]
            Dictionary mapping action sequences to their statistics:
            - count: Number of occurrences of the sequence
            - probability: Probability of next_action following action
        """
        return {
            f"{action}->{next_action}": {
                "count": count,
                "probability": (
                    count / decision_patterns[action]["count"]
                    if action in decision_patterns
                    else 0
                ),
            }
            for action, next_action, count in sequential_patterns
            if action is not None and next_action is not None
        }

    def _format_decision_patterns(
        self, metrics: List[ActionMetrics], rewards_by_type: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, Any]]:
        """Format basic decision patterns from metrics and rewards.

        Parameters
        ----------
        metrics : List[ActionMetrics]
            List of action metrics containing counts and basic stats
        rewards_by_type : Dict[str, List[float]]
            Dictionary mapping action types to lists of reward values

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary containing formatted decision patterns with:
            - count: Number of decisions
            - frequency: Relative frequency of decisions
            - reward_stats: Dictionary of reward statistics
        """
        import numpy as np

        decision_patterns = {}
        total_decisions = sum(m.decision_count for m in metrics)

        for metric in metrics:
            rewards = rewards_by_type.get(metric.action_type, [])
            reward_stddev = float(np.std(rewards)) if len(rewards) > 1 else 0.0

            decision_patterns[metric.action_type] = {
                "count": metric.decision_count,
                "frequency": (
                    float(metric.decision_count) / total_decisions
                    if total_decisions > 0
                    else 0
                ),
                "reward_stats": {
                    "average": metric.avg_reward,
                    "stddev": reward_stddev,
                    "min": metric.min_reward,
                    "max": metric.max_reward,
                },
            }

        return decision_patterns

    def _get_interaction_patterns(self, base_query) -> List[Tuple]:
        """Get interaction patterns from agent actions.

        Parameters
        ----------
        base_query : Query
            Base SQLAlchemy query to build upon

        Returns
        -------
        List[Tuple]
            List of tuples containing:
            - action_type: str
            - is_interaction: bool
            - count: int
            - avg_reward: float
        """
        return (
            base_query.with_entities(
                AgentAction.action_type,
                AgentAction.action_target_id.isnot(None).label("is_interaction"),
                func.count().label("count"),
                func.avg(AgentAction.reward).label("avg_reward"),
            )
            .group_by(
                AgentAction.action_type,
                AgentAction.action_target_id.isnot(None),
            )
            .all()
        )

    def _get_time_patterns(self, base_query) -> List[Tuple]:
        """Get time-based patterns from agent actions.

        Parameters
        ----------
        base_query : Query
            Base SQLAlchemy query to build upon

        Returns
        -------
        List[Tuple]
            List of tuples containing:
            - action_type: str
            - count: int
            - avg_reward: float
            Grouped by action type and time periods (100 steps)
        """
        return (
            base_query.with_entities(
                AgentAction.action_type,
                func.count().label("count"),
                func.avg(AgentAction.reward).label("avg_reward"),
            )
            .group_by(
                AgentAction.action_type,
                func.round(AgentAction.step_number / 100),  # Group by time periods
            )
            .all()
        )

    def _get_resource_patterns(self, base_query) -> List[Tuple]:
        """Get resource-driven patterns from agent actions.

        Parameters
        ----------
        base_query : Query
            Base SQLAlchemy query to build upon

        Returns
        -------
        List[Tuple]
            List of tuples containing:
            - action_type: str
            - avg_resources_before: float
            - avg_resource_change: float
            - count: int
        """
        return (
            base_query.with_entities(
                AgentAction.action_type,
                func.avg(AgentAction.resources_before).label("avg_resources_before"),
                func.avg(
                    AgentAction.resources_after - AgentAction.resources_before
                ).label("avg_resource_change"),
                func.count().label("count"),
            )
            .filter(
                AgentAction.resources_before.isnot(None),
                AgentAction.resources_after.isnot(None),
            )
            .group_by(AgentAction.action_type)
            .all()
        )

    def _get_sequential_patterns(self, base_query) -> List[Tuple]:
        """Get sequential patterns of agent actions.

        Parameters
        ----------
        base_query : Query
            Base query to build upon

        Returns
        -------
        List[Tuple]
            Representing sequences of actions and their frequencies
        """
        # First, get the actions with their next actions using a subquery
        subq = (
            base_query.add_columns(
                AgentAction.step_number,
                AgentAction.action_type,
            )
            .order_by(AgentAction.step_number)
            .subquery()
        )

        # Create an alias for self-join
        next_action = aliased(subq)

        # Join to get action pairs
        return (
            base_query.session.query(
                subq.c.action_type,
                next_action.c.action_type.label("next_action"),
                func.count().label("sequence_count"),
            )
            .join(
                next_action,
                and_(
                    subq.c.step_number < next_action.c.step_number,
                    not_(
                        exists().where(
                            and_(
                                AgentAction.step_number > subq.c.step_number,
                                AgentAction.step_number < next_action.c.step_number,
                                AgentAction.agent_id
                                == base_query.whereclause.right.value,
                            )
                        )
                    ),
                ),
            )
            .group_by(subq.c.action_type, next_action.c.action_type)
            .all()
        )

    def _get_all_patterns(
        self, base_query
    ) -> Tuple[SequencePattern, ResourceImpact, TimePattern, InteractionStats]:
        """Get all pattern types from agent actions.

        Parameters
        ----------
        base_query : Query
            Base SQLAlchemy query to build upon

        Returns
        -------
        Tuple[SequencePattern, ResourceImpact, TimePattern, InteractionStats]
            Tuple containing:
            - sequential_patterns: Sequences of actions and their frequencies
            - resource_patterns: Resource-driven patterns and impacts
            - time_patterns: Time-based patterns and trends
            - interaction_patterns: Interaction-based patterns and statistics
        """
        return (
            self._get_sequential_patterns(base_query),
            self._get_resource_patterns(base_query),
            self._get_time_patterns(base_query),
            self._get_interaction_patterns(base_query),
        )

    def agent_decision_patterns(
        self,
        agent_id: Optional[int] = None,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
    ) -> DecisionPatterns:
        """Get comprehensive analysis of agent decision-making patterns.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.
        start_step : Optional[int]
            Starting step for analysis window
        end_step : Optional[int]
            Ending step for analysis window

        Returns
        -------
        DecisionPatterns
            Comprehensive metrics about agent decision patterns including:
            - Basic decision statistics
            - Sequential patterns
            - Context-based decisions
            - Resource-driven patterns
            - Interaction patterns
            - Learning progression
        """

        def _query(session):
            # Base query for additional analysis
            base_query = session.query(AgentAction)
            if agent_id is not None:
                base_query = base_query.filter(AgentAction.agent_id == agent_id)
            if start_step is not None:
                base_query = base_query.filter(AgentAction.step_number >= start_step)
            if end_step is not None:
                base_query = base_query.filter(AgentAction.step_number <= end_step)

            # Get basic action metrics
            action_metrics: List[ActionMetrics] = self.action_metrics(
                session, agent_id, start_step, end_step
            )

            # Get rewards by action type
            rewards_by_type: Dict[str, List[float]] = self.rewards_by_type(
                session,
                [(m.action_type, m.decision_count) for m in action_metrics],
                agent_id,
            )

            # Get all patterns
            (
                sequential_patterns,
                resource_patterns,
                time_patterns,
                interaction_patterns,
            ) = self._get_all_patterns(base_query)

            # Format decision patterns
            decision_patterns: DecisionPatternStats = self._format_decision_patterns(
                action_metrics, rewards_by_type
            )

            # Format sequential patterns
            sequence_analysis: SequencePattern = self._format_sequence_analysis(
                sequential_patterns, decision_patterns
            )

            # Format resource patterns
            resource_impact: ResourceImpact = self._format_resource_patterns(
                resource_patterns
            )

            # Format time-based patterns
            temporal_patterns: TimePattern = self._format_temporal_patterns(
                time_patterns
            )

            # Format interaction patterns
            interaction_analysis: InteractionStats = self._format_interaction_patterns(
                interaction_patterns
            )

            # Create decision summary
            decision_summary: DecisionSummary = self._create_decision_summary(
                decision_patterns
            )

            # Return with the created decision summary
            return {
                "decision_patterns": decision_patterns,
                "sequence_analysis": sequence_analysis,
                "resource_impact": resource_impact,
                "temporal_patterns": temporal_patterns,
                "interaction_analysis": interaction_analysis,
                "decision_summary": decision_summary,
            }

        return self.db._execute_in_transaction(_query)

    def _create_decision_summary(
        self, decision_patterns: Dict[str, Any]
    ) -> DecisionSummary:
        """Create a summary of decision patterns.

        Parameters
        ----------
        decision_patterns: Dict[str, Any]
            Dictionary containing decision pattern statistics

        Returns
        -------
        DecisionSummary
            Summary containing:
            - total_decisions: Total number of decisions made
            - unique_actions: Number of unique action types
            - most_frequent: Most frequently used action type
            - most_rewarding: Action type with highest average reward
            - action_diversity: Shannon entropy of action distribution
        """
        total_decisions = sum(p["count"] for p in decision_patterns.values())

        return {
            "total_decisions": total_decisions,
            "unique_actions": len(decision_patterns),
            "most_frequent": (
                max(
                    decision_patterns.items(),
                    key=lambda x: x[1]["frequency"],
                )[0]
                if decision_patterns
                else None
            ),
            "most_rewarding": (
                max(
                    decision_patterns.items(),
                    key=lambda x: x[1]["reward_stats"]["average"],
                )[0]
                if decision_patterns
                else None
            ),
            "action_diversity": -sum(
                p["frequency"] * math.log(p["frequency"])
                for p in decision_patterns.values()
                if p["frequency"] > 0
            ),  # Shannon entropy
        }
