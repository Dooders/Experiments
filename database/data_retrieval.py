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
from typing import Any, Dict, List

import pandas as pd
from sqlalchemy import case, func

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


class DataRetriever:
    """Handles data retrieval operations for the simulation database.

    This class provides methods to query and analyze simulation data, including agent states,
    resource distributions, learning statistics, and behavioral patterns.

    Attributes
    ----------
    db : SimulationDatabase
        Database connection instance used for executing queries
    """

    def __init__(self, database):
        """Initialize data retriever with database connection.

        Parameters
        ----------
        database : SimulationDatabase
            Database instance to use for queries
        """
        self.db = database

    def get_simulation_data(self, step_number: int) -> Dict[str, Any]:
        """Retrieve complete simulation state for a specific step.

        Parameters
        ----------
        step_number : int
            The simulation step number to retrieve data for

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - agent_states: List[Tuple] of agent state records
            - resource_states: List[Tuple] of resource state records
            - metrics: Dict[str, Any] of simulation metrics for the step
        """

        def _query(session):
            # Get agent states
            agent_states = (
                session.query(
                    AgentState.agent_id,
                    Agent.agent_type,
                    AgentState.position_x,
                    AgentState.position_y,
                    AgentState.resource_level,
                    AgentState.current_health,
                    AgentState.is_defending,
                )
                .join(Agent)
                .filter(AgentState.step_number == step_number)
                .all()
            )

            # Get resource states
            resource_states = (
                session.query(
                    ResourceState.resource_id,
                    ResourceState.amount,
                    ResourceState.position_x,
                    ResourceState.position_y,
                )
                .filter(ResourceState.step_number == step_number)
                .all()
            )

            # Get metrics
            metrics = (
                session.query(SimulationStep)
                .filter(SimulationStep.step_number == step_number)
                .first()
            )

            return {
                "agent_states": agent_states,
                "resource_states": resource_states,
                "metrics": metrics.as_dict() if metrics else {},
            }

        return self.db._execute_in_transaction(_query)

    def get_agent_lifespan_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive statistics about agent lifespans.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - average_lifespan: float, mean lifespan across all agents
            - lifespan_by_type: Dict[str, float], mean lifespan per agent type
            - lifespan_by_generation: Dict[int, float], mean lifespan per generation
            - survival_rates: Dict[int, float], survival rate per generation
        """

        def _query(session):
            # Calculate lifespans
            lifespans = (
                session.query(
                    Agent.agent_type,
                    Agent.generation,
                    (Agent.death_time - Agent.birth_time).label("lifespan"),
                )
                .filter(Agent.death_time.isnot(None))
                .all()
            )

            # Calculate survival rates
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

            # Process results
            lifespan_data = pd.DataFrame(
                lifespans, columns=["agent_type", "generation", "lifespan"]
            )
            survival_data = pd.DataFrame(
                survival_rates, columns=["generation", "survival_rate"]
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
                "survival_rates": survival_data.set_index("generation")[
                    "survival_rate"
                ].to_dict(),
            }

        return self.db._execute_in_transaction(_query)

    def get_population_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive population statistics using SQLAlchemy.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
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
            - resource_distribution: Dict[str, List[float]]
                - steps: List of step numbers
                - total_resources: Total resources per step
                - average_per_agent: Average resources per agent per step
            - efficiency_metrics: Dict[str, Union[float, List[float]]]
                - average_efficiency: Mean resource utilization efficiency
                - efficiency_trend: Resource efficiency over time
                - distribution_entropy: Resource distribution entropy over time
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

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about agent learning and adaptation.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - learning_progress: Dict[str, Dict[int, float]]
                - average_reward: Mean reward per step
                - average_loss: Mean loss per step
            - module_performance: Dict[str, Dict[str, float]]
                Per module statistics including:
                - avg_reward: Average reward for the module
                - avg_loss: Average loss for the module
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

    def get_historical_data(self) -> Dict[str, Any]:
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

    def get_population_momentum(self) -> float:
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

    def get_advanced_statistics(self) -> Dict[str, Any]:
        """Calculate advanced simulation statistics using optimized queries.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - Population metrics (peak, average, diversity)
            - Interaction statistics (conflict/cooperation ratios)
            - Resource efficiency metrics
            - Agent type distribution
            - Survival and adaptation metrics
        """

        def _query(session):
            # Get basic population stats in one query
            pop_stats = session.query(
                func.avg(SimulationStep.total_agents).label("avg_pop"),
                func.max(SimulationStep.total_agents).label("peak_pop"),
                func.min(SimulationStep.total_agents).label("min_pop"),
                func.count(SimulationStep.step_number).label("total_steps"),
                func.avg(SimulationStep.average_agent_health).label("avg_health"),
            ).first()

            # Get agent type ratios
            type_stats = session.query(
                func.avg(SimulationStep.system_agents).label("avg_system"),
                func.avg(SimulationStep.independent_agents).label("avg_independent"),
                func.avg(SimulationStep.control_agents).label("avg_control"),
                func.avg(SimulationStep.total_agents).label("avg_total"),
            ).first()

            # Get interaction stats with detailed categorization
            interaction_stats = session.query(
                func.count(AgentAction.action_id).label("total_actions"),
                func.sum(
                    case(
                        [(AgentAction.action_type.in_(["attack", "defend"]), 1)],
                        else_=0,
                    )
                ).label("conflicts"),
                func.sum(
                    case(
                        [(AgentAction.action_type.in_(["share", "help"]), 1)],
                        else_=0,
                    )
                ).label("cooperation"),
                func.sum(
                    case(
                        [(AgentAction.action_type == "reproduce", 1)],
                        else_=0,
                    )
                ).label("reproductions"),
            ).first()

            # Get resource efficiency metrics
            resource_stats = session.query(
                func.avg(SimulationStep.resource_efficiency).label("avg_efficiency"),
                func.avg(SimulationStep.total_resources).label("avg_resources"),
                func.avg(SimulationStep.average_agent_resources).label(
                    "avg_agent_resources"
                ),
            ).first()

            if not all([pop_stats, type_stats, interaction_stats, resource_stats]):
                return {}

            # Calculate diversity index using Shannon entropy
            total = float(type_stats[3] or 1)  # Avoid division by zero
            ratios = [float(count or 0) / total for count in type_stats[:3]]

            import math

            diversity = sum(
                -ratio * math.log(ratio) if ratio > 0 else 0 for ratio in ratios
            )

            # Calculate interaction rates and ratios
            total_interactions = float(interaction_stats[0] or 1)
            conflict_rate = float(interaction_stats[1] or 0) / total_interactions
            cooperation_rate = float(interaction_stats[2] or 0) / total_interactions
            reproduction_rate = float(interaction_stats[3] or 0) / total_interactions

            return {
                "population_metrics": {
                    "peak_population": int(pop_stats[1] or 0),
                    "average_population": float(pop_stats[0] or 0),
                    "minimum_population": int(pop_stats[2] or 0),
                    "total_steps": int(pop_stats[3] or 0),
                    "average_health": float(pop_stats[4] or 0),
                    "population_diversity": diversity,
                },
                "interaction_metrics": {
                    "total_actions": int(interaction_stats[0] or 0),
                    "conflict_rate": conflict_rate,
                    "cooperation_rate": cooperation_rate,
                    "reproduction_rate": reproduction_rate,
                    "conflict_cooperation_ratio": (
                        conflict_rate / cooperation_rate
                        if cooperation_rate > 0
                        else float("inf")
                    ),
                },
                "resource_metrics": {
                    "average_efficiency": float(resource_stats[0] or 0),
                    "average_total_resources": float(resource_stats[1] or 0),
                    "average_agent_resources": float(resource_stats[2] or 0),
                    "resource_utilization": (
                        float(resource_stats[2] or 0) / float(resource_stats[1] or 1)
                    ),
                },
                "agent_distribution": {
                    "system_ratio": ratios[0],
                    "independent_ratio": ratios[1],
                    "control_ratio": ratios[2],
                    "type_entropy": diversity,
                },
                "survival_metrics": {
                    "population_stability": (
                        float(pop_stats[2] or 0) / float(pop_stats[1] or 1)
                    ),
                    "health_maintenance": float(pop_stats[4] or 0) / 100.0,
                    "interaction_rate": (
                        float(interaction_stats[0] or 0)
                        / float(pop_stats[3] or 1)
                        / float(pop_stats[0] or 1)
                    ),
                },
            }

        return self.db._execute_in_transaction(_query)

    def get_agent_data(self, agent_id: int) -> Dict[str, Any]:
        """Get comprehensive data for a specific agent.

        Parameters
        ----------
        agent_id : int
            The unique identifier of the agent

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
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
        self, agent_id: int, start_step: int = None, end_step: int = None
    ) -> Dict[str, Any]:
        """Get detailed action history for a specific agent with optional time range.

        Parameters
        ----------
        agent_id : int
            The unique identifier of the agent
        start_step : int, optional
            Starting step number for filtering actions
        end_step : int, optional
            Ending step number for filtering actions

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - action_history: Dict[str, Any]
                - chronological_actions: List[Dict[str, Any]]
                    List of actions in chronological order, each containing:
                    - step_number: Step number
                    - action_type: Action type
                    - action_target_id: Target agent ID (None if no target)
                    - position_before: (x, y) position before the action
                    - position_after: (x, y) position after the action
                    - resources_before: Resource level before the action
                    - resources_after: Resource level after the action
                    - reward: Reward received
                    - details: Additional action details (JSON)
                - total_actions: Total number of actions
                - unique_action_types: Number of unique action types
                - time_range: Dict[str, int]
                    - first_action: Step number of the first action
                    - last_action: Step number of the last action
            - action_statistics: Dict[str, Dict[str, float]]
                Per action type statistics including:
                - count: Number of occurrences
                - frequency: Frequency of the action type
                - avg_reward: Average reward for the action
                - total_reward: Total reward for the action
                - avg_resource_change: Average resource change for the action
            - interaction_patterns: Dict[str, Dict[str, float]]
                Per interaction partner statistics including:
                - interaction_count: Number of interactions
                - avg_reward: Average reward for the interactions
            - reward_analysis: Dict[str, float]
                - total_reward: Total reward received
                - average_reward: Average reward per action
                - max_reward: Maximum reward received
                - min_reward: Minimum reward received
                - reward_variance: Variance of rewards
            - resource_impact: Dict[str, float]
                - total_resource_gain: Total resource gain
                - total_resource_loss: Total resource loss
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

    def get_step_actions(self, step_number: int) -> Dict[str, Any]:
        """Get all actions performed during a specific simulation step.

        Parameters
        ----------
        step_number : int
            The simulation step number to query

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
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
                List of actions in chronological order, each containing:
                - agent_id: Agent ID
                - action_type: Action type
                - action_target_id: Target agent ID (None if no target)
                - position_before: (x, y) position before the action
                - position_after: (x, y) position after the action
                - resources_before: Resource level before the action
                - resources_after: Resource level after the action
                - reward: Reward received
                - details: Additional action details (JSON)
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

            return {
                "step_summary": {
                    "total_actions": len(actions),
                    "unique_agents": len(set(a.agent_id for a in actions)),
                    "action_types": len(set(a.action_type for a in actions)),
                    "total_interactions": len(interactions),
                    "total_reward": sum(
                        a.reward for a in actions if a.reward is not None
                    ),
                },
                "action_statistics": {
                    action_type: {
                        "count": count,
                        "frequency": count / len(actions),
                        "avg_reward": float(avg_reward or 0),
                        "total_reward": float(total_reward or 0),
                    }
                    for action_type, count, avg_reward, total_reward in action_stats
                },
                "resource_metrics": {
                    "net_resource_change": float(resource_changes[0] or 0),
                    "average_resource_change": float(resource_changes[1] or 0),
                    "resource_transactions": len(
                        [a for a in actions if a.resources_before != a.resources_after]
                    ),
                },
                "interaction_network": {
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
                "performance_metrics": {
                    "success_rate": (
                        len(
                            [
                                a
                                for a in actions
                                if a.reward is not None and a.reward > 0
                            ]
                        )
                        / len(actions)
                        if actions
                        else 0
                    ),
                    "average_reward": (
                        sum(a.reward for a in actions if a.reward is not None)
                        / len(actions)
                        if actions
                        else 0
                    ),
                    "action_efficiency": (
                        sum(1 for a in actions if a.position_before != a.position_after)
                        / len(actions)
                        if actions
                        else 0
                    ),
                },
                "detailed_actions": action_list,
            }

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
        self, start_step: int = None, end_step: int = None
    ) -> Dict[str, Any]:
        """Get comprehensive analysis of agent behaviors across the simulation.

        Parameters
        ----------
        start_step : int, optional
            Starting step number for analysis window
        end_step : int, optional
            Ending step number for analysis window

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
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

    def get_agent_decisions(
        self, agent_id: int = None, start_step: int = None, end_step: int = None
    ) -> Dict[str, Any]:
        """Get comprehensive analysis of agent decision-making patterns."""
        def _query(session):
            # Build base query
            base_query = session.query(
                AgentAction.action_type,
                func.count().label('decision_count'),
                func.avg(AgentAction.reward).label('avg_reward'),
                func.min(AgentAction.reward).label('min_reward'),
                func.max(AgentAction.reward).label('max_reward'),
            )
            
            if agent_id is not None:
                base_query = base_query.filter(AgentAction.agent_id == agent_id)
            if start_step is not None:
                base_query = base_query.filter(AgentAction.step_number >= start_step)
            if end_step is not None:
                base_query = base_query.filter(AgentAction.step_number <= end_step)
            
            # Get basic stats grouped by action type
            stats = base_query.group_by(AgentAction.action_type).all()
            
            # Get all rewards for each action type to calculate stddev in Python
            rewards_by_type = {}
            for action_type in set(s[0] for s in stats):
                rewards = session.query(AgentAction.reward).filter(
                    AgentAction.action_type == action_type
                )
                if agent_id is not None:
                    rewards = rewards.filter(AgentAction.agent_id == agent_id)
                rewards = [r[0] for r in rewards.all() if r[0] is not None]
                rewards_by_type[action_type] = rewards

            # Format results with Python-calculated stddev
            decision_patterns = {}
            for stat in stats:
                action_type = stat[0]
                rewards = rewards_by_type.get(action_type, [])
                
                # Calculate stddev in Python
                if rewards:
                    import numpy as np
                    reward_stddev = float(np.std(rewards)) if len(rewards) > 1 else 0.0
                else:
                    reward_stddev = 0.0
                    
                decision_patterns[action_type] = {
                    "count": int(stat[1]),
                    "frequency": float(stat[1]) / sum(s[1] for s in stats),
                    "reward_stats": {
                        "average": float(stat[2] or 0),
                        "stddev": reward_stddev,
                        "min": float(stat[3] or 0),
                        "max": float(stat[4] or 0),
                    }
                }

            # Rest of the method remains the same...
            return {
                "decision_patterns": decision_patterns,
                # ... other return values ...
            }

        return self.db._execute_in_transaction(_query)
