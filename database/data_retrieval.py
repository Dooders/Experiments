"""Data retrieval module for simulation database.

This module handles all data querying operations including retrieving simulation states,
agent data, and calculating various statistics. It provides a clean interface for
accessing stored simulation data.

Features:
- Efficient query optimization
- Data aggregation and statistics
- Historical data analysis
- Performance metrics calculation
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sqlalchemy import case, func, text
from sqlalchemy.exc import SQLAlchemyError

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
    """Handles data retrieval operations for the simulation database."""

    def __init__(self, database):
        """Initialize data retriever with database connection.
        
        Parameters
        ----------
        database : SimulationDatabase
            Database instance to use for queries
        """
        self.db = database

    def get_simulation_data(self, step_number: int) -> Dict:
        """Get complete simulation state for a specific step.

        Parameters
        ----------
        step_number : int
            Step number to retrieve data for

        Returns
        -------
        Dict
            Dictionary containing:
            - agent_states: List of agent states
            - resource_states: List of resource states
            - metrics: Dictionary of simulation metrics
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

    def get_agent_lifespan_statistics(self) -> Dict:
        """Calculate statistics about agent lifespans.

        Returns
        -------
        Dict
            Dictionary containing:
            - average_lifespan: Average agent lifespan
            - lifespan_distribution: Distribution of lifespans
            - survival_rates: Survival rates by generation
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
                    func.count(case([(Agent.death_time.is_(None), 1)])) * 100.0 / func.count(),
                )
                .group_by(Agent.generation)
                .all()
            )

            # Process results
            lifespan_data = pd.DataFrame(lifespans, columns=["agent_type", "generation", "lifespan"])
            survival_data = pd.DataFrame(survival_rates, columns=["generation", "survival_rate"])

            return {
                "average_lifespan": lifespan_data["lifespan"].mean(),
                "lifespan_by_type": lifespan_data.groupby("agent_type")["lifespan"].mean().to_dict(),
                "lifespan_by_generation": lifespan_data.groupby("generation")["lifespan"].mean().to_dict(),
                "survival_rates": survival_data.set_index("generation")["survival_rate"].to_dict(),
            }

        return self.db._execute_in_transaction(_query)

    def get_population_statistics(self) -> Dict:
        """Get detailed population statistics over time.

        Returns
        -------
        Dict
            Dictionary containing:
            - population_over_time: Population counts by type
            - birth_death_rates: Birth and death rates
            - generation_distribution: Distribution across generations
        """
        def _query(session):
            # Get population data over time
            population_data = (
                session.query(
                    SimulationStep.step_number,
                    SimulationStep.total_agents,
                    SimulationStep.system_agents,
                    SimulationStep.independent_agents,
                    SimulationStep.births,
                    SimulationStep.deaths,
                    SimulationStep.current_max_generation,
                )
                .order_by(SimulationStep.step_number)
                .all()
            )

            # Convert to DataFrame for easier processing
            df = pd.DataFrame(population_data, columns=[
                "step", "total", "system", "independent", 
                "births", "deaths", "max_generation"
            ])

            return {
                "population_over_time": {
                    "steps": df["step"].tolist(),
                    "total": df["total"].tolist(),
                    "system": df["system"].tolist(),
                    "independent": df["independent"].tolist(),
                },
                "birth_death_rates": {
                    "birth_rate": df["births"].mean(),
                    "death_rate": df["deaths"].mean(),
                    "net_growth_rate": (df["births"] - df["deaths"]).mean(),
                },
                "generation_stats": {
                    "max_generation": df["max_generation"].max(),
                    "generation_progression": df["max_generation"].tolist(),
                }
            }

        return self.db._execute_in_transaction(_query)

    def get_resource_statistics(self) -> Dict:
        """Get statistics about resource distribution and consumption.

        Returns
        -------
        Dict
            Dictionary containing:
            - resource_distribution: Resource distribution metrics
            - consumption_patterns: Resource consumption patterns
            - efficiency_metrics: Resource utilization efficiency
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

            df = pd.DataFrame(resource_data, columns=[
                "step", "total_resources", "avg_agent_resources",
                "efficiency", "entropy"
            ])

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
                }
            }

        return self.db._execute_in_transaction(_query)

    def get_learning_statistics(self) -> Dict:
        """Get statistics about agent learning and adaptation.

        Returns
        -------
        Dict
            Dictionary containing:
            - learning_curves: Learning progress over time
            - reward_statistics: Reward distribution statistics
            - adaptation_metrics: Measures of agent adaptation
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

            df = pd.DataFrame(learning_data, columns=[
                "step", "module_type", "reward", "loss"
            ])

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
                }
            }

        return self.db._execute_in_transaction(_query) 