from typing import Dict

import pandas as pd
from sqlalchemy import case, func

from database.data_retrieval import execute_query
from database.data_types import AgentLifespanStats
from database.models import Agent


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
    def lifespans(self, session) -> Dict[str, Dict[str, float]]:
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
            "lifespan_by_generation": lifespan_data.groupby("generation")["lifespan"]
            .mean()
            .to_dict(),
        }

    @execute_query
    def survival_rates(self, session) -> Dict[int, float]:
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
    def execute(self, session) -> AgentLifespanStats:
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
        lifespan_stats = self.lifespans()

        # Get survival rates
        survival_rates = self.survival_rates()

        return {
            **lifespan_stats,
            "survival_rates": survival_rates,
        }
