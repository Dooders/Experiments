from typing import List

from sqlalchemy.orm import Session

from database.base_repository import BaseRepository
from database.data_types import (
    AgentLifespanResults,
    LifespanStatistics,
    SurvivalRatesByGeneration,
)
from database.models import Agent
from database.utilities import execute_query


class AgentRepository(BaseRepository[Agent]):
    """Repository for handling agent-related data operations."""

    def __init__(self, db):
        """Initialize the repository with a database connection."""
        super().__init__(db, Agent)

    @execute_query
    def lifespan_statistics(self, session: Session) -> LifespanStatistics:
        """Calculate comprehensive lifespan statistics across agent types and generations."""
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

        return LifespanStatistics(
            average_lifespan=lifespan_data["lifespan"].mean(),
            maximum_lifespan=lifespan_data["lifespan"].max(),
            minimum_lifespan=lifespan_data["lifespan"].min(),
            lifespan_by_type=lifespan_data.groupby("agent_type")["lifespan"]
            .mean()
            .to_dict(),
            lifespan_by_generation=lifespan_data.groupby("generation")["lifespan"]
            .mean()
            .to_dict(),
        )

    @execute_query
    def survival_rates(self, session: Session) -> SurvivalRatesByGeneration:
        """Calculate the survival rates for each generation of agents."""
        survival_rates = (
            session.query(
                Agent.generation,
                func.count(case((Agent.death_time.is_(None), 1)))
                * 100.0
                / func.count(),
            )
            .group_by(Agent.generation)
            .all()
        )

        survival_data = pd.DataFrame(
            survival_rates, columns=["generation", "survival_rate"]
        )

        return SurvivalRatesByGeneration(
            rates=survival_data.set_index("generation")["survival_rate"].to_dict()
        )

    @execute_query
    def execute(self, session: Session) -> AgentLifespanResults:
        """Calculate and combine all agent lifespan statistics."""
        lifespan_stats = self.lifespan_statistics()
        survival_rates = self.survival_rates()

        return AgentLifespanResults(
            lifespan_statistics=lifespan_stats,
            survival_rates=survival_rates,
        )
