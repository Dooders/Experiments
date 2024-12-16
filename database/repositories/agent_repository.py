from typing import List

import pandas as pd
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from database.data_types import (
    AgentLifespanResults,
    LifespanStatistics,
    SurvivalRatesByGeneration,
)
from database.models import AgentModel
from database.repositories.base_repository import BaseRepository
from database.session_manager import SessionManager
from database.utilities import execute_query


class AgentRepository(BaseRepository[AgentModel]):
    """Repository for handling agent-related data operations."""

    def __init__(self, session_manager: SessionManager):
        """Initialize the repository with a database connection."""
        super().__init__(session_manager, AgentModel)

    @execute_query
    def lifespan_statistics(self, session: Session) -> LifespanStatistics:
        """Calculate comprehensive lifespan statistics across agent types and generations."""
        lifespans = (
            session.query(
                AgentModel.agent_type,
                AgentModel.generation,
                (AgentModel.death_time - AgentModel.birth_time).label("lifespan"),
            )
            .filter(AgentModel.death_time.isnot(None))
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
                AgentModel.generation,
                func.count(case((AgentModel.death_time.is_(None), 1)))
                * 100.0
                / func.count(),
            )
            .group_by(AgentModel.generation)
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
