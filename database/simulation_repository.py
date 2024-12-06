from typing import List, Optional
from database.base_repository import BaseRepository
from database.data_types import (
    AgentStates,
    ResourceStates,
    SimulationResults,
    SimulationState,
)
from database.models import Agent, AgentState, ResourceState, SimulationStep


class SimulationRepository(BaseRepository[SimulationStep]):
    """Handles retrieval of simulation state data from the database."""

    def __init__(self, db):
        """Initialize with database connection and model type."""
        super().__init__(db, SimulationStep)

    def agent_states(self, step_number: Optional[int] = None) -> List[AgentStates]:
        """Retrieve agent states for a specific step or all steps."""
        def _query(session):
            query = session.query(
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
                query = query.filter(AgentState.step_number == step_number)
            else:
                query = query.order_by(AgentState.step_number, AgentState.agent_id)

            results = query.all()

            return [
                AgentStates(
                    step_number=row[0],
                    agent_id=row[1],
                    agent_type=row[2],
                    position_x=row[3],
                    position_y=row[4],
                    resource_level=row[5],
                    current_health=row[6],
                    is_defending=row[7],
                )
                for row in results
            ]

        return self._execute_in_transaction(_query)

    def resource_states(self, step_number: int) -> List[ResourceStates]:
        """Retrieve resource states for a specific step."""
        def _query(session):
            query = session.query(
                ResourceState.resource_id,
                ResourceState.amount,
                ResourceState.position_x,
                ResourceState.position_y,
            ).filter(ResourceState.step_number == step_number)

            results = query.all()

            return [
                ResourceStates(
                    resource_id=row[0],
                    amount=row[1],
                    position_x=row[2],
                    position_y=row[3],
                )
                for row in results
            ]

        return self._execute_in_transaction(_query)

    def simulation_state(self, step_number: int) -> SimulationState:
        """Retrieve simulation state for a specific step."""
        def _query(session):
            query = session.query(SimulationStep).filter(
                SimulationStep.step_number == step_number
            )

            result = query.first()

            return SimulationState(**result.as_dict())

        return self._execute_in_transaction(_query)

    def execute(self, step_number: int) -> SimulationResults:
        """Retrieve complete simulation state for a specific step."""
        return SimulationResults(
            agent_states=self.agent_states(step_number),
            resource_states=self.resource_states(step_number),
            simulation_state=self.simulation_state(step_number),
        )
