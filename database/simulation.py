from typing import List, Optional

from database.data_retrieval import execute_query
from database.data_types import (
    AgentStates,
    ResourceStates,
    SimulationResults,
    SimulationState,
)
from database.models import Agent, AgentState, ResourceState, SimulationStep


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

    def execute(self, step_number: int) -> SimulationResults:
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
