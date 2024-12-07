"""Agent retrieval module for simulation database.

This module provides specialized queries and analysis methods for agent-related
data, including state history, performance metrics, and behavioral analysis.

The AgentRetriever class handles agent-specific database operations with
optimized queries and efficient data aggregation methods.

Classes
-------
AgentRetriever
    Handles retrieval and analysis of agent-related data from the simulation database.
"""

from typing import Any, Dict, List, Optional

from sqlalchemy import distinct, func

from database.data_types import (
    ActionStats,
    AgentActionHistory,
    AgentEvolutionMetrics,
    AgentGenetics,
    AgentHistory,
    AgentStateData,
    AgentStates,
    BasicAgentStats,
    HealthIncidentData,
)
from database.models import Agent, AgentAction, AgentState, HealthIncident
from database.retrievers import BaseRetriever
from database.utilities import execute_query
from database.repositories.agent_repository import AgentRepository
from database.unit_of_work import UnitOfWork
from database.dtos.agent_dto import AgentDTO
from database.services.agent_service import AgentService


class AgentRetriever(BaseRetriever):
    """Handles retrieval and analysis of agent-related data.

    A specialized retriever class that provides comprehensive methods for querying
    and analyzing agent data throughout the simulation lifecycle, including state
    tracking, performance metrics, and evolutionary patterns.

    Attributes
    ----------
    session : Session
        SQLAlchemy session for database interactions (inherited from BaseRetriever)

    Methods
    -------
    info(agent_id: int) -> Dict[str, Any]
        Retrieves fundamental agent attributes and configuration
    genetics(agent_id: int) -> Dict[str, Any]
        Retrieves genetic lineage and evolutionary data
    state(agent_id: int) -> Optional[AgentState]
        Retrieves the most recent state for an agent
    history(agent_id: int) -> Dict[str, float]
        Retrieves historical performance metrics
    actions(agent_id: int) -> Dict[str, Dict[str, float]]
        Retrieves detailed action statistics and patterns
    health(agent_id: int) -> List[Dict[str, Any]]
        Retrieves health incident history
    data(agent_id: int) -> AgentStateData
        Retrieves comprehensive agent data
    states(step_number: Optional[int]) -> List[AgentStates]
        Retrieves agent states for specific or all simulation steps
    types() -> List[str]
        Retrieves all unique agent types in the simulation
    evolution(generation: Optional[int]) -> AgentEvolutionMetrics
        Retrieves evolutionary metrics for specific or all generations
    """

    def _execute(self) -> Dict[str, Any]:
        """Execute comprehensive agent analysis.

        Returns
        -------
        Dict[str, Any]
            Complete agent analysis including:
            - agent_types: List of unique agent types
            - evolution_metrics: Evolution statistics
            - performance_metrics: Performance statistics
            - agent_states: Agent states
            - agent_data: Comprehensive agent data
        """
        return {
            "agent_types": self.types(),
            "evolution_metrics": self.evolution(),
            "performance_metrics": self.performance(),
            "agent_states": self.states(),
            "agent_data": self.data(),
        }

    def info(self, agent_id: int) -> BasicAgentStats:
        """Get basic information about an agent.

        Parameters
        ----------
        agent_id : int
            The unique identifier of the agent to query

        Returns
        -------
        BasicAgentStats
            Basic agent information including:
            - agent_id: int
            - agent_type: str
            - birth_time: datetime
            - death_time: Optional[datetime]
            - lifespan: Optional[timedelta]
            - initial_resources: float
            - max_health: float
            - starvation_threshold: float
        """
        with UnitOfWork(self.db.Session) as uow:
            agent = uow.agents.get_by_id(agent_id)
            return BasicAgentStats(
                agent_id=agent.agent_id,
                agent_type=agent.agent_type,
                birth_time=agent.birth_time,
                death_time=agent.death_time,
                lifespan=(
                    (agent.death_time - agent.birth_time) if agent.death_time else None
                ),
                initial_resources=agent.initial_resources,
                max_health=agent.max_health,
                starvation_threshold=agent.starvation_threshold,
            )

    def genetics(self, agent_id: int) -> AgentGenetics:
        """Get genetic information about an agent.

        Parameters
        ----------
        agent_id : int
            The unique identifier of the agent to query

        Returns
        -------
        AgentGenetics
            Genetic information including:
            - genome_id: str
            - parent_id: Optional[int]
            - generation: int
        """
        with UnitOfWork(self.db.Session) as uow:
            agent = uow.agents.get_by_id(agent_id)
            return AgentGenetics(
                genome_id=agent.genome_id,
                parent_id=agent.parent_id,
                generation=agent.generation,
            )

    def state(
        self, agent_id: int, step_number: Optional[int] = None
    ) -> Optional[AgentState]:
        """Get the state for a specific agent. If a step number is provided, the state
        for that specific step is returned. Otherwise, the most recent state is returned.

        Parameters
        ----------
        agent_id : int
            The unique identifier of the agent
        step_number : Optional[int], default=None
            Specific step to get state for. If None, retrieves most recent state.

        Returns
        -------
        Optional[AgentState]
            The most recent state of the agent, or None if no states exist
        """
        with UnitOfWork(self.db.Session) as uow:
            query = uow.session.query(AgentState).filter(AgentState.agent_id == agent_id)
            if step_number is not None:
                query = query.filter(AgentState.step_number == step_number)
            return query.order_by(AgentState.step_number.desc()).first()

    def history(self, agent_id: int) -> AgentHistory:
        """Get historical metrics for a specific agent.

        Parameters
        ----------
        agent_id : int
            The unique identifier of the agent

        Returns
        -------
        AgentHistory
            Historical metrics including:
            - average_health: float
                Mean health value across all states
            - average_resources: float
                Mean resource level across all states
            - total_steps: int
                Total number of simulation steps
            - total_reward: float
                Cumulative reward earned
        """
        with UnitOfWork(self.db.Session) as uow:
            metrics = (
                uow.session.query(
                    func.avg(AgentState.current_health).label("avg_health"),
                    func.avg(AgentState.resource_level).label("avg_resources"),
                    func.count(AgentState.step_number).label("total_steps"),
                    func.max(AgentState.total_reward).label("total_reward"),
                )
                .filter(AgentState.agent_id == agent_id)
                .first()
            )

            return AgentHistory(
                average_health=float(metrics[0] or 0),
                average_resources=float(metrics[1] or 0),
                total_steps=int(metrics[2] or 0),
                total_reward=float(metrics[3] or 0),
            )

    def actions(self, agent_id: int) -> AgentActionHistory:
        """Get action statistics for a specific agent.

        Parameters
        ----------
        agent_id : int
            The unique identifier of the agent

        Returns
        -------
        AgentActionHistory
            Dictionary mapping action types to their statistics:
            - count: Number of times action was taken
            - average_reward: Mean reward for this action
            - total_actions: Total number of actions by agent
            - action_diversity: Number of unique actions used
        """
        with UnitOfWork(self.db.Session) as uow:
            stats = (
                uow.session.query(
                    AgentAction.action_type,
                    func.count().label("count"),
                    func.avg(AgentAction.reward).label("avg_reward"),
                    func.count(AgentAction.action_id).over().label("total_actions"),
                    func.count(distinct(AgentAction.action_type))
                    .over()
                    .label("action_diversity"),
                )
                .filter(AgentAction.agent_id == agent_id)
                .group_by(AgentAction.action_type)
                .all()
            )

            actions = {
                action_type: ActionStats(
                    count=count,
                    average_reward=float(avg_reward or 0),
                    total_actions=int(total_actions),
                    action_diversity=int(action_diversity),
                )
                for action_type, count, avg_reward, total_actions, action_diversity in stats
            }

            return AgentActionHistory(actions=actions)

    def health(self, agent_id: int) -> List[HealthIncidentData]:
        """Get health incident history for a specific agent.

        Parameters
        ----------
        agent_id : int
            The unique identifier of the agent

        Returns
        -------
        List[HealthIncidentData]
            List of health incidents, each containing:
            - step: Simulation step when incident occurred
            - health_before: Health value before incident
            - health_after: Health value after incident
            - cause: Reason for health change
            - details: Additional incident-specific information
        """
        with UnitOfWork(self.db.Session) as uow:
            incidents = (
                uow.session.query(HealthIncident)
                .filter(HealthIncident.agent_id == agent_id)
                .order_by(HealthIncident.step_number)
                .all()
            )

            return [
                HealthIncidentData(
                    step=incident.step_number,
                    health_before=incident.health_before,
                    health_after=incident.health_after,
                    cause=incident.cause,
                    details=incident.details,
                )
                for incident in incidents
            ]

    def data(self, agent_id: int) -> AgentStateData:
        """Get comprehensive data for a specific agent.

        Parameters
        ----------
        agent_id : int
            The unique identifier of the agent

        Returns
        -------
        AgentStateData
            Complete agent data including:
            - basic_info: Dict[str, Any]
                Fundamental agent attributes
            - genetic_info: Dict[str, Any]
                Genetic and evolutionary data
            - current_state: Optional[AgentState]
                Most recent agent state
            - historical_metrics: Dict[str, float]
                Performance statistics
            - action_history: Dict[str, Dict[str, float]]
                Action statistics and patterns
            - health_incidents: List[Dict[str, Any]]
                Health incident records
        """
        with UnitOfWork(self.db.Session) as uow:
            return AgentStateData(
                basic_info=self.info(agent_id),
                genetic_info=self.genetics(agent_id),
                current_state=self.state(agent_id),
                historical_metrics=self.history(agent_id),
                action_history=self.actions(agent_id),
                health_incidents=self.health(agent_id),
            )

    def states(self, step_number: Optional[int] = None) -> List[AgentStates]:
        """Get agent states for a specific step or all steps.

        Parameters
        ----------
        step_number : Optional[int], default=None
            Specific step to get states for. If None, retrieves states for all steps.

        Returns
        -------
        List[AgentStates]
            List of agent states, each containing:
            - step_number: int
            - agent_id: int
            - agent_type: str
            - position_x: float
            - position_y: float
            - resource_level: float
            - current_health: float
            - is_defending: bool
        """
        with UnitOfWork(self.db.Session) as uow:
            query = uow.session.query(
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

    def types(self) -> List[str]:
        """Get list of all unique agent types.

        Returns
        -------
        List[str]
            List of unique agent type identifiers present in the simulation
        """
        with UnitOfWork(self.db.Session) as uow:
            types = uow.session.query(Agent.agent_type).distinct().all()
            return [t[0] for t in types]

    def evolution(
        self, generation: Optional[int] = None
    ) -> AgentEvolutionMetrics:
        """Get evolution metrics for agents.

        Parameters
        ----------
        generation : Optional[int], default=None
            Specific generation to analyze. If None, analyzes all generations.

        Returns
        -------
        AgentEvolutionMetrics
            Evolution metrics including:
            - total_agents: int
                Number of agents in the generation
            - unique_genomes: int
                Number of distinct genetic configurations
            - average_lifespan: timedelta
                Mean survival duration
            - generation: Optional[int]
                Generation number (None if analyzing all generations)
        """
        with UnitOfWork(self.db.Session) as uow:
            query = uow.session.query(Agent)
            if generation is not None:
                query = query.filter(Agent.generation == generation)

            results = query.all()

            # Calculate metrics
            total_agents = len(results)
            unique_genomes = len(set(a.genome_id for a in results if a.genome_id))
            avg_lifespan = (
                sum((a.death_time - a.birth_time) if a.death_time else 0 for a in results)
                / total_agents
                if total_agents > 0
                else 0
            )

            return AgentEvolutionMetrics(
                total_agents=total_agents,
                unique_genomes=unique_genomes,
                average_lifespan=avg_lifespan,
                generation=generation,
            )
