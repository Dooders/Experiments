"""Agent retrieval module for simulation database.

This module provides specialized queries and analysis methods for agent-related
data, including state history, performance metrics, and behavioral analysis.

The AgentRetriever class handles agent-specific database operations with
optimized queries and efficient data aggregation methods.
"""

from typing import Any, Dict, List, Optional

from sqlalchemy import distinct, func

from database.data_types import (
    AgentEvolutionMetrics,
    AgentMetrics,
    AgentPerformance,
    AgentStateData,
    AgentStates,
    BasicAgentStats,
)
from database.models import Agent, AgentAction, AgentState, HealthIncident
from database.retrievers import BaseRetriever
from database.utilities import execute_query


class AgentRetriever(BaseRetriever):
    """Handles retrieval and analysis of agent-related data.

    This class provides methods for analyzing agent states, performance,
    and evolution throughout the simulation.

    Methods
    -------
    get_agent_data(agent_id: int) -> AgentStateData
        Get comprehensive data for a specific agent
    get_agent_states(step_number: Optional[int] = None) -> List[AgentStates]
        Get agent states for a specific step or all steps
    get_agent_types() -> List[str]
        Get list of all unique agent types
    get_agent_metrics(agent_id: int) -> AgentMetrics
        Get performance metrics for a specific agent
    get_agent_evolution(generation: Optional[int] = None) -> AgentEvolutionMetrics
        Get evolution metrics for agents
    execute() -> Dict[str, Any]
        Generate comprehensive agent analysis
    """

    @execute_query
    def basic_info(self, session, agent_id: int) -> Dict[str, Any]:
        """Get basic information about an agent."""
        agent = session.query(Agent).filter(Agent.agent_id == agent_id).first()
        return {
            "agent_id": agent.agent_id,
            "agent_type": agent.agent_type,
            "birth_time": agent.birth_time,
            "death_time": agent.death_time,
            "lifespan": (
                (agent.death_time - agent.birth_time) if agent.death_time else None
            ),
            "initial_resources": agent.initial_resources,
            "max_health": agent.max_health,
            "starvation_threshold": agent.starvation_threshold,
        }

    @execute_query
    def genetic_info(self, session, agent_id: int) -> Dict[str, Any]:
        """Get genetic information about an agent."""
        agent = session.query(Agent).filter(Agent.agent_id == agent_id).first()
        return {
            "genome_id": agent.genome_id,
            "parent_id": agent.parent_id,
            "generation": agent.generation,
        }

    @execute_query
    def state(self, session, agent_id: int) -> Optional[AgentState]:
        """Get the latest state for a specific agent.

        Parameters
        ----------
        agent_id : int
            The unique identifier of the agent

        Returns
        -------
        Optional[AgentState]
            The most recent state of the agent, or None if no states exist
        """
        return (
            session.query(AgentState)
            .filter(AgentState.agent_id == agent_id)
            .order_by(AgentState.step_number.desc())
            .first()
        )

    @execute_query
    def historical(self, session, agent_id: int) -> Dict[str, float]:
        """Get historical metrics for a specific agent.

        Parameters
        ----------
        agent_id : int
            The unique identifier of the agent

        Returns
        -------
        Dict[str, float]
            Dictionary containing average health, resources, total steps, and reward
        """
        metrics = (
            session.query(
                func.avg(AgentState.current_health).label("avg_health"),
                func.avg(AgentState.resource_level).label("avg_resources"),
                func.count(AgentState.step_number).label("total_steps"),
                func.max(AgentState.total_reward).label("total_reward"),
            )
            .filter(AgentState.agent_id == agent_id)
            .first()
        )

        return {
            "average_health": float(metrics[0] or 0),
            "average_resources": float(metrics[1] or 0),
            "total_steps": int(metrics[2] or 0),
            "total_reward": float(metrics[3] or 0),
        }

    @execute_query
    def actions(self, session, agent_id: int) -> Dict[str, Dict[str, float]]:
        """Get action statistics for a specific agent.

        Parameters
        ----------
        agent_id : int
            The unique identifier of the agent

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping action types to their statistics (count and average reward)
        """
        stats = (
            session.query(
                AgentAction.action_type,
                func.count().label("count"),
                func.avg(AgentAction.reward).label("avg_reward"),
            )
            .filter(AgentAction.agent_id == agent_id)
            .group_by(AgentAction.action_type)
            .all()
        )

        return {
            action_type: {
                "count": count,
                "average_reward": float(avg_reward or 0),
            }
            for action_type, count, avg_reward in stats
        }

    @execute_query
    def health(self, session, agent_id: int) -> List[Dict[str, Any]]:
        """Get health incident history for a specific agent.

        Parameters
        ----------
        agent_id : int
            The unique identifier of the agent

        Returns
        -------
        List[Dict[str, Any]]
            List of health incidents with step number, health changes, cause, and details
        """
        incidents = (
            session.query(HealthIncident)
            .filter(HealthIncident.agent_id == agent_id)
            .order_by(HealthIncident.step_number)
            .all()
        )

        return [
            {
                "step": incident.step_number,
                "health_before": incident.health_before,
                "health_after": incident.health_after,
                "cause": incident.cause,
                "details": incident.details,
            }
            for incident in incidents
        ]

    @execute_query
    def data(self, session, agent_id: int) -> AgentStateData:
        """Get comprehensive data for a specific agent.

        Parameters
        ----------
        agent_id : int
            The unique identifier of the agent

        Returns
        -------
        AgentStateData
            Complete agent data including basic info, state history,
            and performance metrics
        """

        return AgentStateData(
            basic_info=self.basic_info(agent_id),
            genetic_info=self.genetic_info(agent_id),
            current_state=self.state(agent_id),
            historical_metrics=self.historical(agent_id),
            action_history=self.actions(agent_id),
            health_incidents=self.health(agent_id),
        )

    @execute_query
    def states(self, session, step_number: Optional[int] = None) -> List[AgentStates]:
        """Get agent states for a specific step or all steps.

        Parameters
        ----------
        step_number : Optional[int]
            Specific step to get states for. If None, gets all steps.

        Returns
        -------
        List[AgentStates]
            List of agent states
        """
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

    @execute_query
    def types(self, session) -> List[str]:
        """Get list of all unique agent types.

        Returns
        -------
        List[str]
            List of unique agent type names
        """
        types = session.query(Agent.agent_type).distinct().all()
        return [t[0] for t in types]

    @execute_query
    def metrics(self, session, agent_id: int) -> AgentMetrics:
        """Get performance metrics for a specific agent.

        Parameters
        ----------
        agent_id : int
            ID of the agent to analyze

        Returns
        -------
        AgentMetrics
            Performance metrics for the agent
        """
        # Get basic stats
        basic_stats = self.basic_info(agent_id)

        # Get performance stats
        performance = (
            session.query(
                func.count(AgentAction.action_id).label("total_actions"),
                func.avg(AgentAction.reward).label("avg_reward"),
                func.count(distinct(AgentAction.action_type)).label("unique_actions"),
            )
            .filter(AgentAction.agent_id == agent_id)
            .first()
        )

        return AgentMetrics(
            basic_stats=BasicAgentStats(
                average_health=float(basic_stats[0] or 0),
                average_resources=float(basic_stats[1] or 0),
                lifespan=int(basic_stats[2] or 0),
                total_reward=float(basic_stats[3] or 0),
            ),
            performance=AgentPerformance(
                total_actions=int(performance[0] or 0),
                average_reward=float(performance[1] or 0),
                action_diversity=int(performance[2] or 0),
            ),
        )

    @execute_query
    def get_agent_evolution(
        self, session, generation: Optional[int] = None
    ) -> AgentEvolutionMetrics:
        """Get evolution metrics for agents.

        Parameters
        ----------
        generation : Optional[int]
            Specific generation to analyze. If None, analyzes all generations.

        Returns
        -------
        AgentEvolutionMetrics
            Evolution metrics for agents
        """
        query = session.query(Agent)
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

    def _execute(self) -> Dict[str, Any]:
        """Execute comprehensive agent analysis.

        Returns
        -------
        Dict[str, Any]
            Complete agent analysis including:
            - agent_types: List of unique agent types
            - evolution_metrics: Evolution statistics
            - performance_metrics: Performance statistics
        """
        return {
            "agent_types": self.get_agent_types(),
            "evolution_metrics": self.get_agent_evolution(),
            "performance_metrics": None,  # Aggregate metrics across all agents
        }
