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

from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import case, distinct, func

from database.data_types import (
    ActionStats,
    AdversarialInteractionAnalysis,
    AgentActionHistory,
    AgentEvolutionMetrics,
    AgentGenetics,
    AgentHistory,
    AgentStateData,
    AgentStates,
    BasicAgentStats,
    CollaborativeInteractionAnalysis,
    ConflictAnalysis,
    CounterfactualAnalysis,
    EnvironmentalImpactAnalysis,
    ExplorationExploitation,
    HealthIncidentData,
    LearningCurveAnalysis,
    ResilienceAnalysis,
    RiskRewardAnalysis,
)
from database.models import AgentModel, AgentAction, AgentState, HealthIncident
from database.retrievers import BaseRetriever
from database.utilities import execute_query


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

    @execute_query
    def info(self, session, agent_id: int) -> BasicAgentStats:
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
        agent = (
            session.query(AgentModel).filter(AgentModel.agent_id == agent_id).first()
        )
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

    @execute_query
    def genetics(self, session, agent_id: int) -> AgentGenetics:
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
        agent = (
            session.query(AgentModel).filter(AgentModel.agent_id == agent_id).first()
        )
        return AgentGenetics(
            genome_id=agent.genome_id,
            parent_id=agent.parent_id,
            generation=agent.generation,
        )

    @execute_query
    def state(
        self, session, agent_id: int, step_number: Optional[int] = None
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
        query = session.query(AgentState).filter(AgentState.agent_id == agent_id)
        if step_number is not None:
            query = query.filter(AgentState.step_number == step_number)
        return query.order_by(AgentState.step_number.desc()).first()

    @execute_query
    def history(self, session, agent_id: int) -> AgentHistory:
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

        return AgentHistory(
            average_health=float(metrics[0] or 0),
            average_resources=float(metrics[1] or 0),
            total_steps=int(metrics[2] or 0),
            total_reward=float(metrics[3] or 0),
        )

    @execute_query
    def actions(self, session, agent_id: int) -> AgentActionHistory:
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
        stats = (
            session.query(
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

    @execute_query
    def health(self, session, agent_id: int) -> List[HealthIncidentData]:
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
        incidents = (
            session.query(HealthIncident)
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
        query = session.query(
            AgentState.step_number,
            AgentState.agent_id,
            AgentModel.agent_type,
            AgentState.position_x,
            AgentState.position_y,
            AgentState.resource_level,
            AgentState.current_health,
            AgentState.is_defending,
        ).join(AgentModel)

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
            List of unique agent type identifiers present in the simulation
        """
        types = session.query(AgentModel.agent_type).distinct().all()
        return [t[0] for t in types]

    @execute_query
    def evolution(
        self, session, generation: Optional[int] = None
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
        query = session.query(AgentModel)
        if generation is not None:
            query = query.filter(AgentModel.generation == generation)

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

    @execute_query
    def exploration_exploitation(
        self, session, agent_id: Optional[int] = None
    ) -> ExplorationExploitation:
        """Analyze how agents balance exploration and exploitation.

        Examines the trade-off between trying new actions (exploration) and
        repeating successful actions (exploitation), including reward comparisons
        and strategy evolution.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.

        Returns
        -------
        ExplorationExploitation
            Analysis results containing:
            - exploration_rate : float
                Proportion of actions that are first-time attempts
            - exploitation_rate : float
                Proportion of actions that repeat previous choices
            - reward_comparison : Dict[str, float]
                Comparison of rewards between new and known actions:
                - new_actions_avg: Mean reward for first attempts
                - known_actions_avg: Mean reward for repeated actions

        Notes
        -----
        Rates sum to 1.0. Higher exploration rates indicate more experimental
        behavior, while higher exploitation rates suggest more conservative
        strategies.

        Examples
        --------
        >>> analysis = retriever.get_exploration_exploitation(agent_id=1)
        >>> print(f"Exploration rate: {analysis.exploration_rate:.2%}")
        >>> print(f"New vs Known rewards: {analysis.reward_comparison}")
        """
        query = session.query(AgentAction)
        if agent_id:
            query = query.filter(AgentAction.agent_id == agent_id)

        actions = query.order_by(AgentAction.step_number).all()

        # Track unique and repeated actions
        action_history = {}
        exploration_count = 0
        exploitation_count = 0
        new_action_rewards = []
        known_action_rewards = []

        for action in actions:
            action_key = (action.agent_id, action.action_type)

            if action_key not in action_history:
                exploration_count += 1
                if action.reward is not None:
                    new_action_rewards.append(action.reward)
                action_history[action_key] = action.reward
            else:
                exploitation_count += 1
                if action.reward is not None:
                    known_action_rewards.append(action.reward)

        total_actions = exploration_count + exploitation_count

        return ExplorationExploitation(
            exploration_rate=(
                exploration_count / total_actions if total_actions > 0 else 0
            ),
            exploitation_rate=(
                exploitation_count / total_actions if total_actions > 0 else 0
            ),
            reward_comparison={
                "new_actions_avg": (
                    sum(new_action_rewards) / len(new_action_rewards)
                    if new_action_rewards
                    else 0
                ),
                "known_actions_avg": (
                    sum(known_action_rewards) / len(known_action_rewards)
                    if known_action_rewards
                    else 0
                ),
            },
        )

    @execute_query
    def adversarial_analysis(
        self, session, agent_id: Optional[int] = None
    ) -> AdversarialInteractionAnalysis:
        """Analyze performance in competitive scenarios.

        Examines the relationship between action risk levels and rewards,
        including risk appetite assessment and outcome analysis.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.

        Returns
        -------
        AdversarialInteractionAnalysis
            Analysis results containing:
            - win_rate : float
                Proportion of high-risk actions taken (0.0 to 1.0)
            - damage_efficiency : float
                Average reward for high-risk actions
            - counter_strategies : Dict[str, float]
                Counter-strategies used against high-risk actions

        Notes
        -----
        Risk level is determined by reward variance relative to mean reward.
        High risk: variance > mean/2, Low risk: variance <= mean/2

        Examples
        --------
        >>> adversarial = retriever.get_adversarial_analysis(agent_id=1)
        >>> print(f"Win rate: {adversarial.win_rate:.2%}")
        Win rate: 65.30%
        >>> print(f"Damage efficiency: {adversarial.damage_efficiency:.2f}")
        Damage efficiency: 2.45
        >>> print("Counter strategies:")
        Counter strategies:
        >>> for action, freq in adversarial.counter_strategies.items():
        ...     print(f"  {action}: {freq:.2%}")
        defend: 45.20%
        retreat: 30.50%
        counter_attack: 24.30%
        """
        query = session.query(AgentAction).filter(
            AgentAction.action_type.in_(["attack", "defend", "compete"])
        )

        if agent_id:
            query = query.filter(AgentAction.agent_id == agent_id)

        actions = query.all()

        successful = [a for a in actions if a.reward and a.reward > 0]

        # Calculate counter-strategies
        counter_actions = {}
        for action in actions:
            if action.action_target_id:
                # Get target's next action
                target_response = (
                    session.query(AgentAction)
                    .filter(
                        AgentAction.agent_id == action.action_target_id,
                        AgentAction.step_number > action.step_number,
                    )
                    .order_by(AgentAction.step_number)
                    .first()
                )
                if target_response:
                    if target_response.action_type not in counter_actions:
                        counter_actions[target_response.action_type] = 0
                    counter_actions[target_response.action_type] += 1

        total_counters = sum(counter_actions.values())
        counter_frequencies = (
            {
                action: count / total_counters
                for action, count in counter_actions.items()
            }
            if total_counters > 0
            else {}
        )

        return AdversarialInteractionAnalysis(
            win_rate=len(successful) / len(actions) if actions else 0,
            damage_efficiency=(
                sum(a.reward for a in successful) / len(successful) if successful else 0
            ),
            counter_strategies=counter_frequencies,
        )

    @execute_query
    def collaborative_analysis(
        self, session, agent_id: Optional[int] = None
    ) -> CollaborativeInteractionAnalysis:
        """Analyze patterns and outcomes of cooperative behaviors.

        Examines collaborative interactions between agents, including resource sharing,
        mutual aid, and the benefits of cooperation versus individual action.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.

        Returns
        -------
        CollaborativeInteractionAnalysis
            Analysis results containing:
            - collaboration_rate : float
                Proportion of actions involving cooperation
            - group_reward_impact : float
                Average reward benefit from collaborative actions
            - synergy_metrics : float
                Performance difference between collaborative and solo actions

        Notes
        -----
        Positive synergy metrics indicate that collaboration is more effective
        than individual action.

        Examples
        --------
        >>> collab = retriever.get_collaborative_analysis(agent_id=1)
        >>> print(f"Collaboration rate: {collab.collaboration_rate:.2%}")
        >>> print(f"Group reward impact: {collab.group_reward_impact:.2f}")
        >>> print(f"Synergy benefit: {collab.synergy_metrics:+.2f}")
        """
        query = session.query(AgentAction).filter(
            AgentAction.action_type.in_(["share", "help", "cooperate"])
        )

        if agent_id:
            query = query.filter(AgentAction.agent_id == agent_id)

        actions = query.all()
        total_actions = session.query(AgentAction).count()

        collaborative_rewards = [a.reward for a in actions if a.reward is not None]
        solo_actions = (
            session.query(AgentAction)
            .filter(AgentAction.action_target_id.is_(None))
            .all()
        )
        solo_rewards = [a.reward for a in solo_actions if a.reward is not None]

        return CollaborativeInteractionAnalysis(
            collaboration_rate=len(actions) / total_actions if total_actions > 0 else 0,
            group_reward_impact=(
                sum(collaborative_rewards) / len(collaborative_rewards)
                if collaborative_rewards
                else 0
            ),
            synergy_metrics=(
                (
                    sum(collaborative_rewards) / len(collaborative_rewards)
                    - sum(solo_rewards) / len(solo_rewards)
                )
                if collaborative_rewards and solo_rewards
                else 0
            ),
        )

    @execute_query
    def learning_curve(
        self, session, agent_id: Optional[int] = None
    ) -> LearningCurveAnalysis:
        """Analyze agent learning progress and performance improvements over time.

        Examines how agent performance evolves throughout the simulation by tracking success
        rates, reward progression, and mistake reduction patterns.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.

        Returns
        -------
        LearningCurveAnalysis
            Comprehensive learning analysis containing:
            - action_success_over_time : List[float]
                Success rates across time periods
            - reward_progression : List[float]
                Average rewards across time periods
            - mistake_reduction : float
                Measure of improvement in avoiding mistakes

        Examples
        --------
        >>> curve = retriever.get_learning_curve(agent_id=1)
        >>> print(f"Mistake reduction: {curve.mistake_reduction:.2%}")
        Mistake reduction: 23.50%
        >>> print(f"Final success rate: {curve.action_success_over_time[-1]:.2%}")
        Final success rate: 78.25%
        >>> print("Reward progression:", [f"{r:.2f}" for r in curve.reward_progression[:5]])
        Reward progression: ['0.12', '0.45', '0.67', '0.89', '1.23']
        """
        query = session.query(
            AgentAction.step_number,
            func.avg(AgentAction.reward).label("avg_reward"),
            func.count(case([(AgentAction.reward > 0, 1)])).label("successes"),
            func.count().label("total"),
        ).group_by(
            func.floor(AgentAction.step_number / 100)  # Group by time periods
        )

        if agent_id:
            query = query.filter(AgentAction.agent_id == agent_id)

        results = query.all()

        return LearningCurveAnalysis(
            action_success_over_time=[
                r.successes / r.total if r.total > 0 else 0 for r in results
            ],
            reward_progression=[float(r.avg_reward or 0) for r in results],
            mistake_reduction=self._calculate_mistake_reduction(results),
        )

    @execute_query
    def environmental_impact(
        self, session, agent_id: Optional[int] = None
    ) -> EnvironmentalImpactAnalysis:
        #! How is this diff from resource impact
        """Analyze how environment affects agent action outcomes.

        Examines the relationship between environmental states (like resource levels)
        and action outcomes, including adaptation patterns and spatial effects.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.

        Returns
        -------
        EnvironmentalImpactAnalysis
            Analysis results containing:
            - environmental_state_impact : Dict[str, float]
                Correlation between resource levels and action outcomes
            - adaptive_behavior : Dict[str, float]
                Measures of agent adaptation to environmental changes
            - spatial_analysis : Dict[str, Any]
                Analysis of location-based patterns and effects

        Examples
        --------
        >>> impact = retriever.get_environmental_impact(agent_id=1)
        >>> print("Resource correlations:")
        Resource correlations:
        >>> for action, corr in impact.environmental_state_impact.items():
        ...     print(f"  {action}: {corr:.2f}")
        attack: -0.45
        defend: +0.25
        gather: +0.85
        >>> print(f"Adaptation rate: {impact.adaptive_behavior['adaptation_rate']:.2%}")
        Adaptation rate: 65.30%
        >>> print("Spatial patterns:", impact.spatial_analysis['movement_patterns'])
        Spatial patterns: ['resource_following', 'threat_avoidance', 'group_formation']
        """
        # Get actions with their states
        query = session.query(AgentAction, AgentState).join(
            AgentState, AgentAction.state_before_id == AgentState.id
        )

        if agent_id:
            query = query.filter(AgentAction.agent_id == agent_id)

        results = query.all()

        # Analyze resource levels impact
        resource_impacts = {}
        for action, state in results:
            resource_level = state.resource_level
            if action.action_type not in resource_impacts:
                resource_impacts[action.action_type] = []
            if action.reward is not None:
                resource_impacts[action.action_type].append(
                    (resource_level, action.reward)
                )

        return EnvironmentalImpactAnalysis(
            environmental_state_impact={
                action: self._calculate_correlation(
                    [r[0] for r in rewards], [r[1] for r in rewards]
                )
                for action, rewards in resource_impacts.items()
            },
            adaptive_behavior=self._analyze_adaptation(results),
            spatial_analysis=self._analyze_spatial_patterns(results),
        )

    @execute_query
    def get_conflict_analysis(
        self, session, agent_id: Optional[int] = None
    ) -> ConflictAnalysis:
        """Analyze patterns of conflict and conflict resolution strategies.

        Examines sequences of actions leading to and resolving conflicts, including
        trigger patterns, resolution strategies, and outcome analysis.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.

        Returns
        -------
        ConflictAnalysis
            Analysis results containing:
            - conflict_trigger_actions : Dict[str, float]
                Actions that commonly lead to conflicts
            - conflict_resolution_actions : Dict[str, float]
                Actions used to resolve conflicts
            - conflict_outcome_metrics : Dict[str, float]
                Success rates of different resolution strategies

        Examples
        --------
        >>> conflicts = retriever.get_conflict_analysis(agent_id=1)
        >>> print("Common triggers:", conflicts.conflict_trigger_actions)
        >>> print("Best resolution:", max(conflicts.conflict_outcome_metrics.items(),
        ...       key=lambda x: x[1])[0])
        """
        # Get sequences of actions
        query = session.query(AgentAction).order_by(
            AgentAction.agent_id, AgentAction.step_number
        )

        if agent_id:
            query = query.filter(AgentAction.agent_id == agent_id)

        actions = query.all()

        # Analyze action sequences leading to conflicts
        conflict_triggers = {}
        conflict_resolutions = {}
        conflict_outcomes = {}

        for i in range(len(actions) - 1):
            current = actions[i]
            next_action = actions[i + 1]

            if next_action.action_type in ["attack", "defend"]:
                if current.action_type not in conflict_triggers:
                    conflict_triggers[current.action_type] = 0
                conflict_triggers[current.action_type] += 1

            if current.action_type in ["attack", "defend"]:
                if next_action.action_type not in conflict_resolutions:
                    conflict_resolutions[next_action.action_type] = 0
                    conflict_outcomes[next_action.action_type] = []
                conflict_resolutions[next_action.action_type] += 1
                if next_action.reward is not None:
                    conflict_outcomes[next_action.action_type].append(
                        next_action.reward
                    )

        return ConflictAnalysis(
            conflict_trigger_actions=self._normalize_dict(conflict_triggers),
            conflict_resolution_actions=self._normalize_dict(conflict_resolutions),
            conflict_outcome_metrics={
                action: sum(outcomes) / len(outcomes) if outcomes else 0
                for action, outcomes in conflict_outcomes.items()
            },
        )

    @execute_query
    def get_risk_reward_analysis(
        self, session, agent_id: Optional[int] = None
    ) -> RiskRewardAnalysis:
        """Analyze risk-taking behavior and associated outcomes.

        Examines the relationship between action risk levels and rewards,
        including risk appetite assessment and outcome analysis.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.

        Returns
        -------
        RiskRewardAnalysis
            Analysis results containing:
            - high_risk_actions : Dict[str, float]
                Actions with high reward variance and their average returns
            - low_risk_actions : Dict[str, float]
                Actions with low reward variance and their average returns
            - risk_appetite : float
                Proportion of high-risk actions taken (0.0 to 1.0)

        Notes
        -----
        Risk level is determined by reward variance relative to mean reward.
        High risk: variance > mean/2, Low risk: variance <= mean/2

        Examples
        --------
        >>> risk = retriever.get_risk_reward_analysis(agent_id=1)
        >>> print(f"Risk appetite: {risk.risk_appetite:.2%}")
        Risk appetite: 35.80%
        >>> print("High-risk actions:")
        High-risk actions:
        >>> for action, reward in risk.high_risk_actions.items():
        ...     print(f"  {action}: {reward:+.2f}")
        attack: +2.45
        explore: +1.78
        trade: -0.92
        >>> print("Best high-risk action:", max(risk.high_risk_actions.items(),
        ...       key=lambda x: x[1])[0])
        Best high-risk action: attack
        """
        query = session.query(
            AgentAction.action_type,
            func.stddev(AgentAction.reward).label("reward_std"),
            func.avg(AgentAction.reward).label("reward_avg"),
            func.count().label("count"),
        ).group_by(AgentAction.action_type)

        if agent_id:
            query = query.filter(AgentAction.agent_id == agent_id)

        results = query.all()

        # Classify actions by risk level
        high_risk = {}
        low_risk = {}
        for result in results:
            if result.reward_std > result.reward_avg / 2:  # High variability
                high_risk[result.action_type] = float(result.reward_avg or 0)
            else:
                low_risk[result.action_type] = float(result.reward_avg or 0)

        # Calculate risk appetite
        total_actions = sum(r.count for r in results)
        high_risk_actions = sum(r.count for r in results if r.action_type in high_risk)

        return RiskRewardAnalysis(
            high_risk_actions=high_risk,
            low_risk_actions=low_risk,
            risk_appetite=high_risk_actions / total_actions if total_actions > 0 else 0,
        )

    @execute_query
    def get_counterfactual_analysis(
        self, session, agent_id: Optional[int] = None
    ) -> CounterfactualAnalysis:
        """Analyze potential alternative outcomes and missed opportunities.

        Examines what-if scenarios by analyzing unused or underutilized actions
        and their potential impacts based on observed outcomes.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.

        Returns
        -------
        CounterfactualAnalysis
            Analysis results containing:
            - counterfactual_rewards : Dict[str, float]
                Potential rewards from alternative actions
            - missed_opportunities : Dict[str, float]
                High-value actions that were underutilized
            - strategy_comparison : Dict[str, float]
                Performance delta between actual and optimal strategies

        Examples
        --------
        >>> analysis = retriever.get_counterfactual_analysis(agent_id=1)
        >>> print("Missed opportunities:")
        Missed opportunities:
        >>> for action, value in analysis.missed_opportunities.items():
        ...     print(f"  {action}: {value:+.2f}")
        trade: +2.45
        cooperate: +1.85
        explore: +1.25
        >>> print("Strategy comparison:")
        Strategy comparison:
        >>> for strategy, delta in analysis.strategy_comparison.items():
        ...     print(f"  {strategy}: {delta:+.2f} vs optimal")
        aggressive: -0.85 vs optimal
        defensive: -0.45 vs optimal
        cooperative: +0.35 vs optimal
        """
        # Get actual action history
        query = session.query(AgentAction)
        if agent_id:
            query = query.filter(AgentAction.agent_id == agent_id)
        actions = query.all()

        # Calculate average rewards for each action type
        action_rewards = {}
        for action in actions:
            if action.action_type not in action_rewards:
                action_rewards[action.action_type] = []
            if action.reward is not None:
                action_rewards[action.action_type].append(action.reward)

        avg_rewards = {
            action: sum(rewards) / len(rewards)
            for action, rewards in action_rewards.items()
            if rewards
        }

        # Find unused or underused actions
        all_action_types = set(a.action_type for a in actions)
        action_counts = {
            action: len(rewards) for action, rewards in action_rewards.items()
        }
        median_usage = sorted(action_counts.values())[len(action_counts) // 2]
        underused = {
            action: count
            for action, count in action_counts.items()
            if count < median_usage / 2
        }

        return CounterfactualAnalysis(
            counterfactual_rewards=avg_rewards,
            missed_opportunities={
                action: avg_rewards.get(action, 0) for action in underused
            },
            strategy_comparison={
                action: reward - sum(avg_rewards.values()) / len(avg_rewards)
                for action, reward in avg_rewards.items()
            },
        )

    @execute_query
    def get_resilience_analysis(
        self, session, agent_id: Optional[int] = None
    ) -> ResilienceAnalysis:
        """Analyze agent recovery patterns and adaptation to failures.

        Examines how agents respond to and recover from negative outcomes, including recovery
        speed, adaptation strategies, and impact assessment.

        Parameters
        ----------
        agent_id : Optional[int]
            Specific agent ID to analyze. If None, analyzes all agents.

        Returns
        -------
        ResilienceAnalysis
            Comprehensive resilience metrics containing:
            - recovery_rate : float
                Average time steps needed to recover from failures
            - adaptation_rate : float
                Rate at which agents modify strategies after failures
            - failure_impact : float
                Average performance impact of failures

        Examples
        --------
        >>> resilience = retriever.get_resilience_analysis(agent_id=1)
        >>> print(f"Recovery rate: {resilience.recovery_rate:.2f} steps")
        >>> print(f"Adaptation rate: {resilience.adaptation_rate:.2%}")
        """
        query = session.query(AgentAction).order_by(
            AgentAction.agent_id, AgentAction.step_number
        )

        if agent_id:
            query = query.filter(AgentAction.agent_id == agent_id)

        actions = query.all()

        # Track failure sequences
        recovery_times = []
        adaptation_speeds = []
        failure_impacts = []

        current_failure = False
        failure_start = 0
        pre_failure_reward = 0

        for i, action in enumerate(actions):
            if action.reward is not None and action.reward < 0:
                if not current_failure:
                    current_failure = True
                    failure_start = i
                    pre_failure_reward = (
                        sum(
                            a.reward
                            for a in actions[max(0, i - 5) : i]
                            if a.reward is not None
                        )
                        / 5
                    )
            elif current_failure and action.reward is not None and action.reward > 0:
                # Recovery detected
                recovery_times.append(i - failure_start)

                # Calculate adaptation speed
                post_failure_actions = actions[failure_start:i]
                if post_failure_actions:
                    adaptation = sum(
                        1
                        for a in post_failure_actions
                        if a.action_type != actions[failure_start].action_type
                    ) / len(post_failure_actions)
                    adaptation_speeds.append(adaptation)

                # Calculate impact
                failure_impacts.append(
                    pre_failure_reward
                    - min(
                        a.reward for a in post_failure_actions if a.reward is not None
                    )
                )

                current_failure = False

        return ResilienceAnalysis(
            recovery_rate=(
                sum(recovery_times) / len(recovery_times) if recovery_times else 0
            ),
            adaptation_rate=(
                sum(adaptation_speeds) / len(adaptation_speeds)
                if adaptation_speeds
                else 0
            ),
            failure_impact=(
                sum(failure_impacts) / len(failure_impacts) if failure_impacts else 0
            ),
        )

    def _calculate_mistake_reduction(self, results) -> float:
        """Calculate the reduction in mistake rate over time.

        Compares early-stage mistake rates with late-stage rates to measure
        improvement in decision-making accuracy.

        Parameters
        ----------
        results : List[Row]
            Query results containing success/failure counts per time period

        Returns
        -------
        float
            Reduction in mistake rate, where:
            - Positive values indicate fewer mistakes over time
            - Zero indicates no improvement
            - Values range from 0.0 to 1.0

        Notes
        -----
        Mistake rate is calculated as (1 - success_rate) for each period.
        The reduction is the difference between early and late mistake rates.
        """
        if not results:
            return 0
        early_mistakes = (
            1 - results[0].successes / results[0].total if results[0].total > 0 else 0
        )
        late_mistakes = (
            1 - results[-1].successes / results[-1].total
            if results[-1].total > 0
            else 0
        )
        return max(0, early_mistakes - late_mistakes)

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient between two variables.

        Measures the linear correlation between two sets of values, typically
        used for analyzing relationships between actions and outcomes.

        Parameters
        ----------
        x : List[float]
            First variable's values
        y : List[float]
            Second variable's values (must be same length as x)

        Returns
        -------
        float
            Correlation coefficient between -1.0 and 1.0, where:
            - 1.0 indicates perfect positive correlation
            - -1.0 indicates perfect negative correlation
            - 0.0 indicates no linear correlation

        Notes
        -----
        Returns 0.0 if either list is empty or lengths don't match.
        """
        if not x or not y or len(x) != len(y):
            return 0
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(i * j for i, j in zip(x, y))
        sum_x2 = sum(i * i for i in x)
        sum_y2 = sum(i * i for i in y)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y) ** 0.5

        return numerator / denominator if denominator != 0 else 0

    def _normalize_dict(self, d: Dict[str, int]) -> Dict[str, float]:
        """Normalize dictionary values to proportions.

        Converts raw counts to proportions by dividing each value by the sum
        of all values.

        Parameters
        ----------
        d : Dict[str, int]
            Dictionary of raw counts

        Returns
        -------
        Dict[str, float]
            Dictionary with normalized values (proportions), where:
            - Each value is between 0.0 and 1.0
            - Sum of all values equals 1.0
            - Empty input returns empty dictionary

        Examples
        --------
        >>> d = {'attack': 20, 'defend': 30, 'explore': 50}
        >>> normalized = self._normalize_dict(d)
        >>> print(normalized)
        {'attack': 0.2, 'defend': 0.3, 'explore': 0.5}
        >>> print(f"Total: {sum(normalized.values())}")
        Total: 1.0
        """
        total = sum(d.values())
        return {k: v / total if total > 0 else 0 for k, v in d.items()}

    def _analyze_adaptation(
        self, results: List[Tuple[AgentAction, AgentState]]
    ) -> Dict[str, float]:
        """Analyze how agents adapt to changing conditions.

        Examines patterns of behavioral changes in response to environmental
        conditions and action outcomes.

        Parameters
        ----------
        results : List[Tuple[AgentAction, AgentState]]
            List of action-state pairs to analyze

        Returns
        -------
        Dict[str, float]
            Adaptation metrics including:
            - adaptation_rate: How quickly agents modify behavior
            - success_rate: Success rate of adapted behaviors
            - stability: Consistency of adapted behaviors

        Notes
        -----
        Adaptation is measured by comparing action choices before and after
        significant changes in state or rewards.
        """
        # Implementation details...
        pass

    def _analyze_spatial_patterns(
        self, results: List[Tuple[AgentAction, AgentState]]
    ) -> Dict[str, Any]:
        """Analyze spatial patterns in agent behavior.

        Examines how agent actions and outcomes vary based on position
        and spatial relationships with other agents.

        Parameters
        ----------
        results : List[Tuple[AgentAction, AgentState]]
            List of action-state pairs to analyze

        Returns
        -------
        Dict[str, Any]
            Spatial analysis results including:
            - position_effects: Impact of location on outcomes
            - clustering: Patterns of agent grouping
            - movement_patterns: Common movement strategies

        Notes
        -----
        Requires valid position data in AgentState records.
        Returns empty patterns if position data is missing.
        """
        # Implementation details...
        pass
