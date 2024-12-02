"""Data types for simulation metrics and statistics.

This module defines dataclasses that represent the structured return types
for various simulation metrics and statistics queries.

Classes
-------
SimulationStateData
    Data for a specific simulation step
AgentLifespanStats
    Statistics about agent lifespans
PopulationStatistics
    Comprehensive population statistics
ResourceMetrics
    Resource distribution and efficiency metrics
PopulationVariance
    Population variance statistics
AgentDistribution
    Distribution of different agent types
SurvivalMetrics
    Population survival metrics
ResourceDistribution
    Resource distribution statistics over time
EfficiencyMetrics
    Resource efficiency metrics
LearningProgress
    Learning and adaptation statistics
ModulePerformance
    Performance metrics for learning modules
HistoricalMetrics
    Historical simulation metrics
InteractionPattern
    Statistics about agent interactions
ResourceBehavior
    Resource-related behavior statistics
RewardStats
    Statistics about action rewards
DecisionPattern
    Pattern of agent decisions
AgentBehaviorMetrics
    Comprehensive analysis of agent behaviors
AgentDecisionMetrics
    Analysis of agent decision-making patterns
StepActionData
    Data about actions in a specific simulation step
AgentStateData
    Comprehensive data about an agent's state
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SimulationResults:
    """Data for a specific simulation step.

    Attributes
    ----------
    agent_states : List[Tuple]
        List of agent state records
    resource_states : List[Tuple]
        List of resource state records
    simulation_state : Dict[str, Any]
        Dictionary of simulation metrics for the step
    """

    agent_states: List[Tuple]
    resource_states: List[Tuple]
    simulation_state: Dict[str, Any]


@dataclass
class AgentLifespanStats:
    """Statistics about agent lifespans.

    Attributes
    ----------
    average_lifespan : float
        Mean lifespan across all agents
    lifespan_by_type : Dict[str, float]
        Mean lifespan per agent type
    lifespan_by_generation : Dict[int, float]
        Mean lifespan per generation
    survival_rates : Dict[int, float]
        Survival rate per generation
    """

    average_lifespan: float
    lifespan_by_type: Dict[str, float]
    lifespan_by_generation: Dict[int, float]
    survival_rates: Dict[int, float]


@dataclass
class ResourceMetrics:
    """Resource distribution and efficiency metrics.

    Attributes
    ----------
    resource_utilization : float
        Resource usage efficiency
    resources_consumed : float
        Total resources consumed
    resources_available : float
        Total resources available
    utilization_per_agent : float
        Average resource usage per agent
    """

    resource_utilization: float
    resources_consumed: float
    resources_available: float
    utilization_per_agent: float


@dataclass
class PopulationVariance:
    """Population variance statistics.

    Attributes
    ----------
    variance : float
        Population size variance
    standard_deviation : float
        Population size standard deviation
    coefficient_variation : float
        Coefficient of variation
    """

    variance: float
    standard_deviation: float
    coefficient_variation: float


@dataclass
class AgentDistribution:
    """Distribution of different agent types.

    Attributes
    ----------
    system_agents : float
        Average number of system agents
    independent_agents : float
        Average number of independent agents
    control_agents : float
        Average number of control agents
    """

    system_agents: float
    independent_agents: float
    control_agents: float


@dataclass
class SurvivalMetrics:
    """Population survival metrics.

    Attributes
    ----------
    survival_rate : float
        Population survival rate
    average_lifespan : float
        Average agent lifespan
    """

    survival_rate: float
    average_lifespan: float


@dataclass
class ResourceDistribution:
    """Resource distribution statistics over time.

    Attributes
    ----------
    steps : List[int]
        List of step numbers
    total_resources : List[float]
        Total resources per step
    average_per_agent : List[float]
        Average resources per agent per step
    """

    steps: List[int]
    total_resources: List[float]
    average_per_agent: List[float]


@dataclass
class EfficiencyMetrics:
    """Resource efficiency metrics.

    Attributes
    ----------
    average_efficiency : float
        Mean resource utilization efficiency
    efficiency_trend : List[float]
        Resource efficiency over time
    distribution_entropy : List[float]
        Resource distribution entropy over time
    """

    average_efficiency: float
    efficiency_trend: List[float]
    distribution_entropy: List[float]


@dataclass
class LearningProgress:
    """Learning and adaptation statistics, sorted by step number.

    Attributes
    ----------
    average_reward : List[float]
        Mean reward per step
    average_loss : List[float]
        Mean loss per step
    """

    average_reward: List[float]
    average_loss: List[float]


@dataclass
class ModulePerformance:
    """Performance metrics for learning modules.

    Attributes
    ----------
    avg_reward : float
        Average reward for the module
    avg_loss : float
        Average loss for the module
    """

    avg_reward: float
    avg_loss: float


@dataclass
class HistoricalMetrics:
    """Historical simulation metrics.

    Attributes
    ----------
    steps : List[int]
        List of step numbers
    metrics : Dict[str, List[float]]
        Dictionary of metric lists including:
        - total_agents
        - system_agents
        - independent_agents
        - control_agents
        - total_resources
        - average_agent_resources
        - births
        - deaths
    """

    steps: List[int]
    metrics: Dict[str, List[float]]


@dataclass
class InteractionPattern:
    """Statistics about agent interactions.

    Attributes
    ----------
    interaction_count : int
        Number of interactions
    average_reward : float
        Average reward for interactions
    """

    interaction_count: int
    average_reward: float


@dataclass
class ResourceBehavior:
    """Resource-related behavior statistics.

    Attributes
    ----------
    average_resource_change : float
        Average change in resources
    action_count : int
        Number of resource-affecting actions
    """

    average_resource_change: float
    action_count: int


@dataclass
class RewardStats:
    """Statistics about action rewards.

    Attributes
    ----------
    average : float
        Mean reward value
    stddev : float
        Standard deviation of rewards
    min : float
        Minimum reward value
    max : float
        Maximum reward value
    """

    average: float
    stddev: float
    min: float
    max: float


@dataclass
class DecisionPattern:
    """Pattern of agent decisions.

    Attributes
    ----------
    count : int
        Number of times this decision was made
    frequency : float
        Frequency of this decision type
    reward_stats : RewardStats
        Statistics about rewards for this decision
    """

    count: int
    frequency: float
    reward_stats: RewardStats


@dataclass
class AgentBehaviorMetrics:
    """Comprehensive analysis of agent behaviors.

    Attributes
    ----------
    temporal_patterns : Dict[int, Dict[str, Dict[str, float]]]
        Behavior patterns over time, indexed by step
    type_behaviors : Dict[str, Dict[str, Any]]
        Behavior patterns by agent type
    interaction_patterns : Dict[str, InteractionPattern]
        Statistics about different types of interactions
    resource_behaviors : Dict[str, ResourceBehavior]
        Resource-related behavior statistics
    behavior_summary : Dict[str, Any]
        Summary statistics about behaviors
    """

    temporal_patterns: Dict[int, Dict[str, Dict[str, float]]]
    type_behaviors: Dict[str, Dict[str, Any]]
    interaction_patterns: Dict[str, InteractionPattern]
    resource_behaviors: Dict[str, ResourceBehavior]
    behavior_summary: Dict[str, Any]


@dataclass
class AgentDecisionMetrics:
    """Analysis of agent decision-making patterns.

    Attributes
    ----------
    decision_patterns : Dict[str, DecisionPattern]
        Statistics about different decision types
    temporal_trends : Dict[int, Dict[str, float]]
        Decision patterns over time
    context_influence : Dict[str, Dict[str, float]]
        Impact of context on decisions
    decision_outcomes : Dict[str, RewardStats]
        Outcome statistics for each decision type
    """

    decision_patterns: Dict[str, DecisionPattern]
    temporal_trends: Dict[int, Dict[str, float]]
    context_influence: Dict[str, Dict[str, float]]
    decision_outcomes: Dict[str, RewardStats]


@dataclass
class StepActionData:
    """Data about actions in a specific simulation step.

    Attributes
    ----------
    step_summary : Dict[str, int]
        Summary statistics for the step
    action_statistics : Dict[str, Dict[str, float]]
        Statistics about different action types
    resource_metrics : Dict[str, float]
        Resource-related metrics
    interaction_network : Dict[str, Any]
        Network of agent interactions
    performance_metrics : Dict[str, float]
        Performance-related metrics
    detailed_actions : List[Dict[str, Any]]
        Detailed list of all actions
    """

    step_summary: Dict[str, int]
    action_statistics: Dict[str, Dict[str, float]]
    resource_metrics: Dict[str, float]
    interaction_network: Dict[str, Any]
    performance_metrics: Dict[str, float]
    detailed_actions: List[Dict[str, Any]]


@dataclass
class AgentStateData:
    """Comprehensive data about an agent's state.

    Attributes
    ----------
    basic_info : Dict[str, Any]
        Basic agent information
    genetic_info : Dict[str, Any]
        Genetic and hereditary information
    current_state : Dict[str, Any]
        Current agent state
    historical_metrics : Dict[str, float]
        Historical performance metrics
    action_history : Dict[str, Dict[str, float]]
        History of agent actions
    health_incidents : List[Dict[str, Any]]
        Record of health-related incidents
    """

    basic_info: Dict[str, Any]
    genetic_info: Dict[str, Any]
    current_state: Dict[str, Any]
    historical_metrics: Dict[str, float]
    action_history: Dict[str, Dict[str, float]]
    health_incidents: List[Dict[str, Any]]


@dataclass
class AgentStatesData:
    """Data representing agent states from database queries.

    Attributes
    ----------
    agent_states : List[Tuple]
        List of tuples containing:
        - step_number: int
        - agent_id: int
        - agent_type: str
        - position_x: float
        - position_y: float
        - resource_level: float
        - current_health: float
        - is_defending: bool
    """
    agent_states: List[Tuple[int, int, str, float, float, float, float, bool]]


@dataclass
class PopulationMetrics:
    """Basic population count metrics.

    Attributes
    ----------
    total_agents : int
        Total number of agents in the population
    system_agents : int
        Number of system agents
    independent_agents : int
        Number of independent agents
    control_agents : int
        Number of control agents
    """

    total_agents: int
    system_agents: int
    independent_agents: int
    control_agents: int


@dataclass
class PopulationStatistics:
    """Combined population statistics.

    Attributes
    ----------
    population_metrics : PopulationMetrics
        Basic population count metrics
    population_variance : PopulationVariance
        Statistical variance metrics for the population
    """

    population_metrics: PopulationMetrics
    population_variance: PopulationVariance


@dataclass
class LearningStatistics:
    """Statistics related to learning performance.

    Attributes
    ----------
    learning_progress : LearningProgress
        Overall learning and adaptation statistics
    module_performance : Dict[str, ModulePerformance]
        Performance metrics for individual learning modules
    """

    learning_progress: LearningProgress
    module_performance: Dict[str, ModulePerformance]


@dataclass
class AdvancedStatistics:
    """Comprehensive simulation statistics.

    Attributes
    ----------
    population_metrics : PopulationStatistics
        Detailed population statistics
    interaction_metrics : InteractionPattern
        Statistics about agent interactions
    resource_metrics : ResourceMetrics
        Resource distribution and efficiency metrics
    agent_type_distribution : AgentDistribution
        Distribution of different agent types
    """

    population_metrics: PopulationStatistics
    interaction_metrics: InteractionPattern
    resource_metrics: ResourceMetrics
    agent_type_distribution: AgentDistribution


@dataclass
class InteractionMetrics:
    """Detailed metrics about agent interactions.

    Attributes
    ----------
    total_actions : int
        Total number of interaction actions
    conflict_rate : float
        Rate of conflict interactions
    cooperation_rate : float
        Rate of cooperative interactions
    reproduction_rate : float
        Rate of reproduction events
    interaction_density : float
        Density of interactions in the population
    avg_reward_conflict : float
        Average reward from conflict interactions
    avg_reward_coop : float
        Average reward from cooperative interactions
    interaction_success : float
        Overall success rate of interactions
    """

    total_actions: int
    conflict_rate: float
    cooperation_rate: float
    reproduction_rate: float
    interaction_density: float
    avg_reward_conflict: float
    avg_reward_coop: float
    interaction_success: float


@dataclass
class ActionMetrics:
    """Metrics for a specific action type.
    
    Attributes
    ----------
    action_type : str
        The type of action
    decision_count : int
        Number of times this action was taken
    avg_reward : float
        Average reward received for this action
    min_reward : float
        Minimum reward received for this action
    max_reward : float
        Maximum reward received for this action
    """
    action_type: str
    decision_count: int
    avg_reward: float
    min_reward: float
    max_reward: float


@dataclass
class DecisionPatternStats:
    """Statistics for a single decision/action type.

    Attributes
    ----------
    count : int
        Number of times this action was taken
    frequency : float
        Frequency of this action relative to total decisions
    reward_stats : Dict[str, float]
        Statistics about rewards for this action type including:
        - average: Mean reward
        - stddev: Standard deviation of rewards
        - min: Minimum reward received
        - max: Maximum reward received
    """
    count: int
    frequency: float
    reward_stats: Dict[str, float]


@dataclass
class SequencePattern:
    """Statistics about action sequences.

    Attributes
    ----------
    count : int
        Number of times this sequence occurred
    probability : float
        Probability of this sequence occurring
    """
    count: int
    probability: float


@dataclass
class ResourceImpact:
    """Impact of an action on resources.

    Attributes
    ----------
    avg_resources_before : float
        Average resources before taking the action
    avg_resource_change : float
        Average change in resources from the action
    resource_efficiency : float
        Efficiency of resource usage for this action
    """
    avg_resources_before: float
    avg_resource_change: float 
    resource_efficiency: float


@dataclass
class TimePattern:
    """Temporal patterns of an action.

    Attributes
    ----------
    time_distribution : List[int]
        Distribution of action counts over time periods
    reward_progression : List[float]
        Progression of rewards over time periods
    """
    time_distribution: List[int]
    reward_progression: List[float]


@dataclass
class InteractionStats:
    """Statistics about agent interactions.

    Attributes
    ----------
    interaction_rate : float
        Rate of interactive vs solo actions
    solo_performance : float
        Average reward for solo actions
    interaction_performance : float
        Average reward for interactive actions
    """
    interaction_rate: float
    solo_performance: float
    interaction_performance: float


@dataclass
class DecisionSummary:
    """Summary statistics about decision making.

    Attributes
    ----------
    total_decisions : int
        Total number of decisions made
    unique_actions : int
        Number of unique action types used
    most_frequent : Optional[str]
        Most frequently taken action
    most_rewarding : Optional[str]
        Action with highest average reward
    action_diversity : float
        Shannon entropy of action distribution
    """
    total_decisions: int
    unique_actions: int
    most_frequent: Optional[str]
    most_rewarding: Optional[str]
    action_diversity: float


@dataclass
class DecisionPatterns:
    """Comprehensive analysis of decision-making patterns.

    Attributes
    ----------
    decision_patterns : Dict[str, DecisionPatternStats]
        Statistics for each action type
    sequence_analysis : Dict[str, SequencePattern]
        Analysis of action sequences
    resource_impact : Dict[str, ResourceImpact]
        Resource impact of different actions
    temporal_patterns : Dict[str, TimePattern]
        Temporal patterns of actions
    interaction_analysis : Dict[str, InteractionStats]
        Analysis of interactive behaviors
    decision_summary : DecisionSummary
        Overall decision-making summary
    """
    decision_patterns: Dict[str, DecisionPatternStats]
    sequence_analysis: Dict[str, SequencePattern]
    resource_impact: Dict[str, ResourceImpact]
    temporal_patterns: Dict[str, TimePattern]
    interaction_analysis: Dict[str, InteractionStats]
    decision_summary: DecisionSummary
