# Repository Pattern Documentation

## Introduction

The Repository Pattern is a design pattern that provides a way to manage data access and persistence in a centralized and abstracted manner. It helps to separate the data access logic from the business logic, making the code more maintainable, testable, and reusable.

## Benefits

- **Abstraction of Data Access**: Centralizes data access logic, simplifying modifications or replacements of data sources (e.g., switching from SQL to NoSQL).
- **Reusability**: Common CRUD operations can be shared across multiple entities.
- **Testability**: Facilitates mocking or stubbing repositories during testing for better isolation.
- **Separation of Concerns**: Keeps business logic separate from database logic, improving maintainability.

## Costs

- **Increased Complexity**: Adds an additional layer to the architecture, which might be unnecessary for simpler applications.
- **Overhead**: May introduce more code and classes for projects with minimal data access logic.
- **Potential Duplication**: Specific repositories might duplicate logic if proper abstraction isn't enforced.

## Implementation

### Base Repository Class

The base repository class provides common CRUD operations and session management. It is designed to be reusable and extendable by specific repositories for different domain entities.

```python
from typing import Type, TypeVar, Generic, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from database.database import SimulationDatabase

T = TypeVar('T')

class BaseRepository(Generic[T]):
    """Base repository class with common CRUD operations and session management."""

    def __init__(self, db: SimulationDatabase, model: Type[T]):
        """Initialize the repository with a database connection and model type.

        Parameters
        ----------
        db : SimulationDatabase
            The database instance to use for executing queries
        model : Type[T]
            The model class representing the database table
        """
        self.db = db
        self.model = model

    def add(self, entity: T) -> None:
        """Add a new entity to the database.

        Parameters
        ----------
        entity : T
            The entity to add
        """
        def _add(session: Session):
            session.add(entity)

        self._execute_in_transaction(_add)

    def get_by_id(self, entity_id: int) -> Optional[T]:
        """Retrieve an entity by its ID.

        Parameters
        ----------
        entity_id : int
            The ID of the entity to retrieve

        Returns
        -------
        Optional[T]
            The retrieved entity, or None if not found
        """
        def _get_by_id(session: Session) -> Optional[T]:
            return session.query(self.model).get(entity_id)

        return self._execute_in_transaction(_get_by_id)

    def update(self, entity: T) -> None:
        """Update an existing entity in the database.

        Parameters
        ----------
        entity : T
            The entity to update
        """
        def _update(session: Session):
            session.merge(entity)

        self._execute_in_transaction(_update)

    def delete(self, entity: T) -> None:
        """Delete an entity from the database.

        Parameters
        ----------
        entity : T
            The entity to delete
        """
        def _delete(session: Session):
            session.delete(entity)

        self._execute_in_transaction(_delete)

    def _execute_in_transaction(self, func: callable) -> Any:
        """Execute database operations within a transaction with error handling.

        Parameters
        ----------
        func : callable
            Function that takes a session argument and performs database operations

        Returns
        -------
        Any
            Result of the executed function

        Raises
        ------
        SQLAlchemyError
            For database-related errors
        """
        session = self.db.Session()
        try:
            result = func(session)
            session.commit()
            return result
        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            self.db.Session.remove()
```

### Specific Repositories

Specific repositories extend the base repository class and provide additional methods for domain-specific data access and query logic.

#### Agent Repository

The `AgentRepository` class handles agent-related data operations and extends the base repository class.

```python
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
```

#### Resource Repository

The `ResourceRepository` class handles resource-related data operations and extends the base repository class.

```python
from typing import List

from sqlalchemy import func

from database.base_repository import BaseRepository
from database.data_types import (
    ConsumptionStats,
    ResourceAnalysis,
    ResourceDistributionStep,
    ResourceEfficiencyMetrics,
    ResourceHotspot,
)
from database.models import ResourceState, SimulationStep
from database.utilities import execute_query


class ResourceRepository(BaseRepository[ResourceState]):
    """Handles retrieval and analysis of resource-related data from simulation database.

    Provides methods for analyzing resource dynamics including distribution patterns,
    consumption statistics, concentration hotspots, and efficiency metrics across
    simulation timesteps.

    Methods
    -------
    resource_distribution()
        Retrieves time series of resource distribution metrics
    consumption_patterns()
        Calculates aggregate resource consumption statistics
    resource_hotspots()
        Identifies areas of high resource concentration
    efficiency_metrics()
        Computes resource utilization and efficiency measures
    execute()
        Performs comprehensive resource analysis
    """

    def __init__(self, database):
        """Initialize the ResourceRepository.

        Parameters
        ----------
        database : SimulationDatabase
            Database connection instance used to execute queries
        """
        super().__init__(database, ResourceState)

    @execute_query
    def resource_distribution(self, session) -> List[ResourceDistributionStep]:
        """Retrieve time series of resource distribution metrics.

        Queries database for spatial resource distribution patterns across simulation
        timesteps, including total quantities, densities, and distribution entropy.

        Returns
        -------
        List[ResourceDistributionStep]
            Sequence of distribution metrics per timestep:
            - step: Simulation timestep number
            - total_resources: Total quantity of resources present
            - average_per_cell: Mean resource density per grid cell
            - distribution_entropy: Shannon entropy of resource distribution
        """
        distribution_data = (
            session.query(
                SimulationStep.step_number,
                SimulationStep.total_resources,
                SimulationStep.average_agent_resources,
                SimulationStep.resource_distribution_entropy,
            )
            .order_by(SimulationStep.step_number)
            .all()
        )

        return [
            ResourceDistributionStep(
                step=step,
                total_resources=total,
                average_per_cell=density,
                distribution_entropy=entropy,
            )
            for step, total, density, entropy in distribution_data
        ]

    @execute_query
    def consumption_patterns(self, session) -> ConsumptionStats:
        """Calculate aggregate resource consumption statistics.

        Analyzes consumption rates and variability across the entire simulation
        timeline, including totals, averages, peaks and variance measures.

        Returns
        -------
        ConsumptionStats
            Statistical measures of resource consumption:
            - total_consumed: Total resources consumed across all steps
            - avg_consumption_rate: Mean consumption rate per timestep
            - peak_consumption: Maximum single-step consumption
            - consumption_variance: Variance in consumption rates
        """
        # First get basic stats
        basic_stats = session.query(
            func.sum(SimulationStep.resources_consumed).label("total"),
            func.avg(SimulationStep.resources_consumed).label("average"),
            func.max(SimulationStep.resources_consumed).label("peak"),
        ).first()

        # Calculate variance manually: VAR = E[(X - μ)²]
        avg_consumption = basic_stats[1] or 0
        variance_calc = session.query(
            func.avg(
                (SimulationStep.resources_consumed - avg_consumption)
                * (SimulationStep.resources_consumed - avg_consumption)
            ).label("variance")
        ).first()

        return ConsumptionStats(
            total_consumed=float(basic_stats[0] or 0),
            avg_consumption_rate=float(basic_stats[1] or 0),
            peak_consumption=float(basic_stats[2] or 0),
            consumption_variance=float(variance_calc[0] or 0),
        )

    @execute_query
    def resource_hotspots(self, session) -> List[ResourceHotspot]:
        """Identify areas of high resource concentration.

        Analyzes spatial resource distribution to locate and rank areas with
        above-average resource concentrations.

        Returns
        -------
        List[ResourceHotspot]
            Resource hotspots sorted by concentration (highest first):
            - position_x: X coordinate of hotspot
            - position_y: Y coordinate of hotspot
            - concentration: Average resource amount at location
        """
        hotspot_data = (
            session.query(
                ResourceState.position_x,
                ResourceState.position_y,
                func.avg(ResourceState.amount).label("concentration"),
            )
            .group_by(ResourceState.position_x, ResourceState.position_y)
            .having(func.avg(ResourceState.amount) > 0)
            .order_by(func.avg(ResourceState.amount).desc())
            .all()
        )

        return [
            ResourceHotspot(
                position_x=x,
                position_y=y,
                concentration=concentration,
            )
            for x, y, concentration in hotspot_data
        ]

    @execute_query
    def efficiency_metrics(self, session) -> ResourceEfficiencyMetrics:
        #! Currently not working
        """Calculate resource utilization and efficiency metrics.

        Computes various efficiency measures related to resource distribution,
        consumption, and regeneration patterns.

        Returns
        -------
        ResourceEfficiencyMetrics
            Collection of efficiency metrics:
            - utilization_rate: Resource usage efficiency
            - distribution_efficiency: Spatial distribution effectiveness
            - consumption_efficiency: Resource consumption optimization
            - regeneration_rate: Resource replenishment speed
        """
        metrics = session.query(
            func.avg(SimulationStep.resource_efficiency).label("utilization"),
            func.avg(SimulationStep.distribution_efficiency).label("distribution"),
            func.avg(SimulationStep.consumption_efficiency).label("consumption"),
            func.avg(SimulationStep.regeneration_rate).label("regeneration"),
        ).first()

        return ResourceEfficiencyMetrics(
            utilization_rate=float(metrics[0] or 0),
            distribution_efficiency=float(metrics[1] or 0),
            consumption_efficiency=float(metrics[2] or 0),
            regeneration_rate=float(metrics[3] or 0),
        )

    @execute_query
    def execute(self, session) -> ResourceAnalysis:
        """Perform comprehensive resource analysis.

        Combines distribution, consumption, hotspot and efficiency analyses
        into a complete resource behavior assessment.

        Returns
        -------
        ResourceAnalysis
            Complete resource analysis including:
            - distribution: Time series of distribution metrics
            - consumption: Aggregate consumption statistics
            - hotspots: Areas of high concentration
            - efficiency: Resource efficiency measures
        """
        return ResourceAnalysis(
            distribution=self.resource_distribution(),
            consumption=self.consumption_patterns(),
            hotspots=self.resource_hotspots(),
            efficiency={},
        )
```

#### Learning Repository

The `LearningRepository` class handles learning-related data operations and extends the base repository class.

```python
from typing import Any, Dict, List, Optional

from database.base_repository import BaseRepository
from database.data_types import (
    AgentLearningStats,
    LearningEfficiencyMetrics,
    LearningProgress,
    LearningStatistics,
    ModulePerformance,
)
from database.models import LearningExperience
from database.utilities import execute_query


class LearningRepository(BaseRepository[LearningExperience]):
    """Handles learning-related data retrieval and analysis.

    This class provides methods for analyzing learning experiences, module performance,
    and adaptation patterns throughout the simulation. It interfaces with the database
    to retrieve and aggregate learning metrics.

    Methods
    -------
    learning_progress() -> LearningProgress
        Retrieves time-series data of learning progress
    module_performance() -> Dict[str, ModulePerformance]
        Calculates performance metrics per learning module
    agent_learning_stats(agent_id: Optional[int]) -> Dict[str, AgentLearningStats]
        Analyzes learning statistics for specific or all agents
    learning_efficiency() -> LearningEfficiencyMetrics
        Computes efficiency metrics across learning experiences
    execute() -> LearningStatistics
        Generates comprehensive learning statistics report
    """

    def __init__(self, db):
        """Initialize with database connection.

        Parameters
        ----------
        db : SimulationDatabase
            Database instance to use for queries
        """
        super().__init__(db, LearningExperience)

    @execute_query
    def learning_progress(self, session) -> List[LearningProgress]:
        """Calculate aggregated learning progress metrics over time.

        Retrieves and aggregates learning metrics for each simulation step, including
        rewards earned and action patterns. Each step's data is returned as a separate
        LearningProgress object.

        Parameters
        ----------
        session : Session
            SQLAlchemy session object (automatically injected by execute_query decorator)

        Returns
        -------
        List[LearningProgress]
            List of learning progress metrics per step, where each object contains:
            - step : int
                Step number in the simulation
            - reward : float
                Average reward achieved in this step
            - action_count : int
                Total number of actions taken in this step
            - unique_actions : int
                Number of distinct actions used in this step
        """
        progress_data = (
            session.query(
                LearningExperience.step_number,
                func.avg(LearningExperience.reward).label("avg_reward"),
                func.count(LearningExperience.action_taken).label("action_count"),
                func.count(distinct(LearningExperience.action_taken_mapped)).label(
                    "unique_actions"
                ),
            )
            .group_by(LearningExperience.step_number)
            .order_by(LearningExperience.step_number)
            .all()
        )

        return [
            LearningProgress(
                step=step,
                reward=float(reward or 0),
                action_count=int(count or 0),
                unique_actions=int(unique or 0),
            )
            for step, reward, count, unique in progress_data
        ]

    @execute_query
    def module_performance(self, session) -> Dict[str, ModulePerformance]:
        """Calculate performance metrics for each learning module type.

        Aggregates and analyzes performance data for each unique learning module,
        including rewards, action counts, and action diversity metrics.

        Parameters
        ----------
        session : Session
            SQLAlchemy session object (automatically injected by execute_query decorator)

        Returns
        -------
        Dict[str, ModulePerformance]
            Dictionary mapping module identifiers to their performance metrics, where each
            ModulePerformance contains:
            - module_type : str
                Type of learning module
            - module_id : str
                Unique identifier for the module
            - avg_reward : float
                Average reward achieved by the module
            - total_actions : int
                Total number of actions taken by the module
            - unique_actions : int
                Number of distinct actions used by the module
        """
        module_stats = (
            session.query(
                LearningExperience.module_type,
                LearningExperience.module_id,
                func.avg(LearningExperience.reward).label("avg_reward"),
                func.count(LearningExperience.action_taken).label("total_actions"),
                func.count(distinct(LearningExperience.action_taken_mapped)).label(
                    "unique_actions"
                ),
            )
            .group_by(LearningExperience.module_type, LearningExperience.module_id)
            .all()
        )

        return {
            f"{module_type}": ModulePerformance(
                module_type=module_type,
                module_id=module_id,
                avg_reward=float(avg_reward or 0),
                total_actions=int(total_actions or 0),
                unique_actions=int(unique_actions or 0),
            )
            for module_type, module_id, avg_reward, total_actions, unique_actions in module_stats
        }

    @execute_query
    def agent_learning_stats(
        self, session, agent_id: Optional[int] = None
    ) -> Dict[str, AgentLearningStats]:
        """Get learning statistics for specific agent or all agents.

        Retrieves and analyzes learning performance metrics either for a specific
        agent or aggregated across all agents.

        Parameters
        ----------
        agent_id : Optional[int]
            If provided, limits analysis to specific agent. If None, includes all agents.

        Returns
        -------
        Dict[str, AgentLearningStats]
            Dictionary mapping agent/module combinations to their statistics:
            - reward_mean: Average reward achieved
            - total_actions: Total number of actions taken
            - actions_used: List of unique actions performed
        """
        query = session.query(
            LearningExperience.agent_id,
            LearningExperience.module_type,
            func.avg(LearningExperience.reward).label("reward_mean"),
            func.count(LearningExperience.action_taken).label("total_actions"),
            func.group_concat(distinct(LearningExperience.action_taken_mapped)).label(
                "actions_used"
            ),
        )

        if agent_id is not None:
            query = query.filter(LearningExperience.agent_id == agent_id)

        results = query.group_by(
            LearningExperience.agent_id, LearningExperience.module_type
        ).all()

        return {
            f"{module_type}": AgentLearningStats(
                agent_id=agent_id,
                reward_mean=float(reward_mean or 0),
                total_actions=int(total_actions or 0),
                actions_used=actions_used.split(",") if actions_used else [],
            )
            for agent_id, module_type, reward_mean, total_actions, actions_used in results
        }

    @execute_query
    def learning_efficiency(self, session) -> LearningEfficiencyMetrics:
        """Calculate learning efficiency metrics.

        Computes various efficiency metrics to evaluate the overall learning
        performance and stability of the system.

        Parameters
        ----------
        session : Session
            SQLAlchemy session object (automatically injected by execute_query decorator)

        Returns
        -------
        LearningEfficiencyMetrics
            Object containing efficiency metrics:
            - reward_efficiency : float
                Average reward across all experiences (0.0 to 1.0)
            - action_diversity : float
                Ratio of unique actions to total actions (0.0 to 1.0)
            - learning_stability : float
                Measure of consistency in learning performance (0.0 to 1.0)
        """
        experiences = pd.read_sql(
            session.query(
                LearningExperience.step_number,
                LearningExperience.module_type,
                LearningExperience.reward,
                LearningExperience.action_taken_mapped,
            )
            .order_by(LearningExperience.step_number)
            .statement,
            session.bind,
        )

        if experiences.empty:
            return LearningEfficiencyMetrics(
                reward_efficiency=0.0,
                action_diversity=0.0,
                learning_stability=0.0,
            )

        # Calculate metrics
        reward_efficiency = experiences["reward"].mean()

        # Calculate action diversity (unique actions / total actions)
        total_actions = len(experiences)
        unique_actions = experiences["action_taken_mapped"].nunique()
        action_diversity = unique_actions / total_actions if total_actions > 0 else 0

        # Calculate learning stability (inverse of reward variance)
        reward_variance = experiences.groupby("module_type")["reward"].var().mean()
        learning_stability = 1 / (1 + reward_variance) if reward_variance > 0 else 1.0

        return LearningEfficiencyMetrics(
            reward_efficiency=float(reward_efficiency or 0),
            action_diversity=float(action_diversity or 0),
            learning_stability=float(learning_stability or 0),
        )

    @execute_query
    def execute(self, session) -> LearningStatistics:
        """Generate a comprehensive learning statistics report.

        Combines multiple analysis methods to create a complete picture of
        learning performance, including progress over time, module-specific
        metrics, and efficiency measures.

        Returns
        -------
        LearningStatistics
            Complete learning statistics including:
            - learning_progress: Time series of rewards and losses
            - module_performance: Per-module performance metrics
            - agent_learning_stats: Per-agent learning statistics
            - learning_efficiency: Overall efficiency metrics
        """
        return LearningStatistics(
            learning_progress=self.learning_progress(),
            module_performance=self.module_performance(),
            agent_learning_stats=self.agent_learning_stats(),
            learning_efficiency=self.learning_efficiency(),
        )
```

#### Population Repository

The `PopulationRepository` class handles population-related data operations and extends the base repository class.

```python
from typing import List, Optional

from sqlalchemy import func

from database.base_repository import BaseRepository
from database.data_types import (
    AgentDistribution,
    BasicPopulationStatistics,
    Population,
    PopulationMetrics,
    PopulationStatistics,
    PopulationVariance,
)
from database.models import AgentState, SimulationStep
from database.utilities import execute_query


class PopulationRepository(BaseRepository):
    """Handles retrieval and analysis of population statistics.

    This class encapsulates methods for analyzing population dynamics, resource utilization,
    and agent distributions across the simulation steps. It provides comprehensive statistics
    about agent populations, resource consumption, and survival metrics.

    Methods
    -------
    population_data()
        Retrieves raw population and resource data for each simulation step
    basic_population_statistics(pop_data)
        Calculates fundamental population statistics from raw data
    agent_type_distribution()
        Analyzes the distribution of different agent types
    execute()
        Generates comprehensive population statistics and metrics
    """

    def __init__(self, database):
        """Initialize with database connection.

        Parameters
        ----------
        database : SimulationDatabase
            Database instance to use for queries
        """
        super().__init__(database, SimulationStep)

    @execute_query
    def population_data(self, session) -> List[Population]:
        """Retrieve base population and resource data for each simulation step.

        Queries the database to get step-wise population metrics including total agents,
        resources, and consumption data.

        Returns
        -------
        List[Population]
            List of Population objects containing:
            - step_number : int
                The simulation step number
            - total_agents : int
                Total number of agents in that step
            - total_resources : float
                Total available resources in that step
            - resources_consumed : float
                Total resources consumed by agents in that step
        """
        pop_data = (
            session.query(
                SimulationStep.step_number,
                SimulationStep.total_agents,
                SimulationStep.total_resources,
                func.sum(AgentState.resource_level).label("resources_consumed"),
            )
            .outerjoin(AgentState, AgentState.step_number == SimulationStep.step_number)
            .filter(SimulationStep.total_agents > 0)
            .group_by(SimulationStep.step_number)
            .all()
        )

        return [
            Population(
                step_number=row[0],
                total_agents=row[1],
                total_resources=row[2],
                resources_consumed=row[3],
            )
            for row in pop_data
        ]

    @execute_query
    def basic_population_statistics(
        self, session, pop_data: Optional[List[Population]] = None
    ) -> BasicPopulationStatistics:
        """Calculate basic population statistics from step data.

        Processes raw population data to compute fundamental statistics about
        the population and resource usage.

        Parameters
        ----------
        pop_data : List[Population]
            List of Population objects containing step-wise simulation data

        Returns
        -------
        BasicPopulationStatistics
            Object containing:
            - avg_population : float
                Average population across all steps
            - death_step : int
                Final step number where agents existed
            - peak_population : int
                Maximum population reached
            - resources_consumed : float
                Total resources consumed across all steps
            - resources_available : float
                Total resources available across all steps
            - sum_squared : float
                Sum of squared population counts (for variance calculation)
            - step_count : int
                Total number of steps with active agents
        """
        if not pop_data:
            pop_data = self.population_data()

        # Calculate statistics directly from Population objects
        stats = {
            "avg_population": sum(p.total_agents for p in pop_data) / len(pop_data),
            "death_step": max(p.step_number for p in pop_data),
            "peak_population": max(p.total_agents for p in pop_data),
            "resources_consumed": sum(p.resources_consumed for p in pop_data),
            "resources_available": sum(p.total_resources for p in pop_data),
            "sum_squared": sum(p.total_agents * p.total_agents for p in pop_data),
            "step_count": len(pop_data),
        }

        return BasicPopulationStatistics(
            avg_population=float(stats["avg_population"] or 0),
            death_step=int(stats["death_step"] or 0),
            peak_population=int(stats["peak_population"] or 0),
            resources_consumed=float(stats["resources_consumed"] or 0),
            resources_available=float(stats["resources_available"] or 0),
            sum_squared=float(stats["sum_squared"] or 0),
            step_count=int(stats["step_count"] or 1),
        )

    @execute_query
    def agent_type_distribution(self, session) -> AgentDistribution:
        """Analyze the distribution of different agent types across the simulation.

        Calculates the average number of each agent type (system, independent, and control)
        across all simulation steps.

        Returns
        -------
        AgentDistribution
            Distribution metrics containing:
            - system_agents : float
                Average number of system-controlled agents
            - independent_agents : float
                Average number of independently operating agents
            - control_agents : float
                Average number of control group agents
        """
        type_stats = session.query(
            func.avg(SimulationStep.system_agents).label("avg_system"),
            func.avg(SimulationStep.independent_agents).label("avg_independent"),
            func.avg(SimulationStep.control_agents).label("avg_control"),
        ).first()

        return AgentDistribution(
            system_agents=float(type_stats[0] or 0),
            independent_agents=float(type_stats[1] or 0),
            control_agents=float(type_stats[2] or 0),
        )

    @execute_query
    def execute(self, session) -> PopulationStatistics:
        """Calculate comprehensive population statistics for the entire simulation.

        Combines data from multiple analyses to create a complete statistical overview
        of the simulation's population dynamics, resource usage, and agent behavior.

        Returns
        -------
        PopulationStatistics
            Comprehensive statistics containing:
            - basic_stats : BasicPopulationStatistics
                Fundamental population metrics (avg, peak, steps, etc.)
            - resource_metrics : ResourceMetrics
                Resource consumption and utilization statistics
            - population_variance : PopulationVariance
                Statistical measures of population variation
            - agent_distribution : AgentDistribution
                Distribution of different agent types
            - survival_metrics : SurvivalMetrics
                Population survival rates and average lifespans

        Notes
        -----
        This method aggregates data from multiple queries and calculations to provide
        a complete statistical analysis of the simulation. If no data is available,
        it returns a PopulationStatistics object with zero values.
        """
        # Get base population data
        pop_data = self.population_data()

        # Get basic statistics
        basic_stats = self.basic_population_statistics(pop_data)
        if not basic_stats:
            return PopulationStatistics(
                population_metrics=PopulationMetrics(
                    total_agents=0,
                    system_agents=0,
                    independent_agents=0,
                    control_agents=0,
                ),
                population_variance=PopulationVariance(
                    variance=0.0, standard_deviation=0.0, coefficient_variation=0.0
                ),
            )

        # Calculate variance statistics
        variance = (basic_stats.sum_squared / basic_stats.step_count) - (
            basic_stats.avg_population**2
        )
        std_dev = variance**0.5
        cv = (
            std_dev / basic_stats.avg_population
            if basic_stats.avg_population > 0
            else 0
        )

        # Get agent type distribution
        type_stats = self.agent_type_distribution()

        # Create PopulationMetrics
        population_metrics = PopulationMetrics(
            total_agents=basic_stats.peak_population,
            system_agents=int(type_stats.system_agents),
            independent_agents=int(type_stats.independent_agents),
            control_agents=int(type_stats.control_agents),
        )

        # Create PopulationVariance
        population_variance = PopulationVariance(
            variance=variance, standard_deviation=std_dev, coefficient_variation=cv
        )

        # Return PopulationStatistics with the correct structure
        return PopulationStatistics(
            population_metrics=population_metrics,
            population_variance=population_variance,
        )
```

#### Simulation Repository

The `SimulationRepository` class handles simulation-related data operations and extends the base repository class.

```python
from typing import List

from sqlalchemy.orm import Session

from database.base_repository import BaseRepository
from database.data_types import (
    SimulationResults,
    AgentStateData,
    ResourceStateData,
    SimulationState,
)
from database.models import AgentState, ResourceState, SimulationStep
from database.utilities import execute_query


class SimulationRepository(BaseRepository[SimulationStep]):
    """Handles retrieval and analysis of simulation state data.

    This class provides methods for retrieving and analyzing the state of the simulation,
    including agent states, resource states, and overall simulation metrics.

    Methods
    -------
    agent_states(step_number)
        Retrieves the states of all agents at a specific simulation step
    resource_states(step_number)
        Retrieves the states of all resources at a specific simulation step
    simulation_state(step_number)
        Retrieves the overall simulation state at a specific simulation step
    execute(step_number)
        Combines agent states, resource states, and simulation state into a single result
    """

    def __init__(self, db):
        """Initialize the repository with a database connection."""
        super().__init__(db, SimulationStep)

    @execute_query
    def agent_states(self, session: Session, step_number: int) -> List[AgentStateData]:
        """Retrieve the states of all agents at a specific simulation step."""
        agent_states = (
            session.query(AgentState)
            .filter(AgentState.step_number == step_number)
            .all()
        )

        return [
            AgentStateData(
                agent_id=state.agent_id,
                agent_type=state.agent_type,
                position_x=state.position_x,
                position_y=state.position_y,
                health=state.health,
                resources=state.resources,
                age=state.age,
                is_defending=state.is_defending,
            )
            for state in agent_states
        ]

    @execute_query
    def resource_states(
        self, session: Session, step_number: int
    ) -> List[ResourceStateData]:
        """Retrieve the states of all resources at a specific simulation step."""
        resource_states = (
            session.query(ResourceState)
            .filter(ResourceState.step_number == step_number)
            .all()
        )

        return [
            ResourceStateData(
                resource_id=state.resource_id,
                position_x=state.position_x,
                position_y=state.position_y,
                amount=state.amount,
            )
            for state in resource_states
        ]

    @execute_query
    def simulation_state(self, session: Session, step_number: int) -> SimulationState:
        """Retrieve the overall simulation state at a specific simulation step."""
        simulation_state = (
            session.query(SimulationStep)
            .filter(SimulationStep.step_number == step_number)
            .first()
        )

        return SimulationState(
            step_number=simulation_state.step_number,
            total_agents=simulation_state.total_agents,
            total_resources=simulation_state.total_resources,
            average_agent_health=simulation_state.average_agent_health,
            average_agent_resources=simulation_state.average_agent_resources,
            births=simulation_state.births,
            deaths=simulation_state.deaths,
        )

    @execute_query
    def execute(self, session: Session, step_number: int) -> SimulationResults:
        """Combine agent states, resource states, and simulation state into a single result."""
        agent_states = self.agent_states(step_number)
        resource_states = self.resource_states(step_number)
        simulation_state = self.simulation_state(step_number)

        return SimulationResults(
            agent_states=agent_states,
            resource_states=resource_states,
            simulation_state=simulation_state,
        )
```

## Usage

To use the Repository Pattern in this project, follow these steps:

1. **Create an instance of the database connection**:
   ```python
   from database.database import SimulationDatabase

   db = SimulationDatabase()
   ```

2. **Create an instance of the specific repository**:
   ```python
   from database.agent_repository import AgentRepository

   agent_repository = AgentRepository(db)
   ```

3. **Use the repository methods to perform data operations**:
   ```python
   # Add a new agent
   new_agent = Agent(agent_id=1, agent_type="TypeA", ...)
   agent_repository.add(new_agent)

   # Get an agent by ID
   agent = agent_repository.get_by_id(1)

   # Update an existing agent
   agent.agent_type = "TypeB"
   agent_repository.update(agent)

   # Delete an agent
   agent_repository.delete(agent)

   # Get lifespan statistics
   lifespan_stats = agent_repository.lifespan_statistics()

   # Get survival rates
   survival_rates = agent_repository.survival_rates()
   ```

4. **Run unit tests to confirm repository functionality and mockability**:
   ```python
   import unittest
   from unittest.mock import MagicMock

   class TestAgentRepository(unittest.TestCase):
       def setUp(self):
           self.db = MagicMock()
           self.repository = AgentRepository(self.db)

       def test_add_agent(self):
           agent = Agent(agent_id=1, agent_type="TypeA", ...)
           self.repository.add(agent)
           self.db.Session().add.assert_called_once_with(agent)

       def test_get_by_id(self):
           self.db.Session().query().get.return_value = Agent(agent_id=1, agent_type="TypeA", ...)
           agent = self.repository.get_by_id(1)
           self.assertEqual(agent.agent_id, 1)

       # Add more tests for other methods

   if __name__ == "__main__":
       unittest.main()
   ```

By following these steps, you can effectively use the Repository Pattern to manage data access and persistence in this project.
