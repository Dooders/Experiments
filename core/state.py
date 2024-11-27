from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional

import numpy as np
import torch
from pydantic import BaseModel, Field, validator

if TYPE_CHECKING:
    from actions.move import MoveModule
    from agents.system_agent import SystemAgent
    from core.environment import Environment


class BaseState(BaseModel):
    """Base class for all state representations in the simulation.

    This class provides common functionality and validation for state objects.
    All state values should be normalized to range [0,1] for stable learning
    and consistent processing across different state implementations.

    Attributes:
        DIMENSIONS (ClassVar[int]): Number of dimensions in the state vector

    Methods:
        to_tensor: Convert state to tensor format for neural network input
        to_dict: Convert state to dictionary representation
        validate_normalized: Ensure values are properly normalized
    """

    class Config:
        """Pydantic configuration.

        Attributes:
            validate_assignment: Validate values when attributes are assigned
            frozen: Make the state immutable after creation
            arbitrary_types_allowed: Allow custom types like numpy arrays
        """

        validate_assignment = True
        frozen = True
        arbitrary_types_allowed = True

    @validator("*")
    def validate_normalized(cls, v: float) -> float:
        """Validate that all state values are normalized between 0 and 1.

        Args:
            v (float): Value to validate

        Returns:
            float: The validated value

        Raises:
            ValueError: If value is not between 0 and 1
        """
        if not 0 <= v <= 1:
            raise ValueError(f"Value {v} must be normalized between 0 and 1")
        return v

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert state to tensor format for neural network input.

        This method should be implemented by subclasses to define how
        their specific state attributes are converted to a tensor.

        Args:
            device (torch.device): Device to place tensor on (CPU/GPU)

        Returns:
            torch.Tensor: State represented as a tensor
        """
        raise NotImplementedError("Subclasses must implement to_tensor")

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary with human-readable keys.

        This method should be implemented by subclasses to provide
        a meaningful dictionary representation of their state.

        Returns:
            Dict[str, Any]: Dictionary representation of state
        """
        raise NotImplementedError("Subclasses must implement to_dict")


class AgentState(BaseState):
    """State representation for agent decision making.

    Represents an agent's state in the environment using normalized values
    for stable learning. All values are constrained to the range [0,1].

    Attributes:
        normalized_distance (float): Distance to nearest resource
        normalized_angle (float): Angle to nearest resource
        normalized_resource_level (float): Agent's current resource amount
        normalized_target_amount (float): Amount in nearest resource
        RESOURCE_NORMALIZER (ClassVar[float]): Constant for resource normalization
        ANGLE_NORMALIZER (ClassVar[float]): Constant for angle normalization
        DIMENSIONS (ClassVar[int]): Number of dimensions in state vector

    Example:
        >>> state = AgentState(
        ...     normalized_distance=0.5,
        ...     normalized_angle=0.25,
        ...     normalized_resource_level=0.8,
        ...     normalized_target_amount=0.6
        ... )
        >>> tensor = state.to_tensor(device)
        >>> state_dict = state.to_dict()
    """

    normalized_distance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Distance to nearest resource, normalized by environment diagonal. "
        "0 = at resource, 1 = maximum possible distance",
    )

    normalized_angle: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Angle to nearest resource, normalized by π. "
        "0 = -π radians, 0.5 = 0 radians, 1 = π radians",
    )

    normalized_resource_level: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent's current resource amount, normalized by max capacity (20). "
        "0 = empty, 1 = full capacity",
    )

    normalized_target_amount: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Amount in nearest resource, normalized by max amount (20). "
        "0 = depleted, 1 = full capacity",
    )

    # Class constants
    RESOURCE_NORMALIZER: ClassVar[float] = 20.0  # Maximum resource capacity
    ANGLE_NORMALIZER: ClassVar[float] = np.pi  # Used for angle normalization
    DIMENSIONS: ClassVar[int] = 4  # State vector dimensionality

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert state to tensor format for neural network input.

        Converts the normalized state values into a tensor suitable for
        neural network processing. The tensor contains all state dimensions
        in a consistent order.

        Args:
            device (torch.device): Device to place tensor on (CPU/GPU)

        Returns:
            torch.Tensor: State tensor of shape (4,) containing:
                [normalized_distance,
                 normalized_angle,
                 normalized_resource_level,
                 normalized_target_amount]

        Example:
            >>> state = AgentState(...)
            >>> device = torch.device("cuda")
            >>> tensor = state.to_tensor(device)
            >>> print(tensor.shape)
            torch.Size([4])
        """
        return torch.FloatTensor(
            [
                self.normalized_distance,
                self.normalized_angle,
                self.normalized_resource_level,
                self.normalized_target_amount,
            ]
        ).to(device)

    @classmethod
    def from_raw_values(
        cls,
        distance: float,
        angle: float,
        resource_level: float,
        target_amount: float,
        env_diagonal: float,
    ) -> "AgentState":
        """Create a normalized state from raw (unnormalized) values.

        Args:
            distance (float): Raw distance to resource
            angle (float): Raw angle in radians (-π to π)
            resource_level (float): Raw resource amount (0 to RESOURCE_NORMALIZER)
            target_amount (float): Raw target resource amount (0 to RESOURCE_NORMALIZER)
            env_diagonal (float): Environment diagonal length for distance normalization

        Returns:
            AgentState: Normalized state instance
        """
        # Clamp values to valid ranges before normalization
        resource_level = max(0.0, min(resource_level, cls.RESOURCE_NORMALIZER))
        target_amount = max(0.0, min(target_amount, cls.RESOURCE_NORMALIZER))
        distance = max(0.0, min(distance, env_diagonal))

        return cls(
            normalized_distance=distance / env_diagonal,
            normalized_angle=(angle + np.pi) / (2 * np.pi),
            normalized_resource_level=resource_level / cls.RESOURCE_NORMALIZER,
            normalized_target_amount=target_amount / cls.RESOURCE_NORMALIZER,
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert state to dictionary with descriptive keys.

        Returns:
            Dict[str, float]: Dictionary containing state values with descriptive keys

        Example:
            >>> state.to_dict()
            {
                'distance_to_resource': 0.5,
                'angle_to_resource': 0.25,
                'agent_resource_level': 0.8,
                'target_resource_amount': 0.6
            }
        """
        return {
            "distance_to_resource": self.normalized_distance,
            "angle_to_resource": self.normalized_angle,
            "agent_resource_level": self.normalized_resource_level,
            "target_resource_amount": self.normalized_target_amount,
        }

    @property
    def raw_angle(self) -> float:
        """Convert normalized angle back to radians.

        Returns:
            float: Angle in radians (-π to π)
        """
        return (self.normalized_angle * 2 * np.pi) - np.pi

    def __eq__(self, other: object) -> bool:
        """Compare two states for equality.

        Parameters
        ----------
        other : object
            Other state to compare with

        Returns
        -------
        bool
            True if states are equal, False otherwise
        """
        if not isinstance(other, AgentState):
            return NotImplemented

        return (
            self.normalized_distance == other.normalized_distance
            and self.normalized_angle == other.normalized_angle
            and self.normalized_resource_level == other.normalized_resource_level
            and self.normalized_target_amount == other.normalized_target_amount
        )

    def to_array(self) -> np.ndarray:
        """Convert agent state to numpy array for neural network input."""
        return np.array([
            self.normalized_distance,
            self.normalized_angle, 
            self.normalized_resource_level,
            self.normalized_target_amount
        ], dtype=np.float32)


class EnvironmentState(BaseState):
    """State representation for the simulation environment.

    Captures the overall state of the environment including resource distribution,
    agent populations, and global metrics. All values are normalized to [0,1].

    Attributes:
        normalized_resource_density (float): Density of resources in environment
        normalized_agent_density (float): Density of agents in environment
        normalized_system_ratio (float): Ratio of system agents to total agents
        normalized_resource_availability (float): Average resource amount availability
        normalized_time (float): Current simulation time relative to max steps
        DIMENSIONS (ClassVar[int]): Number of dimensions in state vector

    Example:
        >>> state = EnvironmentState(
        ...     normalized_resource_density=0.4,
        ...     normalized_agent_density=0.3,
        ...     normalized_system_ratio=0.6,
        ...     normalized_resource_availability=0.7,
        ...     normalized_time=0.5
        ... )
    """

    normalized_resource_density: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Density of resources relative to environment area. "
        "0 = no resources, 1 = maximum expected density",
    )

    normalized_agent_density: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Density of agents relative to environment area. "
        "0 = no agents, 1 = at population capacity",
    )

    normalized_system_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio of system agents to total agents. "
        "0 = all independent, 1 = all system agents",
    )

    normalized_resource_availability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average resource amount across all resources. "
        "0 = all depleted, 1 = all at maximum capacity",
    )

    normalized_time: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Current simulation time normalized by maximum steps. "
        "0 = start, 1 = end of simulation",
    )

    # Class constants
    DIMENSIONS: ClassVar[int] = 5
    MAX_EXPECTED_RESOURCES: ClassVar[int] = 100  # Adjust based on your simulation
    MAX_STEPS: ClassVar[int] = 1000  # Maximum simulation steps

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert environment state to tensor format.

        Args:
            device (torch.device): Device to place tensor on (CPU/GPU)

        Returns:
            torch.Tensor: 1D tensor of shape (DIMENSIONS,) containing state values
        """
        return torch.FloatTensor(
            [
                self.normalized_resource_density,
                self.normalized_agent_density,
                self.normalized_system_ratio,
                self.normalized_resource_availability,
                self.normalized_time,
            ]
        ).to(device)

    @classmethod
    def from_environment(cls, env: "Environment") -> "EnvironmentState":
        """Create a normalized state from an Environment instance.

        Args:
            env (Environment): Environment instance to create state from

        Returns:
            EnvironmentState: Normalized state representation

        Example:
            >>> state = EnvironmentState.from_environment(env)
        """
        # Calculate environment area
        env_area = env.width * env.height

        # Calculate densities
        resource_density = len(env.resources) / cls.MAX_EXPECTED_RESOURCES

        alive_agents = [a for a in env.agents if a.alive]
        agent_density = len(alive_agents) / env.config.max_population

        # Calculate system agent ratio
        system_agents = [a for a in alive_agents if isinstance(a, SystemAgent)]
        system_ratio = len(system_agents) / len(alive_agents) if alive_agents else 0.0

        # Calculate resource availability
        max_possible = env.max_resource or env.config.max_resource_amount
        avg_resource = (
            sum(r.amount for r in env.resources) / (len(env.resources) * max_possible)
            if env.resources
            else 0.0
        )

        return cls(
            normalized_resource_density=resource_density,
            normalized_agent_density=agent_density,
            normalized_system_ratio=system_ratio,
            normalized_resource_availability=avg_resource,
            normalized_time=env.time / cls.MAX_STEPS,
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert state to dictionary with descriptive keys.

        Returns:
            Dict[str, float]: Dictionary containing state values

        Example:
            >>> state.to_dict()
            {
                'resource_density': 0.4,
                'agent_density': 0.3,
                'system_agent_ratio': 0.6,
                'resource_availability': 0.7,
                'simulation_progress': 0.5
            }
        """
        return {
            "resource_density": self.normalized_resource_density,
            "agent_density": self.normalized_agent_density,
            "system_agent_ratio": self.normalized_system_ratio,
            "resource_availability": self.normalized_resource_availability,
            "simulation_progress": self.normalized_time,
        }


class ModelState(BaseModel):
    """State representation for machine learning models in the simulation.

    Captures the current state of a model including its learning parameters,
    performance metrics, and architecture information in their raw form.

    Attributes:
        learning_rate (float): Current learning rate
        epsilon (float): Current exploration rate
        latest_loss (Optional[float]): Most recent training loss
        latest_reward (Optional[float]): Most recent reward
        memory_size (int): Current number of experiences in memory
        memory_capacity (int): Maximum memory capacity
        steps (int): Total training steps taken
        architecture (Dict[str, Any]): Network architecture information
        training_metrics (Dict[str, float]): Recent training performance metrics

    Example:
        >>> state = ModelState.from_move_module(agent.move_module)
        >>> print(state.training_metrics['avg_reward'])
    """

    learning_rate: float = Field(
        ..., description="Current learning rate used by optimizer"
    )

    epsilon: float = Field(..., description="Current exploration rate (epsilon)")

    latest_loss: Optional[float] = Field(
        None, description="Most recent training loss value"
    )

    latest_reward: Optional[float] = Field(
        None, description="Most recent reward received"
    )

    memory_size: int = Field(..., description="Current number of experiences in memory")

    memory_capacity: int = Field(
        ..., description="Maximum capacity of experience memory"
    )

    steps: int = Field(..., description="Total number of training steps taken")

    architecture: Dict[str, Any] = Field(
        ..., description="Summary of network architecture including layer sizes"
    )

    training_metrics: Dict[str, float] = Field(
        ..., description="Recent training performance metrics"
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        frozen = True

    @classmethod
    def from_move_module(cls, move_module: "MoveModule") -> "ModelState":
        """Create a state representation from a MoveModule instance.

        Args:
            move_module (MoveModule): Move module instance to create state from

        Returns:
            ModelState: Current state of the move module

        Example:
            >>> state = ModelState.from_move_module(agent.move_module)
        """
        # Get architecture summary
        architecture = {
            "input_dim": move_module.q_network.network[0].in_features,
            "hidden_sizes": [
                layer.out_features
                for layer in move_module.q_network.network
                if isinstance(layer, torch.nn.Linear)
            ][:-1],
            "output_dim": move_module.q_network.network[-1].out_features,
        }

        # Get recent metrics
        recent_losses = [
            loss for loss in move_module.losses[-1000:] if loss is not None
        ]
        recent_rewards = move_module.episode_rewards[-1000:]

        training_metrics = {
            "avg_loss": (
                sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
            ),
            "avg_reward": (
                sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
            ),
            "min_reward": min(recent_rewards) if recent_rewards else 0.0,
            "max_reward": max(recent_rewards) if recent_rewards else 0.0,
            "std_reward": float(np.std(recent_rewards)) if recent_rewards else 0.0,
        }

        return cls(
            learning_rate=move_module.optimizer.param_groups[0]["lr"],
            epsilon=move_module.epsilon,
            latest_loss=recent_losses[-1] if recent_losses else None,
            latest_reward=recent_rewards[-1] if recent_rewards else None,
            memory_size=len(move_module.memory),
            memory_capacity=move_module.memory.maxlen,
            steps=move_module.steps,
            architecture=architecture,
            training_metrics=training_metrics,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing all state values

        Example:
            >>> state.to_dict()
            {
                'learning_rate': 0.001,
                'epsilon': 0.3,
                'latest_loss': 0.5,
                'latest_reward': 1.2,
                'memory_usage': {'current': 1000, 'capacity': 10000},
                'steps': 5000,
                'architecture': {'input_dim': 4, 'hidden_sizes': [64, 64], 'output_dim': 4},
                'metrics': {'avg_loss': 0.4, 'avg_reward': 1.1, ...}
            }
        """
        return {
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "latest_loss": self.latest_loss,
            "latest_reward": self.latest_reward,
            "memory_usage": {
                "current": self.memory_size,
                "capacity": self.memory_capacity,
            },
            "steps": self.steps,
            "architecture": self.architecture,
            "metrics": self.training_metrics,
        }

    def __str__(self) -> str:
        """Human-readable string representation of model state.

        Returns:
            str: Formatted string with key model information
        """
        return (
            f"ModelState(lr={self.learning_rate:.6f}, "
            f"ε={self.epsilon:.3f}, "
            f"loss={self.latest_loss:.3f if self.latest_loss else 'None'}, "
            f"reward={self.latest_reward:.3f if self.latest_reward else 'None'}, "
            f"memory={self.memory_size}/{self.memory_capacity}, "
            f"steps={self.steps})"
        )


class SimulationState(BaseState):
    """State representation for the overall simulation.

    Captures the current state of the entire simulation including time progression,
    population metrics, resource metrics, and performance indicators. All values 
    are normalized to [0,1] for consistency with other state representations.

    Attributes:
        normalized_time_progress (float): Current simulation progress
        normalized_population_size (float): Current total population relative to capacity
        normalized_survival_rate (float): Portion of original agents still alive
        normalized_resource_efficiency (float): Resource utilization effectiveness
        normalized_system_performance (float): System agents' performance metric
        DIMENSIONS (ClassVar[int]): Number of dimensions in state vector
        MAX_POPULATION (ClassVar[int]): Maximum expected population for normalization
        
    Example:
        >>> state = SimulationState(
        ...     normalized_time_progress=0.5,
        ...     normalized_population_size=0.7,
        ...     normalized_survival_rate=0.8,
        ...     normalized_resource_efficiency=0.6,
        ...     normalized_system_performance=0.75
        ... )
    """

    normalized_time_progress: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Current simulation step relative to total steps. "
        "0 = start, 1 = completion"
    )

    normalized_population_size: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Current population relative to maximum capacity. "
        "0 = empty, 1 = at capacity"
    )

    normalized_survival_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Portion of original agents still alive. "
        "0 = none survived, 1 = all survived"
    )

    normalized_resource_efficiency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Resource utilization effectiveness. "
        "0 = inefficient, 1 = optimal usage"
    )

    normalized_system_performance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="System agents' performance metric. "
        "0 = poor performance, 1 = optimal performance"
    )

    # Class constants
    DIMENSIONS: ClassVar[int] = 5
    MAX_POPULATION: ClassVar[int] = 1000  # Adjust based on simulation needs

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert simulation state to tensor format.

        Args:
            device (torch.device): Device to place tensor on (CPU/GPU)

        Returns:
            torch.Tensor: 1D tensor of shape (DIMENSIONS,) containing state values
        """
        return torch.FloatTensor([
            self.normalized_time_progress,
            self.normalized_population_size,
            self.normalized_survival_rate,
            self.normalized_resource_efficiency,
            self.normalized_system_performance
        ]).to(device)

    def to_dict(self) -> Dict[str, float]:
        """Convert state to dictionary with descriptive keys.

        Returns:
            Dict[str, float]: Dictionary containing state values

        Example:
            >>> state.to_dict()
            {
                'time_progress': 0.5,
                'population_size': 0.7,
                'survival_rate': 0.8,
                'resource_efficiency': 0.6,
                'system_performance': 0.75
            }
        """
        return {
            'time_progress': self.normalized_time_progress,
            'population_size': self.normalized_population_size,
            'survival_rate': self.normalized_survival_rate,
            'resource_efficiency': self.normalized_resource_efficiency,
            'system_performance': self.normalized_system_performance
        }

    @classmethod
    def from_environment(cls, environment: "Environment", num_steps: int) -> "SimulationState":
        """Create a SimulationState instance from current environment state.

        Args:
            environment (Environment): Current simulation environment
            num_steps (int): Total number of simulation steps

        Returns:
            SimulationState: Current state of the simulation
        """
        # Calculate normalized time progress
        time_progress = environment.time / num_steps

        # Calculate population metrics
        current_population = len([a for a in environment.agents if a.alive])
        initial_population = environment.initial_agent_count
        max_population = cls.MAX_POPULATION

        # Calculate survival rate (capped at 1.0 to handle reproduction)
        survival_rate = min(
            current_population / initial_population if initial_population > 0 else 0.0,
            1.0
        )

        # Calculate resource efficiency
        total_resources = sum(resource.amount for resource in environment.resources)
        max_resources = environment.config.max_resource_amount * len(environment.resources)
        resource_efficiency = (
            total_resources / max_resources if max_resources > 0 else 0.0
        )

        # Since we're not using system agents yet, set performance to 0
        system_performance = 0.0

        # Add clamping to ensure normalized_resource_efficiency doesn't exceed 1.0
        normalized_resource_efficiency = min(resource_efficiency, 1.0)

        return cls(
            normalized_time_progress=time_progress,
            normalized_population_size=min(current_population / max_population, 1.0),
            normalized_survival_rate=survival_rate,
            normalized_resource_efficiency=normalized_resource_efficiency,
            normalized_system_performance=system_performance
        )

    def get_agent_genealogy(self) -> Dict[str, Any]:
        """Get genealogical information about the current agent population.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - max_generation: Highest generation number reached
            - generation_counts: Count of agents per generation
            - lineage_lengths: Distribution of lineage lengths
            - survival_rates: Survival rates by generation
        """
        return {
            'normalized_max_generation': self.normalized_time_progress,
            'generation_distribution': self.normalized_population_size,
            'lineage_survival': self.normalized_survival_rate,
            'evolutionary_progress': self.normalized_system_performance
        }
