import copy
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Tuple

import yaml


@dataclass
class VisualizationConfig:
    canvas_size: Tuple[int, int] = (400, 400)
    padding: int = 20
    background_color: str = "black"
    max_animation_frames: int = 5
    animation_min_delay: int = 50
    max_resource_amount: int = 30
    resource_colors: Dict[str, int] = field(
        default_factory=lambda: {"glow_red": 150, "glow_green": 255, "glow_blue": 50}
    )
    resource_size: int = 2
    agent_radius_scale: int = 2
    birth_radius_scale: int = 4
    death_mark_scale: float = 1.5
    agent_colors: Dict[str, str] = field(
        default_factory=lambda: {"SystemAgent": "blue", "IndependentAgent": "red"}
    )
    min_font_size: int = 10
    font_scale_factor: int = 40
    font_family: str = "arial"
    death_mark_color: List[int] = field(default_factory=lambda: [255, 0, 0])
    birth_mark_color: List[int] = field(default_factory=lambda: [255, 255, 255])
    metric_colors: Dict[str, str] = field(
        default_factory=lambda: {
            "total_agents": "#4a90e2",
            "system_agents": "#50c878",
            "independent_agents": "#e74c3c",
            "total_resources": "#f39c12",
            "average_agent_resources": "#9b59b6",
        }
    )


@dataclass
class SimulationConfig:
    # Environment settings
    width: int = 100
    height: int = 100

    # Agent settings
    system_agents: int = 25
    independent_agents: int = 25
    initial_resource_level: int = 12
    max_population: int = 300
    starvation_threshold: int = 0
    max_starvation_time: int = 15
    offspring_cost: int = 6
    min_reproduction_resources: int = 10
    offspring_initial_resources: int = 5

    # Resource settings
    initial_resources: int = 60
    resource_regen_rate: float = 0.1
    resource_regen_amount: int = 2
    max_resource_amount: int = 30

    # Agent behavior settings
    base_consumption_rate: float = 0.1
    max_movement: int = 8
    gathering_range: int = 20
    max_gather_amount: int = 3
    territory_range: int = 30

    # Learning and Movement Module Parameters
    target_update_freq: int = 100
    memory_size: int = 10000
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    dqn_hidden_size: int = 64
    batch_size: int = 32
    training_frequency: int = 50
    tau: float = 0.005

    # Movement Module Parameters
    move_target_update_freq: int = 100
    move_memory_size: int = 10000
    move_learning_rate: float = 0.001
    move_gamma: float = 0.99
    move_epsilon_start: float = 1.0
    move_epsilon_min: float = 0.01
    move_epsilon_decay: float = 0.995
    move_dqn_hidden_size: int = 64
    move_batch_size: int = 32
    move_reward_history_size: int = 100
    move_epsilon_adapt_threshold: float = 0.1
    move_epsilon_adapt_factor: float = 1.5
    move_min_reward_samples: int = 10
    move_tau: float = 0.005
    move_base_cost: float = -0.1  # Base cost for any movement
    move_resource_approach_reward: float = 0.3  # Reward for moving closer to resources
    move_resource_retreat_penalty: float = (
        -0.2
    )  # Penalty for moving away from resources

    # Attack Module Parameters
    attack_target_update_freq: int = 100
    attack_memory_size: int = 10000
    attack_learning_rate: float = 0.001
    attack_gamma: float = 0.99
    attack_epsilon_start: float = 1.0
    attack_epsilon_min: float = 0.01
    attack_epsilon_decay: float = 0.995
    attack_dqn_hidden_size: int = 64
    attack_batch_size: int = 32
    attack_tau: float = 0.005
    attack_base_cost: float = -0.2
    attack_success_reward: float = 1.0
    attack_failure_penalty: float = -0.3
    attack_defense_threshold: float = 0.3
    attack_defense_boost: float = 2.0

    # Combat Parameters
    max_health: float = 100.0
    attack_range: float = 20.0
    attack_base_damage: float = 10.0
    attack_kill_reward: float = 5.0

    # Action Multipliers (existing ones remain unchanged)
    attack_mult_desperate: float = 1.4
    attack_mult_stable: float = 0.6
    attack_starvation_threshold: float = 0.5

    # Visualization settings (separate config)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # Action probability adjustment parameters
    social_range = 30  # Range for social interactions (share/attack)

    # Movement multipliers
    move_mult_no_resources = 1.5  # Multiplier when no resources nearby

    # Gathering multipliers
    gather_mult_low_resources = 1.5  # Multiplier when resources needed

    # Sharing multipliers
    share_mult_wealthy = 1.3  # Multiplier when agent has excess resources
    share_mult_poor = 0.5  # Multiplier when agent needs resources

    # Attack multipliers
    attack_starvation_threshold = (
        0.5  # Starvation risk threshold for desperate behavior
    )
    attack_mult_desperate = 1.4  # Multiplier when desperate for resources
    attack_mult_stable = 0.6  # Multiplier when resource stable

    @classmethod
    def from_yaml(cls, file_path: str) -> "SimulationConfig":
        """Load configuration from a YAML file."""
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Handle visualization config separately
        vis_config = config_dict.pop("visualization", {})
        config_dict["visualization"] = VisualizationConfig(**vis_config)

        return cls(**config_dict)

    def to_yaml(self, file_path: str) -> None:
        """Save configuration to a YAML file."""
        # Convert to dictionary, handling visualization config specially
        config_dict = self.__dict__.copy()
        config_dict["visualization"] = self.visualization.__dict__

        with open(file_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = self.__dict__.copy()
        config_dict["visualization"] = self.visualization.__dict__
        return config_dict

    def copy(self):
        """Create a deep copy of the configuration."""
        return copy.deepcopy(self)
