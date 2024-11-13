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

    # Visualization settings (separate config)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

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
