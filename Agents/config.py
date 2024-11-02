import yaml
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SimulationConfig:
    # Environment settings
    width: int = 100
    height: int = 100
    
    # Agent settings
    system_agents: int = 25
    individual_agents: int = 25
    initial_resource_level: int = 12
    max_population: int = 300
    
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
    
    # Learning parameters
    learning_rate: float = 0.001
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995

    @classmethod
    def from_yaml(cls, file_path: str) -> 'SimulationConfig':
        """Load configuration from a YAML file."""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, file_path: str) -> None:
        """Save configuration to a YAML file."""
        with open(file_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.__dict__ 