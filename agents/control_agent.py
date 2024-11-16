import numpy as np

from action import Action
from actions.attack import attack_action
from actions.gather import gather_action
from actions.move import move_action
from actions.share import share_action

from .base_agent import BaseAgent


class ControlAgent(BaseAgent):
    """
    A balanced agent implementation that maintains equilibrium between
    cooperative and individualistic behaviors.

    This agent:
    - Uses balanced action weights
    - Has moderate resource gathering efficiency
    - Maintains balanced sharing and attack tendencies
    - Adapts behavior based on environmental conditions
    """

    def __init__(self, agent_id, position, resource_level, environment):
        super().__init__(agent_id, position, resource_level, environment)

        # Override default actions with balanced weights
        self.actions = [
            Action("move", 0.30, move_action),  # Balanced movement
            Action("gather", 0.40, gather_action),  # Moderate focus on gathering
            Action("share", 0.15, share_action),  # Moderate sharing
            Action("attack", 0.15, attack_action),  # Moderate aggression
        ]

        # Normalize weights
        total_weight = sum(action.weight for action in self.actions)
        for action in self.actions:
            action.weight /= total_weight

        # Configure gather module with balanced parameters
        self.gather_module.config.gather_efficiency_multiplier = (
            0.55  # Balanced between system (0.4) and independent (0.7)
        )
        self.gather_module.config.gather_cost_multiplier = (
            0.3  # Balanced between system (0.4) and independent (0.2)
        )
        self.gather_module.config.min_resource_threshold = (
            0.125  # Balanced between system (0.2) and independent (0.05)
        )
