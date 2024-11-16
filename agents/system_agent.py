from typing import TYPE_CHECKING, Optional

import numpy as np

from action import Action
from actions.attack import attack_action
from actions.gather import gather_action
from actions.move import move_action
from actions.share import share_action

from .base_agent import BaseAgent


class SystemAgent(BaseAgent):
    """System-oriented agent implementation focused on cooperation."""

    def __init__(self, agent_id, position, resource_level, environment):
        super().__init__(agent_id, position, resource_level, environment)

        # Override default actions with SystemAgent-specific weights
        self.actions = [
            Action("move", 0.3, move_action),
            Action("gather", 0.35, gather_action),
            Action("share", 0.3, share_action),  # Higher weight for sharing
            Action("attack", 0.05, attack_action),  # Lower weight for attacking
        ]

        # Normalize weights
        total_weight = sum(action.weight for action in self.actions)
        for action in self.actions:
            action.weight /= total_weight

        # Configure gather module for more sustainable resource collection
        self.gather_module.config.gather_efficiency_multiplier = (
            0.4  # Lower efficiency reward
        )
        self.gather_module.config.gather_cost_multiplier = (
            0.4  # Higher movement penalty
        )
        self.gather_module.config.min_resource_threshold = (
            0.2  # Higher threshold for gathering
        )
