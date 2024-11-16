import numpy as np

from action import Action
from actions.attack import attack_action
from actions.gather import gather_action
from actions.move import move_action
from actions.share import share_action

from .base_agent import BaseAgent


class IndependentAgent(BaseAgent):
    def __init__(self, agent_id, position, resource_level, environment):
        super().__init__(agent_id, position, resource_level, environment)

        # Override default actions with IndependentAgent-specific weights
        self.actions = [
            Action("move", 0.25, move_action),
            Action("gather", 0.45, gather_action),  # Higher weight for gathering
            Action("share", 0.05, share_action),  # Lower weight for sharing
            Action("attack", 0.25, attack_action),  # Moderate weight for attacking
        ]

        # Normalize weights
        total_weight = sum(action.weight for action in self.actions)
        for action in self.actions:
            action.weight /= total_weight

        # Configure gather module for more aggressive resource collection
        self.gather_module.config.gather_efficiency_multiplier = (
            0.7  # Higher efficiency reward
        )
        self.gather_module.config.gather_cost_multiplier = 0.2  # Lower movement penalty
        self.gather_module.config.min_resource_threshold = (
            0.05  # Lower threshold for gathering
        )
