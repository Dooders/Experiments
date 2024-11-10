import numpy as np

from action import Action, attack_action, gather_action, move_action, share_action

from .base_agent import BaseAgent


class IndependentAgent(BaseAgent):
    def __init__(self, agent_id, position, resource_level, environment):
        super().__init__(agent_id, position, resource_level, environment)

        # Override default actions with IndependentAgent-specific weights
        self.actions = [
            Action("move", 0.25, move_action),
            Action("gather", 0.35, gather_action),
            Action("share", 0.05, share_action),  # Lower weight for sharing
            Action("attack", 0.35, attack_action),  # Higher weight for attacking
        ]

        # Normalize weights
        total_weight = sum(action.weight for action in self.actions)
        for action in self.actions:
            action.weight /= total_weight

    def act(self):
        # First check if agent should die
        if not self.alive:
            return

        super().act()  # Call parent class act() for death check
        if not self.alive:  # If died during death check, skip the rest
            return

        initial_resources = self.resource_level
        self.gather_resources()
        self.consume_resources()

        # Calculate reward based on resource change
        reward = self.resource_level - initial_resources
        self.learn(reward)

    def gather_resources(self):
        for resource in self.environment.resources:
            if not resource.is_depleted():
                dist = np.sqrt(
                    (self.position[0] - resource.position[0]) ** 2
                    + (self.position[1] - resource.position[1]) ** 2
                )

                if dist < 20:  # Increased gathering range
                    gather_amount = min(
                        3, resource.amount
                    )  # Increased gathering amount
                    resource.consume(gather_amount)
                    self.resource_level += gather_amount
                    break

    def consume_resources(self):
        self.resource_level = max(
            0, self.resource_level - 1
        )  # Ensure it doesn't go negative
