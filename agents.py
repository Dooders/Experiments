import numpy as np

from base_agent import BaseAgent


class SystemAgent(BaseAgent):
    def act(self):
        # First check if agent should die
        if not self.alive:
            return

        super().act()  # Call parent class act() for death check
        if not self.alive:  # If died during death check, skip the rest
            return

        initial_resources = self.resource_level
        self.conserve_resources()
        self.share_resources()
        self.gather_resources()

        # Calculate reward based on resource change
        reward = self.resource_level - initial_resources
        self.learn(reward)

    def conserve_resources(self):
        if self.resource_level < 5:  # Example conservation logic
            self.resource_level = max(0, self.resource_level - 1)

    def share_resources(self):
        nearby_agents = self.get_nearby_system_agents()
        for agent in nearby_agents:
            self.transfer_resources(agent)

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

    def get_nearby_system_agents(self):
        # Define the maximum distance for considering agents as "nearby"
        max_distance = 30  # Adjust this value based on your simulation needs

        nearby_agents = []
        for agent in self.environment.agents:
            if isinstance(agent, SystemAgent) and agent != self and agent.alive:
                # Calculate Euclidean distance between agents
                distance = np.sqrt(
                    (self.position[0] - agent.position[0]) ** 2
                    + (self.position[1] - agent.position[1]) ** 2
                )

                # Add agent to nearby list if within range
                if distance <= max_distance:
                    nearby_agents.append(agent)

        return nearby_agents

    def transfer_resources(self, agent):
        # Transfer resources to another agent
        if self.resource_level > 0:
            agent.resource_level += 1
            self.resource_level -= 1


class IndividualAgent(BaseAgent):
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
