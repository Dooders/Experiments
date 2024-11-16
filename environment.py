import os
import random
from typing import List

import numpy as np

from agents import ControlAgent, IndependentAgent, SystemAgent
from database import SimulationDatabase
from resources import Resource
from state import EnvironmentState


class Environment:
    def __init__(
        self,
        width,
        height,
        resource_distribution,
        db_path="simulation_results.db",
        max_resource=None,
        config=None,
    ):
        # Delete existing database file if it exists
        if os.path.exists(db_path):
            os.remove(db_path)

        self.width = width
        self.height = height
        self.agents: List["SystemAgent" | "IndependentAgent" | "ControlAgent"] = []
        self.resources: List[Resource] = []
        self.time = 0
        self.db = SimulationDatabase(db_path)
        self.next_agent_id = 0
        self.next_resource_id = 0
        self.max_resource = max_resource
        self.config = config  # Store configuration
        self.initialize_resources(resource_distribution)
        self.initial_agent_count = 0  # Add this to track initial agents
        self._initialize_agents()  # Changed to use config values

    def get_next_resource_id(self):
        resource_id = self.next_resource_id
        self.next_resource_id += 1
        return resource_id

    def initialize_resources(self, distribution):
        for _ in range(distribution["amount"]):
            position = (random.uniform(0, self.width), random.uniform(0, self.height))
            resource = Resource(
                resource_id=self.get_next_resource_id(),
                position=position,
                amount=random.randint(3, 8),
            )
            self.resources.append(resource)
            # Log resource to database
            self.db.log_resource(
                resource_id=resource.resource_id,
                initial_amount=resource.amount,
                position=resource.position,
            )

    def add_agent(self, agent):
        self.agents.append(agent)
        # Update initial count only during setup (time=0)
        if self.time == 0:
            self.initial_agent_count += 1

    def update(self):
        """Update environment state with batch processing."""
        self.time += 1

        # Vectorize resource regeneration
        regen_mask = (
            np.random.random(len(self.resources)) < self.config.resource_regen_rate
        )
        for resource, should_regen in zip(self.resources, regen_mask):
            if should_regen and (
                self.max_resource is None or resource.amount < self.max_resource
            ):
                resource.amount = min(
                    resource.amount + self.config.resource_regen_amount,
                    self.max_resource or float("inf"),
                )

        # Calculate metrics and log state
        metrics = self._calculate_metrics()

        # Log state with batch processing
        self.db.log_simulation_step(
            step_number=self.time,
            agents=self.agents,
            resources=self.resources,
            metrics=metrics,
        )

    def _calculate_metrics(self):
        """Calculate various metrics for the current simulation state."""
        from agents import ControlAgent, IndependentAgent, SystemAgent  # Local import

        alive_agents = [agent for agent in self.agents if agent.alive]
        system_agents = [a for a in alive_agents if isinstance(a, SystemAgent)]
        independent_agents = [
            a for a in alive_agents if isinstance(a, IndependentAgent)
        ]
        control_agents = [
            a for a in alive_agents if isinstance(a, ControlAgent)
        ]  # Add control agents

        return {
            "total_agents": len(alive_agents),
            "system_agents": len(system_agents),
            "independent_agents": len(independent_agents),
            "control_agents": len(control_agents),  # Add control agents count
            "total_resources": sum(r.amount for r in self.resources),
            "average_agent_resources": (
                sum(a.resource_level for a in alive_agents) / len(alive_agents)
                if alive_agents
                else 0
            ),
        }

    def get_next_agent_id(self):
        agent_id = self.next_agent_id
        self.next_agent_id += 1
        return agent_id

    def get_state(self) -> EnvironmentState:
        """Get current environment state."""
        return EnvironmentState.from_environment(self)

    def is_valid_position(self, position):
        """Check if a position is valid within the environment bounds.

        Parameters
        ----------
        position : tuple
            (x, y) coordinates to check

        Returns
        -------
        bool
            True if position is within bounds, False otherwise
        """
        x, y = position
        return (0 <= x <= self.width) and (0 <= y <= self.height)

    def _initialize_resources(self):
        """Initialize resources in the environment."""
        for i in range(self.config.initial_resources):
            # Random position
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)

            # Create resource with regeneration parameters
            resource = Resource(
                resource_id=i,
                position=(x, y),
                amount=self.config.max_resource_amount,
                max_amount=self.config.max_resource_amount,
                regeneration_rate=self.config.resource_regen_rate,
            )
            self.resources.append(resource)

    def _initialize_agents(self):
        """Initialize starting agent populations."""
        # Create system agents
        for _ in range(self.config.system_agents):
            position = self._get_random_position()
            agent = SystemAgent(
                agent_id=self.get_next_agent_id(),
                position=position,
                resource_level=self.config.initial_resource_level,
                environment=self,
            )
            self.add_agent(agent)

        # Create independent agents
        for _ in range(self.config.independent_agents):
            position = self._get_random_position()
            agent = IndependentAgent(
                agent_id=self.get_next_agent_id(),
                position=position,
                resource_level=self.config.initial_resource_level,
                environment=self,
            )
            self.add_agent(agent)

        # Create control agents
        for _ in range(self.config.control_agents):
            position = self._get_random_position()
            agent = ControlAgent(
                agent_id=self.get_next_agent_id(),
                position=position,
                resource_level=self.config.initial_resource_level,
                environment=self,
            )
            self.add_agent(agent)

    def _get_random_position(self):
        """Get a random position within the environment bounds."""
        x = random.uniform(0, self.width)
        y = random.uniform(0, self.height)
        return (x, y)
