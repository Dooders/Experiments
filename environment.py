import os
import random

import numpy as np

from agents import IndividualAgent, SystemAgent
from database import SimulationDatabase
from resources import Resource


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
        self.agents = []
        self.resources = []
        self.time = 0
        self.db = SimulationDatabase(db_path)
        self.next_agent_id = 0
        self.next_resource_id = 0
        self.max_resource = max_resource
        self.config = config  # Store configuration
        self.initialize_resources(resource_distribution)

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
        alive_agents = [agent for agent in self.agents if agent.alive]
        system_agents = [a for a in alive_agents if isinstance(a, SystemAgent)]
        individual_agents = [a for a in alive_agents if isinstance(a, IndividualAgent)]

        return {
            "total_agents": len(alive_agents),
            "system_agents": len(system_agents),
            "individual_agents": len(individual_agents),
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
