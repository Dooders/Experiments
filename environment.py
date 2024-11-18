import os
import random
from typing import Dict, List

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
            environment=self,
        )

    def _calculate_metrics(self):
        """Calculate various metrics for the current simulation state."""
        from agents import ControlAgent, IndependentAgent, SystemAgent  # Local import

        alive_agents = [agent for agent in self.agents if agent.alive]
        system_agents = [a for a in alive_agents if isinstance(a, SystemAgent)]
        independent_agents = [
            a for a in alive_agents if isinstance(a, IndependentAgent)
        ]
        control_agents = [a for a in alive_agents if isinstance(a, ControlAgent)]

        # Calculate births (agents created this step)
        births = len([a for a in alive_agents if self.time - a.birth_time == 0])

        # Calculate deaths (difference from previous population)
        previous_population = getattr(self, "previous_population", len(alive_agents))
        deaths = max(0, previous_population - len(alive_agents))
        self.previous_population = len(alive_agents)

        # Calculate generation metrics
        current_max_generation = (
            max([a.generation for a in alive_agents]) if alive_agents else 0
        )

        # Calculate health and age metrics
        total_agents = len(alive_agents)
        average_health = (
            sum(a.current_health for a in alive_agents) / total_agents
            if total_agents > 0
            else 0
        )
        average_age = (
            sum(self.time - a.birth_time for a in alive_agents) / total_agents
            if total_agents > 0
            else 0
        )
        average_reward = (
            sum(a.total_reward for a in alive_agents) / total_agents
            if total_agents > 0
            else 0
        )

        # Calculate resource metrics
        total_resources = sum(r.amount for r in self.resources)
        resource_efficiency = (
            total_resources / (len(self.resources) * self.config.max_resource_amount)
            if self.resources
            else 0
        )

        # Calculate genetic diversity
        genome_counts = {}
        for agent in alive_agents:
            genome_counts[agent.genome_id] = genome_counts.get(agent.genome_id, 0) + 1
        genetic_diversity = len(genome_counts) / total_agents if total_agents > 0 else 0
        dominant_genome_ratio = (
            max(genome_counts.values()) / total_agents if genome_counts else 0
        )

        # Get combat and sharing metrics from environment attributes
        combat_encounters = getattr(self, "combat_encounters", 0)
        successful_attacks = getattr(self, "successful_attacks", 0)
        resources_shared = getattr(self, "resources_shared", 0)

        # Calculate resource distribution entropy (simplified)
        resource_amounts = [r.amount for r in self.resources]
        if resource_amounts:
            total = sum(resource_amounts)
            if total > 0:
                probabilities = [amt / total for amt in resource_amounts]
                resource_distribution_entropy = -sum(
                    p * np.log(p) if p > 0 else 0 for p in probabilities
                )
            else:
                resource_distribution_entropy = 0.0
        else:
            resource_distribution_entropy = 0.0

        return {
            "total_agents": total_agents,
            "system_agents": len(system_agents),
            "independent_agents": len(independent_agents),
            "control_agents": len(control_agents),
            "total_resources": total_resources,
            "average_agent_resources": (
                sum(a.resource_level for a in alive_agents) / total_agents
                if total_agents > 0
                else 0
            ),
            "births": births,
            "deaths": deaths,
            "current_max_generation": current_max_generation,
            "resource_efficiency": resource_efficiency,
            "resource_distribution_entropy": resource_distribution_entropy,
            "average_agent_health": average_health,
            "average_agent_age": average_age,
            "average_reward": average_reward,
            "combat_encounters": combat_encounters,
            "successful_attacks": successful_attacks,
            "resources_shared": resources_shared,
            "genetic_diversity": genetic_diversity,
            "dominant_genome_ratio": dominant_genome_ratio,
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

    def get_step_metrics(self) -> Dict:
        """Calculate comprehensive metrics for the current simulation step."""
        alive_agents = [a for a in self.agents if a.alive]

        # Calculate basic population metrics
        total_agents = len(alive_agents)
        system_agents = len([a for a in alive_agents if isinstance(a, SystemAgent)])
        independent_agents = len(
            [a for a in alive_agents if isinstance(a, IndependentAgent)]
        )

        # Calculate new metrics
        births = len([a for a in alive_agents if self.time - a.birth_time == 0])
        deaths = (
            self.previous_population - total_agents
            if hasattr(self, "previous_population")
            else 0
        )
        self.previous_population = total_agents

        # Calculate generation metrics
        current_max_generation = (
            max([a.generation for a in alive_agents]) if alive_agents else 0
        )

        # Calculate health and age metrics
        average_health = (
            sum(a.current_health for a in alive_agents) / total_agents
            if total_agents > 0
            else 0
        )
        average_age = (
            sum(self.time - a.birth_time for a in alive_agents) / total_agents
            if total_agents > 0
            else 0
        )
        average_reward = (
            sum(a.total_reward for a in alive_agents) / total_agents
            if total_agents > 0
            else 0
        )

        # Calculate resource metrics
        total_resources = sum(r.amount for r in self.resources)
        resource_efficiency = (
            total_resources / (len(self.resources) * self.config.max_resource_amount)
            if self.resources
            else 0
        )

        # Calculate genetic diversity (example implementation)
        genome_counts = {}
        for agent in alive_agents:
            genome_counts[agent.genome_id] = genome_counts.get(agent.genome_id, 0) + 1
        genetic_diversity = len(genome_counts) / total_agents if total_agents > 0 else 0
        dominant_genome_ratio = (
            max(genome_counts.values()) / total_agents if total_agents > 0 else 0
        )

        return {
            "total_agents": total_agents,
            "system_agents": system_agents,
            "independent_agents": independent_agents,
            "control_agents": 0,  # If you're not using control agents
            "total_resources": total_resources,
            "average_agent_resources": (
                sum(a.resource_level for a in alive_agents) / total_agents
                if total_agents > 0
                else 0
            ),
            "births": births,
            "deaths": deaths,
            "current_max_generation": current_max_generation,
            "resource_efficiency": resource_efficiency,
            "resource_distribution_entropy": 0.0,  # Implement entropy calculation if needed
            "average_agent_health": average_health,
            "average_agent_age": average_age,
            "average_reward": average_reward,
            "combat_encounters": (
                self.combat_encounters if hasattr(self, "combat_encounters") else 0
            ),
            "successful_attacks": (
                self.successful_attacks if hasattr(self, "successful_attacks") else 0
            ),
            "resources_shared": (
                self.resources_shared if hasattr(self, "resources_shared") else 0
            ),
            "genetic_diversity": genetic_diversity,
            "dominant_genome_ratio": dominant_genome_ratio,
        }
