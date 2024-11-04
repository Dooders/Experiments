import numpy as np
import pandas as pd

from agents import IndividualAgent, SystemAgent


class DataCollector:
    def __init__(self):
        self.data = []
        self.births_this_cycle = 0
        self.deaths_this_cycle = 0

    def collect(self, environment, step):
        alive_agents = [agent for agent in environment.agents if agent.alive]
        system_agents = [
            agent for agent in environment.agents if isinstance(agent, SystemAgent)
        ]
        individual_agents = [
            agent for agent in environment.agents if isinstance(agent, IndividualAgent)
        ]

        data_point = {
            "step": step,
            # Existing metrics
            "system_agent_count": len(system_agents),
            "individual_agent_count": len(individual_agents),
            "total_resources": sum(
                resource.amount for resource in environment.resources
            ),
            "total_consumption": sum(agent.resource_level for agent in alive_agents),
            "average_resource_per_agent": (
                sum(agent.resource_level for agent in alive_agents) / len(alive_agents)
                if alive_agents
                else 0
            ),
            # New metrics
            "births": self.births_this_cycle,
            "deaths": self.deaths_this_cycle,
            "average_lifespan": self._calculate_average_lifespan(environment),
            "resource_efficiency": self._calculate_resource_efficiency(alive_agents),
            "system_agent_territory": self._calculate_territory_control(
                system_agents, environment
            ),
            "individual_agent_territory": self._calculate_territory_control(
                individual_agents, environment
            ),
            "resource_density": self._calculate_resource_density(environment),
            "population_stability": self._calculate_population_stability(),
        }
        self.data.append(data_point)

        # Reset cycle-specific counters
        self.births_this_cycle = 0
        self.deaths_this_cycle = 0

    def _calculate_average_lifespan(self, environment):
        # Calculate average time agents have been alive
        alive_agents = [agent for agent in environment.agents if agent.alive]
        return (
            sum(
                environment.time - getattr(agent, "birth_time", 0)
                for agent in alive_agents
            )
            / len(alive_agents)
            if alive_agents
            else 0
        )

    def _calculate_resource_efficiency(self, agents):
        # Calculate how efficiently agents are using resources
        if not agents:
            return 0
        return sum(
            agent.total_reward / max(agent.resource_level, 1) for agent in agents
        ) / len(agents)

    def _calculate_territory_control(self, agents, environment):
        # Calculate approximate territory control using Voronoi-like approach
        if not agents:
            return 0
        total_area = environment.width * environment.height
        territory_size = sum(
            self._estimate_agent_territory(agent, environment) for agent in agents
        )
        return territory_size / total_area

    def _calculate_resource_density(self, environment):
        # Calculate resource density distribution
        total_area = environment.width * environment.height
        total_resources = sum(resource.amount for resource in environment.resources)
        return total_resources / total_area

    def calculate_average_resources(self, environment):
        """Calculate the average resources per agent in the environment."""
        alive_agents = [agent for agent in environment.agents if agent.alive]
        if not alive_agents:
            return 0
        return sum(agent.resource_level for agent in alive_agents) / len(alive_agents)

    def _estimate_agent_territory(self, agent, environment):
        # Simple territory estimation based on distance to nearest other agent
        nearest_distance = float("inf")
        for other in environment.agents:
            if other != agent and other.alive:
                dist = np.sqrt(
                    (agent.position[0] - other.position[0]) ** 2
                    + (agent.position[1] - other.position[1]) ** 2
                )
                nearest_distance = min(nearest_distance, dist)
        return min(
            np.pi * (nearest_distance / 2) ** 2, environment.width * environment.height
        )

    def _calculate_population_stability(self):
        # Calculate population stability using recent history
        if len(self.data) < 10:
            return 1.0

        recent_population = [
            d["system_agent_count"] + d["individual_agent_count"]
            for d in self.data[-10:]
        ]
        return 1.0 - np.std(recent_population) / max(np.mean(recent_population), 1)

    def to_dataframe(self):
        return pd.DataFrame(self.data)
