import json
from typing import TYPE_CHECKING
import random

from action import Action

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent
    from environment import Environment


class Genome:
    """A utility class for managing agent genome serialization, evolution,
    and manipulation.

    The Genome class provides static methods for converting agents to and from a
    serializable genome representation, as well as methods for genetic operations
    and persistence.

    This enables:
    - Saving and loading agent states
    - Transferring agent configurations between environments
    - Creating new agents based on existing configurations
    - Supporting evolutionary algorithms through genome manipulation
    - Breeding agents through crossover operations
    - Creating mutated variants of existing agents

    A genome is represented as a dictionary containing:
    - action_set: List of (action_name, weight) tuples defining available actions
    - module_states: Dictionary of serialized states for each agent module
    - agent_type: String identifier of the agent class
    - resource_level: Current resource level value
    - current_health: Current health value

    Example:
        # Save an agent's genome
        genome = Genome.from_agent(existing_agent)
        Genome.save(genome, "agent_backup.json")

        # Create a new agent from a genome
        loaded_genome = Genome.load("agent_backup.json")
        new_agent = Genome.to_agent(loaded_genome, agent_id=1, position=(0, 0), environment=env)

        # Perform evolutionary operations
        mutated_genome = Genome.mutate(genome, mutation_rate=0.1)
        child1, child2 = Genome.crossover(genome1, genome2)
        clone = Genome.clone(genome)

    The class supports both basic agent serialization and advanced genetic operations
    for implementing evolutionary algorithms and agent breeding mechanics.
    """

    @staticmethod
    def from_agent(agent: "BaseAgent") -> "Genome":
        """Convert agent's current state and configuration into a genome representation.

        This method extracts all module states, action configurations, and core properties
        from an agent to create a serializable genome dictionary.

        Args:
            agent (BaseAgent): The agent to convert into a genome representation.
                             Must have module attributes ending with '_module' that
                             implement get_state_dict().

        Returns:
            dict: Genome dictionary containing:
                - action_set: List of (action_name, weight) tuples
                - module_states: Dictionary of module states
                - agent_type: String name of agent class
                - resource_level: Current resource level
                - current_health: Current health value
        """
        # Get all attributes that end with '_module'
        module_states = {
            name: getattr(agent, name).get_state_dict()
            for name in dir(agent)
            if name.endswith("_module")
            and hasattr(getattr(agent, name), "get_state_dict")
        }

        genome = {
            "action_set": [(action.name, action.weight) for action in agent.actions],
            "module_states": module_states,
            "agent_type": agent.__class__.__name__,
            "resource_level": agent.resource_level,
            "current_health": agent.current_health,
        }
        return genome

    @staticmethod
    def to_agent(
        genome: dict,
        agent_id: int,
        position: tuple[int, int],
        environment: "Environment",
    ) -> "BaseAgent":
        """Create a new agent from a genome representation.

        This method reconstructs an agent instance using the configuration stored
        in a genome dictionary, including its action set, module states, and core
        properties.

        Args:
            genome (dict): Genome dictionary containing agent configuration with keys:
                - action_set: List of (action_name, weight) tuples
                - module_states: Dictionary of module states
                - agent_type: String name of agent class
                - resource_level: Resource level value
                - current_health: Health value
            agent_id (int): Unique identifier for the new agent
            position (tuple[int, int]): Starting (x, y) coordinates for the agent
            environment (Environment): Reference to the environment instance the
                                    agent will operate in

        Returns:
            BaseAgent: New agent instance initialized with the genome's properties
                      and ready to operate in the environment
        """
        # Reconstruct action set
        action_set = [
            Action(name, weight, globals()[f"{name}_action"])
            for name, weight in genome["action_set"]
        ]

        # Create new agent
        agent = BaseAgent(
            agent_id=agent_id,
            position=position,
            resource_level=genome["resource_level"],
            environment=environment,
            action_set=action_set,
        )

        # Load all module states
        for module_name, state_dict in genome["module_states"].items():
            if hasattr(agent, module_name):
                module = getattr(agent, module_name)
                if hasattr(module, "load_state_dict"):
                    module.load_state_dict(state_dict)

        # Set health
        agent.current_health = genome["current_health"]

        return agent

    @staticmethod
    def save(genome: dict, path: str) -> None:
        """Save genome to a file in JSON format.

        Args:
            genome (dict): Genome dictionary to serialize and save
            path (str): File path where the genome should be saved.
                       Will overwrite if file exists.
        """
        with open(path, "w") as f:
            json.dump(genome, f)

    @staticmethod
    def load(path: str) -> dict:
        """Load genome from a JSON file.

        Args:
            path (str): File path from which to load the genome.
                       File must exist and contain valid JSON.

        Returns:
            dict: Loaded genome dictionary containing agent configuration
        """
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def mutate(genome: dict, mutation_rate: float = 0.1) -> dict:
        """Mutate a genome by randomly adjusting action weights.

        This method creates a modified copy of the input genome with potentially
        altered action weights. Each weight has a chance (determined by mutation_rate)
        to be adjusted by up to ±20%. Weights are renormalized after mutation to
        ensure they sum to 1.0.

        Args:
            genome (dict): Original genome to mutate. Must contain 'action_set' key
                          with list of (action_name, weight) tuples.
            mutation_rate (float, optional): Probability of mutating each weight.
                                           Must be between 0 and 1. Defaults to 0.1.

        Returns:
            dict: New genome with potentially mutated action weights. Original genome
                 remains unchanged.

        Example:
            original = Genome.from_agent(agent)
            mutated = Genome.mutate(original, mutation_rate=0.2)
        """
        mutated = genome.copy()

        # Mutate action weights
        action_set = []
        for action_name, weight in mutated["action_set"]:
            if random.random() < mutation_rate:
                # Adjust weight by up to ±20%
                weight *= 1 + random.uniform(-0.2, 0.2)
            action_set.append((action_name, weight))

        # Normalize weights
        total_weight = sum(weight for _, weight in action_set)
        action_set = [(name, weight / total_weight) for name, weight in action_set]
        mutated["action_set"] = action_set

        return mutated

    @staticmethod
    def crossover(genome1: dict, genome2: dict) -> tuple[dict, dict]:
        """Create two child genomes by crossing over two parent genomes.

        This method performs uniform crossover on the action weights of two parent
        genomes. For each action, there's a 50% chance that the children will swap
        their inherited weights. Other genome properties are copied from the respective
        parents. Weights are renormalized after crossover.

        Args:
            genome1 (dict): First parent genome. Must contain 'action_set' key with
                           list of (action_name, weight) tuples.
            genome2 (dict): Second parent genome. Must contain compatible action_set.

        Returns:
            tuple[dict, dict]: Two new child genomes with mixed properties from both
                              parents. Original genomes remain unchanged.

        Example:
            parent1 = Genome.from_agent(agent1)
            parent2 = Genome.from_agent(agent2)
            child1, child2 = Genome.crossover(parent1, parent2)
        """
        child1 = genome1.copy()
        child2 = genome2.copy()

        # Crossover action weights
        actions1 = dict(genome1["action_set"])
        actions2 = dict(genome2["action_set"])

        for action in actions1.keys():
            if random.random() < 0.5:
                # Swap weights between children
                actions1[action], actions2[action] = actions2[action], actions1[action]

        # Normalize weights for both children
        child1["action_set"] = [(n, w) for n, w in actions1.items()]
        child2["action_set"] = [(n, w) for n, w in actions2.items()]

        return child1, child2

    @staticmethod
    def clone(genome: dict) -> dict:
        """Create an exact deep copy of a genome.

        This method creates a completely independent copy of the input genome,
        ensuring that modifying the clone won't affect the original. Uses JSON
        serialization to ensure a deep copy of all nested structures.

        Args:
            genome (dict): Genome to clone. Must be JSON-serializable.

        Returns:
            dict: New independent copy of the genome with identical structure
                 and values.

        Example:
            original = Genome.from_agent(agent)
            copy = Genome.clone(original)
            # Modifying copy won't affect original
        """
        return json.loads(json.dumps(genome))
