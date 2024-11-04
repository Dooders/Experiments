import numpy as np


class Action:
    def __init__(self, name, weight, function):
        """
        Initialize an action with a name, weight, and associated function.

        Parameters:
        - name (str): The name of the action (e.g., "move", "gather").
        - weight (float): The weight or likelihood of selecting this action.
        - function (callable): The function to execute when this action is chosen.
        """
        self.name = name
        self.weight = weight
        self.function = function

    def execute(self, agent, *args, **kwargs):
        """
        Execute the action's function, passing in the agent and any additional arguments.

        Parameters:
        - agent: The agent performing the action.
        - args, kwargs: Additional arguments for the action function.
        """
        self.function(agent, *args, **kwargs)


# Default action functions
def move_action(agent):
    """Move the agent in the environment."""
    agent.move()


def gather_action(agent):
    """Gather resources from the environment."""
    agent.gather_resources()


def share_action(agent):
    """Share resources with nearby agents."""
    nearby_agents = [
        a
        for a in agent.environment.agents
        if a != agent
        and a.alive
        and np.sqrt(((np.array(a.position) - np.array(agent.position)) ** 2).sum()) < 30
    ]

    if nearby_agents and agent.resource_level > 1:
        target = np.random.choice(nearby_agents)
        agent.resource_level -= 1
        target.resource_level += 1


def attack_action(agent):
    """Attack nearby agents."""
    nearby_agents = [
        a
        for a in agent.environment.agents
        if a != agent
        and a.alive
        and np.sqrt(((np.array(a.position) - np.array(agent.position)) ** 2).sum()) < 20
    ]

    if nearby_agents and agent.resource_level > 2:
        target = np.random.choice(nearby_agents)
        damage = min(2, agent.resource_level - 1)
        target.resource_level = max(0, target.resource_level - damage)
        agent.resource_level -= 1  # Cost of attacking
