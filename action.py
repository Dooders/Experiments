import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


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
    """
    Move the agent in the environment using DQN-based movement.

    This function implements epsilon-greedy DQN-based movement, allowing agents to:
    1. Either explore randomly (with probability epsilon)
    2. Or exploit learned behaviors (with probability 1-epsilon)

    Movement is restricted to four directions (right, left, up, down) and
    constrained by the environment boundaries.

    Parameters:
        agent: The agent performing the move action.
              Must have attributes:
              - model: DQN model with epsilon value
              - max_movement: Maximum distance per move
              - position: Current (x,y) position
              - environment: Contains width and height bounds
              - get_state: Method to get current state
              - last_state: Storage for learning
              - last_action: Storage for learning

    Effects:
        - Updates agent's position within environment bounds
        - Stores state and action for learning purposes

    Movement Map:
        0: Right (max_movement, 0)
        1: Left (-max_movement, 0)
        2: Up (0, max_movement)
        3: Down (0, -max_movement)
    """
    # Get state once and reuse
    state = agent.get_state()
    initial_position = agent.position

    # Epsilon-greedy action selection with vectorized operations
    if random.random() < agent.model.epsilon:
        action = random.randint(0, 3)
        action_type = "exploration"
    else:
        with torch.no_grad():
            action = agent.model(state).argmax().item()
        action_type = "exploitation"

    # Use lookup table instead of if-else
    move_map = {
        0: (agent.max_movement, 0),  # Right
        1: (-agent.max_movement, 0),  # Left
        2: (0, agent.max_movement),  # Up
        3: (0, -agent.max_movement),  # Down
    }
    dx, dy = move_map[action]

    # Update position with vectorized operations
    agent.position = (
        max(0, min(agent.environment.width, agent.position[0] + dx)),
        max(0, min(agent.environment.height, agent.position[1] + dy)),
    )

    # Store for learning
    agent.last_state = state
    agent.last_action = action

    logger.debug(
        f"Agent {id(agent)} moved via {action_type} from {initial_position} to {agent.position}. "
        f"Action: {['Right', 'Left', 'Up', 'Down'][action]}, "
        f"Epsilon: {agent.model.epsilon:.3f}"
    )


def gather_action(agent):
    """
    Initiate resource gathering for an agent and log the results.

    This function serves as a wrapper around the gather_resources function,
    adding logging functionality to track resource gathering activities.

    Parameters:
        agent: The agent performing the gather action.
              Must have attributes:
              - position: Current (x,y) position
              - resource_level: Current resource amount

    Effects:
        - Calls gather_resources to attempt resource collection
        - Logs the outcome of the gathering attempt
    """
    initial_resources = agent.resource_level
    gather_resources(agent)

    if agent.resource_level > initial_resources:
        logger.info(
            f"Agent {id(agent)} successfully gathered {agent.resource_level - initial_resources} "
            f"resources at position {agent.position}. "
            f"Resources: {initial_resources} -> {agent.resource_level}"
        )
    else:
        logger.debug(
            f"Agent {id(agent)} attempted to gather at position {agent.position} "
            f"but found no accessible resources."
        )


def share_action(agent):
    """
    Share resources with nearby agents within a specified range.

    This function allows an agent to share one resource unit with another randomly chosen agent
    within a 30-unit radius. The sharing will only occur if:
    1. There are nearby agents within range
    2. The sharing agent has more than 1 resource unit

    Parameters:
        agent: The agent performing the share action.
              Must have attributes:
              - environment: Contains list of all agents
              - position: Current (x,y) position
              - resource_level: Current resource amount
              - alive: Boolean indicating if agent is alive

    Effects:
        - Transfers 1 resource unit from sharing agent to target
        - Reduces sharer's resource_level by 1
        - Increases target's resource_level by 1

    Range:
        Sharing only possible within 30 distance units
    """
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
        logger.info(
            f"Agent {id(agent)} shared 1 resource with Agent {id(target)} at position {agent.position}. "
            f"Sharer resources: {agent.resource_level + 1} -> {agent.resource_level}, "
            f"Target resources: {target.resource_level - 1} -> {target.resource_level}"
        )
    else:
        logger.debug(
            f"Agent {id(agent)} attempted to share but conditions not met. "
            f"Nearby agents: {len(nearby_agents)}, Resources: {agent.resource_level}"
        )


def attack_action(agent):
    """
    Attack nearby agents within a specified range.

    This function allows an agent to attack another randomly chosen agent within
    a 20-unit radius. The attack will only occur if:
    1. There are nearby agents within range
    2. The attacking agent has more than 2 resource units

    Parameters:
        agent: The agent performing the attack action.
              Must have attributes:
              - environment: Contains list of all agents
              - position: Current (x,y) position
              - resource_level: Current resource amount
              - alive: Boolean indicating if agent is alive

    Effects:
        - Deals 1-2 damage to target's resource_level (capped by attacker's resources)
        - Reduces attacker's resource_level by 1 as attack cost
        - Target's resource_level cannot go below 0

    Range:
        Attacks only possible within 20 distance units
    """
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
        initial_target_resources = target.resource_level
        target.resource_level = max(0, target.resource_level - damage)
        agent.resource_level -= 1  # Cost of attacking

        logger.info(
            f"Agent {id(agent)} attacked Agent {id(target)} at position {agent.position}. "
            f"Damage dealt: {damage}, Attack cost: 1. "
            f"Attacker resources: {agent.resource_level + 1} -> {agent.resource_level}, "
            f"Target resources: {initial_target_resources} -> {target.resource_level}"
        )
    else:
        logger.debug(
            f"Agent {id(agent)} attempted to attack but conditions not met. "
            f"Nearby agents: {len(nearby_agents)}, Resources: {agent.resource_level}"
        )


def gather_resources(agent):
    """
    Gather resources from nearby resource nodes.

    This function implements the core resource gathering logic:
    1. Identifies all resources in the environment
    2. Calculates distances to all resources
    3. Finds the closest resource within gathering range
    4. Attempts to gather from that resource if not depleted

    Parameters:
        agent: The agent performing the gather action.
              Must have attributes:
              - environment: Contains list of resources
              - position: Current (x,y) position
              - resource_level: Current resource amount
              - config: Contains gathering_range and max_gather_amount

    Effects:
        - May increase agent's resource_level
        - May decrease resource node's amount
        - Logs gathering attempts and outcomes
    """
    if not agent.environment.resources:
        logger.debug(f"Agent {id(agent)} found no resources in environment")
        return

    # Convert positions to numpy arrays
    agent_pos = np.array(agent.position)
    resource_positions = np.array([r.position for r in agent.environment.resources])

    # Calculate all distances at once
    distances = np.sqrt(((resource_positions - agent_pos) ** 2).sum(axis=1))

    # Find closest resource within range
    in_range = distances < agent.config.gathering_range
    if not np.any(in_range):
        logger.debug(
            f"Agent {id(agent)} found no resources within range {agent.config.gathering_range} "
            f"at position {agent.position}"
        )
        return

    closest_idx = distances[in_range].argmin()
    resource = agent.environment.resources[closest_idx]

    if not resource.is_depleted():
        gather_amount = min(agent.config.max_gather_amount, resource.amount)
        resource.consume(gather_amount)
        agent.resource_level += gather_amount
        logger.debug(
            f"Agent {id(agent)} gathered {gather_amount} resources from node at {resource.position}. "
            f"Resource node remaining: {resource.amount}"
        )
    else:
        logger.debug(
            f"Agent {id(agent)} found depleted resource node at {resource.position}"
        )
