import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


"""Action management module for agent behaviors in a multi-agent environment.

This module defines the core Action class and implements various agent behaviors
including movement, resource gathering, sharing, and combat. Each action has
associated costs, rewards, and conditions for execution.

Key Components:
    - Action: Base class for defining executable agent behaviors
    - Movement: Deep Q-Learning based movement with rewards
    - Gathering: Resource collection from environment nodes
    - Sharing: Resource distribution between nearby agents
    - Combat: Competitive resource acquisition through attacks

Technical Details:
    - Range-based interactions (gathering: config-based, sharing: 30 units, attack: 20 units)
    - Resource-based action costs and rewards
    - Numpy-based distance calculations
    - Automatic tensor-numpy conversion for state handling
"""


class Action:
    """Base class for defining executable agent behaviors.

    Encapsulates a named action with an associated weight for action selection
    and an execution function that defines the behavior.

    Args:
        name (str): Identifier for the action (e.g., "move", "gather")
        weight (float): Selection probability weight for action choice
        function (callable): Function implementing the action behavior
            Must accept agent as first parameter and support *args, **kwargs

    Example:
        ```python
        move = Action("move", 0.4, move_action)
        move.execute(agent, additional_arg=value)
        ```
    """

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
        """Execute the action's behavior function.

        Calls the associated function with the agent and any additional parameters.

        Args:
            agent: Agent instance performing the action
            *args: Variable positional arguments for the action function
            **kwargs: Variable keyword arguments for the action function
        """
        self.function(agent, *args, **kwargs)


# Default action functions
def move_action(agent):
    """Execute movement action for an agent."""
    # Get current position and state
    old_pos = agent.position

    # Get full state using AgentState
    state = agent.get_state()  # This returns normalized state with all 4 dimensions

    # Select action using move module
    action = agent.move_module.select_action(state.to_tensor(agent.move_module.device))

    # Calculate new position based on selected action
    new_pos = agent.calculate_new_position(action)

    # Update agent position if valid
    if agent.environment.is_valid_position(new_pos):
        agent.position = new_pos

        # Get next state after movement
        next_state = agent.get_state()

        # Calculate reward based on resource gain
        reward = agent.calculate_move_reward(old_pos, new_pos)

        # Store experience
        agent.move_module.store_experience(
            state=state, action=action, reward=reward, next_state=next_state, done=False
        )

        logger.debug(
            f"Agent {id(agent)} moved from {old_pos} to {new_pos}. "
            f"Reward: {reward:.3f}, Epsilon: {agent.move_module.epsilon:.3f}"
        )


def gather_action(agent):
    """Attempt resource gathering from nearby nodes.

    Wrapper for gather_resources that adds logging and tracking.
    Monitors resource collection success and changes in agent's inventory.

    Args:
        agent: Agent performing the gathering
            Required attributes:
                - position: Current (x,y) coordinates
                - resource_level: Current resource amount
                - config: Contains gathering parameters

    Effects:
        - May increase agent's resource_level
        - Logs gathering attempts and results
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
    """Share resources with nearby agents.

    Implements cooperative behavior allowing resource distribution:
    1. Identifies agents within 30-unit radius
    2. Randomly selects one recipient
    3. Transfers 1 resource unit if conditions met

    Args:
        agent: Agent performing the sharing
            Required attributes:
                - environment: Contains all agents
                - position: Current (x,y) coordinates
                - resource_level: Current resource amount
                - alive: Active status flag

    Requirements:
        - Sharing agent must have > 1 resource
        - At least one valid recipient within range
        - Range limit: 30 distance units

    Effects:
        - Decreases sharer's resources by 1
        - Increases recipient's resources by 1
        - Logs sharing activity
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
    """Attack and steal resources from nearby agents.

    Implements competitive behavior:
    1. Identifies agents within 20-unit radius
    2. Randomly selects one target
    3. Deals 1-2 damage if conditions met

    Args:
        agent: Agent performing the attack
            Required attributes:
                - environment: Contains all agents
                - position: Current (x,y) coordinates
                - resource_level: Current resource amount
                - alive: Active status flag

    Requirements:
        - Attacking agent must have > 2 resources
        - At least one valid target within range
        - Range limit: 20 distance units

    Effects:
        - Costs 1 resource to attack
        - Deals 1-2 damage to target's resources
        - Logs combat activity
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
    """Core resource gathering implementation.

    Implements efficient resource collection logic:
    1. Vectorized distance calculation to all resources
    2. Identifies closest accessible resource
    3. Handles resource transfer and depletion

    Args:
        agent: Agent performing the gathering
            Required attributes:
                - environment: Contains resource nodes
                - position: Current (x,y) coordinates
                - resource_level: Current resource amount
                - config: Contains gathering_range and max_gather_amount

    Effects:
        - May increase agent's resource_level
        - May decrease resource node's amount
        - Logs gathering process and results

    Performance:
        Uses numpy vectorization for efficient distance calculations
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
