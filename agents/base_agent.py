import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from action import *
from actions.attack import AttackActionSpace, AttackModule, attack_action
from actions.gather import GatherModule, gather_action
from actions.move import MoveModule, move_action
from actions.select import SelectConfig, SelectModule, create_selection_state
from actions.share import ShareModule, share_action
from genome import Genome
from state import AgentState

if TYPE_CHECKING:
    from environment import Environment

logger = logging.getLogger(__name__)


BASE_ACTION_SET = [
    Action("move", 0.4, move_action),
    Action("gather", 0.3, gather_action),
    Action("share", 0.2, share_action),
    Action("attack", 0.1, attack_action),
]


class BaseAgent:
    def __init__(
        self,
        agent_id: int,
        position: tuple[int, int],
        resource_level: int,
        environment: "Environment",
        action_set: list[Action] = BASE_ACTION_SET,
    ):
        # Add default actions
        self.actions = action_set

        # Normalize weights
        total_weight = sum(action.weight for action in self.actions)
        for action in self.actions:
            action.weight /= total_weight

        self.agent_id = agent_id
        self.position = position
        self.resource_level = resource_level
        self.alive = True
        self.environment = environment
        self.config = environment.config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_state: AgentState | None = None
        self.last_action = None
        self.max_movement = self.config.max_movement
        self.total_reward = 0
        self.episode_rewards = []
        self.losses = []
        self.starvation_threshold = self.config.starvation_threshold
        self.max_starvation = self.config.max_starvation_time
        self.birth_time = environment.time
        logger.info(
            f"Agent {self.agent_id} created at {self.position} during step {environment.time} of type {self.__class__.__name__}"
        )

        # Log agent creation to database
        environment.db.log_agent(
            agent_id=self.agent_id,
            birth_time=environment.time,
            agent_type=self.__class__.__name__,
            position=self.position,
            initial_resources=self.resource_level,
        )

        # Initialize modules with their specific configs
        self.move_module = MoveModule(self.config)
        self.attack_module = AttackModule(self.config)
        self.share_module = ShareModule(self.config)
        self.gather_module = GatherModule(self.config)

        # Add health tracking for combat
        self.max_health = self.config.max_health
        self.current_health = self.max_health
        self.is_defending = False

        # Initialize selection module
        self.select_module = SelectModule(
            num_actions=len(self.actions), config=SelectConfig(), device=self.device
        )

    def get_state(self) -> AgentState:
        """Get the current normalized state of the agent.

        Calculates the agent's state relative to nearest resource and current
        resource levels. Returns None if no resources are available.

        Returns:
            AgentState: Normalized state representation
        """
        # Get closest resource position
        closest_resource = None
        min_distance = float("inf")
        for resource in self.environment.resources:
            if resource.amount > 0:
                dist = np.sqrt(
                    (self.position[0] - resource.position[0]) ** 2
                    + (self.position[1] - resource.position[1]) ** 2
                )
                if dist < min_distance:
                    min_distance = dist
                    closest_resource = resource

        if closest_resource is None:
            # Return zero state if no resources available
            return AgentState(
                normalized_distance=1.0,  # Maximum distance
                normalized_angle=0.5,  # Neutral angle
                normalized_resource_level=0.0,
                normalized_target_amount=0.0,
            )

        # Calculate raw values
        dx = closest_resource.position[0] - self.position[0]
        dy = closest_resource.position[1] - self.position[1]
        angle = np.arctan2(dy, dx)

        # Calculate environment diagonal for distance normalization
        env_diagonal = np.sqrt(self.environment.width**2 + self.environment.height**2)

        # Ensure resource level is non-negative
        resource_level = max(0.0, self.resource_level)

        # Create normalized state using factory method
        return AgentState.from_raw_values(
            distance=min_distance,
            angle=angle,
            resource_level=resource_level,  # Use clamped value
            target_amount=closest_resource.amount,
            env_diagonal=env_diagonal,
        )

    def select_action(self):
        """Select an action using the SelectModule's intelligent decision making.

        Uses both predefined weights and learned preferences to choose optimal actions:
        1. Gets current state representation
        2. Passes state through SelectModule for decision
        3. Returns selected action

        Returns:
            Action: Selected action object to execute
        """
        # Get current state for selection
        state = create_selection_state(self)

        # Select action using selection module
        selected_action = self.select_module.select_action(
            agent=self, actions=self.actions, state=state
        )

        return selected_action

    def act(self):
        """Execute an action based on current state."""
        if not self.alive:
            return

        # Reset defense status at start of turn
        self.is_defending = False

        initial_resources = self.resource_level
        self.resource_level -= self.config.base_consumption_rate

        if self.resource_level <= 0:
            self.starvation_threshold += 1
            if self.starvation_threshold >= self.max_starvation:
                self.die()
                return
        else:
            self.starvation_threshold = 0

        # Get current state before action
        current_state = self.get_state()

        # Select and execute action
        action = self.select_action()
        action.execute(self)

        # Log agent action
        self.environment.db.log_agent_action(
            step_number=self.environment.time,
            agent_id=self.agent_id,
            action_type=action.name,
            action_target_id=None,  # Update with actual target ID if applicable
            position_before=self.position,
            position_after=self.position,  # Update with actual new position if applicable
            resources_before=initial_resources,
            resources_after=self.resource_level,
            reward=0.0,  # Update with actual reward if applicable
        )

        # Store state for learning
        self.last_state = current_state
        self.last_action = action

        # Log learning experience
        self.environment.db.log_learning_experience(
            step_number=self.environment.time,
            agent_id=self.agent_id,
            module_type="act",
            state_before=str(current_state),
            action_taken=action.name,
            reward=0.0,  # Update with actual reward if applicable
            state_after=str(self.get_state()),
            loss=0.0,  # Update with actual loss if applicable
        )

    def reproduce(self):
        if len(self.environment.agents) >= self.config.max_population:
            return

        if self.resource_level >= self.config.min_reproduction_resources:
            if self.resource_level >= self.config.offspring_cost + 2:
                new_agent = self.create_offspring()
                self.environment.add_agent(new_agent)
                self.resource_level -= self.config.offspring_cost

                logger.info(
                    f"Agent {self.agent_id} reproduced at {self.position} during step {self.environment.time} creating agent {new_agent.agent_id}"
                )

    def create_offspring(self):
        return type(self)(
            agent_id=self.environment.get_next_agent_id(),
            position=self.position,
            resource_level=self.config.offspring_initial_resources,
            environment=self.environment,
        )

    def die(self):
        """
        Handle agent death by:
        1. Setting alive status to False
        2. Logging death to database
        3. Notifying environment for visualization updates
        """
        self.alive = False

        # Log death to database
        self.environment.db.update_agent_death(
            agent_id=self.agent_id, death_time=self.environment.time
        )

        # Remove agent from environment's active agents list
        if hasattr(self.environment, "agents"):
            try:
                self.environment.agents.remove(self)
            except ValueError:
                pass  # Agent was already removed

        logger.info(
            f"Agent {self.agent_id} died at {self.position} during step {self.environment.time}"
        )

    def get_environment(self) -> "Environment":
        return self._environment

    def set_environment(self, environment: "Environment") -> None:
        self._environment = environment

    def calculate_new_position(self, action):
        """Calculate new position based on movement action.

        Parameters
        ----------
        action : int
            Movement action index (0: right, 1: left, 2: up, 3: down)

        Returns
        -------
        tuple
            New (x, y) position coordinates
        """
        # Define movement vectors for each action
        action_vectors = {
            0: (1, 0),  # Right
            1: (-1, 0),  # Left
            2: (0, 1),  # Up
            3: (0, -1),  # Down
        }

        # Get movement vector for the action
        dx, dy = action_vectors[action]

        # Scale by max_movement
        dx *= self.config.max_movement
        dy *= self.config.max_movement

        # Calculate new position
        new_x = max(0, min(self.environment.width, self.position[0] + dx))
        new_y = max(0, min(self.environment.height, self.position[1] + dy))

        return (new_x, new_y)

    def calculate_move_reward(self, old_pos, new_pos):
        """Calculate reward for a movement action.

        Parameters
        ----------
        old_pos : tuple
            Previous (x, y) position
        new_pos : tuple
            New (x, y) position

        Returns
        -------
        float
            Movement reward value
        """
        # Base cost for moving
        reward = -0.1

        # Calculate movement distance
        distance_moved = np.sqrt(
            (new_pos[0] - old_pos[0]) ** 2 + (new_pos[1] - old_pos[1]) ** 2
        )

        if distance_moved > 0:
            # Find closest non-depleted resource
            closest_resource = min(
                [r for r in self.environment.resources if not r.is_depleted()],
                key=lambda r: np.sqrt(
                    (r.position[0] - new_pos[0]) ** 2
                    + (r.position[1] - new_pos[1]) ** 2
                ),
                default=None,
            )

            if closest_resource:
                # Calculate distances to resource before and after move
                old_distance = np.sqrt(
                    (closest_resource.position[0] - old_pos[0]) ** 2
                    + (closest_resource.position[1] - old_pos[1]) ** 2
                )
                new_distance = np.sqrt(
                    (closest_resource.position[0] - new_pos[0]) ** 2
                    + (closest_resource.position[1] - new_pos[1]) ** 2
                )

                # Reward for moving closer to resources, penalty for moving away
                reward += 0.3 if new_distance < old_distance else -0.2

        return reward

    def calculate_attack_position(self, action: int) -> tuple[float, float]:
        """Calculate target position for attack based on action.

        Args:
            action (int): Attack action index from AttackActionSpace

        Returns:
            tuple[float, float]: Target (x,y) coordinates for attack
        """
        # Get attack direction vector
        dx, dy = self.attack_module.action_space[action]

        # Scale by attack range
        dx *= self.config.attack_range
        dy *= self.config.attack_range

        # Calculate target position
        target_x = self.position[0] + dx
        target_y = self.position[1] + dy

        return (target_x, target_y)

    def handle_combat(self, attacker: "BaseAgent", damage: float) -> float:
        """Handle incoming attack and calculate actual damage taken.

        Args:
            attacker (BaseAgent): Agent performing the attack
            damage (float): Base damage amount

        Returns:
            float: Actual damage dealt after defense
        """
        # Reduce damage if defending
        if self.is_defending:
            damage *= 0.5  # 50% damage reduction when defending

        # Apply damage
        self.current_health = max(0, self.current_health - damage)

        # Check for death
        if self.current_health <= 0:
            self.die()

        return damage

    def calculate_attack_reward(
        self, target: "BaseAgent", damage_dealt: float, action: int
    ) -> float:
        """Calculate reward for an attack action.

        Args:
            target: The agent that was attacked
            damage_dealt: Amount of damage successfully dealt
            action: The attack action that was taken

        Returns:
            float: The calculated reward value
        """
        # Base reward starts with the attack cost
        reward = self.config.attack_base_cost

        # Defensive action reward
        if action == AttackActionSpace.DEFEND:
            if (
                self.current_health
                < self.max_health * self.config.attack_defense_threshold
            ):
                reward += (
                    self.config.attack_success_reward
                )  # Good decision to defend when health is low
            else:
                reward += self.config.attack_failure_penalty  # Unnecessary defense
            return reward

        # Attack success reward
        if damage_dealt > 0:
            reward += self.config.attack_success_reward * (
                damage_dealt / self.config.attack_base_damage
            )
            if not target.alive:
                reward += self.config.attack_kill_reward
        else:
            reward += self.config.attack_failure_penalty

        return reward

    def to_genome(self) -> "Genome":
        """Convert agent's current state and configuration into a genome representation.

        Returns:
            Genome: Genome containing all necessary information to recreate the agent
        """
        return Genome.from_agent(self)

    @classmethod
    def from_genome(
        cls,
        genome: "Genome",
        agent_id: int,
        position: tuple[int, int],
        environment: "Environment",
    ) -> "BaseAgent":
        """Create a new agent from a genome representation.

        Args:
            genome (Genome): Genome containing agent configuration
            agent_id (int): ID for the new agent
            position (tuple[int, int]): Starting position
            environment (Environment): Environment instance

        Returns:
            BaseAgent: New agent instance with genome's properties
        """
        return Genome.to_agent(genome, agent_id, position, environment)
