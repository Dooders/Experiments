import logging
import random
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from action import *
from actions.attack import AttackActionSpace, AttackModule, attack_action
from actions.gather import GatherConfig, GatherModule, gather_action
from actions.move import MoveModule, move_action
from actions.share import DEFAULT_SHARE_CONFIG, ShareModule, share_action
from state import AgentState

if TYPE_CHECKING:
    from environment import Environment

logger = logging.getLogger(__name__)


class AgentModel(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super(AgentModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = nn.Sequential(
            nn.Linear(input_dim, config.dqn_hidden_size),
            nn.ReLU(),
            nn.Linear(config.dqn_hidden_size, config.dqn_hidden_size),
            nn.ReLU(),
            nn.Linear(config.dqn_hidden_size, output_dim),
        ).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=config.memory_size)
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.config = config

    def forward(self, x):
        return self.network(x)

    def learn(self, batch):
        if len(batch) < self.config.batch_size:
            return None

        # States are already tensors from memory
        states = torch.stack([x[0] for x in batch])
        actions = torch.tensor([x[1] for x in batch], device=self.device)
        rewards = torch.tensor(
            [x[2] for x in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.stack([x[3] for x in batch])

        with torch.no_grad():
            next_q_values = self(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values)

        current_q_values = self(states).gather(1, actions.unsqueeze(1))
        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()


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
        self.model = AgentModel(
            input_dim=AgentState.DIMENSIONS,
            output_dim=len(self.actions),
            config=self.config,
        )
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
            f"Agent {self.agent_id} created at {self.position} during step {environment.time}"
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
        self.share_module = ShareModule(config=DEFAULT_SHARE_CONFIG)

        # Create a GatherConfig instance with DQN parameters from simulation config
        gather_config = GatherConfig()
        gather_config.learning_rate = self.config.learning_rate
        gather_config.memory_size = self.config.memory_size
        gather_config.gamma = self.config.gamma
        gather_config.epsilon_start = self.config.epsilon_start
        gather_config.epsilon_min = self.config.epsilon_min
        gather_config.epsilon_decay = self.config.epsilon_decay
        gather_config.dqn_hidden_size = self.config.dqn_hidden_size
        gather_config.batch_size = self.config.batch_size
        gather_config.tau = self.config.tau

        # Initialize gather module with the configured GatherConfig
        self.gather_module = GatherModule(config=gather_config)

        # Add health tracking for combat
        self.max_health = self.config.max_health
        self.current_health = self.max_health
        self.is_defending = False

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

    def learn(self, reward: float) -> None:
        """Update agent's learning based on received reward.

        Args:
            reward (float): Reward value from last action
        """
        if self.last_state is None:
            return

        self.total_reward += reward
        self.episode_rewards.append(reward)

        # Store experience with proper state objects
        current_state = self.get_state()

        # Convert states to tensors before storing in memory
        last_state_tensor = self.last_state.to_tensor(self.device)
        current_state_tensor = current_state.to_tensor(self.device)

        # Get action index instead of storing Action object
        action_idx = self.actions.index(self.last_action)

        self.model.memory.append(
            (last_state_tensor, action_idx, reward, current_state_tensor)
        )

        # Only train on larger batches less frequently
        if (
            len(self.model.memory) >= self.config.batch_size * 4
            and len(self.model.memory) % (self.config.training_frequency * 4) == 0
        ):
            batch = random.sample(self.model.memory, self.config.batch_size * 4)
            loss = self.model.learn(batch)
            if loss is not None:
                self.losses.append(loss)

    def select_action(self):
        """Select an action using a combination of weighted probabilities and state awareness.

        Uses both predefined weights and current state to make intelligent decisions:
        1. Gets base probabilities from action weights
        2. Adjusts probabilities based on current state
        3. Applies epsilon-greedy exploration

        Returns:
            Action: Selected action object to execute
        """
        # Get base probabilities from weights
        actions = [action for action in self.actions]
        action_weights = [action.weight for action in actions]

        # Normalize base weights
        total_weight = sum(action_weights)
        base_probs = [weight / total_weight for weight in action_weights]

        # State-based adjustments
        adjusted_probs = self._adjust_probabilities(base_probs)

        # Epsilon-greedy exploration
        if random.random() < self.model.epsilon:
            # Random exploration
            return random.choice(actions)
        else:
            # Weighted selection using adjusted probabilities
            return random.choices(actions, weights=adjusted_probs, k=1)[0]

    def _adjust_probabilities(self, base_probs):
        """Adjust action probabilities based on agent's current state.

        Uses configurable multipliers to adjust probabilities based on:
        - Resource levels
        - Nearby resources
        - Nearby agents
        - Current health/starvation

        Args:
            base_probs (list[float]): Original action probabilities

        Returns:
            list[float]: Adjusted probability distribution
        """
        adjusted_probs = base_probs.copy()

        # Get relevant state information
        state = self.get_state()
        resource_level = self.resource_level
        starvation_risk = self.starvation_threshold / self.max_starvation

        # Find nearby entities
        nearby_resources = [
            r
            for r in self.environment.resources
            if not r.is_depleted()
            and np.sqrt(((np.array(r.position) - np.array(self.position)) ** 2).sum())
            < self.config.gathering_range
        ]

        nearby_agents = [
            a
            for a in self.environment.agents
            if a != self
            and a.alive
            and np.sqrt(((np.array(a.position) - np.array(self.position)) ** 2).sum())
            < self.config.social_range
        ]

        # Adjust move probability
        move_idx = next(i for i, a in enumerate(self.actions) if a.name == "move")
        if not nearby_resources:
            # Increase move probability if no resources nearby
            adjusted_probs[move_idx] *= self.config.move_mult_no_resources

        # Adjust gather probability
        gather_idx = next(i for i, a in enumerate(self.actions) if a.name == "gather")
        if nearby_resources and resource_level < self.config.min_reproduction_resources:
            # Increase gather probability if resources needed
            adjusted_probs[gather_idx] *= self.config.gather_mult_low_resources

        # Adjust share probability
        share_idx = next(i for i, a in enumerate(self.actions) if a.name == "share")
        if resource_level > self.config.min_reproduction_resources and nearby_agents:
            # Increase share probability if wealthy and agents nearby
            adjusted_probs[share_idx] *= self.config.share_mult_wealthy
        else:
            # Decrease share probability if resources needed
            adjusted_probs[share_idx] *= self.config.share_mult_poor

        # Adjust attack probability
        attack_idx = next(i for i, a in enumerate(self.actions) if a.name == "attack")
        if (
            starvation_risk > self.config.attack_starvation_threshold
            and nearby_agents
            and resource_level > 2
        ):
            # Increase attack probability if desperate
            adjusted_probs[attack_idx] *= self.config.attack_mult_desperate
        else:
            # Decrease attack probability if stable
            adjusted_probs[attack_idx] *= self.config.attack_mult_stable

        # Get health ratio for combat decisions
        health_ratio = self.current_health / self.max_health

        # Adjust attack probability based on health
        if health_ratio < self.config.attack_defense_threshold:
            # Reduce attack probability when health is low
            adjusted_probs[attack_idx] *= 0.5
        elif (
            health_ratio > 0.8
            and self.resource_level > self.config.min_reproduction_resources
        ):
            # Increase attack probability when healthy and wealthy
            adjusted_probs[attack_idx] *= 1.5

        # Renormalize probabilities
        total = sum(adjusted_probs)
        adjusted_probs = [p / total for p in adjusted_probs]

        return adjusted_probs

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

        # Store state for learning
        self.last_state = current_state
        self.last_action = action

        # Calculate reward and learn
        reward = self.resource_level - initial_resources
        self.learn(reward)

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
