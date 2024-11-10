import random
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from action import *

if TYPE_CHECKING:
    from environment import Environment


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


class AgentState:
    def __init__(self, distance, angle, resource_level, target_resource_amount):
        self.normalized_distance = distance  # Distance to nearest resource (normalized by diagonal of environment)
        self.normalized_angle = angle  # Angle to nearest resource (normalized by π)
        self.normalized_resource_level = (
            resource_level  # Agent's current resources (normalized by 20)
        )
        self.normalized_target_amount = (
            target_resource_amount  # Target resource amount (normalized by 20)
        )

    def to_tensor(self, device):
        return torch.FloatTensor(
            [
                self.normalized_distance,
                self.normalized_angle,
                self.normalized_resource_level,
                self.normalized_target_amount,
            ]
        ).to(device)


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
            input_dim=len(self.get_state()),
            output_dim=len(self.actions),
            config=self.config,
        )
        self.last_state = None
        self.last_action = None
        self.max_movement = self.config.max_movement
        self.total_reward = 0
        self.episode_rewards = []
        self.losses = []
        self.starvation_threshold = self.config.starvation_threshold
        self.max_starvation = self.config.max_starvation_time
        self.birth_time = environment.time

        # Log agent creation to database
        environment.db.log_agent(
            agent_id=self.agent_id,
            birth_time=environment.time,
            agent_type=self.__class__.__name__,
            position=self.position,
            initial_resources=self.resource_level,
        )

    def get_state(self):
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
            return torch.zeros(4, device=self.device)

        # Calculate normalized values
        dx = closest_resource.position[0] - self.position[0]
        dy = closest_resource.position[1] - self.position[1]
        angle = np.arctan2(dy, dx)

        normalized_distance = min_distance / np.sqrt(
            self.environment.width**2 + self.environment.height**2
        )
        normalized_angle = angle / np.pi
        normalized_resource_level = self.resource_level / 20
        normalized_target_amount = closest_resource.amount / 20

        state = AgentState(
            distance=normalized_distance,
            angle=normalized_angle,
            resource_level=normalized_resource_level,
            target_resource_amount=normalized_target_amount,
        )

        return state.to_tensor(self.device)

    def move(self):
        # Get state once and reuse
        state = self.get_state()

        # Epsilon-greedy action selection with vectorized operations
        if random.random() < self.model.epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                action = self.model(state).argmax().item()

        # Use lookup table instead of if-else
        move_map = {
            0: (self.max_movement, 0),  # Right
            1: (-self.max_movement, 0),  # Left
            2: (0, self.max_movement),  # Up
            3: (0, -self.max_movement),  # Down
        }
        dx, dy = move_map[action]

        # Update position with vectorized operations
        self.position = (
            max(0, min(self.environment.width, self.position[0] + dx)),
            max(0, min(self.environment.height, self.position[1] + dy)),
        )

        # Store for learning
        self.last_state = state
        self.last_action = action

    def learn(self, reward):
        if self.last_state is None:
            return

        self.total_reward += reward
        self.episode_rewards.append(reward)

        # Store experience
        self.model.memory.append(
            (self.last_state, self.last_action, reward, self.get_state())
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
        # Select an action based on weights
        actions = [action for action in self.actions]
        action_weights = [action.weight for action in actions]

        # Normalize weights to make them probabilities
        total_weight = sum(action_weights)
        action_probs = [weight / total_weight for weight in action_weights]

        # Choose an action based on the weighted probabilities
        chosen_action = random.choices(actions, weights=action_probs, k=1)[0]
        return chosen_action

    def act(self):
        # First check if agent should die
        if not self.alive:
            return

        # Base resource consumption
        self.resource_level -= self.config.base_consumption_rate

        if self.resource_level <= 0:
            self.starvation_threshold += 1
            if self.starvation_threshold >= self.max_starvation:
                self.die()
                return
        else:
            self.starvation_threshold = 0

        # Select and execute an action
        action = self.select_action()
        action.execute(self)

    def reproduce(self):
        if len(self.environment.agents) >= self.config.max_population:
            return

        if self.resource_level >= self.config.min_reproduction_resources:
            if self.resource_level >= self.config.offspring_cost + 2:
                new_agent = self.create_offspring()
                self.environment.add_agent(new_agent)
                self.resource_level -= self.config.offspring_cost

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

    def gather_resources(self):
        if not self.environment.resources:
            return

        # Convert positions to numpy arrays
        agent_pos = np.array(self.position)
        resource_positions = np.array([r.position for r in self.environment.resources])

        # Calculate all distances at once
        distances = np.sqrt(((resource_positions - agent_pos) ** 2).sum(axis=1))

        # Find closest resource within range
        in_range = distances < self.config.gathering_range
        if not np.any(in_range):
            return

        closest_idx = distances[in_range].argmin()
        resource = self.environment.resources[closest_idx]

        if not resource.is_depleted():
            gather_amount = min(self.config.max_gather_amount, resource.amount)
            resource.consume(gather_amount)
            self.resource_level += gather_amount

    def get_environment(self) -> "Environment":
        return self._environment

    def set_environment(self, environment: "Environment") -> None:
        self._environment = environment
