import logging
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=24):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class BaseAgent:
    def __init__(self, agent_id, position, resource_level, environment):
        self.agent_id = agent_id
        self.position = position
        self.resource_level = resource_level
        self.alive = True
        self.environment = environment
        self.config = environment.config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(
            input_dim=4, output_dim=4, hidden_size=self.config.dqn_hidden_size
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=self.config.memory_size)
        self.gamma = self.config.gamma
        self.epsilon = self.config.epsilon_start
        self.epsilon_min = self.config.epsilon_min
        self.epsilon_decay = self.config.epsilon_decay
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

        # State: [distance_to_resource, angle_to_resource, current_resources, resource_amount]
        dx = closest_resource.position[0] - self.position[0]
        dy = closest_resource.position[1] - self.position[1]
        angle = np.arctan2(dy, dx)

        state = torch.FloatTensor(
            [
                min_distance
                / np.sqrt(self.environment.width**2 + self.environment.height**2),
                angle / np.pi,
                self.resource_level / 20,
                closest_resource.amount / 20,
            ]
        ).to(self.device)

        return state

    def move(self):
        # Get state once and reuse
        state = self.get_state()

        # Epsilon-greedy action selection with vectorized operations
        if random.random() < self.epsilon:
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

        # Store experience more efficiently
        self.memory.append(
            (self.last_state, self.last_action, reward, self.get_state())
        )

        # Only train if we have enough samples and it's time to train
        if (
            len(self.memory) >= self.config.batch_size
            and len(self.memory) % self.config.training_frequency == 0
        ):
            # Sample batch efficiently
            indices = random.sample(range(len(self.memory)), self.config.batch_size)
            batch = [self.memory[i] for i in indices]

            # Batch process tensors
            states = torch.stack([x[0] for x in batch])
            actions = torch.tensor([x[1] for x in batch], device=self.device)
            rewards = torch.tensor(
                [x[2] for x in batch], dtype=torch.float32, device=self.device
            )
            next_states = torch.stack([x[3] for x in batch])

            # Compute Q values in one batch
            with torch.no_grad():
                next_q_values = self.model(next_states)
                max_next_q_values = next_q_values.max(1)[0]
                target_q_values = rewards + (self.gamma * max_next_q_values)

            # Compute current Q values and loss
            current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
            loss = self.criterion(current_q_values.squeeze(), target_q_values)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Store loss and decay epsilon
            self.losses.append(loss.item())
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def act(self):
        # Base resource consumption
        self.resource_level -= self.config.base_consumption_rate

        if self.resource_level <= 0:
            self.starvation_threshold += 1
            if self.starvation_threshold >= self.max_starvation:
                self.die()
                return
        else:
            self.starvation_threshold = 0

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
        self.alive = False
        # Log death to database
        self.environment.db.update_agent_death(
            agent_id=self.agent_id, death_time=self.environment.time
        )
