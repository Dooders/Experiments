import logging
import os
import random
import tkinter as tk
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from colorama import Fore, Style, init
from config import SimulationConfig
from database import SimulationDatabase
from PIL import Image, ImageDraw, ImageFont
from visualization import SimulationVisualizer

# ==============================
# Environment Setup
# ==============================


class Environment:
    def __init__(
        self,
        width,
        height,
        resource_distribution,
        db_path="simulation_results.db",
        max_resource=None,
        config=None
    ):
        # Delete existing database file if it exists
        if os.path.exists(db_path):
            os.remove(db_path)

        self.width = width
        self.height = height
        self.agents = []
        self.resources = []
        self.time = 0
        self.db = SimulationDatabase(db_path)
        self.next_agent_id = 0
        self.next_resource_id = 0
        self.max_resource = max_resource
        self.config = config  # Store configuration
        self.initialize_resources(resource_distribution)

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

    def update(self):
        self.time += 1
        self.regenerate_resources()

        # Calculate metrics
        metrics = self._calculate_metrics()

        # Log current state to database
        self.db.log_simulation_step(
            step_number=self.time,
            agents=self.agents,
            resources=self.resources,
            metrics=metrics,
        )

    def regenerate_resources(self):
        for resource in self.resources:
            # Only check max_resource if it's set
            if random.random() < 0.1:  # 10% chance to regenerate
                if self.max_resource is None:
                    # No maximum, just add the regeneration amount
                    resource.amount += 2
                else:
                    # Only regenerate if below maximum
                    if resource.amount < self.max_resource:
                        resource.amount = min(resource.amount + 2, self.max_resource)

    def _calculate_metrics(self):
        """Calculate various metrics for the current simulation state."""
        alive_agents = [agent for agent in self.agents if agent.alive]
        system_agents = [a for a in alive_agents if isinstance(a, SystemAgent)]
        individual_agents = [a for a in alive_agents if isinstance(a, IndividualAgent)]

        return {
            "total_agents": len(alive_agents),
            "system_agents": len(system_agents),
            "individual_agents": len(individual_agents),
            "total_resources": sum(r.amount for r in self.resources),
            "average_agent_resources": (
                sum(a.resource_level for a in alive_agents) / len(alive_agents)
                if alive_agents
                else 0
            ),
        }

    def get_next_agent_id(self):
        agent_id = self.next_agent_id
        self.next_agent_id += 1
        return agent_id


# ==============================
# Resource Model
# ==============================


class Resource:
    def __init__(self, resource_id, position, amount):
        self.resource_id = resource_id
        self.position = position  # (x, y) coordinates
        self.amount = amount

    def is_depleted(self):
        return self.amount <= 0

    def consume(self, consumption_amount):
        self.amount -= consumption_amount
        if self.amount < 0:
            self.amount = 0


# ==============================
# Agent Definitions
# ==============================


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


class Agent:
    def __init__(self, agent_id, position, resource_level, environment):
        self.agent_id = agent_id
        self.position = position
        self.resource_level = resource_level
        self.alive = True
        self.environment = environment
        self.config = environment.config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(input_dim=4, output_dim=4, hidden_size=self.config.dqn_hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
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
        state = self.get_state()

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                q_values = self.model(state)
                action = q_values.argmax().item()

        # Convert action to movement - reduced movement distance
        move_distance = self.max_movement
        if action == 0:  # Move right
            dx, dy = move_distance, 0
        elif action == 1:  # Move left
            dx, dy = -move_distance, 0
        elif action == 2:  # Move up
            dx, dy = 0, move_distance
        else:  # Move down
            dx, dy = 0, -move_distance

        # Update position while keeping within bounds
        new_x = max(0, min(self.environment.width, self.position[0] + dx))
        new_y = max(0, min(self.environment.height, self.position[1] + dy))
        self.position = (new_x, new_y)

        # Store state and action for learning
        self.last_state = state
        self.last_action = action

    def learn(self, reward):
        if self.last_state is None:
            return

        self.total_reward += reward
        self.episode_rewards.append(reward)

        # Store experience in memory
        self.memory.append(
            (self.last_state, self.last_action, reward, self.get_state())
        )

        # Only train every N steps
        if (
            len(self.memory) >= 32 and len(self.memory) % 4 == 0
        ):  # Reduce training frequency
            batch = random.sample(self.memory, 32)
            states = torch.stack([x[0] for x in batch])
            actions = torch.tensor([x[1] for x in batch], device=self.device)
            rewards = torch.tensor(
                [x[2] for x in batch], dtype=torch.float32, device=self.device
            )
            next_states = torch.stack([x[3] for x in batch])

            # Compute Q values in one batch
            with torch.no_grad():
                max_next_q_values = self.model(next_states).max(1)[0]
                target_q_values = rewards + (self.gamma * max_next_q_values)

            current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
            loss = self.criterion(current_q_values.squeeze(), target_q_values)
            self.losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log training progress (now in green in console)
            if len(self.losses) % 100 == 0:
                avg_loss = sum(self.losses[-100:]) / 100
                avg_reward = sum(self.episode_rewards[-100:]) / 100
                logging.info(
                    f"Agent {self.agent_id} - "
                    f"Epsilon: {self.epsilon:.3f}, "
                    f"Avg Loss: {avg_loss:.3f}, "
                    f"Avg Reward: {avg_reward:.3f}, "
                    f"Total Reward: {self.total_reward:.3f}"
                )

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self):
        # Reduced base resource consumption from config
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


class SystemAgent(Agent):
    def act(self):
        # First check if agent should die
        if not self.alive:
            return

        super().act()  # Call parent class act() for death check
        if not self.alive:  # If died during death check, skip the rest
            return

        initial_resources = self.resource_level
        self.conserve_resources()
        self.share_resources()
        self.gather_resources()

        # Calculate reward based on resource change
        reward = self.resource_level - initial_resources
        self.learn(reward)

    def conserve_resources(self):
        if self.resource_level < 5:  # Example conservation logic
            self.resource_level = max(0, self.resource_level - 1)

    def share_resources(self):
        nearby_agents = self.get_nearby_system_agents()
        for agent in nearby_agents:
            self.transfer_resources(agent)

    def gather_resources(self):
        for resource in self.environment.resources:
            if not resource.is_depleted():
                dist = np.sqrt(
                    (self.position[0] - resource.position[0]) ** 2
                    + (self.position[1] - resource.position[1]) ** 2
                )

                if dist < 20:  # Increased gathering range
                    gather_amount = min(
                        3, resource.amount
                    )  # Increased gathering amount
                    resource.consume(gather_amount)
                    self.resource_level += gather_amount
                    break

    def get_nearby_system_agents(self):
        # Define the maximum distance for considering agents as "nearby"
        max_distance = 30  # Adjust this value based on your simulation needs

        nearby_agents = []
        for agent in self.environment.agents:
            if isinstance(agent, SystemAgent) and agent != self and agent.alive:
                # Calculate Euclidean distance between agents
                distance = np.sqrt(
                    (self.position[0] - agent.position[0]) ** 2
                    + (self.position[1] - agent.position[1]) ** 2
                )

                # Add agent to nearby list if within range
                if distance <= max_distance:
                    nearby_agents.append(agent)

        return nearby_agents

    def transfer_resources(self, agent):
        # Transfer resources to another agent
        if self.resource_level > 0:
            agent.resource_level += 1
            self.resource_level -= 1


class IndividualAgent(Agent):
    def act(self):
        # First check if agent should die
        if not self.alive:
            return

        super().act()  # Call parent class act() for death check
        if not self.alive:  # If died during death check, skip the rest
            return

        initial_resources = self.resource_level
        self.gather_resources()
        self.consume_resources()

        # Calculate reward based on resource change
        reward = self.resource_level - initial_resources
        self.learn(reward)

    def gather_resources(self):
        for resource in self.environment.resources:
            if not resource.is_depleted():
                dist = np.sqrt(
                    (self.position[0] - resource.position[0]) ** 2
                    + (self.position[1] - resource.position[1]) ** 2
                )

                if dist < 20:  # Increased gathering range
                    gather_amount = min(
                        3, resource.amount
                    )  # Increased gathering amount
                    resource.consume(gather_amount)
                    self.resource_level += gather_amount
                    break

    def consume_resources(self):
        self.resource_level = max(
            0, self.resource_level - 1
        )  # Ensure it doesn't go negative


# ==============================
# Data Collection and Metrics
# ==============================


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


# ==============================
# Simulation Loop
# ==============================


def setup_logging():
    # Initialize colorama for Windows support
    init()

    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Setup logging configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Custom formatter for colored console output
    class ColoredFormatter(logging.Formatter):
        FORMATS = {
            logging.DEBUG: Fore.CYAN
            + "%(asctime)s - %(levelname)s - %(message)s"
            + Style.RESET_ALL,
            logging.INFO: Fore.GREEN
            + "%(asctime)s - %(levelname)s - %(message)s"
            + Style.RESET_ALL,
            logging.WARNING: Fore.YELLOW
            + "%(asctime)s - %(levelname)s - %(message)s"
            + Style.RESET_ALL,
            logging.ERROR: Fore.RED
            + "%(asctime)s - %(levelname)s - %(message)s"
            + Style.RESET_ALL,
            logging.CRITICAL: Fore.RED
            + Style.BRIGHT
            + "%(asctime)s - %(levelname)s - %(message)s"
            + Style.RESET_ALL,
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)

    # Create handlers
    file_handler = logging.FileHandler(f"logs/simulation_{timestamp}.log")
    console_handler = logging.StreamHandler()

    # Set formatters
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    console_handler.setFormatter(ColoredFormatter())

    # Setup root logger
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)


def create_resource_grid(environment, step_number):
    # Set scaling factor based on window size (assuming typical display)
    scale = 8  # Increased from 4 to 8 for better visibility
    width = environment.width * scale
    height = environment.height * scale

    # Create a black background
    grid = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw resources with larger radius
    for resource in environment.resources:
        x = int(resource.position[0] * scale)
        y = int(resource.position[1] * scale)
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)
        value = min(255, resource.amount * 8)  # Scale resource amount to visible range

        # Make resources appear larger by filling a larger circle
        radius = 3  # Increased from 1 to 3
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:  # Circular shape
                    new_x = min(max(x + dx, 0), width - 1)
                    new_y = min(max(y + dy, 0), height - 1)
                    grid[new_y, new_x] = [0, value, 0]  # Green color for resources

    # Draw agents with larger radius
    for agent in environment.agents:
        if agent.alive:
            x = int(agent.position[0] * scale)
            y = int(agent.position[1] * scale)
            x = min(max(x, 0), width - 1)
            y = min(max(y, 0), height - 1)

            # Different colors for different agent types
            color = [255, 0, 0] if isinstance(agent, SystemAgent) else [0, 0, 255]

            # Make agents appear as larger dots
            radius = 4  # Increased from 2 to 4
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= radius * radius:  # Circular shape
                        new_x = min(max(x + dx, 0), width - 1)
                        new_y = min(max(y + dy, 0), height - 1)
                        grid[new_y, new_x] = color

    # Convert to PIL Image
    img = Image.fromarray(grid)

    # Add cycle number text
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)  # Increased font size
    except:
        font = ImageFont.load_default()

    draw.text((10, 10), f"Cycle: {step_number}", fill=(255, 255, 255), font=font)

    return img


def run_simulation(environment, num_steps, data_collector):
    logging.info(f"Starting simulation with {len(environment.agents)} agents")
    logging.info(f"Initial resources: {sum(r.amount for r in environment.resources)}")

    frames = []

    for step in range(num_steps):
        # Create and store the current frame with step number
        img = create_resource_grid(environment, step)
        frames.append(img)

        if step % 100 == 0:  # Log every 100 steps
            alive_agents = sum(1 for agent in environment.agents if agent.alive)
            total_resources = sum(r.amount for r in environment.resources)
            avg_agent_resources = sum(
                a.resource_level for a in environment.agents if a.alive
            ) / max(alive_agents, 1)

            logging.info(f"\nStep {step}/{num_steps}:")
            logging.info(f"Alive agents: {alive_agents}")
            logging.info(f"Total resources: {total_resources}")
            logging.info(f"Average agent resources: {avg_agent_resources:.2f}")

        for agent in environment.agents[:]:
            if agent.alive:
                agent.act()
                agent.move()
                agent.reproduce()
            else:
                environment.agents.remove(agent)
                agent.die()

        # Environment updates
        environment.update()

        # Data collection
        data_collector.collect(environment, step)

    # Save the animation with longer duration per frame
    frames[0].save(
        "Agents/resource_distribution.gif",
        save_all=True,
        append_images=frames[1:],
        duration=200,  # Changed from 50 to 200 milliseconds per frame
        loop=0,
    )


# ==============================
# Experiment Scenarios
# ==============================


def setup_experiment_scenario(scenario_params, initial_resource_level):
    environment = Environment(
        width=scenario_params["environment_size"][0],
        height=scenario_params["environment_size"][1],
        resource_distribution=scenario_params["resource_distribution"],
    )

    # Initialize agents
    agent_id = 0
    for _ in range(scenario_params["agent_population"]["system_agents"]):
        position = (
            random.uniform(0, environment.width),
            random.uniform(0, environment.height),
        )
        agent = SystemAgent(agent_id, position, initial_resource_level, environment)
        environment.add_agent(agent)
        agent_id += 1

    for _ in range(scenario_params["agent_population"]["individual_agents"]):
        position = (
            random.uniform(0, environment.width),
            random.uniform(0, environment.height),
        )
        agent = IndividualAgent(agent_id, position, initial_resource_level, environment)
        environment.add_agent(agent)
        agent_id += 1

    return environment


# ==============================
# Data Analysis and Visualization
# ==============================


def analyze_data(data_collector):
    df = data_collector.to_dataframe()
    return df


def visualize_results(df):
    # Population chart
    plt.figure(figsize=(12, 6))
    plt.plot(df["step"], df["system_agent_count"], label="System Agents", color="blue")
    plt.plot(
        df["step"],
        df["individual_agent_count"],
        label="Individual Agents",
        color="orange",
    )
    plt.xlabel("Simulation Step")
    plt.ylabel("Agent Count")
    plt.title("Agent Population Over Time")
    plt.legend()
    plt.savefig("Agents/population_over_time.png")
    plt.close()

    # Combined Resources and Consumption chart
    plt.figure(figsize=(12, 6))
    plt.plot(df["step"], df["total_resources"], label="Total Resources", color="green")
    plt.plot(
        df["step"],
        df["total_consumption"],
        label="Total Resource Consumption",
        color="red",
    )
    plt.xlabel("Simulation Step")
    plt.ylabel("Resource Units")
    plt.title("Resource Levels and Consumption Over Time")
    plt.legend()
    plt.savefig("Agents/resource_metrics.png")
    plt.close()

    # Average resources per agent chart
    plt.figure(figsize=(12, 6))
    plt.plot(
        df["step"],
        df["average_resource_per_agent"],
        label="Average Resource per Agent",
        color="purple",
    )
    plt.xlabel("Simulation Step")
    plt.ylabel("Average Resource per Agent")
    plt.title("Average Resource per Agent Over Time")
    plt.legend()
    plt.savefig("Agents/average_resource_per_agent.png")
    plt.close()


# ==============================
# Main Function
# ==============================


def main(num_steps=500, config=None, db_path="simulation_results.db"):
    """Run the simulation with the given parameters."""
    # Setup logging
    setup_logging()

    # Use default parameters if no config provided
    if config is None:
        config = SimulationConfig()

    # Setup experiment
    environment = Environment(
        width=config.width,
        height=config.height,
        resource_distribution={"type": "random", "amount": config.initial_resources},
        db_path=db_path,
        max_resource=config.max_resource_amount,
        config=config
    )

    # Initialize agents
    for _ in range(config.system_agents):
        position = (
            random.uniform(0, environment.width),
            random.uniform(0, environment.height),
        )
        agent = SystemAgent(
            agent_id=environment.get_next_agent_id(),
            position=position,
            resource_level=config.initial_resource_level,
            environment=environment,
        )
        environment.add_agent(agent)

    for _ in range(config.individual_agents):
        position = (
            random.uniform(0, environment.width),
            random.uniform(0, environment.height),
        )
        agent = IndividualAgent(
            agent_id=environment.get_next_agent_id(),
            position=position,
            resource_level=config.initial_resource_level,
            environment=environment,
        )
        environment.add_agent(agent)

    # Run simulation
    for step in range(num_steps):
        if step % 100 == 0:
            logging.info(f"Step {step}/{num_steps}")

        # Check if any agents are still alive
        alive_agents = [agent for agent in environment.agents if agent.alive]
        if not alive_agents:
            logging.info(f"Simulation ended at step {step}: All agents have died")
            break

        # Update all agents
        for agent in environment.agents[:]:
            if agent.alive:
                agent.act()
                agent.move()
                agent.reproduce()
            else:
                environment.agents.remove(agent)
                agent.die()

        # Update environment
        environment.update()

    # Close database connection
    environment.db.close()

    return environment


if __name__ == "__main__":
    # When run directly, start visualization after simulation
    env = main()
    root = tk.Tk()
    visualizer = SimulationVisualizer(root, db_path="simulation_results.db")
    visualizer.run()
