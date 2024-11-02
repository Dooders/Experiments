import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import logging
from datetime import datetime
import os
from colorama import init, Fore, Style  # Add color support for console
from PIL import Image
import io
from PIL import ImageDraw, ImageFont

# ==============================
# Environment Setup
# ==============================


class Environment:
    def __init__(self, width, height, resource_distribution):
        self.width = width
        self.height = height
        self.agents = []
        self.resources = []
        self.time = 0
        self.initialize_resources(resource_distribution)

    def initialize_resources(self, distribution):
        for i in range(distribution["amount"]):
            position = (random.uniform(0, self.width), random.uniform(0, self.height))
            resource = Resource(
                resource_id=i, 
                position=position, 
                amount=random.randint(10, 30)  # Increased initial resource amounts
            )
            self.resources.append(resource)

    def add_agent(self, agent):
        self.agents.append(agent)

    def update(self):
        self.time += 1
        self.regenerate_resources()

    def regenerate_resources(self):
        for resource in self.resources:
            if resource.amount < 30 and random.random() < 0.1:  # Increased regen chance and max amount
                resource.amount += 2  # Increased regen amount


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
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class Agent:
    def __init__(self, agent_id, position, resource_level, environment):
        self.agent_id = agent_id
        self.position = position  # (x, y) coordinates
        self.resource_level = resource_level
        self.alive = True
        self.environment = environment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(input_dim=4, output_dim=4).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=2)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.last_state = None
        self.last_action = None
        self.max_movement = 8  # Increased movement range
        self.total_reward = 0
        self.episode_rewards = []
        self.losses = []
        self.starvation_threshold = 0  # New attribute to track low resources
        self.max_starvation = 15  # Increased survival time without resources

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
        # Reduced base resource consumption
        self.resource_level -= 0.1  # Reduced from 0.2
        
        if self.resource_level <= 0:
            self.starvation_threshold += 1
            if self.starvation_threshold >= self.max_starvation:
                self.die()
                return
        else:
            self.starvation_threshold = 0

    def reproduce(self):
        if len(self.environment.agents) >= 300:
            return

        # Made reproduction easier
        if self.resource_level >= 10:  # Reduced from 12
            offspring_cost = 6  # Reduced from 8
            if self.resource_level >= offspring_cost + 2:
                new_agent = self.create_offspring()
                self.environment.add_agent(new_agent)
                self.resource_level -= offspring_cost

    def create_offspring(self):
        return type(self)(
            agent_id=len(self.environment.agents),
            position=self.position,
            resource_level=5,  # Increased starting resources for offspring
            environment=self.environment,
        )

    def die(self):
        self.alive = False


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
                    gather_amount = min(3, resource.amount)  # Increased gathering amount
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
                    (self.position[0] - agent.position[0]) ** 2 +
                    (self.position[1] - agent.position[1]) ** 2
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
                    gather_amount = min(3, resource.amount)  # Increased gathering amount
                    resource.consume(gather_amount)
                    self.resource_level += gather_amount
                    break

    def consume_resources(self):
        self.resource_level = max(0, self.resource_level - 1)  # Ensure it doesn't go negative


# ==============================
# Data Collection and Metrics
# ==============================


class DataCollector:
    def __init__(self):
        self.data = []

    def collect(self, environment, step):
        data_point = {
            "step": step,
            "system_agent_count": sum(
                isinstance(agent, SystemAgent) and agent.alive
                for agent in environment.agents
            ),
            "individual_agent_count": sum(
                isinstance(agent, IndividualAgent) and agent.alive
                for agent in environment.agents
            ),
            "total_resources": sum(
                resource.amount for resource in environment.resources
            ),
            "total_consumption": sum(
                agent.resource_level for agent in environment.agents if agent.alive
            ),
            "average_resource_per_agent": (
                (
                    sum(
                        agent.resource_level
                        for agent in environment.agents
                        if agent.alive
                    )
                    / len(environment.agents)
                )
                if len(environment.agents) > 0
                else 0
            ),
        }
        self.data.append(data_point)

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
    # Set scaling factor (e.g., 4x larger)
    scale = 4
    width = environment.width * scale
    height = environment.height * scale
    
    # Create a larger grid
    grid = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Convert resource positions to scaled integer coordinates
    for resource in environment.resources:
        x = int(resource.position[0] * scale)
        y = int(resource.position[1] * scale)
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)
        value = resource.amount
        
        # Make resources appear larger by filling a small square
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                new_x = min(max(x + dx, 0), width - 1)
                new_y = min(max(y + dy, 0), height - 1)
                grid[new_y, new_x] = [value, value, value]
    
    # Normalize to 0-255 range
    if grid.max() > 0:
        grid = (grid / grid.max() * 255).astype(np.uint8)
    
    # Add agents as larger red dots
    for agent in environment.agents:
        if agent.alive:
            x = int(agent.position[0] * scale)
            y = int(agent.position[1] * scale)
            x = min(max(x, 0), width - 1)
            y = min(max(y, 0), height - 1)
            
            # Make agents appear as larger dots
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    new_x = min(max(x + dx, 0), width - 1)
                    new_y = min(max(y + dy, 0), height - 1)
                    grid[new_y, new_x] = [255, 0, 0]
    
    # Convert to PIL Image
    img = Image.fromarray(grid)
    
    # Add cycle number text
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
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
    
    # Save the animation
    frames[0].save(
        'Agents/resource_distribution.gif',
        save_all=True,
        append_images=frames[1:],
        duration=50,  # Duration for each frame in milliseconds
        loop=0
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


def main():
    # Setup logging
    setup_logging()

    # Define simulation parameters
    num_steps = 500
    environment_size = (100, 100)
    resource_distribution = {
        "type": "random", 
        "amount": 60  # Increased number of resources
    }
    agent_population = {
        "system_agents": 25, 
        "individual_agents": 25
    }
    initial_resource_level = 12  # Increased starting resources
    scenario_params = {
        "environment_size": environment_size,
        "resource_distribution": resource_distribution,
        "agent_population": agent_population,
    }

    # Setup experiment
    environment = setup_experiment_scenario(scenario_params, initial_resource_level)

    # Data collector
    data_collector = DataCollector()

    # Run simulation
    run_simulation(environment, num_steps, data_collector)

    # Analyze data
    df = analyze_data(data_collector)

    # Visualize results
    visualize_results(df)

    logging.info("Simulation completed")


if __name__ == "__main__":
    main()
