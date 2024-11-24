import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.solve.ShortestPath import ShortestPath

# Create maze
m = Maze()
m.generator = Prims(5, 5)
m.generate()

# Manually create openings at the top and bottom
m.grid[0, 1] = 0  # Top opening (entrance)
m.grid[-1, -2] = 0  # Bottom opening (exit)

# Set start and end coordinates that match our openings
m.start = (0, 1)  # Top opening coordinates
m.end = (m.grid.shape[0] - 1, m.grid.shape[1] - 2)  # Bottom opening coordinates

# Set the solving algorithm
m.solver = ShortestPath()

# # Solve the maze
# m.solve()

# # Plot the maze with solution
# plt.figure(figsize=(10, 10))
# plt.imshow(m.grid, cmap="binary", interpolation="nearest")

# # Plot the solution path in red
# if len(m.solutions) > 0:
#     solution_path = np.array(m.solutions[0])
#     plt.plot(solution_path[:, 1], solution_path[:, 0], "r-", linewidth=3, alpha=0.7)

# plt.axis("off")
# plt.show()


class DiamondMazeEnv(gym.Env):
    def __init__(self, maze, start, end):
        super().__init__()
        self.maze = maze
        self.start = start
        self.end = end
        self.state = start
        self.action_space = spaces.Discrete(4)  # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.observation_space = spaces.Box(
            low=0, high=maze.shape[0] - 1, shape=(2,), dtype=np.int32
        )

    def reset(self):
        self.state = self.start
        return np.array(self.state, dtype=np.int32)

    def step(self, action):
        x, y = self.state
        old_distance = np.linalg.norm(np.array(self.end) - np.array(self.state))
        
        # Update position based on action (only move if the new position is a path (0))
        if action == 0 and y > 0 and self.maze[y - 1, x] == 0:  # Up
            y -= 1
        elif action == 1 and y < self.maze.shape[0] - 1 and self.maze[y + 1, x] == 0:  # Down
            y += 1
        elif action == 2 and x > 0 and self.maze[y, x - 1] == 0:  # Left
            x -= 1
        elif action == 3 and x < self.maze.shape[1] - 1 and self.maze[y, x + 1] == 0:  # Right
            x += 1

        self.state = (x, y)
        new_distance = np.linalg.norm(np.array(self.end) - np.array(self.state))

        # Reward structure
        if self.state == self.end:
            reward = 100  # Reduced from 1000 to better scale with step penalties
            done = True
        else:
            # Stronger reward/penalty for distance changes
            reward = 5 * (old_distance - new_distance)
            # Smaller step penalty
            reward -= 0.05
            # Additional penalty for hitting walls (when position doesn't change)
            if (x, y) == self.state:
                reward -= 1
            done = False

        return np.array(self.state, dtype=np.int32), reward, done, False, {}

    def render(self):
        maze_copy = self.maze.copy()
        x, y = self.state
        maze_copy[y, x] = 2
        print(maze_copy)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.gamma = 0.99
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        self.steps_done = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.update_target_network()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
        )

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Convert batch to numpy arrays first
        batch = random.sample(self.memory, self.batch_size)
        states = np.vstack([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.vstack([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])

        # Convert to tensors efficiently
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        # Get current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Get next Q values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Update model
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network and epsilon
        if self.steps_done % 100 == 0:
            self.update_target_network()
        self.steps_done += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# Generate the maze
size = 5
maze = m.grid
start = (0, size - 1)
end = (size * 2 - 2, size - 1)

# Initialize environment and agent
env = DiamondMazeEnv(maze, start, end)
state_dim = 2  # (x, y) position
action_dim = 4  # Up, Down, Left, Right
agent = DQNAgent(state_dim, action_dim)

# Simplified training loop
episodes = 2000
max_steps = 200
print_freq = 50

for e in range(episodes):
    state = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, truncated, info = env.step(action)
        
        # Store experience
        agent.remember(state, action, reward, next_state, done)
        
        # Train every 4 steps if we have enough samples
        if len(agent.memory) >= agent.batch_size and step % 4 == 0:
            agent.replay()
            
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    if (e + 1) % print_freq == 0:
        print(f"Episode {e+1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

print("Training Complete!")
