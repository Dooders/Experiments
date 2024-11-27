import matplotlib.pyplot as plt
import numpy as np
import torch
from mazelib import Maze
from mazelib.generate.Prims import Prims

from agents.maze_agent import MazeAgent
from config import SimulationConfig
from environments.maze_environment import MazeEnv

# Generate the maze
m = Maze()
m.generator = Prims(5, 5)
m.generate()

# # Manually create openings at the top and bottom
m.grid[0, 1] = 0  # Top opening (entrance)
m.grid[-1, -2] = 0  # Bottom opening (exit)

# Set start and end coordinates that match our openings
m.start = (0, 1)  # Top opening coordinates
m.end = (m.grid.shape[0] - 1, m.grid.shape[1] - 2)  # Bottom opening coordinates

maze = m.grid

# Initialize environment and agent
config = SimulationConfig.from_yaml("config.yaml")
env = MazeEnv(maze, m.start, m.end, config=config)
state_dim = 2  # (x, y) position
action_dim = 4  # Up, Down, Left, Right
agent = MazeAgent(124, m.start, 0, env, config)

# Simplified training loop
episodes = 500
max_steps = 100  # Reduced from 200
print_freq = 10  # More frequent printing

for e in range(episodes):
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, truncated, info = env.step(action)

        # Store experience
        agent.remember(state, action, reward, next_state, done)

        # Train every step if we have enough samples
        if len(agent.memory) >= agent.batch_size:  # Removed step % 4 condition
            agent.replay()

        state = next_state
        total_reward += reward

        if done:
            break

    if (e + 1) % print_freq == 0:
        print(
            f"Episode {e+1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}"
        )

print("Training Complete!")


# Visualization code
def create_agent_path_visualization(env, agent):
    # Set epsilon to 0 for pure exploitation
    original_epsilon = agent.epsilon
    agent.epsilon = 0

    # Reset environment
    state = env.reset()
    path_positions = [state]

    max_steps = 100
    step_count = 0

    # Collect path positions
    while step_count < max_steps:
        with torch.no_grad():
            action = agent.act(state)
        next_state, _, done, _, _ = env.step(action)
        path_positions.append(next_state)
        state = next_state

        step_count += 1
        if done:
            break

    # Create visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(env.maze, cmap="binary")

    # Plot start and end points
    plt.plot(env.start[0], env.start[1], "go", markersize=10, label="Start")
    plt.plot(env.end[0], env.end[1], "bo", markersize=10, label="End")

    # Create path with less extreme gradient (starting from 0.3 instead of near-zero)
    path = np.array(path_positions)
    for i in range(len(path) - 1):
        alpha = 0.3 + (0.7 * i / (len(path) - 1))  # Gradient from 0.3 to 1.0
        plt.plot(
            path[i : i + 2, 0],
            path[i : i + 2, 1],
            color="red",
            alpha=alpha,
            linewidth=2,
        )

    plt.grid(True)
    plt.legend()
    plt.savefig("maze_solution_path.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Restore original epsilon
    agent.epsilon = original_epsilon
    print(f"Path visualization completed after {step_count} steps")


# Create the visualization
print("Creating path visualization...")
create_agent_path_visualization(env, agent)
print("Visualization saved as 'maze_solution_path.png'")
