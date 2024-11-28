import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.database import SimulationDatabase


def analyze_learning_experiences(db_path: str):
    """Analyze learning experiences from the simulation database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file
    """
    # Initialize database connection
    db = SimulationDatabase(db_path)

    # Query RL data using the existing schema
    session = db.session
    query = session.query(
        LearningExperience.step_number,
        LearningExperience.agent_id,
        LearningExperience.module_type,
        LearningExperience.state_before,
        LearningExperience.action_taken,
        LearningExperience.reward,
        LearningExperience.state_after,
        LearningExperience.loss,
    ).order_by(LearningExperience.agent_id, LearningExperience.step_number)

    rl_data = pd.read_sql(query.statement, session.bind)

    # Show data preview
    print("\nData Preview:")
    print(rl_data.head())

    # If we have no data, exit early
    if rl_data.empty:
        print("No learning experience data found in the database.")
        db.close()
        return

    # Convert reward and loss columns to numeric
    rl_data['reward'] = pd.to_numeric(rl_data['reward'], errors='coerce')
    rl_data['loss'] = pd.to_numeric(rl_data['loss'], errors='coerce')

    # Ensure data is clean
    rl_data.fillna(0, inplace=True)

    # Decode JSON states into dictionaries
    import json

    def decode_state(state):
        if isinstance(state, str):
            return json.loads(state)
        return state

    rl_data["state_before"] = rl_data["state_before"].apply(decode_state)
    rl_data["state_after"] = rl_data["state_after"].apply(decode_state)

    # Convert states into feature vectors
    def state_to_vector(state):
        if state:
            return [
                state.get("position_x", 0),
                state.get("position_y", 0),
                state.get("resources", 0),
            ]
        return [0, 0, 0]

    rl_data["state_vector_before"] = rl_data["state_before"].apply(state_to_vector)
    rl_data["state_vector_after"] = rl_data["state_after"].apply(state_to_vector)

    # Calculate state deltas
    rl_data["state_delta"] = rl_data.apply(
        lambda row: np.array(row["state_vector_after"])
        - np.array(row["state_vector_before"]),
        axis=1,
    )

    # Calculate cumulative rewards per agent
    rl_data["cumulative_reward"] = rl_data.groupby("agent_id")["reward"].cumsum()

    # Create visualizations
    plot_learning_metrics(rl_data)

    # Close database connection
    db.close()


def plot_learning_metrics(rl_data: pd.DataFrame):
    """Generate plots for learning metrics analysis.

    Parameters
    ----------
    rl_data : pd.DataFrame
        DataFrame containing learning experience data
    """
    # Check if the dataframe is empty
    if rl_data.empty:
        print("No learning experience data found in the database.")
        return

    # Plot 1: Reward vs. Loss
    plt.figure(figsize=(10, 6))
    plt.scatter(rl_data["reward"], rl_data["loss"], alpha=0.6)
    plt.title("Reward vs. Loss")
    plt.xlabel("Reward")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # Plot 2: State Delta vs. Reward
    state_deltas = np.array(rl_data["state_delta"].tolist())
    if len(state_deltas) > 0:  # Only plot if we have data
        plt.figure(figsize=(10, 6))
        plt.scatter(
            [delta[0] for delta in state_deltas],  # Position X Delta
            rl_data["reward"],
            alpha=0.6,
            label="Position X Delta"
        )
        plt.scatter(
            [delta[1] for delta in state_deltas],  # Position Y Delta
            rl_data["reward"],
            alpha=0.6,
            label="Position Y Delta"
        )
        plt.title("State Change vs. Reward")
        plt.xlabel("State Change")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Plot 3: Reward Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(rl_data["reward"], bins=50, alpha=0.7, color="blue", label="Rewards")
    plt.axvline(
        rl_data["reward"].mean(), color="red", linestyle="--", label="Mean Reward"
    )
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Plot 4: Action Frequencies
    plt.figure(figsize=(10, 6))
    rl_data["action_taken"].value_counts().plot(kind="bar", color="purple", alpha=0.7)
    plt.title("Action Frequencies")
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # Plot 5: Reward Over Time
    plt.figure(figsize=(10, 6))
    reward_time = rl_data.groupby("step_number")["reward"].mean()
    plt.plot(reward_time, label="Average Reward")
    plt.title("Reward Over Time")
    plt.xlabel("Simulation Step")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    db_path = "simulations/simulation.db"  # Replace with your database path
    analyze_learning_experiences(db_path)
