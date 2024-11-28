import matplotlib.pyplot as plt
import pandas as pd

from core.database import SimulationDatabase
from sqlalchemy.orm import sessionmaker


def fetch_reward_efficiency_data(db: SimulationDatabase) -> pd.DataFrame:
    """
    Fetch reward efficiency data for each action type.
    :param db: SimulationDatabase instance
    :return: DataFrame with reward efficiency by action type and agent group
    """
    session = db.session
    query = session.query(
        AgentAction.action_type,
        Agent.agent_type,
        func.count(AgentAction.action_id).label("frequency"),
        func.sum(AgentAction.reward).label("total_reward"),
        (func.sum(AgentAction.reward) * 1.0 / func.count(AgentAction.action_id)).label("reward_efficiency")
    ).join(Agent, AgentAction.agent_id == Agent.agent_id).group_by(
        AgentAction.action_type, Agent.agent_type
    )
    data = pd.read_sql(query.statement, session.bind)
    return data


def analyze_reward_efficiency(data: pd.DataFrame):
    """
    Analyze reward efficiency data and create visualizations.
    :param data: DataFrame with columns ['action_type', 'agent_type', 'frequency', 'total_reward', 'reward_efficiency']
    :return: Summary statistics and visualizations
    """
    plt.figure(figsize=(12, 5))

    # Plot 1: Overall reward efficiency by action type
    plt.subplot(1, 2, 1)
    efficiency_by_action = data.groupby("action_type")["reward_efficiency"].mean()
    efficiency_by_action.plot(
        kind="bar",
        title="Reward Efficiency by Action Type",
        ylabel="Reward Efficiency",
        xlabel="Action Type",
        color="skyblue",
        edgecolor="black",
    )
    plt.xticks(rotation=45)

    # Plot 2: Reward efficiency by agent type and action
    plt.subplot(1, 2, 2)
    efficiency_pivot = data.pivot(
        index="action_type", columns="agent_type", values="reward_efficiency"
    )
    efficiency_pivot.plot(
        kind="bar",
        title="Reward Efficiency by Agent Type and Action",
        ylabel="Reward Efficiency",
        xlabel="Action Type",
    )
    plt.xticks(rotation=45)
    plt.legend(title="Agent Type", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

    return efficiency_by_action, efficiency_pivot


def reward_efficiency_pipeline(db_path: str):
    """
    Full pipeline to analyze reward efficiency.
    :param db_path: Path to SQLite database
    """
    # Initialize database connection
    db = SimulationDatabase(db_path)

    try:
        # Fetch and analyze data
        data = fetch_reward_efficiency_data(db)
        efficiency_by_action, efficiency_by_group = analyze_reward_efficiency(data)

        # Print summary statistics
        print("Reward Efficiency by Action Type:\n", efficiency_by_action)
        print("\nReward Efficiency by Agent Type and Action:\n", efficiency_by_group)

    finally:
        # Ensure database connection is closed
        db.close()


def main(db_path: str):
    """
    Main entry point for reward efficiency analysis.
    :param db_path: Path to SQLite database
    """
    reward_efficiency_pipeline(db_path)


if __name__ == "__main__":
    db_path = "simulations/simulation.db"
    main(db_path)
