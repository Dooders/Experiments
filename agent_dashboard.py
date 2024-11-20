import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

# Database connection
DB_PATH = "simulation.db"

@st.cache_data
def load_data(query, params=None, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    if params:
        df = pd.read_sql_query(query, conn, params=params)
    else:
        df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Query data for dashboard
agents_query = "SELECT agent_id, birth_time, death_time, agent_type FROM Agents"
agent_states_query = """
SELECT step_number, agent_id, position_x, position_y, resource_level, current_health, total_reward
FROM AgentStates
WHERE agent_id = ?
"""

agent_actions_query = """
SELECT step_number, action_type, reward
FROM AgentActions
WHERE agent_id = ?
"""

# Sidebar: Agent Selection
st.sidebar.title("Agent Dashboard")
st.sidebar.header("Select an Agent")
agents = load_data(agents_query)
selected_agent_id = st.sidebar.selectbox(
    "Choose Agent ID", agents["agent_id"].unique()
)

# Fetch data for the selected agent
agent_states = load_data(agent_states_query, params=(selected_agent_id,))
agent_actions = load_data(agent_actions_query, params=(selected_agent_id,))

# Dashboard Header
st.title(f"Agent {selected_agent_id} Dashboard")
st.markdown(f"""
**Agent Type**: {agents.loc[agents['agent_id'] == selected_agent_id, 'agent_type'].values[0]}  
**Birth Time**: {agents.loc[agents['agent_id'] == selected_agent_id, 'birth_time'].values[0]}  
**Death Time**: {agents.loc[agents['agent_id'] == selected_agent_id, 'death_time'].values[0]}  
""")

# Resource and Health Trends
st.header("Resource and Health Trends")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(agent_states["step_number"], agent_states["resource_level"], label="Resources", marker="o")
ax.plot(agent_states["step_number"], agent_states["current_health"], label="Health", marker="o")
ax.set_title("Resource and Health Over Time")
ax.set_xlabel("Simulation Step")
ax.set_ylabel("Value")
ax.legend()
st.pyplot(fig)

# Action Distribution
st.header("Action Distribution")
action_counts = agent_actions["action_type"].value_counts()
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=action_counts.index, y=action_counts.values, ax=ax)
ax.set_title("Actions Taken")
ax.set_xlabel("Action Type")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Reward Trends
st.header("Reward Trends")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(agent_states["step_number"], agent_states["total_reward"], label="Cumulative Reward", marker="o", color="green")
ax.set_title("Rewards Over Time")
ax.set_xlabel("Simulation Step")
ax.set_ylabel("Reward")
st.pyplot(fig)

# Movement Heatmap
st.header("Movement Heatmap")
heatmap_data = agent_states.pivot(index="position_y", columns="position_x", values="step_number").fillna(0)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(heatmap_data, cmap="coolwarm", ax=ax, cbar_kws={"label": "Step Number"})
ax.set_title("Agent Movement Heatmap")
st.pyplot(fig)

# Actions Table
st.header("Actions Log")
st.dataframe(agent_actions)
