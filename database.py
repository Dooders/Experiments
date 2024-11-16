import logging
import sqlite3
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class SimulationDatabase:
    def __init__(self, db_path: str):
        """Initialize database connection and create tables if they don't exist."""
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        self.cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS Agents (
                agent_id INTEGER PRIMARY KEY,
                birth_time INTEGER,
                death_time INTEGER,
                agent_type TEXT,
                initial_position TEXT,
                initial_resources REAL
            );

            CREATE TABLE IF NOT EXISTS AgentStates (
                step_number INTEGER,
                agent_id INTEGER,
                position_x REAL,
                position_y REAL,
                resource_level REAL,
                FOREIGN KEY(agent_id) REFERENCES Agents(agent_id)
            );

            CREATE TABLE IF NOT EXISTS ResourceStates (
                step_number INTEGER,
                resource_id INTEGER,
                amount REAL,
                position_x REAL,
                position_y REAL
            );

            CREATE TABLE IF NOT EXISTS SimulationSteps (
                step_number INTEGER PRIMARY KEY,
                total_agents INTEGER,
                system_agents INTEGER,
                independent_agents INTEGER,
                control_agents INTEGER,
                total_resources REAL,
                average_agent_resources REAL
            );
        """
        )
        self.conn.commit()

    def log_agent(
        self,
        agent_id: int,
        birth_time: int,
        agent_type: str,
        position: Tuple[float, float],
        initial_resources: float,
    ):
        """Log new agent creation."""
        self.cursor.execute(
            """
            INSERT INTO Agents (
                agent_id, birth_time, agent_type, initial_position, initial_resources
            ) VALUES (?, ?, ?, ?, ?)
        """,
            (agent_id, birth_time, agent_type, str(position), initial_resources),
        )
        self.conn.commit()

    def update_agent_death(self, agent_id: int, death_time: int):
        """Record agent death time."""
        self.cursor.execute(
            """
            UPDATE Agents 
            SET death_time = ? 
            WHERE agent_id = ?
        """,
            (death_time, agent_id),
        )
        self.conn.commit()

    def log_step(
        self,
        step_number: int,
        agent_states: List[Tuple],
        resource_states: List[Tuple],
        metrics: Dict,
    ):
        """Log simulation step data."""
        # Log agent states
        self.cursor.executemany(
            """
            INSERT INTO AgentStates (
                step_number, agent_id, position_x, position_y, resource_level
            ) VALUES (?, ?, ?, ?, ?)
        """,
            [(step_number, *state) for state in agent_states],
        )

        # Log resource states
        self.cursor.executemany(
            """
            INSERT INTO ResourceStates (
                step_number, resource_id, amount, position_x, position_y
            ) VALUES (?, ?, ?, ?, ?)
        """,
            [(step_number, *state) for state in resource_states],
        )

        # Log step metrics
        self.cursor.execute(
            """
            INSERT INTO SimulationSteps (
                step_number, total_agents, system_agents, independent_agents,
                control_agents, total_resources, average_agent_resources
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                step_number,
                metrics["total_agents"],
                metrics["system_agents"],
                metrics["independent_agents"],
                metrics["control_agents"],
                metrics["total_resources"],
                metrics["average_agent_resources"],
            ),
        )
        self.conn.commit()

    def get_simulation_data(self, step_number: int) -> Dict:
        """Get all data for a specific simulation step."""
        # Get agent states
        self.cursor.execute(
            """
            SELECT a.agent_id, ag.agent_type, a.position_x, a.position_y, a.resource_level
            FROM AgentStates a
            JOIN Agents ag ON a.agent_id = ag.agent_id
            WHERE a.step_number = ?
        """,
            (step_number,),
        )
        agent_states = self.cursor.fetchall()

        # Get resource states
        self.cursor.execute(
            """
            SELECT resource_id, amount, position_x, position_y
            FROM ResourceStates
            WHERE step_number = ?
        """,
            (step_number,),
        )
        resource_states = self.cursor.fetchall()

        # Get metrics
        self.cursor.execute(
            """
            SELECT total_agents, system_agents, independent_agents, control_agents,
                   total_resources, average_agent_resources
            FROM SimulationSteps
            WHERE step_number = ?
        """,
            (step_number,),
        )
        metrics_row = self.cursor.fetchone()

        metrics = {
            "total_agents": metrics_row[0] if metrics_row else 0,
            "system_agents": metrics_row[1] if metrics_row else 0,
            "independent_agents": metrics_row[2] if metrics_row else 0,
            "control_agents": metrics_row[3] if metrics_row else 0,
            "total_resources": metrics_row[4] if metrics_row else 0,
            "average_agent_resources": metrics_row[5] if metrics_row else 0,
        }

        return {
            "agent_states": agent_states,
            "resource_states": resource_states,
            "metrics": metrics,
        }

    def get_historical_data(self) -> Dict:
        """Get historical data for plotting."""
        self.cursor.execute(
            """
            SELECT step_number, total_agents, system_agents, independent_agents,
                   control_agents, total_resources, average_agent_resources
            FROM SimulationSteps
            ORDER BY step_number
        """
        )
        rows = self.cursor.fetchall()

        return {
            "steps": [row[0] for row in rows],
            "metrics": {
                "total_agents": [row[1] for row in rows],
                "system_agents": [row[2] for row in rows],
                "independent_agents": [row[3] for row in rows],
                "control_agents": [row[4] for row in rows],
                "total_resources": [row[5] for row in rows],
                "average_agent_resources": [row[6] for row in rows],
            },
        }

    def export_data(self, filepath: str):
        """Export simulation data to CSV."""
        # Get all simulation data
        df = pd.read_sql_query(
            """
            SELECT s.step_number, s.total_agents, s.system_agents, s.independent_agents,
                   s.control_agents, s.total_resources, s.average_agent_resources
            FROM SimulationSteps s
            ORDER BY s.step_number
        """,
            self.conn,
        )

        df.to_csv(filepath, index=False)

    def flush_all_buffers(self):
        """Flush all database buffers and ensure data is written to disk."""
        try:
            self.conn.commit()  # Commit any pending transactions
            self.cursor.execute(
                "PRAGMA wal_checkpoint(FULL)"
            )  # Force write-ahead log checkpoint
        except Exception as e:
            logger.error(f"Error flushing database buffers: {e}")
            raise

    def close(self):
        """Close database connection."""
        try:
            self.flush_all_buffers()  # Ensure all data is written before closing
            self.conn.close()
        except Exception as e:
            logger.error(f"Error closing database: {e}")
            raise

    def log_resource(
        self, resource_id: int, initial_amount: float, position: Tuple[float, float]
    ):
        """Log new resource creation.

        Parameters
        ----------
        resource_id : int
            Unique identifier for the resource
        initial_amount : float
            Starting amount of the resource
        position : Tuple[float, float]
            (x, y) coordinates of the resource
        """
        self.cursor.execute(
            """
            INSERT INTO ResourceStates (
                step_number, resource_id, amount, position_x, position_y
            ) VALUES (?, ?, ?, ?, ?)
        """,
            (0, resource_id, initial_amount, position[0], position[1]),
        )
        self.conn.commit()

    def log_simulation_step(
        self,
        step_number: int,
        agents: List,
        resources: List,
        metrics: Dict,
    ):
        """Log simulation step data.

        Parameters
        ----------
        step_number : int
            Current simulation step number
        agents : List
            List of agents in the simulation
        resources : List
            List of resources in the simulation
        metrics : Dict
            Dictionary of metrics to log
        """
        # Convert agents to state tuples
        agent_states = [
            (agent.agent_id, agent.position[0], agent.position[1], agent.resource_level)
            for agent in agents
            if agent.alive
        ]

        # Convert resources to state tuples
        resource_states = [
            (
                resource.resource_id,
                resource.amount,
                resource.position[0],
                resource.position[1],
            )
            for resource in resources
        ]

        # Use existing log_step method
        self.log_step(step_number, agent_states, resource_states, metrics)
