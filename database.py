import sqlite3
from typing import Any, Dict, List, Tuple

import pandas as pd


class SimulationDatabase:
    def __init__(
        self, db_path: str = "simulation_results.db", table_names: Dict[str, str] = None
    ) -> None:
        """
        Initialize database connection and setup tables.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database file
        table_names : Dict[str, str], optional
            Custom table names for the database schema
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        # Default table names
        self.tables = {
            "agents": "Agents",
            "resources": "Resources",
            "steps": "SimulationSteps",
            "agent_states": "AgentStates",
            "resource_states": "ResourceStates",
            "metrics": "SimulationMetrics",
        }

        # Update with custom table names if provided
        if table_names:
            self.tables.update(table_names)

        # Add batch processing parameters
        self.batch_size = 1000
        self.agent_state_buffer = []
        self.resource_state_buffer = []
        self.metric_buffer = []

        self.setup_tables()

    def setup_tables(self):
        """Create necessary database tables if they don't exist."""
        self.cursor.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS {self.tables['agents']} (
                agent_id INTEGER PRIMARY KEY,
                birth_time INTEGER,
                death_time INTEGER,
                agent_type TEXT,
                initial_position_x REAL,
                initial_position_y REAL,
                initial_resources REAL
            );

            CREATE TABLE IF NOT EXISTS {self.tables['resources']} (
                resource_id INTEGER PRIMARY KEY,
                initial_amount INTEGER,
                position_x REAL,
                position_y REAL
            );

            CREATE TABLE IF NOT EXISTS {self.tables['steps']} (
                step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                step_number INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS {self.tables['agent_states']} (
                state_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id INTEGER,
                step_id INTEGER,
                position_x REAL,
                position_y REAL,
                resource_level REAL,
                alive BOOLEAN,
                agent_type TEXT,
                FOREIGN KEY(agent_id) REFERENCES {self.tables['agents']}(agent_id),
                FOREIGN KEY(step_id) REFERENCES {self.tables['steps']}(step_id)
            );

            CREATE TABLE IF NOT EXISTS {self.tables['resource_states']} (
                state_id INTEGER PRIMARY KEY AUTOINCREMENT,
                resource_id INTEGER,
                step_id INTEGER,
                position_x REAL,
                position_y REAL,
                amount INTEGER,
                FOREIGN KEY(resource_id) REFERENCES {self.tables['resources']}(resource_id),
                FOREIGN KEY(step_id) REFERENCES {self.tables['steps']}(step_id)
            );

            CREATE TABLE IF NOT EXISTS {self.tables['metrics']} (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                step_id INTEGER,
                metric_name TEXT,
                metric_value REAL,
                FOREIGN KEY(step_id) REFERENCES {self.tables['steps']}(step_id)
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
    ) -> None:
        """Log a new agent to the database."""
        self.cursor.execute(
            """
            INSERT INTO Agents (agent_id, birth_time, agent_type, 
                              initial_position_x, initial_position_y, initial_resources)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                agent_id,
                birth_time,
                agent_type,
                position[0],
                position[1],
                initial_resources,
            ),
        )
        self.conn.commit()

    def log_resource(
        self, resource_id: int, initial_amount: int, position: Tuple[float, float]
    ) -> None:
        """Log a new resource to the database."""
        self.cursor.execute(
            """
            INSERT INTO Resources (resource_id, initial_amount, position_x, position_y)
            VALUES (?, ?, ?, ?)
        """,
            (resource_id, initial_amount, position[0], position[1]),
        )
        self.conn.commit()

    def log_simulation_step(
        self,
        step_number: int,
        agents: List[Any],
        resources: List[Any],
        metrics: Dict[str, float],
    ) -> None:
        """Log the current state of the simulation with batch processing."""
        try:
            # Insert step first
            self.cursor.execute(
                "INSERT INTO SimulationSteps (step_number) VALUES (?)", (step_number,)
            )
            step_id = self.cursor.lastrowid

            # Add metrics to buffer
            for metric_name, value in metrics.items():
                self.metric_buffer.append((step_id, metric_name, value))

            # Add agent states to buffer
            self.agent_state_buffer.extend(
                [
                    (
                        agent.agent_id,
                        step_id,
                        agent.position[0],
                        agent.position[1],
                        agent.resource_level,
                        agent.alive,
                        agent.__class__.__name__,
                    )
                    for agent in agents
                ]
            )

            # Add resource states to buffer
            self.resource_state_buffer.extend(
                [
                    (
                        resource.resource_id,
                        step_id,
                        resource.position[0],
                        resource.position[1],
                        resource.amount,
                    )
                    for resource in resources
                ]
            )

            # Flush if buffer size exceeds batch_size OR if it's a batch interval
            should_flush = (
                len(self.agent_state_buffer) >= self.batch_size
                or step_number % self.batch_size == 0
            )

            if should_flush:
                self._flush_agent_states()
                self._flush_resource_states()
                self._flush_metrics()

        except Exception as e:
            print(f"Error logging simulation step {step_number}: {e}")
            import traceback

            traceback.print_exc()

    def _flush_agent_states(self) -> None:
        """Flush agent state buffer to database."""
        if self.agent_state_buffer:
            self.cursor.executemany(
                """
                INSERT INTO AgentStates 
                (agent_id, step_id, position_x, position_y, resource_level, alive, agent_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                self.agent_state_buffer,
            )
            self.agent_state_buffer = []
            self.conn.commit()

    def _flush_resource_states(self) -> None:
        """Flush resource state buffer to database."""
        if self.resource_state_buffer:
            self.cursor.executemany(
                """
                INSERT INTO ResourceStates 
                (resource_id, step_id, position_x, position_y, amount)
                VALUES (?, ?, ?, ?, ?)
                """,
                self.resource_state_buffer,
            )
            self.resource_state_buffer = []
            self.conn.commit()

    def _flush_metrics(self) -> None:
        """Flush metrics buffer to database."""
        if self.metric_buffer:
            self.cursor.executemany(
                """
                INSERT INTO SimulationMetrics (step_id, metric_name, metric_value)
                VALUES (?, ?, ?)
                """,
                self.metric_buffer,
            )
            self.metric_buffer = []
            self.conn.commit()

    def update_agent_death(self, agent_id: int, death_time: int) -> None:
        """Update the death time of an agent."""
        self.cursor.execute(
            """
            UPDATE Agents 
            SET death_time = ? 
            WHERE agent_id = ?
        """,
            (death_time, agent_id),
        )
        self.conn.commit()

    def get_simulation_data(self, step_number: int) -> Dict[str, Any]:
        """Retrieve simulation data for a specific step."""
        # Get agent states
        self.cursor.execute(
            f"""
            SELECT a.agent_id, a.agent_type, s.position_x, s.position_y, 
                   s.resource_level, s.alive
            FROM {self.tables['agent_states']} s
            JOIN {self.tables['agents']} a ON a.agent_id = s.agent_id
            JOIN {self.tables['steps']} st ON st.step_id = s.step_id
            WHERE st.step_number = ?
        """,
            (step_number,),
        )
        agent_states = self.cursor.fetchall()

        # Get resource states
        self.cursor.execute(
            f"""
            SELECT r.resource_id, s.amount, r.position_x, r.position_y
            FROM {self.tables['resource_states']} s
            JOIN {self.tables['resources']} r ON r.resource_id = s.resource_id
            JOIN {self.tables['steps']} st ON st.step_id = s.step_id
            WHERE st.step_number = ?
        """,
            (step_number,),
        )
        resource_states = self.cursor.fetchall()

        # Get metrics
        self.cursor.execute(
            f"""
            SELECT metric_name, metric_value
            FROM {self.tables['metrics']} m
            JOIN {self.tables['steps']} s ON s.step_id = m.step_id
            WHERE s.step_number = ?
        """,
            (step_number,),
        )
        metrics = dict(self.cursor.fetchall())

        return {
            "agent_states": agent_states,
            "resource_states": resource_states,
            "metrics": metrics,
        }

    def close(self):
        """Close the database connection after flushing all buffers."""
        self._flush_agent_states()
        self._flush_resource_states()
        self._flush_metrics()
        self.conn.close()

    def get_historical_data(self):
        """
        Fetch historical data for all metrics up to the current step.
        Returns a dictionary with steps and metrics.
        """
        try:
            # Get all metrics grouped by step
            self.cursor.execute(
                """
                SELECT s.step_number,
                       m.metric_name,
                       m.metric_value
                FROM SimulationSteps s
                JOIN SimulationMetrics m ON m.step_id = s.step_id
                ORDER BY s.step_number ASC
            """
            )

            rows = self.cursor.fetchall()

            if not rows:
                return {
                    "steps": [],
                    "metrics": {
                        "system_agents": [],
                        "independent_agents": [],
                        "total_resources": [],
                    },
                }

            # Process the data
            steps = []
            metrics = {
                "system_agents": [],
                "independent_agents": [],
                "total_resources": [],
                "average_agent_resources": [],
            }

            current_step = None
            step_metrics = {}

            for step, metric_name, metric_value in rows:
                if step != current_step:
                    if current_step is not None:
                        # Store the completed step data
                        steps.append(current_step)
                        for metric_key in metrics:
                            metrics[metric_key].append(step_metrics.get(metric_key, 0))
                    # Start new step
                    current_step = step
                    step_metrics = {}

                step_metrics[metric_name] = metric_value

            # Don't forget to add the last step
            if current_step is not None:
                steps.append(current_step)
                for metric_key in metrics:
                    metrics[metric_key].append(step_metrics.get(metric_key, 0))

            return {"steps": steps, "metrics": metrics}

        except Exception as e:
            # Only log critical errors
            return {
                "steps": [],
                "metrics": {
                    "system_agents": [],
                    "independent_agents": [],
                    "total_resources": [],
                },
            }

    def export_data(self, output_file: str = "simulation_data.csv") -> None:
        """Export simulation data to a CSV file."""
        query = """
            SELECT s.step_number, m.metric_name, m.metric_value
            FROM SimulationMetrics m
            JOIN SimulationSteps s ON s.step_id = m.step_id
            ORDER BY s.step_number, m.metric_name
        """

        self.cursor.execute(query)
        results = self.cursor.fetchall()

        # Convert to DataFrame and pivot
        df = pd.DataFrame(results, columns=["step", "metric", "value"])
        df_pivot = df.pivot(index="step", columns="metric", values="value")

        # Export to CSV
        df_pivot.to_csv(output_file)
        return df_pivot
