import logging
import sqlite3
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class SimulationDatabase:
    """A database manager for storing and retrieving simulation state data.

    This class provides an interface to an SQLite database for persisting simulation
    data, including agent states, resource states, and aggregate metrics over time.
    It handles the creation and management of the database schema, as well as
    providing methods for logging and retrieving simulation data.

    The database schema consists of four main tables:
    - Agents: Stores agent metadata and lifecycle information
    - AgentStates: Tracks agent positions and resources over time
    - ResourceStates: Tracks resource positions and amounts over time
    - SimulationSteps: Stores aggregate metrics for each simulation step

    Attributes
    ----------
    conn : sqlite3.Connection
        Connection to the SQLite database
    cursor : sqlite3.Cursor
        Cursor for executing SQL commands

    Example
    -------
    >>> db = SimulationDatabase("simulation.db")
    >>> db.log_agent(1, 0, "system", (0.5, 0.5), 100.0)
    >>> db.log_step(0, agent_states, resource_states, metrics)
    >>> data = db.get_simulation_data(0)
    >>> db.close()
    """

    def __init__(self, db_path: str):
        """Initialize a new SimulationDatabase instance.

        Creates a connection to an SQLite database and initializes the required tables
        if they don't exist. The database will store agent states, resource states,
        and simulation metrics over time.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database file. If the file doesn't exist,
            it will be created.
        """
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        """Create the required database schema if tables don't exist.

        Creates four main tables:
        - Agents: Stores agent metadata and lifecycle information
        - AgentStates: Tracks agent positions and resources over time
        - ResourceStates: Tracks resource positions and amounts over time
        - SimulationSteps: Stores aggregate metrics for each simulation step
        """
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
        """Log the creation of a new agent in the simulation.

        Parameters
        ----------
        agent_id : int
            Unique identifier for the agent
        birth_time : int
            Simulation step when the agent was created
        agent_type : str
            Type of agent (e.g., 'system', 'independent', 'control')
        position : Tuple[float, float]
            Initial (x, y) coordinates of the agent
        initial_resources : float
            Starting resource level of the agent

        Example
        -------
        >>> db.log_agent(1, 0, 'system', (0.5, 0.5), 100.0)
        """
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
        """Record the death time of an agent in the simulation.

        Parameters
        ----------
        agent_id : int
            Unique identifier of the agent that died
        death_time : int
            Simulation step when the agent died

        Example
        -------
        >>> db.update_agent_death(agent_id=1, death_time=150)
        """
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
        """Log comprehensive simulation state data for a single time step.

        Parameters
        ----------
        step_number : int
            Current simulation step number
        agent_states : List[Tuple]
            List of agent state tuples, each containing:
            (agent_id, position_x, position_y, resource_level)
        resource_states : List[Tuple]
            List of resource state tuples, each containing:
            (resource_id, amount, position_x, position_y)
        metrics : Dict
            Dictionary containing simulation metrics:
            - total_agents: int
            - system_agents: int
            - independent_agents: int
            - control_agents: int
            - total_resources: float
            - average_agent_resources: float

        Notes
        -----
        This method performs a batch insert of all simulation state data
        for efficiency. It commits the transaction after all data is written.
        """
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
        """Retrieve all simulation data for a specific time step.

        Parameters
        ----------
        step_number : int
            The simulation step to retrieve data for

        Returns
        -------
        Dict
            Dictionary containing:
            - agent_states: List of tuples (agent_id, type, x, y, resources)
            - resource_states: List of tuples (resource_id, amount, x, y)
            - metrics: Dict of aggregate metrics for the step

        Example
        -------
        >>> data = db.get_simulation_data(100)
        >>> print(f"Number of agents: {len(data['agent_states'])}")
        >>> print(f"Total resources: {data['metrics']['total_resources']}")
        """
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
        """Retrieve historical metrics for the entire simulation.

        Returns
        -------
        Dict
            Dictionary containing:
            - steps: List of step numbers
            - metrics: Dict of metric lists including:
                - total_agents
                - system_agents
                - independent_agents
                - control_agents
                - total_resources
                - average_agent_resources

        Notes
        -----
        This method is useful for generating time series plots of simulation metrics.
        The returned lists are ordered by step number.
        """
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
        """Export simulation metrics data to a CSV file.

        Parameters
        ----------
        filepath : str
            Path where the CSV file should be saved

        Notes
        -----
        Exports the following columns:
        - step_number
        - total_agents
        - system_agents
        - independent_agents
        - control_agents
        - total_resources
        - average_agent_resources

        Example
        -------
        >>> db.export_data("simulation_results.csv")
        """
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
        """Flush all database buffers and ensure data is written to disk.

        This method ensures that all pending changes are written to the database file
        by:
        1. Committing any pending transactions
        2. Forcing a write-ahead log checkpoint

        Raises
        ------
        Exception
            If there is an error during the flush operation

        Notes
        -----
        This is particularly important to call before closing the database
        or when ensuring data persistence is critical.
        """
        try:
            self.conn.commit()  # Commit any pending transactions
            self.cursor.execute(
                "PRAGMA wal_checkpoint(FULL)"
            )  # Force write-ahead log checkpoint
        except Exception as e:
            logger.error(f"Error flushing database buffers: {e}")
            raise

    def close(self):
        """Close the database connection safely.

        Performs the following cleanup operations:
        1. Flushes all database buffers
        2. Closes the database connection

        Raises
        ------
        Exception
            If there is an error during the closing operation

        Notes
        -----
        Always call this method when finished with the database to prevent
        resource leaks and ensure data integrity.
        """
        try:
            self.flush_all_buffers()  # Ensure all data is written before closing
            self.conn.close()
        except Exception as e:
            logger.error(f"Error closing database: {e}")
            raise

    def log_resource(
        self, resource_id: int, initial_amount: float, position: Tuple[float, float]
    ):
        """Log the creation of a new resource in the simulation.

        Parameters
        ----------
        resource_id : int
            Unique identifier for the resource
        initial_amount : float
            Starting amount of the resource
        position : Tuple[float, float]
            (x, y) coordinates of the resource location

        Notes
        -----
        Resources are logged at step_number=0 to represent their initial state
        in the simulation.

        Example
        -------
        >>> db.log_resource(resource_id=1, initial_amount=1000.0, position=(0.3, 0.7))
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
        """Log complete simulation state for a single time step.

        This is a high-level method that handles the conversion of agent and
        resource objects into their database representation before logging.

        Parameters
        ----------
        step_number : int
            Current simulation step number
        agents : List
            List of agent objects, each must have attributes:
            - agent_id: int
            - position: Tuple[float, float]
            - resource_level: float
            - alive: bool
        resources : List
            List of resource objects, each must have attributes:
            - resource_id: int
            - amount: float
            - position: Tuple[float, float]
        metrics : Dict
            Dictionary containing simulation metrics:
            - total_agents: int
            - system_agents: int
            - independent_agents: int
            - control_agents: int
            - total_resources: float
            - average_agent_resources: float

        Notes
        -----
        This method automatically filters out dead agents and converts
        complex objects into their database representation before logging.

        Example
        -------
        >>> metrics = {
        ...     "total_agents": 10,
        ...     "system_agents": 5,
        ...     "independent_agents": 3,
        ...     "control_agents": 2,
        ...     "total_resources": 1000.0,
        ...     "average_agent_resources": 100.0
        ... }
        >>> db.log_simulation_step(step_number=1, agents=agent_list,
        ...                       resources=resource_list, metrics=metrics)
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
