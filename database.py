import logging
import sqlite3
from typing import Dict, List, Tuple, TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from environment import Environment

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
    - SimulationSteps: Stores comprehensive simulation metrics over time including:
        * Population metrics (total agents, births, deaths, etc.)
        * Resource metrics (efficiency, distribution)
        * Performance metrics (health, age, rewards)
        * Combat and cooperation metrics
        * Evolutionary metrics (genetic diversity)

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
        - SimulationSteps: Stores comprehensive simulation metrics over time including:
            * Population metrics (total agents, births, deaths, etc.)
            * Resource metrics (efficiency, distribution)
            * Performance metrics (health, age, rewards)
            * Combat and cooperation metrics
            * Evolutionary metrics (genetic diversity)
        """
        self.cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS Agents (
                agent_id INTEGER PRIMARY KEY,
                birth_time INTEGER,
                death_time INTEGER,
                agent_type TEXT,
                initial_position TEXT,
                initial_resources REAL,
                max_health REAL,
                starvation_threshold INTEGER,
                genome_id TEXT,
                parent_id INTEGER,
                generation INTEGER,
                FOREIGN KEY(parent_id) REFERENCES Agents(agent_id)
            );

            CREATE TABLE IF NOT EXISTS AgentStates (
                step_number INTEGER,
                agent_id INTEGER,
                position_x REAL,
                position_y REAL,
                resource_level REAL,
                current_health REAL,
                max_health REAL,
                starvation_threshold INTEGER,
                is_defending BOOLEAN,
                total_reward REAL,
                age INTEGER,
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
                average_agent_resources REAL,
                
                -- Add population dynamics metrics
                births INTEGER,
                deaths INTEGER,
                current_max_generation INTEGER,
                
                -- Add resource metrics
                resource_efficiency REAL,
                resource_distribution_entropy REAL,
                
                -- Add agent performance metrics
                average_agent_health REAL,
                average_agent_age INTEGER,
                average_reward REAL,
                
                -- Add combat metrics
                combat_encounters INTEGER,
                successful_attacks INTEGER,
                resources_shared REAL,
                
                -- Add evolutionary metrics
                genetic_diversity REAL,
                dominant_genome_ratio REAL
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
        max_health: float,
        starvation_threshold: int,
        genome_id: str = None,
        parent_id: Optional[int] = None,
        generation: int = 0,
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
        max_health : float
            Maximum health points of the agent
        starvation_threshold : int
            Number of steps agent can survive without resources
        genome_id : str, optional
            Unique identifier for agent's genome
        parent_id : int, optional
            ID of parent agent if this agent was created through reproduction
        generation : int, optional
            Generation number of the agent in evolutionary lineage
        """
        self.cursor.execute(
            """
            INSERT INTO Agents (
                agent_id, birth_time, agent_type, initial_position, initial_resources,
                max_health, starvation_threshold, genome_id, parent_id, generation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                agent_id,
                birth_time,
                agent_type,
                str(position),
                initial_resources,
                max_health,
                starvation_threshold,
                genome_id,
                parent_id,
                generation,
            ),
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
            (agent_id, position_x, position_y, resource_level, current_health,
             max_health, starvation_threshold, is_defending, total_reward, age)
        resource_states : List[Tuple]
            List of resource state tuples, each containing:
            (resource_id, amount, position_x, position_y)
        metrics : Dict
            Dictionary containing simulation metrics:
            - Population metrics:
                * total_agents: Total number of alive agents
                * system_agents: Number of system agents
                * independent_agents: Number of independent agents
                * control_agents: Number of control agents
                * births: Number of new agents this step
                * deaths: Number of agent deaths this step
                * current_max_generation: Highest generation number
            - Resource metrics:
                * total_resources: Total resources in environment
                * average_agent_resources: Mean resources per agent
                * resource_efficiency: Resource utilization effectiveness
                * resource_distribution_entropy: Measure of resource distribution
            - Performance metrics:
                * average_agent_health: Mean health across agents
                * average_agent_age: Mean age of agents
                * average_reward: Mean reward accumulated
            - Combat metrics:
                * combat_encounters: Number of combat interactions
                * successful_attacks: Number of successful attacks
                * resources_shared: Amount of resources shared
            - Evolutionary metrics:
                * genetic_diversity: Measure of genome variety
                * dominant_genome_ratio: Prevalence of most common genome
        """
        # Log agent states
        self.cursor.executemany(
            """
            INSERT INTO AgentStates (
                step_number, agent_id, position_x, position_y, resource_level,
                current_health, max_health, starvation_threshold, is_defending, total_reward,
                age
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [(step_number, *state[:4], *state[4:]) for state in agent_states],
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

        # Enhanced metrics logging
        self.cursor.execute(
            """
            INSERT INTO SimulationSteps (
                step_number, total_agents, system_agents, independent_agents,
                control_agents, total_resources, average_agent_resources,
                births, deaths, current_max_generation,
                resource_efficiency, resource_distribution_entropy,
                average_agent_health, average_agent_age, average_reward,
                combat_encounters, successful_attacks, resources_shared,
                genetic_diversity, dominant_genome_ratio
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step_number,
                metrics["total_agents"],
                metrics["system_agents"],
                metrics["independent_agents"],
                metrics["control_agents"],
                metrics["total_resources"],
                metrics["average_agent_resources"],
                metrics["births"],
                metrics["deaths"],
                metrics["current_max_generation"],
                metrics["resource_efficiency"],
                metrics["resource_distribution_entropy"],
                metrics["average_agent_health"],
                metrics["average_agent_age"],
                metrics["average_reward"],
                metrics["combat_encounters"],
                metrics["successful_attacks"],
                metrics["resources_shared"],
                metrics["genetic_diversity"],
                metrics["dominant_genome_ratio"],
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
            - agent_states: List of tuples (agent_id, type, x, y, resources, health, max_health,
                                          starvation_threshold, is_defending, total_reward, age)
            - resource_states: List of tuples (resource_id, amount, x, y)
            - metrics: Dict of aggregate metrics for the step
        """
        # Get agent states
        self.cursor.execute(
            """
            SELECT a.agent_id, ag.agent_type, a.position_x, a.position_y, a.resource_level,
                   a.current_health, a.max_health, a.starvation_threshold, 
                   a.is_defending, a.total_reward, a.age
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
        environment: "Environment",  # Add environment parameter
    ):
        """Log complete simulation state for a single time step.

        Parameters
        ----------
        step_number : int
            Current simulation step number
        agents : List
            List of agent objects, each must have attributes:
            - agent_id: int
            - position: Tuple[float, float]
            - resource_level: float
            - current_health: float
            - max_health: float
            - starvation_threshold: int
            - is_defending: bool
            - total_reward: float
            - birth_time: int
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
        environment : Environment
            Reference to the environment for time tracking

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
        ...                       resources=resource_list, metrics=metrics,
        ...                       environment=env)
        """
        # Convert agents to state tuples
        agent_states = [
            (
                agent.agent_id,
                agent.position[0],
                agent.position[1],
                agent.resource_level,
                agent.current_health,
                agent.max_health,
                agent.starvation_threshold,
                1 if agent.is_defending else 0,
                agent.total_reward,
                environment.time
                - agent.birth_time,  # Calculate age using environment time
            )
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

    def get_agent_lifespan_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate lifespan statistics for different agent types.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary containing lifespan statistics per agent type:
            {
                'agent_type': {
                    'average_lifespan': float,
                    'max_lifespan': float,
                    'min_lifespan': float
                }
            }
        """
        self.cursor.execute(
            """
            WITH AgentLifespans AS (
                SELECT 
                    ag.agent_type,
                    CASE 
                        WHEN ag.death_time IS NULL THEN MAX(a.age)
                        ELSE ag.death_time - ag.birth_time 
                    END as lifespan
                FROM Agents ag
                LEFT JOIN AgentStates a ON ag.agent_id = a.agent_id
                GROUP BY ag.agent_id
            )
            SELECT 
                agent_type,
                AVG(lifespan) as avg_lifespan,
                MAX(lifespan) as max_lifespan,
                MIN(lifespan) as min_lifespan
            FROM AgentLifespans
            GROUP BY agent_type
        """
        )

        results = {}
        for row in self.cursor.fetchall():
            results[row[0]] = {
                "average_lifespan": row[1],
                "max_lifespan": row[2],
                "min_lifespan": row[3],
            }
        return results
