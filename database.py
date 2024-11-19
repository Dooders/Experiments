import logging
import sqlite3
import threading
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd

if TYPE_CHECKING:
    from environment import Environment

logger = logging.getLogger(__name__)


class SimulationDatabase:
    
    _thread_local = threading.local()

    def __init__(self, db_path: str):
        """Initialize a new SimulationDatabase instance."""
        self.db_path = db_path
        self._setup_connection()

    def _setup_connection(self):
        """Create a new connection for the current thread."""
        if not hasattr(self._thread_local, "conn"):
            self._thread_local.conn = sqlite3.connect(self.db_path)
            self._thread_local.cursor = self._thread_local.conn.cursor()
            self._create_tables()

    @property
    def conn(self):
        """Get thread-local connection."""
        self._setup_connection()
        return self._thread_local.conn

    @property
    def cursor(self):
        """Get thread-local cursor."""
        self._setup_connection()
        return self._thread_local.cursor

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
                position_x REAL,
                position_y REAL,
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
                is_defending INTEGER,
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
            
            CREATE TABLE IF NOT EXISTS AgentActions (
                action_id INTEGER PRIMARY KEY,
                step_number INTEGER NOT NULL,
                agent_id INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                action_target_id INTEGER,
                position_before TEXT,
                position_after TEXT,
                resources_before REAL,
                resources_after REAL,
                reward REAL,
                details TEXT, -- JSON-encoded dictionary for action-specific details
                FOREIGN KEY(agent_id) REFERENCES Agents(agent_id)
            );
            
            -- Track learning experiences
            CREATE TABLE IF NOT EXISTS LearningExperiences (
                experience_id INTEGER PRIMARY KEY,
                step_number INTEGER,
                agent_id INTEGER,
                module_type TEXT,
                state_before TEXT,
                action_taken INTEGER,
                reward REAL,
                state_after TEXT,
                loss REAL,
                FOREIGN KEY(agent_id) REFERENCES Agents(agent_id)
            );
        """
        )
        self.conn.commit()

    def _execute_in_transaction(self, func):
        """Execute database operations within a transaction."""
        try:
            # Ensure we have a connection for this thread
            self._setup_connection()

            with self.conn:  # This automatically handles commit/rollback
                return func()
        except Exception as e:
            logger.error(
                f"Database transaction failed in thread {threading.current_thread().name}: {e}"
            )
            raise

    def batch_log_agents(self, agent_data: List[Dict]):
        """Batch insert multiple agents.

        Parameters
        ----------
        agent_data : List[Dict]
            List of dictionaries containing agent data
        """
        def _insert():
            values = [
                (
                    data["agent_id"],
                    data["birth_time"],
                    data["agent_type"],
                    data["position"][0],    # Extract x coordinate
                    data["position"][1],    # Extract y coordinate
                    data["initial_resources"],
                    data["max_health"],
                    data["starvation_threshold"],
                    data.get("genome_id"),
                    data.get("parent_id"),
                    data.get("generation", 0),
                )
                for data in agent_data
            ]

            self.cursor.executemany(
                """
                INSERT INTO Agents (
                    agent_id, birth_time, agent_type, position_x, position_y,
                    initial_resources, max_health, starvation_threshold, 
                    genome_id, parent_id, generation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )

        self._execute_in_transaction(_insert)

    def log_step(
        self,
        step_number: int,
        agent_states: List[Tuple],
        resource_states: List[Tuple],
        metrics: Dict,
    ):
        """Log comprehensive simulation state data for a single time step."""

        def _insert():
            # Log agent states
            self.cursor.executemany(
                """
                INSERT INTO AgentStates (
                    step_number, agent_id, position_x, position_y, resource_level,
                    current_health, max_health, starvation_threshold, is_defending, 
                    total_reward, age
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

            # Log metrics
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

        self._execute_in_transaction(_insert)

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
        """Close the database connection for the current thread."""
        if hasattr(self._thread_local, "conn"):
            try:
                self.flush_all_buffers()
                self._thread_local.conn.close()
                del self._thread_local.conn
                del self._thread_local.cursor
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

    def log_agent_action(
        self,
        step_number: int,
        agent_id: int,
        action_type: str,
        action_target_id: Optional[int] = None,
        position_before: Optional[Tuple[float, float]] = None,
        position_after: Optional[Tuple[float, float]] = None,
        resources_before: Optional[float] = None,
        resources_after: Optional[float] = None,
        reward: Optional[float] = None,
        details: Optional[Dict] = None,
    ):
        """Log an agent's action during the simulation."""
        import json

        # Convert positions to string representation if provided
        pos_before_str = str(position_before) if position_before is not None else None
        pos_after_str = str(position_after) if position_after is not None else None

        # Convert details dict to JSON string if provided
        details_json = json.dumps(details) if details is not None else None

        self.cursor.execute(
            """
            INSERT INTO AgentActions (
                step_number, agent_id, action_type, action_target_id,
                position_before, position_after, resources_before,
                resources_after, reward, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step_number,
                agent_id,
                action_type,
                action_target_id,
                pos_before_str,
                pos_after_str,
                resources_before,
                resources_after,
                reward,
                details_json,
            ),
        )
        # Remove the commit from here - we'll commit in batches instead

    def batch_log_agent_actions(self, actions: List[Dict]):
        """Batch insert multiple agent actions at once.

        Parameters
        ----------
        actions : List[Dict]
            List of action dictionaries containing:
            - step_number: int
            - agent_id: int
            - action_type: str
            - action_target_id: Optional[int]
            - position_before: Optional[Tuple[float, float]]
            - position_after: Optional[Tuple[float, float]]
            - resources_before: Optional[float]
            - resources_after: Optional[float]
            - reward: Optional[float]
            - details: Optional[Dict]
        """
        import json

        # Prepare all actions for batch insert
        values = []
        for action in actions:
            pos_before_str = (
                str(action["position_before"])
                if action.get("position_before") is not None
                else None
            )
            pos_after_str = (
                str(action["position_after"])
                if action.get("position_after") is not None
                else None
            )
            details_json = (
                json.dumps(action.get("details"))
                if action.get("details") is not None
                else None
            )

            values.append(
                (
                    action["step_number"],
                    action["agent_id"],
                    action["action_type"],
                    action.get("action_target_id"),
                    pos_before_str,
                    pos_after_str,
                    action.get("resources_before"),
                    action.get("resources_after"),
                    action.get("reward"),
                    details_json,
                )
            )

        # Batch insert
        self.cursor.executemany(
            """
            INSERT INTO AgentActions (
                step_number, agent_id, action_type, action_target_id,
                position_before, position_after, resources_before,
                resources_after, reward, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )

    def log_learning_experience(
        self,
        step_number: int,
        agent_id: int,
        module_type: str,
        state_before: str,
        action_taken: int,
        reward: float,
        state_after: str,
        loss: float,
    ):
        """Log a single learning experience during the simulation.

        Parameters
        ----------
        step_number : int
            Current simulation step number
        agent_id : int
            ID of the agent that had the learning experience
        module_type : str
            Type of learning module (e.g., 'movement', 'combat', etc.)
        state_before : str
            String representation of the state before action
        action_taken : int
            Integer representing the action chosen
        reward : float
            Reward received for the action
        state_after : str
            String representation of the state after action
        loss : float
            Loss value from the learning update
        """
        self.cursor.execute(
            """
            INSERT INTO LearningExperiences (
                step_number, agent_id, module_type, state_before,
                action_taken, reward, state_after, loss
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step_number,
                agent_id,
                module_type,
                state_before,
                action_taken,
                reward,
                state_after,
                loss,
            ),
        )

    def batch_log_learning_experiences(self, experiences: List[Dict]):
        """Batch insert multiple learning experiences at once.

        Parameters
        ----------
        experiences : List[Dict]
            List of experience dictionaries containing:
            - step_number: int
            - agent_id: int
            - module_type: str
            - state_before: str
            - action_taken: int
            - reward: float
            - state_after: str
            - loss: float
        """
        values = [
            (
                exp["step_number"],
                exp["agent_id"],
                exp["module_type"],
                exp["state_before"],
                exp["action_taken"],
                exp["reward"],
                exp["state_after"],
                exp["loss"],
            )
            for exp in experiences
        ]

        self.cursor.executemany(
            """
            INSERT INTO LearningExperiences (
                step_number, agent_id, module_type, state_before,
                action_taken, reward, state_after, loss
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        self.conn.commit()

    def log_agent(
        self,
        agent_id: int,
        birth_time: int,
        agent_type: str,
        position: tuple,
        initial_resources: float,
        max_health: float,
        starvation_threshold: int,
        genome_id: Optional[str] = None,
        parent_id: Optional[int] = None,
        generation: int = 0,
    ) -> None:
        """Log a single agent's creation to the database.

        Parameters
        ----------
        agent_id : int
            Unique identifier for the agent
        birth_time : int
            Time step when agent was created
        agent_type : str
            Type of agent (e.g., 'SystemAgent', 'IndependentAgent')
        position : tuple
            Initial (x, y) coordinates
        initial_resources : float
            Starting resource level
        max_health : float
            Maximum health points
        starvation_threshold : int
            Steps agent can survive without resources
        genome_id : Optional[str]
            Unique identifier for agent's genome
        parent_id : Optional[int]
            ID of parent agent if created through reproduction
        generation : int
            Generation number in evolutionary lineage
        """

        def _insert():
            self.cursor.execute(
                """
                INSERT INTO Agents (
                    agent_id, birth_time, agent_type, position_x, position_y,
                    initial_resources, max_health, starvation_threshold, 
                    genome_id, parent_id, generation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_id,
                    birth_time,
                    agent_type,
                    position[0],
                    position[1],
                    initial_resources,
                    max_health,
                    starvation_threshold,
                    genome_id,
                    parent_id,
                    generation,
                ),
            )

        self._execute_in_transaction(_insert)

    def update_agent_death(self, agent_id: int, death_time: int) -> None:
        """Update an agent's death time in the database.

        Parameters
        ----------
        agent_id : int
            ID of the agent that died
        death_time : int
            Time step when the agent died
        """

        def _update():
            self.cursor.execute(
                """
                UPDATE Agents 
                SET death_time = ? 
                WHERE agent_id = ?
                """,
                (death_time, agent_id),
            )

        self._execute_in_transaction(_update)
