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
    _tables_created = False
    _tables_creation_lock = threading.Lock()

    def __init__(self, db_path: str) -> None:
        """Initialize a new SimulationDatabase instance."""
        self.db_path = db_path
        self._setup_connection()

        # Add batch operation buffers
        self._action_buffer = []
        self._learning_exp_buffer = []
        self._health_incident_buffer = []
        self._buffer_size = 1000  # Adjust based on your needs

        # Create tables only once in a thread-safe manner
        with SimulationDatabase._tables_creation_lock:
            if not SimulationDatabase._tables_created:
                self._create_tables()
                SimulationDatabase._tables_created = True

        # Verify foreign key constraints are working
        if not self.verify_foreign_keys():
            logger.warning(
                "Foreign key constraints are not working properly. "
                "Data integrity cannot be guaranteed."
            )

    def _setup_connection(self):
        """Create a new connection for the current thread and enable foreign key constraints."""
        if not hasattr(self._thread_local, "conn"):
            self._thread_local.conn = sqlite3.connect(self.db_path)
            # Enable foreign key constraints
            self._thread_local.conn.execute("PRAGMA foreign_keys = ON")
            self._thread_local.cursor = self._thread_local.conn.cursor()

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
        """Create the required database schema if tables don't exist."""
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

            CREATE TABLE IF NOT EXISTS HealthIncidents (
                incident_id INTEGER PRIMARY KEY,
                step_number INTEGER NOT NULL,
                agent_id INTEGER NOT NULL,
                health_before REAL NOT NULL,
                health_after REAL NOT NULL,
                cause TEXT NOT NULL,
                details TEXT, -- JSON-encoded additional details
                FOREIGN KEY(agent_id) REFERENCES Agents(agent_id)
            );

            CREATE INDEX IF NOT EXISTS idx_health_incidents_agent_id 
            ON HealthIncidents(agent_id);

            CREATE INDEX IF NOT EXISTS idx_health_incidents_step_number 
            ON HealthIncidents(step_number);

            CREATE TABLE IF NOT EXISTS SimulationConfig (
                config_id INTEGER PRIMARY KEY,
                timestamp INTEGER NOT NULL,
                config_data TEXT NOT NULL  -- JSON-encoded configuration
            );
        """
        )

        # Add indexes for performance optimization
        self.cursor.executescript(
            """
            -- Indexes for AgentStates table
            CREATE INDEX IF NOT EXISTS idx_agent_states_agent_id 
            ON AgentStates(agent_id);
            
            CREATE INDEX IF NOT EXISTS idx_agent_states_step_number 
            ON AgentStates(step_number);
            
            CREATE INDEX IF NOT EXISTS idx_agent_states_composite 
            ON AgentStates(step_number, agent_id);
            
            -- Indexes for Agents table
            CREATE INDEX IF NOT EXISTS idx_agents_agent_type 
            ON Agents(agent_type);
            
            CREATE INDEX IF NOT EXISTS idx_agents_birth_time 
            ON Agents(birth_time);
            
            CREATE INDEX IF NOT EXISTS idx_agents_death_time 
            ON Agents(death_time);
            
            -- Indexes for ResourceStates table
            CREATE INDEX IF NOT EXISTS idx_resource_states_step_number 
            ON ResourceStates(step_number);
            
            CREATE INDEX IF NOT EXISTS idx_resource_states_resource_id 
            ON ResourceStates(resource_id);
            
            -- Indexes for SimulationSteps table
            CREATE INDEX IF NOT EXISTS idx_simulation_steps_step_number 
            ON SimulationSteps(step_number);
            
            -- Indexes for AgentActions table
            CREATE INDEX IF NOT EXISTS idx_agent_actions_step_number 
            ON AgentActions(step_number);
            
            CREATE INDEX IF NOT EXISTS idx_agent_actions_agent_id 
            ON AgentActions(agent_id);
            
            CREATE INDEX IF NOT EXISTS idx_agent_actions_action_type 
            ON AgentActions(action_type);
            
            -- Indexes for LearningExperiences table
            CREATE INDEX IF NOT EXISTS idx_learning_experiences_step_number 
            ON LearningExperiences(step_number);
            
            CREATE INDEX IF NOT EXISTS idx_learning_experiences_agent_id 
            ON LearningExperiences(agent_id);
            
            CREATE INDEX IF NOT EXISTS idx_learning_experiences_module_type 
            ON LearningExperiences(module_type);
            """
        )

    def _execute_in_transaction(self, func):
        """Execute database operations within a transaction."""
        try:
            # Ensure we have a connection for this thread
            self._setup_connection()

            with self.conn:  # This automatically handles commit/rollback
                return func()
        except sqlite3.IntegrityError as e:
            logger.error(f"Database integrity error: {e}")
            if "FOREIGN KEY constraint failed" in str(e):
                logger.error("Foreign key constraint violation detected")
            raise
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
                    data["position"][0],  # Extract x coordinate
                    data["position"][1],  # Extract y coordinate
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

    def _prepare_metrics_values(self, step_number: int, metrics: Dict) -> Tuple:
        """Prepare metrics values for database insertion.

        Parameters
        ----------
        step_number : int
            Current simulation step number
        metrics : Dict
            Dictionary containing simulation metrics

        Returns
        -------
        Tuple
            Values ready for database insertion
        """
        metric_keys = (
            "total_agents",
            "system_agents",
            "independent_agents",
            "control_agents",
            "total_resources",
            "average_agent_resources",
            "births",
            "deaths",
            "current_max_generation",
            "resource_efficiency",
            "resource_distribution_entropy",
            "average_agent_health",
            "average_agent_age",
            "average_reward",
            "combat_encounters",
            "successful_attacks",
            "resources_shared",
            "genetic_diversity",
            "dominant_genome_ratio",
        )
        return (step_number, *[metrics[key] for key in metric_keys])

    def _prepare_agent_state(self, agent, current_time: int) -> Tuple:
        """Prepare agent state data for database insertion.

        Parameters
        ----------
        agent : Agent
            Agent object containing state data
        current_time : int
            Current simulation time

        Returns
        -------
        Tuple
            Agent state values ready for database insertion
        """
        return (
            agent.agent_id,
            *agent.position,
            agent.resource_level,
            agent.current_health,
            agent.max_health,
            agent.starvation_threshold,
            int(agent.is_defending),
            agent.total_reward,
            current_time - agent.birth_time,
        )

    def _prepare_resource_state(self, resource) -> Tuple:
        """Prepare resource state data for database insertion.

        Parameters
        ----------
        resource : Resource
            Resource object containing state data

        Returns
        -------
        Tuple
            Resource state values ready for database insertion
        """
        return (
            resource.resource_id,
            resource.amount,
            *resource.position,
        )

    def _prepare_action_data(self, action: Dict) -> Tuple:
        """Prepare action data for database insertion.

        Parameters
        ----------
        action : Dict
            Dictionary containing action data

        Returns
        -------
        Tuple
            Action values ready for database insertion
        """
        import json

        return (
            action["step_number"],
            action["agent_id"],
            action["action_type"],
            action.get("action_target_id"),
            str(action["position_before"]) if action.get("position_before") else None,
            str(action["position_after"]) if action.get("position_after") else None,
            action.get("resources_before"),
            action.get("resources_after"),
            action.get("reward"),
            json.dumps(action["details"]) if action.get("details") else None,
        )

    def log_step(
        self,
        step_number: int,
        agent_states: List[Tuple],
        resource_states: List[Tuple],
        metrics: Dict,
    ):
        """Log comprehensive simulation state data for a single time step."""

        def _insert():
            # Bulk insert agent states
            if agent_states:
                self._bulk_insert_agent_states(step_number, agent_states)

            # Bulk insert resource states
            if resource_states:
                self._bulk_insert_resource_states(step_number, resource_states)

            # Insert metrics
            self._insert_metrics(step_number, metrics)

        self._execute_in_transaction(_insert)

    def _bulk_insert_agent_states(self, step_number: int, agent_states: List[Tuple]):
        """Bulk insert agent states into database."""
        self.cursor.executemany(
            """
            INSERT INTO AgentStates (
                step_number, agent_id, position_x, position_y, resource_level,
                current_health, max_health, starvation_threshold, is_defending, 
                total_reward, age
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [(step_number, *state) for state in agent_states],
        )

    def _bulk_insert_resource_states(
        self, step_number: int, resource_states: List[Tuple]
    ):
        """Bulk insert resource states into database."""
        self.cursor.executemany(
            """
            INSERT INTO ResourceStates (
                step_number, resource_id, amount, position_x, position_y
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [(step_number, *state) for state in resource_states],
        )

    def _insert_metrics(self, step_number: int, metrics: Dict):
        """Insert metrics data into database."""
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
            self._prepare_metrics_values(step_number, metrics),
        )

    def get_simulation_data(self, step_number: int) -> Dict:
        """Retrieve all simulation data for a specific time step."""
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
                   total_resources, average_agent_resources, births, deaths
            FROM SimulationSteps
            WHERE step_number = ?
        """,
            (step_number,)
        )
        metrics_row = self.cursor.fetchone()

        metrics = {
            "total_agents": metrics_row[0] if metrics_row else 0,
            "system_agents": metrics_row[1] if metrics_row else 0,
            "independent_agents": metrics_row[2] if metrics_row else 0,
            "control_agents": metrics_row[3] if metrics_row else 0,
            "total_resources": metrics_row[4] if metrics_row else 0,
            "average_agent_resources": metrics_row[5] if metrics_row else 0,
            "births": metrics_row[6] if metrics_row else 0,
            "deaths": metrics_row[7] if metrics_row else 0
        }

        return {
            "agent_states": agent_states,
            "resource_states": resource_states,
            "metrics": metrics
        }

    def get_historical_data(self) -> Dict:
        """Retrieve historical metrics for the entire simulation."""
        self.cursor.execute(
            """
            SELECT 
                step_number, 
                total_agents, 
                system_agents, 
                independent_agents,
                control_agents, 
                total_resources, 
                average_agent_resources,
                births,    -- Add births
                deaths     -- Add deaths
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
                "births": [row[7] for row in rows],      # Add births
                "deaths": [row[8] for row in rows]       # Add deaths
            }
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
        """Flush all data buffers and ensure data is written to disk."""
        try:
            self.flush_action_buffer()
            self.flush_learning_buffer()
            self.flush_health_buffer()

            # Only commit if we're not already in a transaction
            if not self.conn.in_transaction:
                self.conn.commit()
            self.cursor.execute("PRAGMA wal_checkpoint(FULL)")
        except Exception as e:
            logger.error(f"Error flushing database buffers: {e}")
            raise

    def close(self):
        """Close the database connection for the current thread."""
        if hasattr(self._thread_local, "conn"):
            try:
                self.flush_all_buffers()  # Ensure all buffered data is written
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

        def _insert():
            self.cursor.execute(
                """
                INSERT INTO ResourceStates (
                    step_number, resource_id, amount, position_x, position_y
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (0, resource_id, initial_amount, position[0], position[1]),
            )

        self._execute_in_transaction(_insert)

    def log_simulation_step(
        self,
        step_number: int,
        agents: List,
        resources: List,
        metrics: Dict,
        environment: "Environment",
    ):
        """Log complete simulation state for a single time step."""
        current_time = environment.time

        # Convert agents and resources to state tuples
        agent_states = [
            self._prepare_agent_state(agent, current_time)
            for agent in agents
            if agent.alive
        ]

        resource_states = [
            self._prepare_resource_state(resource) for resource in resources
        ]

        # Use existing log_step method with prepared data
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
        """Buffer an agent action for batch processing."""
        action_data = {
            "step_number": step_number,
            "agent_id": agent_id,
            "action_type": action_type,
            "action_target_id": action_target_id,
            "position_before": position_before,
            "position_after": position_after,
            "resources_before": resources_before,
            "resources_after": resources_after,
            "reward": reward,
            "details": details,
        }
        self._action_buffer.append(action_data)

        # Flush buffer if it reaches the size limit
        if len(self._action_buffer) >= self._buffer_size:
            self.flush_action_buffer()

    def flush_action_buffer(self):
        """Flush the action buffer by batch inserting all buffered actions."""
        if self._action_buffer:
            self.batch_log_agent_actions(self._action_buffer)
            self._action_buffer.clear()

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
        """Buffer a learning experience for batch processing."""
        self._learning_exp_buffer.append(
            {
                "step_number": step_number,
                "agent_id": agent_id,
                "module_type": module_type,
                "state_before": state_before,
                "action_taken": action_taken,
                "reward": reward,
                "state_after": state_after,
                "loss": loss,
            }
        )

        if len(self._learning_exp_buffer) >= self._buffer_size:
            self.flush_learning_buffer()

    def flush_learning_buffer(self):
        """Flush the learning experience buffer."""
        if self._learning_exp_buffer:
            self.batch_log_learning_experiences(self._learning_exp_buffer)
            self._learning_exp_buffer.clear()

    def log_health_incident(
        self,
        step_number: int,
        agent_id: int,
        health_before: float,
        health_after: float,
        cause: str,
        details: Optional[Dict] = None,
    ):
        """Buffer a health incident for batch processing."""
        self._health_incident_buffer.append(
            {
                "step_number": step_number,
                "agent_id": agent_id,
                "health_before": health_before,
                "health_after": health_after,
                "cause": cause,
                "details": details,
            }
        )

        if len(self._health_incident_buffer) >= self._buffer_size:
            self.flush_health_buffer()

    def flush_health_buffer(self):
        """Flush the health incident buffer."""
        if self._health_incident_buffer:

            def _batch_insert():
                import json

                values = [
                    (
                        incident["step_number"],
                        incident["agent_id"],
                        incident["health_before"],
                        incident["health_after"],
                        incident["cause"],
                        (
                            json.dumps(incident["details"])
                            if incident.get("details")
                            else None
                        ),
                    )
                    for incident in self._health_incident_buffer
                ]

                self.cursor.executemany(
                    """
                    INSERT INTO HealthIncidents (
                        step_number, agent_id, health_before, health_after,
                        cause, details
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )

            self._execute_in_transaction(_batch_insert)
            self._health_incident_buffer.clear()

    def batch_log_agent_actions(self, actions: List[Dict]):
        """Batch insert multiple agent actions at once."""

        def _insert():
            values = [self._prepare_action_data(action) for action in actions]
            if values:
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

        self._execute_in_transaction(_insert)

    def batch_log_learning_experiences(self, experiences: List[Dict]):
        """Batch insert multiple learning experiences at once."""

        def _insert():
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

        self._execute_in_transaction(_insert)

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
        # Add type validation
        if parent_id is not None and not isinstance(parent_id, int):
            logger.warning(f"Invalid parent_id type: {type(parent_id)}. Setting to None.")
            parent_id = None

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

    def verify_foreign_keys(self) -> bool:
        """
        Verify that foreign key constraints are enabled and working.

        Returns
        -------
        bool
            True if foreign keys are enabled and working, False otherwise
        """
        try:
            # Check if foreign keys are enabled
            self.cursor.execute("PRAGMA foreign_keys")
            foreign_keys_enabled = bool(self.cursor.fetchone()[0])

            if not foreign_keys_enabled:
                logger.warning("Foreign key constraints are not enabled")
                return False

            # Test foreign key constraint
            try:
                # Try to insert a record with an invalid foreign key
                self.cursor.execute(
                    """
                    INSERT INTO AgentStates (
                        step_number, agent_id, position_x, position_y, 
                        resource_level, current_health, max_health,
                        starvation_threshold, is_defending, total_reward, age
                    ) VALUES (0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                    """
                )
                # If we get here, foreign keys aren't working
                logger.warning("Foreign key constraints failed validation test")
                return False
            except sqlite3.IntegrityError:
                # This is what we want - constraint violation
                return True

        except Exception as e:
            logger.error(f"Error verifying foreign keys: {e}")
            return False

    def get_agent_health_incidents(self, agent_id: int) -> List[Dict]:
        """Get all health incidents for an agent.

        Parameters
        ----------
        agent_id : int
            ID of the agent

        Returns
        -------
        List[Dict]
            List of health incidents with their details
        """
        import json

        self.cursor.execute(
            """
            SELECT step_number, health_before, health_after, 
                   cause, details
            FROM HealthIncidents
            WHERE agent_id = ?
            ORDER BY step_number
        """,
            (agent_id,),
        )

        incidents = []
        for row in self.cursor.fetchall():
            incident = {
                "step_number": row[0],
                "health_before": row[1],
                "health_after": row[2],
                "cause": row[3],
                "details": json.loads(row[4]) if row[4] else None,
            }
            incidents.append(incident)

        return incidents

    def get_step_actions(self, agent_id: int, step_number: int) -> Dict:
        """Get detailed action information for an agent at a specific step.

        Parameters
        ----------
        agent_id : int
            ID of the agent
        step_number : int
            Simulation step number

        Returns
        -------
        Dict
            Dictionary containing action details including:
            - action_type: Type of action taken
            - action_target_id: ID of target agent/resource if applicable
            - resources_before: Resource level before action
            - resources_after: Resource level after action
            - reward: Reward received for action
            - details: Additional action-specific details
        """
        try:
            self.cursor.execute(
                """
                SELECT 
                    action_type,
                    action_target_id,
                    position_before,
                    position_after,
                    resources_before,
                    resources_after,
                    reward,
                    details
                FROM AgentActions
                WHERE agent_id = ? AND step_number = ?
            """,
                (agent_id, step_number),
            )

            row = self.cursor.fetchone()
            if row:
                return {
                    "action_type": row[0],
                    "action_target_id": row[1],
                    "position_before": row[2],
                    "position_after": row[3],
                    "resources_before": row[4],
                    "resources_after": row[5],
                    "reward": row[6],
                    "details": row[7],
                }
            return None

        except Exception as e:
            logger.error(f"Error getting step actions: {e}")
            return None

    def get_configuration(self) -> Dict:
        """Retrieve the simulation configuration from the database.

        Returns
        -------
        Dict
            Dictionary containing simulation configuration parameters
            If no configuration is found, returns an empty dictionary
        """
        try:
            self.cursor.execute(
                """
                SELECT config_data
                FROM SimulationConfig
                ORDER BY timestamp DESC
                LIMIT 1
                """
            )
            row = self.cursor.fetchone()
            if row and row[0]:
                import json
                return json.loads(row[0])
            return {}
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return {}

    def save_configuration(self, config: Dict) -> None:
        """Save simulation configuration to the database.

        Parameters
        ----------
        config : Dict
            Dictionary containing simulation configuration parameters
        """
        def _insert():
            import json
            import time
            self.cursor.execute(
                """
                INSERT INTO SimulationConfig (timestamp, config_data)
                VALUES (?, ?)
                """,
                (int(time.time()), json.dumps(config))
            )

        self._execute_in_transaction(_insert)

    def get_population_momentum(self) -> float:
        """Calculate population momentum (death_step * max_count)."""
        try:
            query = """
                WITH PopulationData AS (
                    SELECT 
                        step_number,
                        total_agents
                    FROM SimulationSteps
                    WHERE total_agents > 0
                )
                SELECT 
                    MAX(step_number) as death_step,
                    MAX(total_agents) as max_count
                FROM PopulationData
            """
            
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            
            if result and result[0] and result[1]:
                death_step, max_count = result
                momentum = death_step * max_count
                return momentum
            return 0
            
        except Exception as e:
            logging.error(f"Error calculating population momentum: {e}")
            return 0

    def get_population_statistics(self) -> Dict:
        """Calculate comprehensive population statistics."""
        try:
            # Query for population data over time
            query = """
                WITH PopulationData AS (
                    SELECT 
                        step_number,
                        total_agents,
                        total_resources,
                        (SELECT SUM(resource_level) 
                         FROM AgentStates 
                         WHERE step_number = s.step_number) as resources_consumed
                    FROM SimulationSteps s
                    WHERE total_agents > 0
                )
                SELECT 
                    -- Basic stats
                    AVG(total_agents) as avg_population,
                    MAX(step_number) as death_step,
                    MAX(total_agents) as peak_population,
                    
                    -- Resource stats
                    SUM(resources_consumed) as total_resources_consumed,
                    SUM(total_resources) as total_resources_available,
                    
                    -- For variance calculation
                    SUM(CAST(total_agents AS FLOAT) * total_agents) as sum_squared,
                    COUNT(*) as step_count
                FROM PopulationData
            """
            
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            
            if result:
                avg_pop = result[0] or 0
                death_step = result[1] or 0
                peak_pop = result[2] or 0
                resources_consumed = result[3] or 0
                resources_available = result[4] or 0
                sum_squared = result[5] or 0
                step_count = result[6] or 1  # Avoid division by zero
                
                # Calculate variance
                variance = (sum_squared / step_count) - (avg_pop * avg_pop)
                std_dev = variance ** 0.5
                
                # Calculate metrics
                resource_utilization = (
                    resources_consumed / resources_available 
                    if resources_available > 0 else 0
                )
                
                cv = std_dev / avg_pop if avg_pop > 0 else 0
                
                return {
                    "average_population": avg_pop,
                    "peak_population": peak_pop,
                    "death_step": death_step,
                    "resource_utilization": resource_utilization,
                    "population_variance": variance,
                    "coefficient_variation": cv,
                    "resources_consumed": resources_consumed,
                    "resources_available": resources_available,
                    "utilization_per_agent": (
                        resources_consumed / (avg_pop * death_step)
                        if avg_pop * death_step > 0 else 0
                    )
                }
                
            return {}
            
        except Exception as e:
            logging.error(f"Error calculating population statistics: {e}")
            return {}

    def get_advanced_statistics(self) -> Dict:
        """Calculate advanced simulation statistics."""
        try:
            # Population dynamics query
            query = """
            WITH PopulationTimeline AS (
                SELECT 
                    step_number,
                    total_agents,
                    total_resources,
                    system_agents,
                    independent_agents,
                    control_agents,
                    (SELECT AVG(current_health) 
                     FROM AgentStates 
                     WHERE step_number = s.step_number) as avg_health,
                    (SELECT COUNT(DISTINCT agent_id) 
                     FROM AgentStates 
                     WHERE step_number = s.step_number) as unique_agents
                FROM SimulationSteps s
                WHERE total_agents > 0
                ORDER BY step_number
            ),
            AgentCounts AS (
                SELECT COUNT(*) as total_created
                FROM Agents
            ),
            InteractionStats AS (
                SELECT 
                    COUNT(*) as total_interactions,
                    SUM(CASE 
                        WHEN action_type IN ('attack', 'defend') THEN 1 
                        ELSE 0 
                    END) as conflict_count,
                    SUM(CASE 
                        WHEN action_type IN ('share', 'help') THEN 1 
                        ELSE 0 
                    END) as cooperation_count
                FROM AgentActions
            ),
            AgentTypeData AS (
                SELECT 
                    step_number,
                    CAST(system_agents AS FLOAT) / NULLIF(total_agents, 0) as sys_ratio,
                    CAST(independent_agents AS FLOAT) / NULLIF(total_agents, 0) as ind_ratio,
                    CAST(control_agents AS FLOAT) / NULLIF(total_agents, 0) as ctrl_ratio
                FROM PopulationTimeline
                WHERE total_agents > 0
            ),
            PopStats AS (
                SELECT
                    FIRST_VALUE(total_agents) OVER (ORDER BY step_number) as initial_pop,
                    FIRST_VALUE(total_agents) OVER (ORDER BY step_number DESC) as final_pop,
                    MAX(total_agents) as peak_pop,
                    AVG(avg_health) as average_health,
                    AVG(CASE WHEN total_resources < (total_agents * 0.5) THEN 1 ELSE 0 END) as scarcity_index,
                    COUNT(*) as total_steps
                FROM PopulationTimeline
            )
            SELECT 
                p.*,
                -- Interaction metrics
                (SELECT CAST(total_interactions AS FLOAT) / p.total_steps / p.peak_pop 
                 FROM InteractionStats) as interaction_rate,
                (SELECT CAST(conflict_count AS FLOAT) / NULLIF(cooperation_count, 0) 
                 FROM InteractionStats) as conflict_cooperation_ratio,
                
                -- Agent type ratios (for diversity calculation)
                AVG(a.sys_ratio) as avg_system_ratio,
                AVG(a.ind_ratio) as avg_independent_ratio,
                AVG(a.ctrl_ratio) as avg_control_ratio,
                
                -- Survivor metrics
                (SELECT CAST(p.final_pop AS FLOAT) / total_created FROM AgentCounts) as survivor_ratio,
                
                -- Critical thresholds
                MIN(CASE 
                    WHEN pt.total_agents <= (p.peak_pop * 0.1) THEN pt.step_number 
                    ELSE NULL 
                END) as extinction_threshold_time
                
            FROM PopStats p
            CROSS JOIN PopulationTimeline pt
            LEFT JOIN AgentTypeData a ON pt.step_number = a.step_number
            GROUP BY p.initial_pop, p.final_pop, p.peak_pop, p.average_health, 
                     p.scarcity_index, p.total_steps
            """
            
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            
            if result:
                initial_pop = result[0] or 0
                final_pop = result[1] or 0
                peak_pop = result[2] or 0
                total_steps = result[5] or 1  # Avoid division by zero
                
                # Calculate diversity using Python's math.log
                import math
                
                # Get agent type ratios
                ratios = [result[8], result[9], result[10]]  # sys, ind, ctrl ratios
                
                # Calculate Shannon entropy
                diversity = 0
                for ratio in ratios:
                    if ratio and ratio > 0:
                        diversity -= ratio * math.log(ratio)
                
                return {
                    # Population dynamics
                    "peak_to_end_ratio": peak_pop / final_pop if final_pop > 0 else float('inf'),
                    "growth_rate": (final_pop - initial_pop) / total_steps if total_steps > 0 else 0,
                    "extinction_threshold_time": result[12],
                    
                    # Health and survival
                    "average_health": result[3],
                    "survivor_ratio": result[11],
                    
                    # Diversity and interaction
                    "agent_diversity": diversity,
                    "interaction_rate": result[6] if result[6] is not None else 0,
                    "conflict_cooperation_ratio": result[7] if result[7] is not None else 0,
                    
                    # Resource dynamics
                    "scarcity_index": result[4] if result[4] is not None else 0
                }
                
            return {}
            
        except Exception as e:
            logging.error(f"Error calculating advanced statistics: {e}")
            return {}
