"""Database module for simulation state persistence and analysis.

This module provides a SQLAlchemy-based database implementation for storing and 
analyzing simulation data. It handles all database operations including state 
logging, configuration management, and data analysis.

Key Components
-------------
- SimulationDatabase : Main database interface class
- SQLAlchemy Models : ORM models for different data types
    - Agent : Agent entity data
    - AgentState : Time-series agent state data
    - ResourceState : Resource position and amount tracking
    - SimulationStep : Per-step simulation metrics
    - AgentAction : Agent behavior logging
    - HealthIncident : Health-related events
    - LearningExperience : Agent learning data

Features
--------
- Efficient batch operations with buffering
- Transaction safety with rollback support
- Comprehensive error handling
- Data export in multiple formats
- Advanced statistical analysis
- Configuration management
- Performance optimized queries

Usage Example
------------
>>> db = SimulationDatabase("simulation_results.db")
>>> db.save_configuration(config_dict)
>>> db.log_step(step_number, agent_states, resource_states, metrics)
>>> db.export_data("results.csv")
>>> db.close()

Notes
-----
- Uses SQLite as the backend database
- Implements foreign key constraints
- Includes indexes for performance
- Supports concurrent access through session management
"""

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    case,
    create_engine,
    event,
    func,
    text,
)
from sqlalchemy.exc import (
    IntegrityError,
    OperationalError,
    ProgrammingError,
    SQLAlchemyError,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import joinedload, relationship, scoped_session, sessionmaker

if TYPE_CHECKING:
    from core.environment import Environment

logger = logging.getLogger(__name__)

Base = declarative_base()


# Define SQLAlchemy Models
class Agent(Base):
    __tablename__ = "agents"
    __table_args__ = (
        Index("idx_agents_agent_type", "agent_type"),
        Index("idx_agents_birth_time", "birth_time"),
        Index("idx_agents_death_time", "death_time"),
    )

    agent_id = Column(Integer, primary_key=True)
    birth_time = Column(Integer)
    death_time = Column(Integer)
    agent_type = Column(String(50))
    position_x = Column(Float(precision=6))
    position_y = Column(Float(precision=6))
    initial_resources = Column(Float(precision=6))
    max_health = Column(Float(precision=4))
    starvation_threshold = Column(Integer)
    genome_id = Column(String(64))
    parent_id = Column(Integer, ForeignKey("agents.agent_id"))
    generation = Column(Integer)

    # Relationships
    states = relationship("AgentState", back_populates="agent")
    actions = relationship("AgentAction", back_populates="agent")
    health_incidents = relationship("HealthIncident", back_populates="agent")
    learning_experiences = relationship("LearningExperience", back_populates="agent")


class AgentState(Base):
    __tablename__ = "agent_states"
    __table_args__ = (
        Index("idx_agent_states_agent_id", "agent_id"),
        Index("idx_agent_states_step_number", "step_number"),
        Index("idx_agent_states_composite", "step_number", "agent_id"),
    )

    id = Column(Integer, primary_key=True)
    step_number = Column(Integer)
    agent_id = Column(Integer, ForeignKey("agents.agent_id"))
    position_x = Column(Float)
    position_y = Column(Float)
    resource_level = Column(Float)
    current_health = Column(Float)
    max_health = Column(Float)
    starvation_threshold = Column(Integer)
    is_defending = Column(Boolean)
    total_reward = Column(Float)
    age = Column(Integer)

    agent = relationship("Agent", back_populates="states")

    def as_dict(self) -> Dict[str, Any]:
        """Convert agent state to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent.agent_type,
            "position": (self.position_x, self.position_y),
            "resource_level": self.resource_level,
            "current_health": self.current_health,
            "max_health": self.max_health,
            "starvation_threshold": self.starvation_threshold,
            "is_defending": self.is_defending,
            "total_reward": self.total_reward,
            "age": self.age,
        }


# Additional SQLAlchemy Models
class ResourceState(Base):
    __tablename__ = "resource_states"
    __table_args__ = (
        Index("idx_resource_states_step_number", "step_number"),
        Index("idx_resource_states_resource_id", "resource_id"),
    )

    id = Column(Integer, primary_key=True)
    step_number = Column(Integer)
    resource_id = Column(Integer)
    amount = Column(Float)
    position_x = Column(Float)
    position_y = Column(Float)

    def as_dict(self) -> Dict[str, Any]:
        """Convert resource state to dictionary."""
        return {
            "resource_id": self.resource_id,
            "amount": self.amount,
            "position": (self.position_x, self.position_y),
        }


class SimulationStep(Base):
    __tablename__ = "simulation_steps"
    __table_args__ = (Index("idx_simulation_steps_step_number", "step_number"),)

    step_number = Column(Integer, primary_key=True)
    total_agents = Column(Integer)
    system_agents = Column(Integer)
    independent_agents = Column(Integer)
    control_agents = Column(Integer)
    total_resources = Column(Float)
    average_agent_resources = Column(Float)
    births = Column(Integer)
    deaths = Column(Integer)
    current_max_generation = Column(Integer)
    resource_efficiency = Column(Float)
    resource_distribution_entropy = Column(Float)
    average_agent_health = Column(Float)
    average_agent_age = Column(Integer)
    average_reward = Column(Float)
    combat_encounters = Column(Integer)
    successful_attacks = Column(Integer)
    resources_shared = Column(Float)
    genetic_diversity = Column(Float)
    dominant_genome_ratio = Column(Float)

    def as_dict(self) -> Dict[str, Any]:
        """Convert simulation step to dictionary."""
        return {
            "total_agents": self.total_agents,
            "system_agents": self.system_agents,
            "independent_agents": self.independent_agents,
            "control_agents": self.control_agents,
            "total_resources": self.total_resources,
            "average_agent_resources": self.average_agent_resources,
            "births": self.births,
            "deaths": self.deaths,
            "current_max_generation": self.current_max_generation,
            "resource_efficiency": self.resource_efficiency,
            "resource_distribution_entropy": self.resource_distribution_entropy,
            "average_agent_health": self.average_agent_health,
            "average_agent_age": self.average_agent_age,
            "average_reward": self.average_reward,
            "combat_encounters": self.combat_encounters,
            "successful_attacks": self.successful_attacks,
            "resources_shared": self.resources_shared,
            "genetic_diversity": self.genetic_diversity,
            "dominant_genome_ratio": self.dominant_genome_ratio,
        }


class AgentAction(Base):
    __tablename__ = "agent_actions"
    __table_args__ = (
        Index("idx_agent_actions_step_number", "step_number"),
        Index("idx_agent_actions_agent_id", "agent_id"),
        Index("idx_agent_actions_action_type", "action_type"),
    )

    action_id = Column(Integer, primary_key=True)
    step_number = Column(Integer, nullable=False)
    agent_id = Column(Integer, ForeignKey("agents.agent_id"), nullable=False)
    action_type = Column(String(20), nullable=False)
    action_target_id = Column(Integer)
    position_before = Column(String(32))
    position_after = Column(String(32))
    resources_before = Column(Float(precision=6))
    resources_after = Column(Float(precision=6))
    reward = Column(Float(precision=6))
    details = Column(String(1024))

    agent = relationship("Agent", back_populates="actions")


class LearningExperience(Base):
    __tablename__ = "learning_experiences"
    __table_args__ = (
        Index("idx_learning_experiences_step_number", "step_number"),
        Index("idx_learning_experiences_agent_id", "agent_id"),
        Index("idx_learning_experiences_module_type", "module_type"),
    )

    experience_id = Column(Integer, primary_key=True)
    step_number = Column(Integer)
    agent_id = Column(Integer, ForeignKey("agents.agent_id"))
    module_type = Column(String(50))
    state_before = Column(String(512))
    action_taken = Column(Integer)
    reward = Column(Float(precision=6))
    state_after = Column(String(512))
    loss = Column(Float(precision=6))

    agent = relationship("Agent", back_populates="learning_experiences")


class HealthIncident(Base):
    __tablename__ = "health_incidents"
    __table_args__ = (
        Index("idx_health_incidents_step_number", "step_number"),
        Index("idx_health_incidents_agent_id", "agent_id"),
    )

    incident_id = Column(Integer, primary_key=True)
    step_number = Column(Integer, nullable=False)
    agent_id = Column(Integer, ForeignKey("agents.agent_id"), nullable=False)
    health_before = Column(Float(precision=4))
    health_after = Column(Float(precision=4))
    cause = Column(String(50), nullable=False)
    details = Column(String(512))

    agent = relationship("Agent", back_populates="health_incidents")


class SimulationConfig(Base):
    __tablename__ = "simulation_config"

    config_id = Column(Integer, primary_key=True)
    timestamp = Column(Integer, nullable=False)
    config_data = Column(String(4096), nullable=False)


class SimulationDatabase:
    """Database interface for simulation state persistence and analysis.

    This class provides a high-level interface for storing and retrieving simulation
    data using SQLAlchemy ORM. It handles all database operations including state
    logging, configuration management, and data analysis with transaction safety
    and efficient batch operations.

    Features
    --------
    - Buffered batch operations for performance
    - Transaction safety with automatic rollback
    - Comprehensive error handling
    - Multi-format data export
    - Statistical analysis functions
    - Thread-safe session management

    Attributes
    ----------
    db_path : str
        Path to the SQLite database file
    engine : sqlalchemy.engine.Engine
        SQLAlchemy database engine instance
    Session : sqlalchemy.orm.scoped_session
        Thread-local session factory
    _action_buffer : List[Dict]
        Buffer for agent actions before batch insert
    _learning_exp_buffer : List[Dict]
        Buffer for learning experiences before batch insert
    _health_incident_buffer : List[Dict]
        Buffer for health incidents before batch insert
    _buffer_size : int
        Maximum size of buffers before auto-flush (default: 1000)

    Methods
    -------
    log_step(step_number, agent_states, resource_states, metrics)
        Log complete simulation state for a time step
    get_simulation_data(step_number)
        Retrieve comprehensive state data for a time step
    export_data(filepath, format='csv', ...)
        Export simulation data to various file formats
    get_advanced_statistics()
        Calculate advanced simulation metrics
    flush_all_buffers()
        Write all buffered data to database
    close()
        Clean up database connections and resources

    Example
    -------
    >>> db = SimulationDatabase("simulation_results.db")
    >>> db.save_configuration({"agents": 100, "resources": 1000})
    >>> db.log_step(1, agent_states, resource_states, metrics)
    >>> simulation_data = db.get_simulation_data(1)
    >>> db.export_data("results.csv")
    >>> db.close()

    Notes
    -----
    - Uses SQLite as the backend database
    - Implements foreign key constraints
    - Creates required tables automatically
    - Handles concurrent access through thread-local sessions
    - Buffers operations for better performance
    - Provides automatic cleanup through context management
    """

    def __init__(self, db_path: str) -> None:
        """Initialize a new SimulationDatabase instance with SQLAlchemy.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database file

        Notes
        -----
        - Enables foreign key support for SQLite
        - Creates session factory with thread-local scope
        - Initializes tables and indexes
        - Sets up batch operation buffers
        """
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")

        # Enable foreign key support for SQLite
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        # Create session factory
        session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(session_factory)

        # Create tables and indexes
        self._create_tables()

        # Add batch operation buffers
        self._action_buffer = []
        self._learning_exp_buffer = []
        self._health_incident_buffer = []
        self._buffer_size = 1000

    def _execute_in_transaction(self, func: callable) -> Any:
        """Execute database operations within a transaction with error handling.

        Parameters
        ----------
        func : callable
            Function that takes a session argument and performs database operations

        Returns
        -------
        Any
            Result of the executed function

        Raises
        ------
        IntegrityError
            If there's a database constraint violation
        OperationalError
            If there's a database connection issue
        ProgrammingError
            If there's a SQL syntax error
        SQLAlchemyError
            For other database-related errors
        """
        session = self.Session()
        try:
            result = func(session)
            session.commit()
            return result
        except IntegrityError as e:
            session.rollback()
            logger.error(f"Database integrity error: {e}")
            raise
        except OperationalError as e:
            session.rollback()
            logger.error(f"Database operational error (connection/timeout): {e}")
            raise
        except ProgrammingError as e:
            session.rollback()
            logger.error(f"Database programming error (SQL syntax/schema): {e}")
            raise
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"General database error: {e}")
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error during database operation: {e}")
            raise
        finally:
            self.Session.remove()

    def batch_log_agents(self, agent_data: List[Dict]):
        """Batch insert multiple agents into the database.

        Parameters
        ----------
        agent_data : List[Dict]
            List of dictionaries containing agent data:
            - agent_id: int
            - birth_time: int
            - agent_type: str
            - position: Tuple[float, float]
            - initial_resources: float
            - max_health: float
            - starvation_threshold: int
            - genome_id: Optional[str]
            - parent_id: Optional[int]
            - generation: Optional[int]

        Raises
        ------
        SQLAlchemyError
            If there's an error during batch insertion
        ValueError
            If agent data is malformed
        """

        def _insert(session):
            # Format data as mappings
            mappings = [
                {
                    "agent_id": data["agent_id"],
                    "birth_time": data["birth_time"],
                    "agent_type": data["agent_type"],
                    "position_x": data["position"][0],
                    "position_y": data["position"][1],
                    "initial_resources": data["initial_resources"],
                    "max_health": data["max_health"],
                    "starvation_threshold": data["starvation_threshold"],
                    "genome_id": data.get("genome_id"),
                    "parent_id": data.get("parent_id"),
                    "generation": data.get("generation", 0),
                }
                for data in agent_data
            ]
            session.bulk_insert_mappings(Agent, mappings)

        self._execute_in_transaction(_insert)

    def get_simulation_data(self, step_number: int) -> Dict[str, Any]:
        """Retrieve simulation data for a specific time step.

        Fetches comprehensive state data including agent states, resource states,
        and simulation metrics for the specified step number.

        Parameters
        ----------
        step_number : int
            The simulation step number to retrieve data for

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - step_number: int
            - agent_states: List[Tuple] - Agent state data tuples
            - resource_states: List[Tuple] - Resource state data tuples
            - metrics: Dict - Simulation metrics
            - timestamp: str - ISO format timestamp

        Raises
        ------
        SQLAlchemyError
            If there's a database error
        ValueError
            If step_number is invalid
        """

        def _query(session):
            # Optimize queries with joins and specific column selection
            agent_states = (
                session.query(AgentState)
                .join(Agent)
                .filter(AgentState.step_number == step_number)
                .options(joinedload(AgentState.agent))  # Eager load agent relationship
                .all()
            )

            resource_states = (
                session.query(ResourceState)
                .filter(ResourceState.step_number == step_number)
                .all()
            )

            metrics = (
                session.query(SimulationStep)
                .filter(SimulationStep.step_number == step_number)
                .first()
            )

            # Convert to tuples for visualization compatibility
            agent_state_tuples = [
                (
                    state.agent_id,
                    state.agent.agent_type,
                    state.position_x,
                    state.position_y,
                    state.resource_level,
                    state.current_health,
                    state.max_health,
                    state.starvation_threshold,
                    state.is_defending,
                    state.total_reward,
                    state.age,
                )
                for state in agent_states
            ]

            resource_state_tuples = [
                (
                    state.resource_id,
                    state.amount,
                    state.position_x,
                    state.position_y,
                )
                for state in resource_states
            ]

            return {
                "step_number": step_number,
                "agent_states": agent_state_tuples,
                "resource_states": resource_state_tuples,
                "metrics": metrics.as_dict() if metrics else {},
                "timestamp": datetime.now().isoformat(),
            }

        return self._execute_in_transaction(_query)

    def log_step(
        self,
        step_number: int,
        agent_states: List[Tuple],
        resource_states: List[Tuple],
        metrics: Dict,
    ) -> None:
        """Log comprehensive simulation state data for a single time step.

        Records the complete state of the simulation including all agent states,
        resource states, and aggregate metrics for the given step.

        Parameters
        ----------
        step_number : int
            Current simulation step number
        agent_states : List[Tuple]
            List of agent state tuples containing:
            (agent_id, position_x, position_y, resource_level, current_health,
             max_health, starvation_threshold, is_defending, total_reward, age)
        resource_states : List[Tuple]
            List of resource state tuples containing:
            (resource_id, amount, position_x, position_y)
        metrics : Dict
            Dictionary of simulation metrics including:
            - total_agents: int
            - system_agents: int
            - independent_agents: int
            - control_agents: int
            - total_resources: float
            - average_agent_resources: float
            - births: int
            - deaths: int
            And other relevant metrics

        Raises
        ------
        SQLAlchemyError
            If there's an error during database insertion
        ValueError
            If input data is malformed or invalid
        """

        def _insert(session):
            # Bulk insert agent states
            if agent_states:
                agent_state_mappings = [
                    {
                        "step_number": step_number,
                        "agent_id": state[0],
                        "position_x": state[1],
                        "position_y": state[2],
                        "resource_level": state[3],
                        "current_health": state[4],
                        "max_health": state[5],
                        "starvation_threshold": state[6],
                        "is_defending": bool(state[7]),
                        "total_reward": state[8],
                        "age": state[9],
                    }
                    for state in agent_states
                ]
                session.bulk_insert_mappings(AgentState, agent_state_mappings)

            # Bulk insert resource states
            if resource_states:
                resource_state_mappings = [
                    {
                        "step_number": step_number,
                        "resource_id": state[0],
                        "amount": state[1],
                        "position_x": state[2],
                        "position_y": state[3],
                    }
                    for state in resource_states
                ]
                session.bulk_insert_mappings(ResourceState, resource_state_mappings)

            # Insert metrics
            simulation_step = SimulationStep(step_number=step_number, **metrics)
            session.add(simulation_step)

        self._execute_in_transaction(_insert)

    def get_historical_data(self) -> Dict:
        """Retrieve historical metrics for the entire simulation."""

        def _query(session):
            steps = (
                session.query(SimulationStep).order_by(SimulationStep.step_number).all()
            )

            return {
                "steps": [step.step_number for step in steps],
                "metrics": {
                    "total_agents": [step.total_agents for step in steps],
                    "system_agents": [step.system_agents for step in steps],
                    "independent_agents": [step.independent_agents for step in steps],
                    "control_agents": [step.control_agents for step in steps],
                    "total_resources": [step.total_resources for step in steps],
                    "average_agent_resources": [
                        step.average_agent_resources for step in steps
                    ],
                    "births": [step.births for step in steps],
                    "deaths": [step.deaths for step in steps],
                },
            }

        return self._execute_in_transaction(_query)

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
    ) -> None:
        """Buffer an agent action with enhanced validation and error handling."""
        try:
            # Input validation
            if step_number < 0:
                raise ValueError("step_number must be non-negative")
            if agent_id < 0:
                raise ValueError("agent_id must be non-negative")
            if not isinstance(action_type, str):
                action_type = str(action_type)

            action_data = {
                "step_number": step_number,
                "agent_id": agent_id,
                "action_type": action_type,
                "action_target_id": action_target_id,
                "position_before": str(position_before) if position_before else None,
                "position_after": str(position_after) if position_after else None,
                "resources_before": resources_before,
                "resources_after": resources_after,
                "reward": reward,
                "details": json.dumps(details) if details else None,
            }

            self._action_buffer.append(action_data)

            if len(self._action_buffer) >= self._buffer_size:
                self.flush_action_buffer()

        except ValueError as e:
            logger.error(f"Invalid input for agent action: {e}")
            raise
        except TypeError as e:
            logger.error(f"Type error in agent action data: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error logging agent action: {e}")
            raise

    def flush_action_buffer(self):
        """Flush the action buffer by batch inserting all buffered actions."""
        if not self._action_buffer:
            return

        buffer_copy = list(self._action_buffer)  # Create a copy of the buffer
        try:

            def _insert(session):
                session.bulk_insert_mappings(AgentAction, buffer_copy)

            self._execute_in_transaction(_insert)
        except SQLAlchemyError as e:
            logger.error(f"Failed to flush action buffer: {e}")
            raise
        else:
            self._action_buffer.clear()  # Only clear if successful

    def flush_learning_buffer(self):
        """Flush the learning experience buffer with transaction safety."""
        if not self._learning_exp_buffer:
            return

        buffer_copy = list(self._learning_exp_buffer)
        try:

            def _insert(session):
                session.bulk_insert_mappings(LearningExperience, buffer_copy)

            self._execute_in_transaction(_insert)
        except SQLAlchemyError as e:
            logger.error(f"Failed to flush learning buffer: {e}")
            raise
        else:
            self._learning_exp_buffer.clear()

    def flush_health_buffer(self):
        """Flush the health incident buffer with transaction safety."""
        if not self._health_incident_buffer:
            return

        buffer_copy = list(self._health_incident_buffer)
        try:

            def _insert(session):
                session.bulk_insert_mappings(HealthIncident, buffer_copy)

            self._execute_in_transaction(_insert)
        except SQLAlchemyError as e:
            logger.error(f"Failed to flush health buffer: {e}")
            raise
        else:
            self._health_incident_buffer.clear()

    def flush_all_buffers(self) -> None:
        """Flush all data buffers to the database.

        Safely writes all buffered data (actions, learning experiences, health incidents)
        to the database in transactions. Maintains data integrity by only clearing
        buffers after successful writes.

        Raises
        ------
        SQLAlchemyError
            If any buffer flush fails, with details about which buffer(s) failed
        IntegrityError
            If there are foreign key or unique constraint violations
        OperationalError
            If there are database connection issues
        """
        buffers = {
            "action": (self._action_buffer, self.flush_action_buffer),
            "learning": (self._learning_exp_buffer, self.flush_learning_buffer),
            "health": (self._health_incident_buffer, self.flush_health_buffer),
        }

        errors = []
        for buffer_name, (buffer, flush_func) in buffers.items():
            if buffer:  # Only attempt flush if buffer has data
                try:
                    flush_func()
                except SQLAlchemyError as e:
                    logger.error(f"Error flushing {buffer_name} buffer: {e}")
                    errors.append((buffer_name, str(e)))
                except Exception as e:
                    logger.error(f"Unexpected error flushing {buffer_name} buffer: {e}")
                    errors.append((buffer_name, str(e)))

        if errors:
            error_msg = "; ".join(f"{name}: {err}" for name, err in errors)
            raise SQLAlchemyError(f"Failed to flush buffers: {error_msg}")

    def close(self) -> None:
        """Close the database connection with enhanced error handling."""
        try:
            # Flush pending changes
            self.flush_all_buffers()

            # Clean up sessions
            self.Session.remove()

            # Dispose engine connections
            if hasattr(self, "engine"):
                try:
                    self.engine.dispose()
                except SQLAlchemyError as e:
                    logger.error(f"Error disposing database engine: {e}")

        except SQLAlchemyError as e:
            logger.error(f"Database error during close: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during database close: {e}")
        finally:
            # Ensure critical resources are released
            if hasattr(self, "Session"):
                try:
                    self.Session.remove()
                except Exception as e:
                    logger.error(f"Final cleanup error: {e}")

    def export_data(
        self,
        filepath: str,
        format: str = "csv",
        data_types: List[str] = None,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        include_metadata: bool = True,
    ) -> None:
        """Export simulation data to various file formats with filtering options.

        Exports selected simulation data to a file in the specified format. Supports
        filtering by data type and time range, with optional metadata inclusion.

        Parameters
        ----------
        filepath : str
            Path where the export file will be saved
        format : str, optional
            Export format, one of: 'csv', 'excel', 'json', 'parquet'
            Default is 'csv'
        data_types : List[str], optional
            List of data types to export, can include:
            'metrics', 'agents', 'resources', 'actions'
            If None, exports all types
        start_step : int, optional
            Starting step number for data range
        end_step : int, optional
            Ending step number for data range
        include_metadata : bool, optional
            Whether to include simulation metadata, by default True

        Raises
        ------
        ValueError
            If format is unsupported or filepath is invalid
        SQLAlchemyError
            If there's an error retrieving data
        IOError
            If there's an error writing to the file
        """

        def _query(session):
            data = {}

            # Build step number filter
            step_filter = []
            if start_step is not None:
                step_filter.append(SimulationStep.step_number >= start_step)
            if end_step is not None:
                step_filter.append(SimulationStep.step_number <= end_step)

            # Default to all data types if none specified
            export_types = data_types or ["metrics", "agents", "resources", "actions"]

            # Collect requested data
            if "metrics" in export_types:
                metrics_query = session.query(SimulationStep)
                if step_filter:
                    metrics_query = metrics_query.filter(*step_filter)
                data["metrics"] = pd.read_sql(
                    metrics_query.statement, session.bind, index_col="step_number"
                )

            if "agents" in export_types:
                agents_query = (
                    session.query(AgentState)
                    .join(Agent)
                    .options(joinedload(AgentState.agent))
                )
                if step_filter:
                    agents_query = agents_query.filter(*step_filter)
                data["agents"] = pd.read_sql(agents_query.statement, session.bind)

            if "resources" in export_types:
                resources_query = session.query(ResourceState)
                if step_filter:
                    resources_query = resources_query.filter(*step_filter)
                data["resources"] = pd.read_sql(resources_query.statement, session.bind)

            if "actions" in export_types:
                actions_query = session.query(AgentAction)
                if step_filter:
                    actions_query = actions_query.filter(*step_filter)
                data["actions"] = pd.read_sql(actions_query.statement, session.bind)

            # Add metadata if requested
            if include_metadata:
                config = self.get_configuration()
                data["metadata"] = {
                    "config": config,
                    "export_timestamp": datetime.now().isoformat(),
                    "data_range": {"start_step": start_step, "end_step": end_step},
                    "exported_types": export_types,
                }

            # Export based on format
            if format == "csv":
                # Create separate CSV files for each data type
                base_path = filepath.rsplit(".", 1)[0]
                for data_type, df in data.items():
                    if isinstance(df, pd.DataFrame):
                        df.to_csv(f"{base_path}_{data_type}.csv", index=False)
                    elif data_type == "metadata":
                        with open(f"{base_path}_metadata.json", "w") as f:
                            json.dump(df, f, indent=2)

            elif format == "excel":
                # Save all data types as separate sheets in one Excel file
                with pd.ExcelWriter(filepath) as writer:
                    for data_type, df in data.items():
                        if isinstance(df, pd.DataFrame):
                            df.to_excel(writer, sheet_name=data_type, index=False)
                        elif data_type == "metadata":
                            pd.DataFrame([df]).to_excel(
                                writer, sheet_name="metadata", index=False
                            )

            elif format == "json":
                # Convert all data to JSON format
                json_data = {
                    k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v)
                    for k, v in data.items()
                }
                with open(filepath, "w") as f:
                    json.dump(json_data, f, indent=2)

            elif format == "parquet":
                # Save as parquet format (good for large datasets)
                if len(data) == 1:
                    # Single dataframe case
                    next(iter(data.values())).to_parquet(filepath)
                else:
                    # Multiple dataframes case
                    base_path = filepath.rsplit(".", 1)[0]
                    for data_type, df in data.items():
                        if isinstance(df, pd.DataFrame):
                            df.to_parquet(f"{base_path}_{data_type}.parquet")

            else:
                raise ValueError(f"Unsupported export format: {format}")

        self._execute_in_transaction(_query)

    def get_population_momentum(self) -> float:
        """Calculate population momentum using simpler SQL queries."""

        def _query(session):
            # Get initial population
            initial = (
                session.query(SimulationStep.total_agents)
                .order_by(SimulationStep.step_number)
                .first()
            )

            # Get max population and final step
            stats = session.query(
                func.max(SimulationStep.total_agents).label("max_count"),
                func.max(SimulationStep.step_number).label("final_step"),
            ).first()

            if initial and stats and initial[0] > 0:
                return (float(stats[1]) * float(stats[0])) / float(initial[0])
            return 0.0

        return self._execute_in_transaction(_query)

    def get_population_statistics(self) -> Dict:
        """Calculate comprehensive population statistics using SQLAlchemy."""

        def _query(session):
            from sqlalchemy import case, func

            # Subquery for population data
            pop_data = (
                session.query(
                    SimulationStep.step_number,
                    SimulationStep.total_agents,
                    SimulationStep.total_resources,
                    func.sum(AgentState.resource_level).label("resources_consumed"),
                )
                .outerjoin(
                    AgentState, AgentState.step_number == SimulationStep.step_number
                )
                .filter(SimulationStep.total_agents > 0)
                .group_by(SimulationStep.step_number)
                .subquery()
            )

            # Calculate statistics
            stats = session.query(
                func.avg(pop_data.c.total_agents).label("avg_population"),
                func.max(pop_data.c.step_number).label("death_step"),
                func.max(pop_data.c.total_agents).label("peak_population"),
                func.sum(pop_data.c.resources_consumed).label(
                    "total_resources_consumed"
                ),
                func.sum(pop_data.c.total_resources).label("total_resources_available"),
                func.sum(pop_data.c.total_agents * pop_data.c.total_agents).label(
                    "sum_squared"
                ),
                func.count().label("step_count"),
            ).first()

            if stats:
                avg_pop = float(stats[0] or 0)
                death_step = int(stats[1] or 0)
                peak_pop = int(stats[2] or 0)
                resources_consumed = float(stats[3] or 0)
                resources_available = float(stats[4] or 0)
                sum_squared = float(stats[5] or 0)
                step_count = int(stats[6] or 1)

                # Calculate derived statistics
                variance = (sum_squared / step_count) - (avg_pop * avg_pop)
                std_dev = variance**0.5

                resource_utilization = (
                    resources_consumed / resources_available
                    if resources_available > 0
                    else 0
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
                        if avg_pop * death_step > 0
                        else 0
                    ),
                }

            return {}

        return self._execute_in_transaction(_query)

    def get_advanced_statistics(self) -> Dict:
        """Calculate advanced simulation statistics using optimized queries."""

        def _query(session):
            # Get basic population stats in one query
            pop_stats = session.query(
                func.avg(SimulationStep.total_agents).label("avg_pop"),
                func.max(SimulationStep.total_agents).label("peak_pop"),
                func.min(SimulationStep.total_agents).label("min_pop"),
                func.count(SimulationStep.step_number).label("total_steps"),
                func.avg(SimulationStep.average_agent_health).label("avg_health"),
            ).first()

            # Get agent type ratios in a separate query
            type_stats = session.query(
                func.avg(SimulationStep.system_agents).label("avg_system"),
                func.avg(SimulationStep.independent_agents).label("avg_independent"),
                func.avg(SimulationStep.control_agents).label("avg_control"),
                func.avg(SimulationStep.total_agents).label("avg_total"),
            ).first()

            # Get interaction stats
            interaction_stats = session.query(
                func.count(AgentAction.action_id).label("total_actions"),
                func.sum(
                    case(
                        [(AgentAction.action_type.in_(["attack", "defend"]), 1)],
                        else_=0,
                    )
                ).label("conflicts"),
                func.sum(
                    case([(AgentAction.action_type.in_(["share", "help"]), 1)], else_=0)
                ).label("cooperation"),
            ).first()

            if not all([pop_stats, type_stats, interaction_stats]):
                return {}

            # Calculate diversity index
            total = float(type_stats[3] or 1)  # Avoid division by zero
            ratios = [float(count or 0) / total for count in type_stats[:3]]

            import math

            diversity = sum(
                -ratio * math.log(ratio) if ratio > 0 else 0 for ratio in ratios
            )

            return {
                "peak_population": int(pop_stats[1] or 0),
                "average_population": float(pop_stats[0] or 0),
                "total_steps": int(pop_stats[3] or 0),
                "average_health": float(pop_stats[4] or 0),
                "agent_diversity": diversity,
                "interaction_rate": (
                    float(interaction_stats[0] or 0)
                    / float(pop_stats[3] or 1)
                    / float(pop_stats[1] or 1)
                ),
                "conflict_cooperation_ratio": (
                    float(interaction_stats[1] or 0)
                    / float(interaction_stats[2] or 1)  # Avoid division by zero
                ),
            }

        return self._execute_in_transaction(_query)

    def log_resource(
        self, resource_id: int, initial_amount: float, position: Tuple[float, float]
    ):
        """Log a new resource in the simulation.

        Parameters
        ----------
        resource_id : int
            Unique identifier for the resource
        initial_amount : float
            Starting amount of the resource
        position : Tuple[float, float]
            (x, y) coordinates of the resource location
        """

        def _insert(session):
            resource_state = ResourceState(
                step_number=0,  # Initial state
                resource_id=resource_id,
                amount=initial_amount,
                position_x=position[0],
                position_y=position[1],
            )
            session.add(resource_state)

        self._execute_in_transaction(_insert)

    def get_agent_data(self, agent_id: int) -> Dict:
        """Get all data for a specific agent."""

        def _query(session):
            agent = session.query(Agent).filter(Agent.agent_id == agent_id).first()

            if not agent:
                return {}

            return {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "birth_time": agent.birth_time,
                "death_time": agent.death_time,
                "initial_resources": agent.initial_resources,
                "max_health": agent.max_health,
                "starvation_threshold": agent.starvation_threshold,
                "genome_id": agent.genome_id,
                "parent_id": agent.parent_id,
                "generation": agent.generation,
            }

        return self._execute_in_transaction(_query)

    def get_agent_actions(self, agent_id: int) -> List[Dict]:
        """Get all actions performed by an agent.

        Retrieves a chronological list of all actions performed by a specific agent,
        including details about each action's outcome and impact.

        Parameters
        ----------
        agent_id : int
            The unique identifier of the agent

        Returns
        -------
        List[Dict]
            List of action records, each containing:
            - step_number: int
            - action_type: str
            - action_target_id: Optional[int]
            - position_before: Optional[str]
            - position_after: Optional[str]
            - resources_before: Optional[float]
            - resources_after: Optional[float]
            - reward: Optional[float]
            - details: Optional[Dict]

        Raises
        ------
        SQLAlchemyError
            If there's a database error during retrieval
        """

        def _query(session):
            actions = (
                session.query(AgentAction)
                .filter(AgentAction.agent_id == agent_id)
                .order_by(AgentAction.step_number)
                .all()
            )

            return [
                {
                    "step_number": action.step_number,
                    "action_type": action.action_type,
                    "action_target_id": action.action_target_id,
                    "position_before": action.position_before,
                    "position_after": action.position_after,
                    "resources_before": action.resources_before,
                    "resources_after": action.resources_after,
                    "reward": action.reward,
                    "details": json.loads(action.details) if action.details else None,
                }
                for action in actions
            ]

        return self._execute_in_transaction(_query)

    def get_step_actions(self, agent_id: int, step_number: int) -> Optional[Dict]:
        """Get actions performed by an agent at a specific step."""

        def _query(session):
            action = (
                session.query(AgentAction)
                .filter(
                    AgentAction.agent_id == agent_id,
                    AgentAction.step_number == step_number,
                )
                .first()
            )

            if not action:
                return None

            return {
                "action_type": action.action_type,
                "action_target_id": action.action_target_id,
                "position_before": action.position_before,
                "position_after": action.position_after,
                "resources_before": action.resources_before,
                "resources_after": action.resources_after,
                "reward": action.reward,
                "details": json.loads(action.details) if action.details else None,
            }

        return self._execute_in_transaction(_query)

    def get_agent_types(self) -> List[str]:
        """Get list of all agent types in the simulation."""

        def _query(session):
            types = session.query(Agent.agent_type).distinct().all()
            return [t[0] for t in types]

        return self._execute_in_transaction(_query)

    def get_agent_behaviors(self) -> List[Dict]:
        """Get behavior patterns for all agents."""

        def _query(session):
            behaviors = (
                session.query(
                    AgentAction.step_number,
                    AgentAction.action_type,
                    func.count().label("count"),
                )
                .group_by(AgentAction.step_number, AgentAction.action_type)
                .order_by(AgentAction.step_number)
                .all()
            )

            return [
                {"step_number": b[0], "action_type": b[1], "count": b[2]}
                for b in behaviors
            ]

        return self._execute_in_transaction(_query)

    def get_agent_decisions(self) -> List[Dict]:
        """Get decision-making patterns for all agents."""

        def _query(session):
            decisions = (
                session.query(
                    AgentAction.action_type,
                    func.avg(AgentAction.reward).label("avg_reward"),
                    func.count().label("count"),
                )
                .group_by(AgentAction.action_type)
                .all()
            )

            return [
                {
                    "action_type": d[0],
                    "avg_reward": float(d[1]) if d[1] is not None else 0.0,
                    "count": d[2],
                }
                for d in decisions
            ]

        return self._execute_in_transaction(_query)

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

        def _insert(session):
            agent = Agent(
                agent_id=agent_id,
                birth_time=birth_time,
                agent_type=agent_type,
                position_x=position[0],
                position_y=position[1],
                initial_resources=initial_resources,
                max_health=max_health,
                starvation_threshold=starvation_threshold,
                genome_id=genome_id,
                parent_id=parent_id,
                generation=generation,
            )
            session.add(agent)

        self._execute_in_transaction(_insert)

    def update_agent_death(
        self, agent_id: int, death_time: int, cause: str = "starvation"
    ):
        """Update agent record with death information.

        Parameters
        ----------
        agent_id : int
            ID of the agent that died
        death_time : int
            Time step when death occurred
        cause : str, optional
            Cause of death, defaults to "starvation"
        """

        def _update(session):
            # Update agent record with death time
            agent = session.query(Agent).filter(Agent.agent_id == agent_id).first()

            if agent:
                agent.death_time = death_time

                # Log health incident for death
                health_incident = HealthIncident(
                    step_number=death_time,
                    agent_id=agent_id,
                    health_before=0,  # Death implies health reached 0
                    health_after=0,
                    cause=cause,
                    details=json.dumps({"final_state": "dead"}),
                )
                session.add(health_incident)

        self._execute_in_transaction(_update)

    def _load_agent_stats(self, conn, agent_id) -> Dict:
        """Load current agent statistics from database."""

        def _query(session):
            # Get latest state for the agent using lowercase table names
            latest_state = (
                session.query(
                    AgentState.current_health,
                    AgentState.resource_level,
                    AgentState.total_reward,
                    AgentState.age,
                    AgentState.is_defending,
                    AgentState.position_x,
                    AgentState.position_y,
                    AgentState.step_number,
                )
                .filter(AgentState.agent_id == agent_id)
                .order_by(AgentState.step_number.desc())
                .first()
            )

            if latest_state:
                return {
                    "current_health": latest_state[0],
                    "resource_level": latest_state[1],
                    "total_reward": latest_state[2],
                    "age": latest_state[3],
                    "is_defending": latest_state[4],
                    "current_position": f"{latest_state[5]}, {latest_state[6]}",
                }

            return {
                "current_health": 0,
                "resource_level": 0,
                "total_reward": 0,
                "age": 0,
                "is_defending": 0,
                "current_position": "0, 0",
            }

        return self._execute_in_transaction(_query)

    def update_agent_state(self, agent_id: int, step_number: int, state_data: Dict):
        """Update agent state in the database.

        Parameters
        ----------
        agent_id : int
            ID of the agent to update
        step_number : int
            Current simulation step
        state_data : Dict
            Dictionary containing state data:
            - current_health: float
            - max_health: float
            - resource_level: float
            - position: Tuple[float, float]
            - is_defending: bool
            - total_reward: float
            - starvation_threshold: int
        """

        def _update(session):
            # Get the agent to access its properties
            agent = session.query(Agent).filter(Agent.agent_id == agent_id).first()
            if not agent:
                logger.error(f"Agent {agent_id} not found")
                return

            agent_state = AgentState(
                step_number=step_number,
                agent_id=agent_id,
                current_health=state_data["current_health"],
                max_health=state_data.get(
                    "max_health", agent.max_health
                ),  # Get from agent if not provided
                resource_level=state_data["resource_level"],
                position_x=state_data["position"][0],
                position_y=state_data["position"][1],
                is_defending=state_data["is_defending"],
                total_reward=state_data["total_reward"],
                starvation_threshold=state_data.get(
                    "starvation_threshold", agent.starvation_threshold
                ),
                age=step_number,  # Age is calculated from step number
            )
            session.add(agent_state)

        self._execute_in_transaction(_update)

    def _create_tables(self):
        """Create the required database schema.

        Creates all tables defined in the SQLAlchemy models if they don't exist.
        Also creates necessary indexes for performance optimization.

        Raises
        ------
        SQLAlchemyError
            If there's an error creating the tables
        OperationalError
            If there's a database connection issue
        """

        def _create(session):
            # Create all tables defined in the models
            Base.metadata.create_all(self.engine)

        self._execute_in_transaction(_create)

    def update_notes(self, notes_data: Dict):
        """Update simulation notes in the database.

        Parameters
        ----------
        notes_data : Dict
            Dictionary of notes to update, where:
            - keys are note identifiers
            - values are note content objects

        Raises
        ------
        SQLAlchemyError
            If there's an error updating the notes
        """

        def _update(session):
            # Use merge instead of update to handle both inserts and updates
            for key, value in notes_data.items():
                session.merge(value)

        self._execute_in_transaction(_update)

    def cleanup(self):
        """Clean up database and GUI resources.

        Performs cleanup operations including:
        - Flushing all data buffers
        - Closing database connections
        - Disposing of the engine
        - Removing session

        This method should be called before application shutdown.

        Raises
        ------
        Exception
            If cleanup operations fail
        """
        try:
            # Flush any pending changes
            self.flush_all_buffers()

            # Close database connections
            self.Session.remove()
            self.engine.dispose()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise
        finally:
            # Ensure critical resources are released
            if hasattr(self, "Session"):
                self.Session.remove()

    def get_configuration(self) -> Dict:
        """Retrieve the simulation configuration from the database."""

        def _query(session):
            config = (
                session.query(SimulationConfig)
                .order_by(SimulationConfig.timestamp.desc())
                .first()
            )

            if config and config.config_data:
                return json.loads(config.config_data)
            return {}

        return self._execute_in_transaction(_query)

    def save_configuration(self, config: Dict) -> None:
        """Save simulation configuration to the database."""

        def _insert(session):
            import time

            config_obj = SimulationConfig(
                timestamp=int(time.time()), config_data=json.dumps(config)
            )
            session.add(config_obj)

        self._execute_in_transaction(_insert)
