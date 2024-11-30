import json
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any, Union

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    case,
    create_engine,
    event,
    func,
    text,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, scoped_session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError, ProgrammingError

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
    agent_type = Column(String)
    position_x = Column(Float)
    position_y = Column(Float)
    initial_resources = Column(Float)
    max_health = Column(Float)
    starvation_threshold = Column(Integer)
    genome_id = Column(String)
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


class SimulationStep(Base):
    __tablename__ = "simulation_steps"
    __table_args__ = (
        Index("idx_simulation_steps_step_number", "step_number"),
    )

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
    action_type = Column(String, nullable=False)
    action_target_id = Column(Integer)
    position_before = Column(String)
    position_after = Column(String)
    resources_before = Column(Float)
    resources_after = Column(Float)
    reward = Column(Float)
    details = Column(String)  # JSON-encoded dictionary

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
    module_type = Column(String)
    state_before = Column(String)
    action_taken = Column(Integer)
    reward = Column(Float)
    state_after = Column(String)
    loss = Column(Float)

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
    health_before = Column(Float, nullable=False)
    health_after = Column(Float, nullable=False)
    cause = Column(String, nullable=False)
    details = Column(String)  # JSON-encoded details

    agent = relationship("Agent", back_populates="health_incidents")


class SimulationConfig(Base):
    __tablename__ = "simulation_config"

    config_id = Column(Integer, primary_key=True)
    timestamp = Column(Integer, nullable=False)
    config_data = Column(String, nullable=False)  # JSON-encoded configuration


class SimulationDatabase:
    def __init__(self, db_path: str) -> None:
        """Initialize a new SimulationDatabase instance with SQLAlchemy."""
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
        """Execute database operations within a transaction with specific error handling."""
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
        """Batch insert multiple agents using SQLAlchemy."""

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

    def get_simulation_data(self, step_number: int) -> Dict:
        """Retrieve simulation data using SQLAlchemy."""

        def _query(session):
            # Get agent states
            agent_states = (
                session.query(AgentState)
                .join(Agent)
                .filter(AgentState.step_number == step_number)
                .all()
            )

            # Get resource states
            resource_states = (
                session.query(ResourceState)
                .filter(ResourceState.step_number == step_number)
                .all()
            )

            # Get metrics
            metrics = (
                session.query(SimulationStep)
                .filter(SimulationStep.step_number == step_number)
                .first()
            )

            return {
                "agent_states": [
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
                ],
                "resource_states": [
                    (
                        state.resource_id,
                        state.amount,
                        state.position_x,
                        state.position_y,
                    )
                    for state in resource_states
                ],
                "metrics": (
                    {
                        "total_agents": metrics.total_agents if metrics else 0,
                        "system_agents": metrics.system_agents if metrics else 0,
                        "independent_agents": (
                            metrics.independent_agents if metrics else 0
                        ),
                        "control_agents": metrics.control_agents if metrics else 0,
                        "total_resources": metrics.total_resources if metrics else 0,
                        "average_agent_resources": (
                            metrics.average_agent_resources if metrics else 0
                        ),
                        "births": metrics.births if metrics else 0,
                        "deaths": metrics.deaths if metrics else 0,
                    }
                    if metrics
                    else {}
                ),
            }

        return self._execute_in_transaction(_query)

    def log_step(
        self,
        step_number: int,
        agent_states: List[Tuple],
        resource_states: List[Tuple],
        metrics: Dict,
    ):
        """Log comprehensive simulation state data for a single time step."""

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
        if self._action_buffer:

            def _insert(session):
                session.bulk_insert_mappings(AgentAction, self._action_buffer)

            self._execute_in_transaction(_insert)
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
        experience_data = {
            "step_number": step_number,
            "agent_id": agent_id,
            "module_type": module_type,
            "state_before": state_before,
            "action_taken": action_taken,
            "reward": reward,
            "state_after": state_after,
            "loss": loss,
        }
        self._learning_exp_buffer.append(experience_data)

        if len(self._learning_exp_buffer) >= self._buffer_size:
            self.flush_learning_buffer()

    def flush_learning_buffer(self):
        """Flush the learning experience buffer."""
        if self._learning_exp_buffer:

            def _insert(session):
                session.bulk_insert_mappings(
                    LearningExperience, self._learning_exp_buffer
                )

            self._execute_in_transaction(_insert)
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
        incident_data = {
            "step_number": step_number,
            "agent_id": agent_id,
            "health_before": health_before,
            "health_after": health_after,
            "cause": cause,
            "details": json.dumps(details) if details else None,
        }
        self._health_incident_buffer.append(incident_data)

        if len(self._health_incident_buffer) >= self._buffer_size:
            self.flush_health_buffer()

    def flush_health_buffer(self):
        """Flush the health incident buffer."""
        if self._health_incident_buffer:

            def _insert(session):
                session.bulk_insert_mappings(
                    HealthIncident, self._health_incident_buffer
                )

            self._execute_in_transaction(_insert)
            self._health_incident_buffer.clear()

    def get_agent_health_incidents(self, agent_id: int) -> List[Dict]:
        """Get all health incidents for an agent."""

        def _query(session):
            incidents = (
                session.query(HealthIncident)
                .filter(HealthIncident.agent_id == agent_id)
                .order_by(HealthIncident.step_number)
                .all()
            )

            return [
                {
                    "step_number": incident.step_number,
                    "health_before": incident.health_before,
                    "health_after": incident.health_after,
                    "cause": incident.cause,
                    "details": (
                        json.loads(incident.details) if incident.details else None
                    ),
                }
                for incident in incidents
            ]

        return self._execute_in_transaction(_query)

    def get_agent_lifespan_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate lifespan statistics for different agent types using simpler SQL."""
        def _query(session):
            # First get stats for agents that have died
            dead_stats = (
                session.query(
                    Agent.agent_type,
                    func.avg(Agent.death_time - Agent.birth_time).label("avg_lifespan"),
                    func.max(Agent.death_time - Agent.birth_time).label("max_lifespan"),
                    func.min(Agent.death_time - Agent.birth_time).label("min_lifespan"),
                )
                .filter(Agent.death_time.isnot(None))
                .group_by(Agent.agent_type)
                .all()
            )

            # Then get stats for living agents using their current age
            living_stats = (
                session.query(
                    Agent.agent_type,
                    func.avg(AgentState.age).label("avg_lifespan"),
                    func.max(AgentState.age).label("max_lifespan"),
                    func.min(AgentState.age).label("min_lifespan"),
                )
                .join(AgentState)
                .filter(Agent.death_time.is_(None))
                .group_by(Agent.agent_type)
                .all()
            )

            # Combine the results
            stats_dict = {}
            for row in dead_stats + living_stats:
                agent_type = row[0]
                if agent_type not in stats_dict:
                    stats_dict[agent_type] = {
                        "average_lifespan": float(row[1] or 0),
                        "max_lifespan": float(row[2] or 0),
                        "min_lifespan": float(row[3] or 0),
                    }
                else:
                    # Update existing stats if needed
                    stats_dict[agent_type]["average_lifespan"] = max(
                        stats_dict[agent_type]["average_lifespan"], 
                        float(row[1] or 0)
                    )
                    stats_dict[agent_type]["max_lifespan"] = max(
                        stats_dict[agent_type]["max_lifespan"], 
                        float(row[2] or 0)
                    )
                    stats_dict[agent_type]["min_lifespan"] = min(
                        stats_dict[agent_type]["min_lifespan"], 
                        float(row[3] or 0)
                    )

            return stats_dict

        return self._execute_in_transaction(_query)

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

    def flush_all_buffers(self) -> None:
        """Flush all data buffers with enhanced error handling."""
        buffers = {
            'action': self.flush_action_buffer,
            'learning': self.flush_learning_buffer,
            'health': self.flush_health_buffer
        }
        
        for buffer_name, flush_func in buffers.items():
            try:
                flush_func()
            except IntegrityError as e:
                logger.error(f"Data integrity error flushing {buffer_name} buffer: {e}")
                raise
            except OperationalError as e:
                logger.error(f"Database operation error flushing {buffer_name} buffer: {e}")
                raise
            except SQLAlchemyError as e:
                logger.error(f"Database error flushing {buffer_name} buffer: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error flushing {buffer_name} buffer: {e}")
                raise

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

    def export_data(self, filepath: str):
        """Export simulation metrics data to a CSV file."""

        def _query(session):
            # Query all simulation steps with their metrics
            query = session.query(SimulationStep).order_by(SimulationStep.step_number)

            # Convert to pandas DataFrame using SQLAlchemy
            df = pd.read_sql(query.statement, session.bind, index_col="step_number")

            # Export to CSV
            df.to_csv(filepath)

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
                func.max(SimulationStep.step_number).label("final_step")
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
                func.avg(SimulationStep.total_agents).label("avg_total")
            ).first()

            # Get interaction stats
            interaction_stats = session.query(
                func.count(AgentAction.action_id).label("total_actions"),
                func.sum(case(
                    [(AgentAction.action_type.in_(["attack", "defend"]), 1)],
                    else_=0
                )).label("conflicts"),
                func.sum(case(
                    [(AgentAction.action_type.in_(["share", "help"]), 1)],
                    else_=0
                )).label("cooperation")
            ).first()

            if not all([pop_stats, type_stats, interaction_stats]):
                return {}

            # Calculate diversity index
            total = float(type_stats[3] or 1)  # Avoid division by zero
            ratios = [
                float(count or 0) / total 
                for count in type_stats[:3]
            ]
            
            import math
            diversity = sum(
                -ratio * math.log(ratio) if ratio > 0 else 0 
                for ratio in ratios
            )

            return {
                "peak_population": int(pop_stats[1] or 0),
                "average_population": float(pop_stats[0] or 0),
                "total_steps": int(pop_stats[3] or 0),
                "average_health": float(pop_stats[4] or 0),
                "agent_diversity": diversity,
                "interaction_rate": (
                    float(interaction_stats[0] or 0) / 
                    float(pop_stats[3] or 1) / 
                    float(pop_stats[1] or 1)
                ),
                "conflict_cooperation_ratio": (
                    float(interaction_stats[1] or 0) / 
                    float(interaction_stats[2] or 1)  # Avoid division by zero
                )
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
        """Get all actions performed by an agent."""

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
                max_health=state_data.get("max_health", agent.max_health),  # Get from agent if not provided
                resource_level=state_data["resource_level"],
                position_x=state_data["position"][0],
                position_y=state_data["position"][1],
                is_defending=state_data["is_defending"],
                total_reward=state_data["total_reward"],
                starvation_threshold=state_data.get("starvation_threshold", agent.starvation_threshold),
                age=step_number,  # Age is calculated from step number
            )
            session.add(agent_state)

        self._execute_in_transaction(_update)

    def _create_tables(self):
        """Create the required database schema."""

        def _create(session):
            # Create all tables defined in the models
            Base.metadata.create_all(self.engine)

        self._execute_in_transaction(_create)

    def update_notes(self, notes_data: Dict):
        """Update notes in the database."""

        def _update(session):
            # Use merge instead of update to handle both inserts and updates
            for key, value in notes_data.items():
                session.merge(value)

        self._execute_in_transaction(_update)

    def cleanup(self):
        """Clean up database and GUI resources."""
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
