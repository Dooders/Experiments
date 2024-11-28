import logging
import threading
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session

if TYPE_CHECKING:
    from core.environment import Environment

logger = logging.getLogger(__name__)

Base = declarative_base()

class Agent(Base):
    __tablename__ = "Agents"
    agent_id = Column(Integer, primary_key=True)
    birth_time = Column(Integer, nullable=False)
    death_time = Column(Integer)
    agent_type = Column(String, nullable=False)
    position_x = Column(Float, nullable=False)
    position_y = Column(Float, nullable=False)
    initial_resources = Column(Float, nullable=False)
    max_health = Column(Float, nullable=False)
    starvation_threshold = Column(Integer, nullable=False)
    genome_id = Column(String)
    parent_id = Column(Integer, ForeignKey("Agents.agent_id"))
    generation = Column(Integer, nullable=False, default=0)

    parent = relationship("Agent", remote_side=[agent_id], backref="children")

class AgentState(Base):
    __tablename__ = "AgentStates"
    step_number = Column(Integer, primary_key=True)
    agent_id = Column(Integer, ForeignKey("Agents.agent_id"), primary_key=True)
    position_x = Column(Float, nullable=False)
    position_y = Column(Float, nullable=False)
    resource_level = Column(Float, nullable=False)
    current_health = Column(Float, nullable=False)
    max_health = Column(Float, nullable=False)
    starvation_threshold = Column(Integer, nullable=False)
    is_defending = Column(Boolean, nullable=False)
    total_reward = Column(Float, nullable=False)
    age = Column(Integer, nullable=False)

    agent = relationship("Agent", backref="states")

class ResourceState(Base):
    __tablename__ = "ResourceStates"
    step_number = Column(Integer, primary_key=True)
    resource_id = Column(Integer, primary_key=True)
    amount = Column(Float, nullable=False)
    position_x = Column(Float, nullable=False)
    position_y = Column(Float, nullable=False)

class SimulationStep(Base):
    __tablename__ = "SimulationSteps"
    step_number = Column(Integer, primary_key=True)
    total_agents = Column(Integer, nullable=False)
    system_agents = Column(Integer, nullable=False)
    independent_agents = Column(Integer, nullable=False)
    control_agents = Column(Integer, nullable=False)
    total_resources = Column(Float, nullable=False)
    average_agent_resources = Column(Float, nullable=False)
    births = Column(Integer, nullable=False)
    deaths = Column(Integer, nullable=False)
    current_max_generation = Column(Integer, nullable=False)
    resource_efficiency = Column(Float, nullable=False)
    resource_distribution_entropy = Column(Float, nullable=False)
    average_agent_health = Column(Float, nullable=False)
    average_agent_age = Column(Integer, nullable=False)
    average_reward = Column(Float, nullable=False)
    combat_encounters = Column(Integer, nullable=False)
    successful_attacks = Column(Integer, nullable=False)
    resources_shared = Column(Float, nullable=False)
    genetic_diversity = Column(Float, nullable=False)
    dominant_genome_ratio = Column(Float, nullable=False)

class AgentAction(Base):
    __tablename__ = "AgentActions"
    action_id = Column(Integer, primary_key=True)
    step_number = Column(Integer, nullable=False)
    agent_id = Column(Integer, ForeignKey("Agents.agent_id"), nullable=False)
    action_type = Column(String, nullable=False)
    action_target_id = Column(Integer)
    position_before = Column(Text)
    position_after = Column(Text)
    resources_before = Column(Float)
    resources_after = Column(Float)
    reward = Column(Float)
    details = Column(Text)

class LearningExperience(Base):
    __tablename__ = "LearningExperiences"
    experience_id = Column(Integer, primary_key=True)
    step_number = Column(Integer, nullable=False)
    agent_id = Column(Integer, ForeignKey("Agents.agent_id"), nullable=False)
    module_type = Column(String, nullable=False)
    state_before = Column(Text, nullable=False)
    action_taken = Column(Integer, nullable=False)
    reward = Column(Float, nullable=False)
    state_after = Column(Text, nullable=False)
    loss = Column(Float, nullable=False)

class HealthIncident(Base):
    __tablename__ = "HealthIncidents"
    incident_id = Column(Integer, primary_key=True)
    step_number = Column(Integer, nullable=False)
    agent_id = Column(Integer, ForeignKey("Agents.agent_id"), nullable=False)
    health_before = Column(Float, nullable=False)
    health_after = Column(Float, nullable=False)
    cause = Column(String, nullable=False)
    details = Column(Text)

class SimulationConfig(Base):
    __tablename__ = "SimulationConfig"
    config_id = Column(Integer, primary_key=True)
    timestamp = Column(Integer, nullable=False)
    config_data = Column(Text, nullable=False)

class SimulationDatabase:

    _thread_local = threading.local()
    _tables_creation_lock = threading.Lock()

    def __init__(self, db_path: str) -> None:
        """Initialize a new SimulationDatabase instance."""
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        self._setup_connection()

        # Add batch operation buffers
        self._action_buffer = []
        self._learning_exp_buffer = []
        self._health_incident_buffer = []
        self._buffer_size = 1000  # Adjust based on your needs

        # Create tables in a thread-safe manner
        with SimulationDatabase._tables_creation_lock:
            Base.metadata.create_all(self.engine)

    def _setup_connection(self):
        """Create a new session for the current thread."""
        if not hasattr(self._thread_local, "session"):
            self._thread_local.session = self.Session()

    @property
    def session(self):
        """Get thread-local session."""
        self._setup_connection()
        return self._thread_local.session

    def _execute_in_transaction(self, func):
        """Execute database operations within a transaction."""
        session = self.session
        try:
            result = func()
            session.commit()
            return result
        except Exception as e:
            session.rollback()
            logger.error(f"Database transaction failed: {e}")
            raise

    def batch_log_agents(self, agent_data: List[Dict]):
        """Batch insert multiple agents.

        Parameters
        ----------
        agent_data : List[Dict]
            List of dictionaries containing agent data
        """
        def _insert():
            agents = [
                Agent(
                    agent_id=data["agent_id"],
                    birth_time=data["birth_time"],
                    agent_type=data["agent_type"],
                    position_x=data["position"][0],
                    position_y=data["position"][1],
                    initial_resources=data["initial_resources"],
                    max_health=data["max_health"],
                    starvation_threshold=data["starvation_threshold"],
                    genome_id=data.get("genome_id"),
                    parent_id=data.get("parent_id"),
                    generation=data.get("generation", 0),
                )
                for data in agent_data
            ]
            self.session.add_all(agents)

        self._execute_in_transaction(_insert)

    def _prepare_metrics_values(self, step_number: int, metrics: Dict) -> SimulationStep:
        """Prepare metrics values for database insertion.

        Parameters
        ----------
        step_number : int
            Current simulation step number
        metrics : Dict
            Dictionary containing simulation metrics

        Returns
        -------
        SimulationStep
            SimulationStep object ready for database insertion
        """
        return SimulationStep(
            step_number=step_number,
            total_agents=metrics["total_agents"],
            system_agents=metrics["system_agents"],
            independent_agents=metrics["independent_agents"],
            control_agents=metrics["control_agents"],
            total_resources=metrics["total_resources"],
            average_agent_resources=metrics["average_agent_resources"],
            births=metrics["births"],
            deaths=metrics["deaths"],
            current_max_generation=metrics["current_max_generation"],
            resource_efficiency=metrics["resource_efficiency"],
            resource_distribution_entropy=metrics["resource_distribution_entropy"],
            average_agent_health=metrics["average_agent_health"],
            average_agent_age=metrics["average_agent_age"],
            average_reward=metrics["average_reward"],
            combat_encounters=metrics["combat_encounters"],
            successful_attacks=metrics["successful_attacks"],
            resources_shared=metrics["resources_shared"],
            genetic_diversity=metrics["genetic_diversity"],
            dominant_genome_ratio=metrics["dominant_genome_ratio"],
        )

    def _prepare_agent_state(self, agent, current_time: int) -> AgentState:
        """Prepare agent state data for database insertion.

        Parameters
        ----------
        agent : Agent
            Agent object containing state data
        current_time : int
            Current simulation time

        Returns
        -------
        AgentState
            AgentState object ready for database insertion
        """
        return AgentState(
            step_number=current_time,
            agent_id=agent.agent_id,
            position_x=agent.position[0],
            position_y=agent.position[1],
            resource_level=agent.resource_level,
            current_health=agent.current_health,
            max_health=agent.max_health,
            starvation_threshold=agent.starvation_threshold,
            is_defending=agent.is_defending,
            total_reward=agent.total_reward,
            age=current_time - agent.birth_time,
        )

    def _prepare_resource_state(self, resource) -> ResourceState:
        """Prepare resource state data for database insertion.

        Parameters
        ----------
        resource : Resource
            Resource object containing state data

        Returns
        -------
        ResourceState
            ResourceState object ready for database insertion
        """
        return ResourceState(
            step_number=resource.step_number,
            resource_id=resource.resource_id,
            amount=resource.amount,
            position_x=resource.position[0],
            position_y=resource.position[1],
        )

    def _prepare_action_data(self, action: Dict) -> AgentAction:
        """Prepare action data for database insertion.

        Parameters
        ----------
        action : Dict
            Dictionary containing action data

        Returns
        -------
        AgentAction
            AgentAction object ready for database insertion
        """
        import json

        return AgentAction(
            step_number=action["step_number"],
            agent_id=action["agent_id"],
            action_type=action["action_type"],
            action_target_id=action.get("action_target_id"),
            position_before=json.dumps(action["position_before"]) if action.get("position_before") else None,
            position_after=json.dumps(action["position_after"]) if action.get("position_after") else None,
            resources_before=action.get("resources_before"),
            resources_after=action.get("resources_after"),
            reward=action.get("reward"),
            details=json.dumps(action["details"]) if action.get("details") else None,
        )

    def log_step(
        self,
        step_number: int,
        agent_states: List[AgentState],
        resource_states: List[ResourceState],
        metrics: Dict,
    ):
        """Log comprehensive simulation state data for a single time step."""

        def _insert():
            # Bulk insert agent states
            if agent_states:
                self.session.add_all(agent_states)

            # Bulk insert resource states
            if resource_states:
                self.session.add_all(resource_states)

            # Insert metrics
            self.session.add(self._prepare_metrics_values(step_number, metrics))

        self._execute_in_transaction(_insert)

    def get_simulation_data(self, step_number: int) -> Dict:
        """Retrieve all simulation data for a specific time step."""
        # Get agent states
        agent_states = self.session.query(AgentState).filter_by(step_number=step_number).all()

        # Get resource states
        resource_states = self.session.query(ResourceState).filter_by(step_number=step_number).all()

        # Get metrics
        metrics = self.session.query(SimulationStep).filter_by(step_number=step_number).first()

        return {
            "agent_states": agent_states,
            "resource_states": resource_states,
            "metrics": metrics
        }

    def get_historical_data(self) -> Dict:
        """Retrieve historical metrics for the entire simulation."""
        steps = self.session.query(SimulationStep).order_by(SimulationStep.step_number).all()

        return {
            "steps": [step.step_number for step in steps],
            "metrics": {
                "total_agents": [step.total_agents for step in steps],
                "system_agents": [step.system_agents for step in steps],
                "independent_agents": [step.independent_agents for step in steps],
                "control_agents": [step.control_agents for step in steps],
                "total_resources": [step.total_resources for step in steps],
                "average_agent_resources": [step.average_agent_resources for step in steps],
                "births": [step.births for step in steps],
                "deaths": [step.deaths for step in steps]
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
            self.engine,
        )

        df.to_csv(filepath, index=False)

    def flush_all_buffers(self):
        """Flush all data buffers and ensure data is written to disk."""
        try:
            self.flush_action_buffer()
            self.flush_learning_buffer()
            self.flush_health_buffer()

            self.session.commit()
        except Exception as e:
            logger.error(f"Error flushing database buffers: {e}")
            raise

    def close(self):
        """Close the database connection for the current thread."""
        if hasattr(self._thread_local, "session"):
            try:
                self.flush_all_buffers()  # Ensure all buffered data is written
                self._thread_local.session.close()
                del self._thread_local.session
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
            resource_state = ResourceState(
                step_number=0,
                resource_id=resource_id,
                amount=initial_amount,
                position_x=position[0],
                position_y=position[1],
            )
            self.session.add(resource_state)

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
        lifespans = self.session.query(
            Agent.agent_type,
            func.avg(case([(Agent.death_time.isnot(None), Agent.death_time - Agent.birth_time)], else_=func.max(AgentState.age))).label("avg_lifespan"),
            func.max(case([(Agent.death_time.isnot(None), Agent.death_time - Agent.birth_time)], else_=func.max(AgentState.age))).label("max_lifespan"),
            func.min(case([(Agent.death_time.isnot(None), Agent.death_time - Agent.birth_time)], else_=func.max(AgentState.age))).label("min_lifespan")
        ).join(AgentState, Agent.agent_id == AgentState.agent_id).group_by(Agent.agent_type).all()

        return {
            lifespan.agent_type: {
                "average_lifespan": lifespan.avg_lifespan,
                "max_lifespan": lifespan.max_lifespan,
                "min_lifespan": lifespan.min_lifespan,
            }
            for lifespan in lifespans
        }

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

                incidents = [
                    HealthIncident(
                        step_number=incident["step_number"],
                        agent_id=incident["agent_id"],
                        health_before=incident["health_before"],
                        health_after=incident["health_after"],
                        cause=incident["cause"],
                        details=json.dumps(incident["details"]) if incident.get("details") else None,
                    )
                    for incident in self._health_incident_buffer
                ]
                self.session.add_all(incidents)

            self._execute_in_transaction(_batch_insert)
            self._health_incident_buffer.clear()

    def batch_log_agent_actions(self, actions: List[Dict]):
        """Batch insert multiple agent actions at once."""
        def _insert():
            action_objects = [self._prepare_action_data(action) for action in actions]
            self.session.add_all(action_objects)

        self._execute_in_transaction(_insert)

    def batch_log_learning_experiences(self, experiences: List[Dict]):
        """Batch insert multiple learning experiences at once."""
        def _insert():
            experience_objects = [
                LearningExperience(
                    step_number=exp["step_number"],
                    agent_id=exp["agent_id"],
                    module_type=exp["module_type"],
                    state_before=exp["state_before"],
                    action_taken=exp["action_taken"],
                    reward=exp["reward"],
                    state_after=exp["state_after"],
                    loss=exp["loss"],
                )
                for exp in experiences
            ]
            self.session.add_all(experience_objects)

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
            self.session.add(agent)

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
            agent = self.session.query(Agent).filter_by(agent_id=agent_id).first()
            if agent:
                agent.death_time = death_time

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
            result = self.session.execute("PRAGMA foreign_keys").fetchone()
            foreign_keys_enabled = bool(result[0])

            if not foreign_keys_enabled:
                logger.warning("Foreign key constraints are not enabled")
                return False

            # Test foreign key constraint
            try:
                # Try to insert a record with an invalid foreign key
                self.session.execute(
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
            except Exception:
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

        incidents = self.session.query(HealthIncident).filter_by(agent_id=agent_id).order_by(HealthIncident.step_number).all()

        return [
            {
                "step_number": incident.step_number,
                "health_before": incident.health_before,
                "health_after": incident.health_after,
                "cause": incident.cause,
                "details": json.loads(incident.details) if incident.details else None,
            }
            for incident in incidents
        ]

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
            action = self.session.query(AgentAction).filter_by(agent_id=agent_id, step_number=step_number).first()
            if action:
                return {
                    "action_type": action.action_type,
                    "action_target_id": action.action_target_id,
                    "position_before": action.position_before,
                    "position_after": action.position_after,
                    "resources_before": action.resources_before,
                    "resources_after": action.resources_after,
                    "reward": action.reward,
                    "details": action.details,
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
            config = self.session.query(SimulationConfig).order_by(SimulationConfig.timestamp.desc()).first()
            if config:
                import json
                return json.loads(config.config_data)
            return {}
        except Exception:
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
            config_entry = SimulationConfig(
                timestamp=int(time.time()),
                config_data=json.dumps(config)
            )
            self.session.add(config_entry)

        self._execute_in_transaction(_insert)

    def get_population_momentum(self) -> float:
        """Calculate population momentum ((death_step * max_count) / initial_count).
        
        Returns
        -------
        float
            Population momentum score. A higher score indicates the population
            maintained higher numbers for longer relative to its starting size.
            Returns 0 if there's no data or initial count is 0.
        """
        try:
            result = self.session.query(
                SimulationStep.step_number.label("death_step"),
                SimulationStep.total_agents.label("max_count"),
                self.session.query(SimulationStep.total_agents).order_by(SimulationStep.step_number).limit(1).scalar().label("initial_count"),
            ).order_by(SimulationStep.step_number.desc()).first()
            
            if result and result.initial_count > 0:
                return (result.death_step * result.max_count) / result.initial_count
            return 0
            
        except Exception as e:
            logging.error(f"Error calculating population momentum: {e}")
            return 0

    def get_population_statistics(self) -> Dict:
        """Calculate comprehensive population statistics."""
        try:
            result = self.session.query(
                func.avg(SimulationStep.total_agents).label("avg_population"),
                func.max(SimulationStep.step_number).label("death_step"),
                func.max(SimulationStep.total_agents).label("peak_population"),
                func.sum(SimulationStep.total_resources).label("total_resources_available"),
                func.sum(SimulationStep.total_agents * SimulationStep.total_resources).label("resources_consumed"),
                func.sum(SimulationStep.total_agents * SimulationStep.total_agents).label("sum_squared"),
                func.count(SimulationStep.step_number).label("step_count")
            ).filter(SimulationStep.total_agents > 0).first()
            
            if result:
                avg_pop = result.avg_population or 0
                death_step = result.death_step or 0
                peak_pop = result.peak_population or 0
                resources_consumed = result.resources_consumed or 0
                resources_available = result.total_resources_available or 0
                sum_squared = result.sum_squared or 0
                step_count = result.step_count or 1  # Avoid division by zero
                
                variance = (sum_squared / step_count) - (avg_pop * avg_pop)
                std_dev = variance ** 0.5
                
                resource_utilization = resources_consumed / resources_available if resources_available > 0 else 0
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
                    "utilization_per_agent": resources_consumed / (avg_pop * death_step) if avg_pop * death_step > 0 else 0
                }
                
            return {}
            
        except Exception as e:
            logging.error(f"Error calculating population statistics: {e}")
            return {}

    def get_advanced_statistics(self) -> Dict:
        """Calculate advanced simulation statistics."""
        try:
            result = self.session.query(
                func.first_value(SimulationStep.total_agents).over(order_by=SimulationStep.step_number).label("initial_pop"),
                func.first_value(SimulationStep.total_agents).over(order_by=SimulationStep.step_number.desc()).label("final_pop"),
                func.max(SimulationStep.total_agents).label("peak_pop"),
                func.avg(SimulationStep.total_resources).label("average_health"),
                func.avg(case([(SimulationStep.total_resources < (SimulationStep.total_agents * 0.5), 1)], else_=0)).label("scarcity_index"),
                func.count(SimulationStep.step_number).label("total_steps"),
                (func.sum(SimulationStep.total_agents) / func.count(SimulationStep.step_number)).label("interaction_rate"),
                (func.sum(case([(AgentAction.action_type.in_(['attack', 'defend']), 1], else_=0))) / func.sum(case([(AgentAction.action_type.in_(['share', 'help']), 1], else_=0)))).label("conflict_cooperation_ratio"),
                func.avg(SimulationStep.system_agents / SimulationStep.total_agents).label("avg_system_ratio"),
                func.avg(SimulationStep.independent_agents / SimulationStep.total_agents).label("avg_independent_ratio"),
                func.avg(SimulationStep.control_agents / SimulationStep.total_agents).label("avg_control_ratio"),
                (func.first_value(SimulationStep.total_agents).over(order_by=SimulationStep.step_number.desc()) / func.count(Agent.agent_id)).label("survivor_ratio"),
                func.min(case([(SimulationStep.total_agents <= (func.max(SimulationStep.total_agents) * 0.1), SimulationStep.step_number)], else_=None)).label("extinction_threshold_time")
            ).filter(SimulationStep.total_agents > 0).first()
            
            if result:
                initial_pop = result.initial_pop or 0
                final_pop = result.final_pop or 0
                peak_pop = result.peak_pop or 0
                total_steps = result.total_steps or 1  # Avoid division by zero
                
                import math
                ratios = [result.avg_system_ratio, result.avg_independent_ratio, result.avg_control_ratio]
                diversity = -sum(ratio * math.log(ratio) for ratio in ratios if ratio and ratio > 0)
                
                return {
                    "peak_to_end_ratio": peak_pop / final_pop if final_pop > 0 else float('inf'),
                    "growth_rate": (final_pop - initial_pop) / total_steps if total_steps > 0 else 0,
                    "extinction_threshold_time": result.extinction_threshold_time,
                    "average_health": result.average_health,
                    "survivor_ratio": result.survivor_ratio,
                    "agent_diversity": diversity,
                    "interaction_rate": result.interaction_rate if result.interaction_rate is not None else 0,
                    "conflict_cooperation_ratio": result.conflict_cooperation_ratio if result.conflict_cooperation_ratio is not None else 0,
                    "scarcity_index": result.scarcity_index if result.scarcity_index is not None else 0
                }
                
            return {}
            
        except Exception as e:
            logging.error(f"Error calculating advanced statistics: {e}")
            return {}
