from typing import List, Optional

from sqlalchemy.orm import Session

from database.models import ActionModel, AgentModel, AgentStateModel
from database.repositories.base_repository import BaseRepository
from database.session_manager import SessionManager


class AgentRepository(BaseRepository[AgentModel]):
    """Repository for handling agent-related data operations.

    This class provides methods to query and retrieve agents and their related data
    such as actions and states.

    Args:
        session_manager (SessionManager): Session manager for database operations.
    """

    def __init__(self, session_manager: SessionManager):
        """Initialize repository with session manager.

        Parameters
        ----------
        session_manager : SessionManager
            Session manager instance for database operations
        """
        self.session_manager = session_manager

    def get_agent_by_id(self, agent_id: str) -> Optional[AgentModel]:
        #! make this the agent info return???
        """Retrieve an agent by their unique identifier.

        Parameters
        ----------
        agent_id : str
            The unique identifier of the agent

        Returns
        -------
        Optional[AgentModel]
            The agent if found, None otherwise
            Fields:
            - agent_id: str (primary key)
            - birth_time: int
            - death_time: Optional[int]
            - agent_type: str
            - position_x: float
            - position_y: float
            - initial_resources: float
            - starting_health: float
            - starvation_threshold: int
            - genome_id: str
            - generation: int

            Relationships:
            - states: List[AgentStateModel]
            - actions: List[ActionModel]
            - health_incidents: List[HealthIncident]
            - learning_experiences: List[LearningExperience]
            - targeted_actions: List[ActionModel]
        """

        def query_agent(session: Session) -> Optional[AgentModel]:
            return session.query(AgentModel).get(agent_id)

        return self.session_manager.execute_with_retry(query_agent)

    def get_actions_by_agent_id(self, agent_id: str) -> List[ActionModel]:
        """Retrieve actions by agent ID.

        Parameters
        ----------
        agent_id : str
            The unique identifier of the agent

        Returns
        -------
        List[ActionModel]
            List of actions performed by the agent
            Fields:
            - action_id: int (primary key)
            - step_number: int
            - agent_id: str
            - action_type: str
            - action_target_id: Optional[str]
            - state_before_id: Optional[str]
            - state_after_id: Optional[str]
            - resources_before: float
            - resources_after: float
            - reward: float
            - details: Optional[str]

            Relationships:
            - agent: AgentModel
            - state_before: Optional[AgentStateModel]
            - state_after: Optional[AgentStateModel]
        """

        def query_actions(session: Session) -> List[ActionModel]:
            return (
                session.query(ActionModel)
                .filter(ActionModel.agent_id == agent_id)
                .all()
            )

        return self.session_manager.execute_with_retry(query_actions)

    def get_states_by_agent_id(self, agent_id: str) -> List[AgentStateModel]:
        """Retrieve states by agent ID.

        Parameters
        ----------
        agent_id : str
            The unique identifier of the agent

        Returns
        -------
        List[AgentStateModel]
            List of states associated with the agent
            Fields:
            - id: str (primary key, format: "agent_id-step_number")
            - step_number: int
            - agent_id: str
            - position_x: float
            - position_y: float
            - position_z: float
            - resource_level: float
            - current_health: float
            - is_defending: bool
            - total_reward: float
            - age: int

            Relationships:
            - agent: AgentModel
        """

        def query_states(session: Session) -> List[AgentStateModel]:
            return (
                session.query(AgentStateModel)
                .filter(AgentStateModel.agent_id == agent_id)
                .all()
            )

        return self.session_manager.execute_with_retry(query_states)
