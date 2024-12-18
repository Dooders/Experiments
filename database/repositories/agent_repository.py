from typing import List, Optional

from sqlalchemy.orm import Session

from database.data_types import AgentGenetics
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
        """Retrieve an agent by their unique identifier.

        Parameters
        ----------
        agent_id : str
            The unique identifier of the agent

        Returns
        -------
        Optional[AgentModel]
            The agent if found, None otherwise
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
        """

        def query_states(session: Session) -> List[AgentStateModel]:
            return (
                session.query(AgentStateModel)
                .filter(AgentStateModel.agent_id == agent_id)
                .all()
            )

        return self.session_manager.execute_with_retry(query_states)

    def get_genetics_by_agent_id(self, agent_id: str) -> AgentGenetics:
        """Get genetic information about an agent.
        #! better define the ids for genome and parent

        Parameters
        ----------
        agent_id : str
            The unique identifier of the agent to query

        Returns
        -------
        AgentGenetics
            Genetic information including:
            - genome_id: str
            - parent_id: Optional[int]
            - generation: int
        """

        def query_genetics(session: Session) -> AgentGenetics:
            agent = (
                session.query(AgentModel)
                .filter(AgentModel.agent_id == agent_id)
                .first()
            )
            return AgentGenetics(
                genome_id=agent.genome_id,
                parent_id=agent.parent_id,
                generation=agent.generation,
            )

        return self.session_manager.execute_with_retry(query_genetics)
