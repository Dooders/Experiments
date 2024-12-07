from database.models import Agent
from database.repositories.base import BaseRepository

class AgentRepository(BaseRepository):
    def get_by_id(self, agent_id: int) -> Agent:
        def operation(session):
            return session.query(Agent).filter(Agent.agent_id == agent_id).first()
        return self._execute_transaction(operation)
