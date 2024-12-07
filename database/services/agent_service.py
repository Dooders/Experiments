from database.unit_of_work import UnitOfWork
from database.dtos.agent_dto import AgentDTO

class AgentService:
    def __init__(self, unit_of_work: UnitOfWork):
        self.uow = unit_of_work

    def get_agent_stats(self, agent_id: int) -> AgentDTO:
        with self.uow:
            agent = self.uow.agents.get_by_id(agent_id)
            return AgentDTO.from_entity(agent)

    def update_agent_position(self, agent_id: int, x: float, y: float):
        with self.uow:
            agent = self.uow.agents.get_by_id(agent_id)
            agent.position_x = x
            agent.position_y = y
            self.uow.commit()
