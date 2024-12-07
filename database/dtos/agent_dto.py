from dataclasses import dataclass
from typing import Optional
from database.models import Agent

@dataclass
class AgentDTO:
    id: int
    type: str
    health: float
    position_x: float
    position_y: float
    resources: float

    @classmethod
    def from_entity(cls, agent: Agent) -> "AgentDTO":
        return cls(
            id=agent.agent_id,
            type=agent.agent_type,
            health=agent.current_health,
            position_x=agent.position_x,
            position_y=agent.position_y,
            resources=agent.resource_level
        )
