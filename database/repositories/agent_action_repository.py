from typing import List, Optional, Tuple, Union

from sqlalchemy.orm import Session
from sqlalchemy import func

from database.models import AgentAction
from database.data_types import AgentActionData
from database.scope_utils import filter_scope


class AgentActionRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_actions_by_scope(
        self,
        scope: str,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[AgentActionData]:
        query = self.session.query(AgentAction).order_by(
            AgentAction.step_number, AgentAction.agent_id
        )

        query = filter_scope(query, scope, agent_id, step, step_range)

        actions = query.all()
        return [
            AgentActionData(
                agent_id=action.agent_id,
                action_type=action.action_type,
                step_number=action.step_number,
                action_target_id=action.action_target_id,
                resources_before=action.resources_before,
                resources_after=action.resources_after,
                state_before_id=action.state_before_id,
                state_after_id=action.state_after_id,
                reward=action.reward,
                details=action.details if action.details else None,
            )
            for action in actions
        ]
