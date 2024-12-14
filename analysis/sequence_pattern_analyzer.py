from typing import List, Optional, Tuple, Union
from database.data_types import SequencePattern, AgentActionData
from database.repositories.agent_action_repository import AgentActionRepository
from database.enums import AnalysisScope

class SequencePatternAnalyzer:
    def __init__(self, repository: AgentActionRepository):
        self.repository = repository

    def analyze(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[SequencePattern]:
        actions = self.repository.get_actions_by_scope(scope, agent_id, step, step_range)
        sequences = {}
        action_counts = {}

        for i in range(len(actions) - 1):
            current = actions[i]
            next_action = actions[i + 1]

            if current.agent_id == next_action.agent_id:
                sequence_key = f"{current.action_type}->{next_action.action_type}"
                sequences[sequence_key] = sequences.get(sequence_key, 0) + 1
                action_counts[current.action_type] = action_counts.get(current.action_type, 0) + 1

        return [
            SequencePattern(
                sequence=sequence,
                count=count,
                probability=count / action_counts[sequence.split("->")[0]] if sequence.split("->")[0] in action_counts else 0,
            )
            for sequence, count in sequences.items()
        ]
