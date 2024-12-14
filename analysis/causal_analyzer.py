from typing import List, Optional, Tuple, Union
from database.data_types import CausalAnalysis, AgentActionData
from database.repositories.agent_action_repository import AgentActionRepository
from database.enums import AnalysisScope

class CausalAnalyzer:
    def __init__(self, repository: AgentActionRepository):
        self.repository = repository

    def analyze(
        self,
        action_type: str,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> CausalAnalysis:
        actions = self.repository.get_actions_by_scope(scope, agent_id, step_range=step_range)
        filtered_actions = [action for action in actions if action.action_type == action_type]

        if not filtered_actions:
            return CausalAnalysis(
                action_type=action_type,
                causal_impact=0.0,
                state_transition_probs={},
            )

        causal_impact = sum(a.reward or 0 for a in filtered_actions) / len(filtered_actions)

        state_transitions = {}
        for i in range(len(filtered_actions) - 1):
            current = filtered_actions[i]
            next_action = filtered_actions[i + 1]

            context_parts = []

            success = current.details.get("success", False) if current.details else False
            next_success = next_action.details.get("success", False) if next_action.details else False
            context_parts.append(f"success_{success}")
            context_parts.append(f"next_success_{next_success}")

            if current.resources_before is not None and current.resources_after is not None:
                resource_change = current.resources_after - current.resources_before
                if abs(resource_change) > 0:
                    context_parts.append(f"resource_change_{resource_change:+.1f}")

            if current.action_target_id:
                context_parts.append("targeted")

            if current.action_type == "gather" and current.details:
                if "amount_gathered" in current.details:
                    context_parts.append(f"gathered_{current.details['amount_gathered']}")
            elif current.action_type == "share" and current.details:
                if "amount_shared" in current.details:
                    context_parts.append(f"shared_{current.details['amount_shared']}")

            if next_action.action_type == "gather" and next_action.details:
                if "amount_gathered" in next_action.details:
                    context_parts.append(f"next_gathered_{next_action.details['amount_gathered']}")
            elif next_action.action_type == "share" and next_action.details:
                if "amount_shared" in next_action.details:
                    context_parts.append(f"next_shared_{next_action.details['amount_shared']}")

            transition_key = f"{next_action.action_type}|{','.join(context_parts)}"

            if transition_key not in state_transitions:
                state_transitions[transition_key] = 0
            state_transitions[transition_key] += 1

        total_transitions = sum(state_transitions.values())
        state_transition_probs = (
            {k: v / total_transitions for k, v in state_transitions.items()}
            if total_transitions > 0
            else {}
        )

        return CausalAnalysis(
            action_type=action_type,
            causal_impact=causal_impact,
            state_transition_probs=state_transition_probs,
        )
