"""Attack learning and execution module using Deep Q-Learning (DQN).

This module implements a Deep Q-Learning approach for agents to learn optimal attack
policies in a multi-agent environment. It provides both the neural network architecture
and the training/execution logic.

Key Components:
    - AttackQNetwork: Neural network architecture for Q-value approximation
    - AttackModule: Main class handling training, action selection, and attack execution
    - Experience Replay: Stores and samples past experiences for stable learning
    - Target Network: Separate network for computing stable Q-value targets

Technical Details:
    - State Space: N-dimensional vector representing agent's current state
    - Action Space: 5 discrete actions (attack up/down/left/right, defend)
    - Learning Algorithm: Deep Q-Learning with experience replay
    - Exploration: Epsilon-greedy strategy with decay
    - Hardware Acceleration: Automatic GPU usage when available
"""

import logging
from typing import TYPE_CHECKING

from actions.base_dqn import BaseDQNConfig, BaseDQNModule, BaseQNetwork

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttackConfig(BaseDQNConfig):
    """Configuration specific to attacks."""

    attack_base_cost: float = -0.2
    attack_success_reward: float = 1.0
    attack_failure_penalty: float = -0.3
    attack_defense_threshold: float = 0.3
    attack_defense_boost: float = 2.0


DEFAULT_ATTACK_CONFIG = AttackConfig()


class AttackActionSpace:
    """Define available attack actions."""

    ATTACK_RIGHT: int = 0
    ATTACK_LEFT: int = 1
    ATTACK_UP: int = 2
    ATTACK_DOWN: int = 3
    DEFEND: int = 4


class AttackQNetwork(BaseQNetwork):
    """Attack-specific Q-network."""

    def __init__(self, input_dim: int, hidden_size: int = 64) -> None:
        super().__init__(
            input_dim, output_dim=5, hidden_size=hidden_size
        )  # 5 attack actions


class AttackModule(BaseDQNModule):
    """Attack-specific DQN module."""

    def __init__(
        self,
        config: AttackConfig = DEFAULT_ATTACK_CONFIG,
        device: torch.device = DEVICE,
    ) -> None:
        super().__init__(input_dim=6, output_dim=5, config=config, device=device)
        self._setup_action_space()

    def _setup_action_space(self) -> None:
        """Initialize attack-specific action space."""
        self.action_space = {
            AttackActionSpace.ATTACK_RIGHT: (1, 0),
            AttackActionSpace.ATTACK_LEFT: (-1, 0),
            AttackActionSpace.ATTACK_UP: (0, 1),
            AttackActionSpace.ATTACK_DOWN: (0, -1),
            AttackActionSpace.DEFEND: (0, 0),
        }

    def select_action(self, state: torch.Tensor, health_ratio: float) -> int:
        """Override select_action to include health-based defense boost."""
        if torch.rand(1).item() > self.epsilon:
            with torch.no_grad():
                q_values = self.q_network(state)
                if health_ratio < self.config.attack_defense_threshold:
                    q_values[
                        AttackActionSpace.DEFEND
                    ] *= self.config.attack_defense_boost
                return q_values.argmax().item()
        return torch.randint(0, len(self.action_space), (1,)).item()


def attack_action(agent: "BaseAgent") -> None:
    """Execute attack action using the AttackModule.

    This function handles the complete attack action workflow for an agent, including:
    - State evaluation and action selection
    - Target identification and selection
    - Damage calculation and application
    - Experience storage for learning

    The function implements several key mechanics:
    - Defense stance when DEFEND action is selected
    - Reduced damage when agent has low resources
    - Closest target selection within attack range
    - Experience replay storage for reinforcement learning

    Args:
        agent (BaseAgent): The agent executing the attack action

    Returns:
        None

    Side Effects:
        - Updates agent's defensive stance
        - Modifies target agent's health
        - Stores experience in agent's attack module
        - Logs attack outcomes
    """
    # Get current state and health ratio
    state = agent.get_state()
    health_ratio = agent.current_health / agent.max_health

    # Select attack action
    action = agent.attack_module.select_action(
        state.to_tensor(agent.attack_module.device), health_ratio
    )

    # Handle defense action
    if action == AttackActionSpace.DEFEND:
        agent.is_defending = True
        logger.debug(f"Agent {id(agent)} took defensive stance")
        return

    # Calculate attack target position
    target_pos = agent.calculate_attack_position(action)

    # Find potential targets
    targets = [
        other
        for other in agent.environment.agents
        if other != agent
        and other.alive
        and np.sqrt(((np.array(other.position) - np.array(target_pos)) ** 2).sum())
        < agent.config.attack_range
    ]
    if not targets:
        logger.debug(f"Agent {id(agent)} attack found no targets")
        return

    # Select closest target
    target = min(
        targets,
        key=lambda t: np.sqrt(
            ((np.array(t.position) - np.array(target_pos)) ** 2).sum()
        ),
    )

    # Calculate base damage
    base_damage = agent.config.attack_base_damage
    if agent.resource_level < agent.config.min_reproduction_resources:
        base_damage *= 0.7  # Reduced damage when low on resources

    # Execute attack
    damage_dealt = target.handle_combat(agent, base_damage)

    # Calculate reward and store experience
    reward = agent.calculate_attack_reward(target, damage_dealt, action)
    agent.total_reward += reward

    # Store experience
    next_state = agent.get_state()
    agent.attack_module.store_experience(
        state.to_tensor(agent.attack_module.device),
        action,
        reward,
        next_state.to_tensor(agent.attack_module.device),
        not target.alive,
    )

    logger.info(
        f"Agent {id(agent)} attacked Agent {id(target)} for {damage_dealt:.1f} damage. "
        f"Target health: {target.current_health:.1f}/{target.max_health}"
    )
