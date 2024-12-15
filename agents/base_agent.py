import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from core.action import *
from actions.attack import AttackActionSpace, AttackModule, attack_action
from actions.gather import GatherModule, gather_action
from actions.move import MoveModule, move_action
from actions.reproduce import ReproduceModule, reproduce_action
from actions.select import SelectConfig, SelectModule, create_selection_state
from actions.share import ShareModule, share_action
from core.genome import Genome
from core.state import AgentState

if TYPE_CHECKING:
    from core.environment import Environment

logger = logging.getLogger(__name__)


BASE_ACTION_SET = [
    Action("move", 0.4, move_action),
    Action("gather", 0.3, gather_action),
    Action("share", 0.2, share_action),
    Action("attack", 0.1, attack_action),
    Action("reproduce", 0.15, reproduce_action),
]


class BaseAgent:
    """Base agent class representing an autonomous entity in the simulation environment.

    This agent can move, gather resources, share with others, and engage in combat.
    It maintains its own state including position, resources, and health, while making
    decisions through various specialized modules.

    Attributes:
        actions (list[Action]): Available actions the agent can take
        agent_id (int): Unique identifier for this agent
        position (tuple[int, int]): Current (x,y) coordinates
        resource_level (int): Current amount of resources held
        alive (bool): Whether the agent is currently alive
        environment (Environment): Reference to the simulation environment
        device (torch.device): Computing device (CPU/GPU) for neural operations
        total_reward (float): Cumulative reward earned
        current_health (float): Current health points
        max_health (float): Maximum possible health points
    """

    def __init__(
        self,
        agent_id: str,
        position: tuple[int, int],
        resource_level: int,
        environment: "Environment",
        action_set: list[Action] = BASE_ACTION_SET,
        parent_id: Optional[str] = None,
        generation: int = 0,
        skip_logging: bool = False,
    ):
        """Initialize a new agent with given parameters."""
        # Add default actions
        self.actions = action_set

        # Normalize weights
        total_weight = sum(action.weight for action in self.actions)
        for action in self.actions:
            action.weight /= total_weight

        self.agent_id = agent_id
        self.position = position
        self.resource_level = resource_level
        self.alive = True
        self.environment = environment
        self.config = environment.config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_state: AgentState | None = None #! change to previous_state
        self.last_action = None #! change to previous_action
        self.max_movement = self.config.max_movement
        self.total_reward = 0
        self.episode_rewards = []
        self.losses = []
        self.starvation_threshold = self.config.starvation_threshold
        self.max_starvation = self.config.max_starvation_time
        self.birth_time = environment.time

        # Initialize health tracking first
        self.max_health = self.config.max_health #! change to starting_health
        self.current_health = self.max_health
        self.is_defending = False

        # Generate genome info
        #! make this 'parent_a_id-parent_b_id'
        self.genome_id = f"{self.__class__.__name__}_{agent_id}_{environment.time}"
        self.parent_id = parent_id
        self.generation = generation

        # Initialize all modules first
        #! make this a list of action modules that can be provided to the agent at init
        self.move_module = MoveModule(self.config, db=environment.db)
        self.attack_module = AttackModule(self.config)
        self.share_module = ShareModule(self.config)
        self.gather_module = GatherModule(self.config)
        self.reproduce_module = ReproduceModule(self.config)
        self.select_module = SelectModule(
            num_actions=len(self.actions), config=SelectConfig(), device=self.device
        )

        # Log agent creation to database only if not skipped
        if not skip_logging:
            environment.db.logger.log_agent(
                agent_id=self.agent_id,
                birth_time=environment.time,
                agent_type=self.__class__.__name__,
                position=self.position,
                initial_resources=self.resource_level,
                max_health=self.max_health,
                starvation_threshold=self.starvation_threshold,
                genome_id=self.genome_id,
                parent_id=self.parent_id,
                generation=self.generation,
            )

            logger.info(
                f"Agent {self.agent_id} created at {self.position} during step {environment.time} of type {self.__class__.__name__}"
            )
            
    def get_perception(self) -> AgentState:
        #! make this a list of perception modules that can be provided to the agent at init
        pass

    def get_state(self) -> AgentState:
        #! rethink state, needs to return a full state with an option to normalize and/or return a smaller state representation
        #! then need to update the state in the database and the input tensor to the neural network
        #! include perception with state? or just the state and perception is seperate?
        #! also incorporate 3rd dimension for position
        """Get the current normalized state of the agent.

        Calculates the agent's state relative to the nearest available resource,
        including normalized values for:
        - Distance to nearest resource
        - Angle to nearest resource
        - Current resource level
        - Amount of resources at nearest location

        If no resources are available, returns a default state with maximum distance
        and neutral angle values.

        Returns:
            AgentState: Normalized state representation containing all relevant metrics
        """
        # Get closest resource position
        #! this is the perception module
        closest_resource = None
        min_distance = float("inf")
        for resource in self.environment.resources:
            if resource.amount > 0:
                dist = np.sqrt(
                    (self.position[0] - resource.position[0]) ** 2
                    + (self.position[1] - resource.position[1]) ** 2
                )
                if dist < min_distance:
                    min_distance = dist
                    closest_resource = resource

        if closest_resource is None:
            # Return zero state if no resources available
            return AgentState(
                normalized_distance=1.0,  # Maximum distance
                normalized_angle=0.5,  # Neutral angle
                normalized_resource_level=0.0,
                normalized_target_amount=0.0,
            )

        # Calculate raw values
        dx = closest_resource.position[0] - self.position[0]
        dy = closest_resource.position[1] - self.position[1]
        angle = np.arctan2(dy, dx)

        # Calculate environment diagonal for distance normalization
        env_diagonal = np.sqrt(self.environment.width**2 + self.environment.height**2)

        # Ensure resource level is non-negative
        resource_level = max(0.0, self.resource_level)

        # Create normalized state using factory method
        return AgentState.from_raw_values(
            distance=min_distance,
            angle=angle,
            resource_level=resource_level,  # Use clamped value
            target_amount=closest_resource.amount,
            env_diagonal=env_diagonal,
        )

    def select_action(self):
        """Select an action using the SelectModule's intelligent decision making.

        The selection process involves:
        1. Getting current state representation
        2. Passing state through SelectModule's neural network
        3. Combining learned preferences with predefined action weights
        4. Choosing optimal action based on current circumstances

        Returns:
            Action: Selected action object to execute
        """
        # Get current state for selection
        state = create_selection_state(self)

        # Select action using selection module
        selected_action = self.select_module.select_action(
            agent=self, actions=self.actions, state=state
        )

        return selected_action

    def act(self):
        """Execute the agent's turn in the simulation.

        This method handles the core action loop including:
        1. Resource consumption and starvation checks
        2. State observation
        3. Action selection and execution
        4. State/action memory for learning

        The agent will not act if it's not alive. Each turn consumes base resources
        and can potentially lead to death if resources are depleted.
        """
        if not self.alive:
            return

        # Reset defense status at start of turn
        self.is_defending = False

        starting_resources = self.resource_level #! whats this for?
        self.resource_level -= self.config.base_consumption_rate


        #! encapsulate this in a method
        #! maybe even change the logic
        if self.resource_level <= 0:
            self.starvation_threshold += 1
            if self.starvation_threshold >= self.max_starvation:
                self.die()
                return
        else:
            self.starvation_threshold = 0

        # Get current state before action
        current_state = self.get_state()

        # Select and execute action
        action = self.select_action()
        action.execute(self)

        # Store state for learning
        self.last_state = current_state
        self.last_action = action

    def clone(self) -> "BaseAgent":
        """Create a mutated copy of this agent.

        Creates a new agent by:
        1. Cloning the current agent's genome
        2. Applying random mutations with 10% probability
        3. Converting mutated genome back to agent instance

        Returns:
            BaseAgent: A new agent with slightly modified characteristics
        """
        cloned_genome = Genome.clone(self.to_genome())
        mutated_genome = Genome.mutate(cloned_genome, mutation_rate=0.1)
        return Genome.to_agent(
            mutated_genome, self.agent_id, self.position, self.environment
        )

    def reproduce(self) -> bool:
        """Attempt reproduction. Returns True if successful."""
        if len(self.environment.agents) >= self.config.max_population:
            return False

        if self.resource_level >= self.config.min_reproduction_resources:
            if self.resource_level >= self.config.offspring_cost + 2:
                new_agent = self.create_offspring()
                self.environment.add_agent(new_agent)
                self.resource_level -= self.config.offspring_cost

                logger.info(
                    f"Agent {self.agent_id} reproduced at {self.position} during step {self.environment.time} creating agent {new_agent.agent_id}"
                )
                return True
        return False

    def create_offspring(self):
        """Create a new agent as offspring."""
        # Get the agent's class (IndependentAgent, SystemAgent, etc)
        agent_class = type(self)

        # Generate unique ID and genome info first
        new_id = self.environment.get_next_agent_id()
        generation = self.generation + 1
        genome_id = f"{agent_class.__name__}_{new_id}_{self.environment.time}"

        # Create new agent with all info
        new_agent = agent_class(
            agent_id=new_id,
            position=self.position,
            resource_level=self.config.offspring_initial_resources,
            environment=self.environment,
            parent_id=self.agent_id,
            generation=generation,
            skip_logging=True,  # Skip individual logging since we'll batch it
        )

        # Set additional attributes
        new_agent.genome_id = genome_id

        # Add to environment using batch operation
        self.environment.batch_add_agents([new_agent])

        # Log creation
        logger.info(
            f"Agent {new_id} created at {self.position} during step {self.environment.time} of type {agent_class.__name__}"
        )

        return new_agent

    def die(self):
        """Handle the agent's death process.

        Performs cleanup operations including:
        1. Setting alive status to False
        2. Logging death event to database
        3. Removing agent from environment's active agents
        4. Logging death information
        """
        self.alive = False

        # Log death to database
        self.environment.db.update_agent_death(
            agent_id=self.agent_id, death_time=self.environment.time
        )

        # Remove agent from environment's active agents list
        if hasattr(self.environment, "agents"):
            try:
                self.environment.agents.remove(self)
            except ValueError:
                pass  # Agent was already removed

        logger.info(
            f"Agent {self.agent_id} died at {self.position} during step {self.environment.time}"
        )

    def get_environment(self) -> "Environment":
        return self._environment

    def set_environment(self, environment: "Environment") -> None:
        self._environment = environment

    def calculate_new_position(self, action):
        """Calculate new position based on movement action.

        Takes into account:
        1. Environment boundaries
        2. Maximum movement distance
        3. Direction vectors for each action type

        Args:
            action (int): Movement direction index
                0: Right
                1: Left
                2: Up
                3: Down

        Returns:
            tuple: New (x, y) position coordinates, bounded by environment limits
        """
        # Define movement vectors for each action
        action_vectors = {
            0: (1, 0),  # Right
            1: (-1, 0),  # Left
            2: (0, 1),  # Up
            3: (0, -1),  # Down
        }

        # Get movement vector for the action
        dx, dy = action_vectors[action]

        # Scale by max_movement
        dx *= self.config.max_movement
        dy *= self.config.max_movement

        # Calculate new position
        new_x = max(0, min(self.environment.width, self.position[0] + dx))
        new_y = max(0, min(self.environment.height, self.position[1] + dy))

        return (new_x, new_y)

    def calculate_move_reward(self, old_pos, new_pos):
        """Calculate reward for a movement action.

        Reward calculation considers:
        1. Base movement cost (-0.1)
        2. Distance to nearest resource before and after move
        3. Positive reward (0.3) for moving closer to resources
        4. Negative reward (-0.2) for moving away from resources

        Args:
            old_pos (tuple): Previous (x, y) position
            new_pos (tuple): New (x, y) position

        Returns:
            float: Movement reward value
        """
        # Base cost for moving
        reward = -0.1

        # Calculate movement distance
        distance_moved = np.sqrt(
            (new_pos[0] - old_pos[0]) ** 2 + (new_pos[1] - old_pos[1]) ** 2
        )

        if distance_moved > 0:
            # Find closest non-depleted resource
            closest_resource = min(
                [r for r in self.environment.resources if not r.is_depleted()],
                key=lambda r: np.sqrt(
                    (r.position[0] - new_pos[0]) ** 2
                    + (r.position[1] - new_pos[1]) ** 2
                ),
                default=None,
            )

            if closest_resource:
                # Calculate distances to resource before and after move
                old_distance = np.sqrt(
                    (closest_resource.position[0] - old_pos[0]) ** 2
                    + (closest_resource.position[1] - old_pos[1]) ** 2
                )
                new_distance = np.sqrt(
                    (closest_resource.position[0] - new_pos[0]) ** 2
                    + (closest_resource.position[1] - new_pos[1]) ** 2
                )

                # Reward for moving closer to resources, penalty for moving away
                reward += 0.3 if new_distance < old_distance else -0.2

        return reward

    def calculate_attack_position(self, action: int) -> tuple[float, float]:
        """Calculate target position for attack based on action.

        Determines attack target location by:
        1. Getting direction vector from action space
        2. Scaling by attack range from config
        3. Adding to current position

        Args:
            action (int): Attack action index from AttackActionSpace

        Returns:
            tuple[float, float]: Target (x,y) coordinates for attack
        """
        # Get attack direction vector
        dx, dy = self.attack_module.action_space[action]

        # Scale by attack range
        dx *= self.config.attack_range
        dy *= self.config.attack_range

        # Calculate target position
        target_x = self.position[0] + dx
        target_y = self.position[1] + dy

        return (target_x, target_y)

    def handle_combat(self, attacker: "BaseAgent", damage: float) -> float:
        """Handle incoming attack and calculate actual damage taken.

        Processes combat mechanics including:
        - Damage reduction from defensive stance
        - Health reduction
        - Death checking if health drops to 0

        Args:
            attacker (BaseAgent): Agent performing the attack
            damage (float): Base damage amount before modifications

        Returns:
            float: Actual damage dealt after defensive calculations
        """
        # Reduce damage if defending
        if self.is_defending:
            damage *= 0.5  # 50% damage reduction when defending

        # Apply damage
        self.current_health = max(0, self.current_health - damage)

        # Check for death
        if self.current_health <= 0:
            self.die()

        return damage

    def calculate_attack_reward(
        self, target: "BaseAgent", damage_dealt: float, action: int
    ) -> float:
        """Calculate reward for an attack action based on outcome.

        Rewards are based on:
        - Base attack cost (negative)
        - Successful hits (positive, scaled by damage)
        - Killing blows (bonus reward)
        - Defensive actions (contextual based on health)
        - Missed attacks (penalty)

        Args:
            target: The agent that was attacked
            damage_dealt: Amount of damage successfully dealt
            action: The attack action that was taken

        Returns:
            float: The calculated reward value
        """
        # Base reward starts with the attack cost
        reward = self.config.attack_base_cost

        # Defensive action reward
        if action == AttackActionSpace.DEFEND:
            if (
                self.current_health
                < self.max_health * self.config.attack_defense_threshold
            ):
                reward += (
                    self.config.attack_success_reward
                )  # Good decision to defend when health is low
            else:
                reward += self.config.attack_failure_penalty  # Unnecessary defense
            return reward

        # Attack success reward
        if damage_dealt > 0:
            reward += self.config.attack_success_reward * (
                damage_dealt / self.config.attack_base_damage
            )
            if not target.alive:
                reward += self.config.attack_kill_reward
        else:
            reward += self.config.attack_failure_penalty

        return reward

    def to_genome(self) -> "Genome":
        """Convert agent's current state into a genome representation.

        Creates a genetic encoding of the agent's:
        - Neural network weights
        - Action preferences
        - Other learnable parameters

        Returns:
            Genome: Complete genetic representation of agent
        """
        return Genome.from_agent(self)

    @classmethod
    def from_genome(
        cls,
        genome: "Genome",
        agent_id: int,
        position: tuple[int, int],
        environment: "Environment",
    ) -> "BaseAgent":
        """Create a new agent instance from a genome.

        Factory method that:
        1. Decodes genome into agent parameters
        2. Initializes new agent with those parameters
        3. Sets up required environment connections

        Args:
            genome (Genome): Genetic encoding of agent parameters
            agent_id (int): Unique identifier for new agent
            position (tuple[int, int]): Starting coordinates
            environment (Environment): Simulation environment reference

        Returns:
            BaseAgent: New agent instance with genome's characteristics
        """
        return Genome.to_agent(genome, agent_id, position, environment)

    def encode_parameters(self) -> dict:
        """Encode agent parameters into gene representations.

        Returns:
            dict: Encoded parameters as genes.
        """
        parameters = {
            "learning_rate": self.config.learning_rate,
            "gamma": self.config.gamma,
            "epsilon_start": self.config.epsilon_start,
            "epsilon_min": self.config.epsilon_min,
            "epsilon_decay": self.config.epsilon_decay,
            "memory_size": self.config.memory_size,
            "batch_size": self.config.batch_size,
            "training_frequency": self.config.training_frequency,
            "dqn_hidden_size": self.config.dqn_hidden_size,
            "tau": self.config.tau,
        }
        return Genome.encode_parameters(parameters)

    def decode_parameters(self, genes: dict) -> None:
        """Decode genes back into agent parameters.

        Args:
            genes (dict): Dictionary of genes to decode.
        """
        parameters = Genome.decode_parameters(genes)
        self.config.learning_rate = parameters["learning_rate"]
        self.config.gamma = parameters["gamma"]
        self.config.epsilon_start = parameters["epsilon_start"]
        self.config.epsilon_min = parameters["epsilon_min"]
        self.config.epsilon_decay = parameters["epsilon_decay"]
        self.config.memory_size = parameters["memory_size"]
        self.config.batch_size = parameters["batch_size"]
        self.config.training_frequency = parameters["training_frequency"]
        self.config.dqn_hidden_size = parameters["dqn_hidden_size"]
        self.config.tau = parameters["tau"]

    def integrate_genetic_processes(self) -> None:
        """Integrate genetic processes for evolving agent parameters."""
        # Example: Apply mutation to agent's genome
        genome = self.to_genome()
        mutated_genome = Genome.mutate(genome, mutation_rate=0.1)
        self.decode_parameters(mutated_genome)
