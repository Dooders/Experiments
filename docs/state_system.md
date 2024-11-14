## Overview
The state system provides a standardized way to represent and manage different types of states in the simulation. It uses Pydantic for validation and type safety, with a hierarchical structure of state classes.

## Base State (`BaseState`)
The foundation class for normalized state representations.

### Key Features
- Enforces value normalization (0-1 range)
- Immutable after creation
- Required tensor conversion interface
- Standard dictionary serialization
- Built-in validation

### Core Methods
```python
def to_tensor(self, device: torch.device) -> torch.Tensor:
    """Convert state to tensor format for neural networks"""

def to_dict(self) -> Dict[str, Any]:
    """Convert state to dictionary representation"""
```

## State Types

### 1. Agent State (`AgentState`)
Represents an agent's perception of its environment.

#### Attributes
- `normalized_distance`: Distance to nearest resource (0-1)
- `normalized_angle`: Angle to nearest resource (0-1)
- `normalized_resource_level`: Agent's current resources (0-1)
- `normalized_target_amount`: Target resource amount (0-1)

#### Factory Method
```python
@classmethod
def from_raw_values(
    cls,
    distance: float,
    angle: float,
    resource_level: float,
    target_amount: float,
    env_diagonal: float
) -> "AgentState"
```

### 2. Environment State (`EnvironmentState`)
Captures global simulation state.

#### Attributes
- `normalized_resource_density`: Resource concentration
- `normalized_agent_density`: Agent population density
- `normalized_system_ratio`: System vs Independent agents
- `normalized_resource_availability`: Resource levels
- `normalized_time`: Simulation progress

### 3. Model State (`ModelState`)
Tracks ML model parameters and performance.

#### Attributes
- `learning_rate`: Current learning rate
- `epsilon`: Exploration rate
- `latest_loss`: Most recent training loss
- `latest_reward`: Most recent reward
- `memory_size`: Experience buffer size
- `memory_capacity`: Maximum memory size
- `steps`: Training steps taken
- `architecture`: Network structure
- `training_metrics`: Performance metrics

#### Factory Method
```python
@classmethod
def from_move_module(cls, move_module: "MoveModule") -> "ModelState"
```

## Usage Examples

### Creating Agent State
```python
state = AgentState.from_raw_values(
    distance=10.0,
    angle=1.57,
    resource_level=16.0,
    target_amount=12.0,
    env_diagonal=100.0
)
```

### Getting Model State
```python
model_state = ModelState.from_move_module(move_module)
print(f"Current epsilon: {model_state.epsilon}")
print(f"Training metrics: {model_state.training_metrics}")
```

### Converting to Tensor
```python
tensor = state.to_tensor(device)
# Use tensor for neural network input
```

## Key Benefits

1. **Standardization**
   - Consistent state representation
   - Normalized values for stable learning
   - Standard interfaces across state types

2. **Type Safety**
   - Pydantic validation
   - Clear attribute definitions
   - Runtime type checking

3. **Flexibility**
   - Easy conversion between formats
   - Support for different state types
   - Extensible base class

4. **Monitoring**
   - Training metrics tracking
   - Performance monitoring
   - State serialization

5. **Documentation**
   - Comprehensive docstrings
   - Usage examples
   - Clear attribute descriptions

## Best Practices

1. Always use factory methods for creating states
2. Handle None/missing values appropriately
3. Validate state normalization
4. Use type hints consistently
5. Document state transformations

## Common Patterns

1. **State Creation**
```python
state = StateClass.from_raw_values(...)
```

2. **Neural Network Input**
```python
tensor = state.to_tensor(device)
```

3. **Logging/Serialization**
```python
state_dict = state.to_dict()
```

4. **Monitoring**
```python
print(f"Model State: {model_state}")
```
