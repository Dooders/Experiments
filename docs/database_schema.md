# Simulation Database Schema

This document describes the database schema used to store simulation data. The database uses SQLite and consists of four main tables tracking different aspects of the simulation.

## Tables Overview

### Agents Table
Stores metadata and lifecycle information for each agent.

| Column | Type | Description |
|--------|------|-------------|
| agent_id | INTEGER PRIMARY KEY | Unique identifier for each agent |
| birth_time | INTEGER | Simulation step when agent was created |
| death_time | INTEGER | Simulation step when agent died (NULL if alive) |
| agent_type | TEXT | Type of agent (system/independent/control) |
| initial_position | TEXT | Starting (x,y) coordinates as string |
| initial_resources | REAL | Starting resource amount |
| max_health | REAL | Maximum health points |
| starvation_threshold | INTEGER | Steps agent can survive without resources |
| genome_id | TEXT | Unique identifier for agent's genome |
| parent_id | INTEGER | ID of parent agent (NULL for initial agents) |
| generation | INTEGER | Generation number in evolutionary lineage |

### AgentStates Table
Tracks the state of each agent at each simulation step.

| Column | Type | Description |
|--------|------|-------------|
| step_number | INTEGER | Simulation step number |
| agent_id | INTEGER | Reference to Agents table |
| position_x | REAL | X coordinate |
| position_y | REAL | Y coordinate |
| resource_level | REAL | Current resource amount |
| current_health | REAL | Current health points |
| max_health | REAL | Maximum health points |
| starvation_threshold | INTEGER | Current starvation counter |
| is_defending | BOOLEAN | Whether agent is in defensive stance |
| total_reward | REAL | Accumulated reward |
| age | INTEGER | Agent's current age in steps |

### ResourceStates Table
Tracks the state of resources at each simulation step.

| Column | Type | Description |
|--------|------|-------------|
| step_number | INTEGER | Simulation step number |
| resource_id | INTEGER | Unique resource identifier |
| amount | REAL | Current resource amount |
| position_x | REAL | X coordinate |
| position_y | REAL | Y coordinate |

### SimulationSteps Table
Stores aggregate metrics for each simulation step.

| Column | Type | Description |
|--------|------|-------------|
| step_number | INTEGER PRIMARY KEY | Simulation step number |
| total_agents | INTEGER | Total number of alive agents |
| system_agents | INTEGER | Number of system agents |
| independent_agents | INTEGER | Number of independent agents |
| control_agents | INTEGER | Number of control agents |
| total_resources | REAL | Total resources in environment |
| average_agent_resources | REAL | Mean resources per agent |
| births | INTEGER | New agents this step |
| deaths | INTEGER | Agent deaths this step |
| current_max_generation | INTEGER | Highest generation number |
| resource_efficiency | REAL | Resource utilization (0-1) |
| resource_distribution_entropy | REAL | Measure of resource distribution evenness |
| average_agent_health | REAL | Mean health across agents |
| average_agent_age | INTEGER | Mean age of agents |
| average_reward | REAL | Mean reward accumulated |
| combat_encounters | INTEGER | Number of combat interactions |
| successful_attacks | INTEGER | Number of successful attacks |
| resources_shared | REAL | Amount of resources shared |
| genetic_diversity | REAL | Measure of genome variety (0-1) |
| dominant_genome_ratio | REAL | Prevalence of most common genome (0-1) |

## Relationships

- `AgentStates.agent_id` → `Agents.agent_id`: Links agent states to their agent records
- `Agents.parent_id` → `Agents.agent_id`: Tracks parent-child relationships for reproduction

## Key Metrics Explained

### Population Dynamics
- **births/deaths**: Track population turnover
- **current_max_generation**: Indicates evolutionary progression
- **agent type counts**: Monitor population composition

### Resource Metrics
- **resource_efficiency**: How effectively resources are being utilized
- **resource_distribution_entropy**: Higher values indicate more even distribution

### Performance Metrics
- **average_agent_health/age**: Indicators of population health
- **average_reward**: Measure of agent success

### Combat & Cooperation
- **combat_encounters/successful_attacks**: Measure conflict levels
- **resources_shared**: Indicates cooperation levels

### Evolutionary Metrics
- **genetic_diversity**: Higher values indicate more varied population
- **dominant_genome_ratio**: Lower values suggest more balanced evolution

## Usage Examples 


### Basic Database Operations
```python
# Initialize database
db = SimulationDatabase("simulation.db")

# Log a new agent
db.log_agent(
    agent_id=1,
    birth_time=0,
    agent_type="system",
    position=(0.5, 0.5),
    initial_resources=100.0,
    max_health=100.0,
    starvation_threshold=10
)

# Log agent death
db.update_agent_death(agent_id=1, death_time=150)

# Log a resource
db.log_resource(
    resource_id=1,
    initial_amount=1000.0,
    position=(0.3, 0.7)
)

# Close database connection
db.close()
```

### Logging Simulation Steps
```python
# Prepare metrics dictionary
metrics = {
    "total_agents": 10,
    "system_agents": 5,
    "independent_agents": 3,
    "control_agents": 2,
    "total_resources": 1000.0,
    "average_agent_resources": 100.0,
    "births": 1,
    "deaths": 0,
    "current_max_generation": 3,
    "resource_efficiency": 0.8,
    "resource_distribution_entropy": 0.7,
    "average_agent_health": 90.0,
    "average_agent_age": 25,
    "average_reward": 150.0,
    "combat_encounters": 5,
    "successful_attacks": 2,
    "resources_shared": 50.0,
    "genetic_diversity": 0.85,
    "dominant_genome_ratio": 0.3
}

# Log complete simulation step
db.log_simulation_step(
    step_number=1,
    agents=agent_list,
    resources=resource_list,
    metrics=metrics,
    environment=env
)
```

### Data Retrieval and Analysis
```python
# Get data for specific step
step_data = db.get_simulation_data(step_number=10)

# Get historical trends
history = db.get_historical_data()

# Export data to CSV
db.export_data("simulation_results.csv")

# Get agent lifespan statistics
lifespan_stats = db.get_agent_lifespan_statistics()
```

## Notes

- Always ensure proper database closure using `db.close()` to prevent resource leaks
- Use `flush_all_buffers()` when immediate data persistence is required
- The database uses SQLite which supports single-writer and multiple-reader access
- Consider using context managers for safer database handling
- Large simulations may benefit from periodic commits and buffer flushes

## Performance Considerations

- Batch operations using `log_step()` instead of individual inserts
- Use transactions for multiple related operations
- Consider indexing heavily queried columns
- Monitor database size and perform maintenance as needed

## Error Handling

The database implements basic error handling for:
- Connection errors
- Write failures
- Buffer flush issues
- Invalid data types
- Constraint violations

Always wrap database operations in try-except blocks in production code.
