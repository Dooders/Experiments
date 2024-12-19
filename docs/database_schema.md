# Simulation Database Schema

This document describes the database schema used to store simulation data. The database uses SQLite with SQLAlchemy ORM and consists of eight main tables tracking different aspects of the simulation.

## Tables Overview

### Agents Table

Stores metadata and lifecycle information for each agent.

| Column               | Type                | Description                                     |
| -------------------- | ------------------- | ----------------------------------------------- |
| agent_id             | INTEGER PRIMARY KEY | Unique identifier for each agent                |
| birth_time           | INTEGER             | Simulation step when agent was created          |
| death_time           | INTEGER             | Simulation step when agent died (NULL if alive) |
| agent_type           | STRING(50)          | Type of agent (system/independent/control)      |
| position_x           | FLOAT               | Initial X coordinate                            |
| position_y           | FLOAT               | Initial Y coordinate                            |
| initial_resources    | FLOAT               | Starting resource amount                        |
| starting_health      | FLOAT               | Starting health points                           |
| starvation_threshold | INTEGER             | Steps agent can survive without resources       |
| genome_id            | STRING(64)          | Unique identifier for agent's genome            |
| generation           | INTEGER             | Generation number in evolutionary lineage       |

### AgentStates Table

Tracks the state of each agent at each simulation step.

| Column               | Type                | Description                          |
| -------------------- | ------------------- | ------------------------------------ |
| id                   | INTEGER PRIMARY KEY | Unique identifier for state record   |
| step_number          | INTEGER             | Simulation step number               |
| agent_id             | INTEGER             | Reference to Agents table            |
| position_x           | FLOAT               | X coordinate                         |
| position_y           | FLOAT               | Y coordinate                         |
| resource_level       | FLOAT               | Current resource amount              |
| current_health       | FLOAT               | Current health points                |
| starting_health      | FLOAT               | Starting health points                |
| starvation_threshold | INTEGER             | Current starvation counter           |
| is_defending         | BOOLEAN             | Whether agent is in defensive stance |
| total_reward         | FLOAT               | Accumulated reward                   |
| age                  | INTEGER             | Agent's current age in steps         |

### ResourceStates Table

Tracks the state of resources at each simulation step.

| Column      | Type                | Description                        |
| ----------- | ------------------- | ---------------------------------- |
| id          | INTEGER PRIMARY KEY | Unique identifier for state record |
| step_number | INTEGER             | Simulation step number             |
| resource_id | INTEGER             | Unique resource identifier         |
| amount      | FLOAT               | Current resource amount            |
| position_x  | FLOAT               | X coordinate                       |
| position_y  | FLOAT               | Y coordinate                       |

### SimulationSteps Table

Stores aggregate metrics for each simulation step.

| Column                        | Type                | Description                               |
| ----------------------------- | ------------------- | ----------------------------------------- |
| step_number                   | INTEGER PRIMARY KEY | Simulation step number                    |
| total_agents                  | INTEGER             | Total number of alive agents              |
| system_agents                 | INTEGER             | Number of system agents                   |
| independent_agents            | INTEGER             | Number of independent agents              |
| control_agents                | INTEGER             | Number of control agents                  |
| total_resources               | FLOAT               | Total resources in environment            |
| average_agent_resources       | FLOAT               | Mean resources per agent                  |
| births                        | INTEGER             | New agents this step                      |
| deaths                        | INTEGER             | Agent deaths this step                    |
| current_max_generation        | INTEGER             | Highest generation number                 |
| resource_efficiency           | FLOAT               | Resource utilization (0-1)                |
| resource_distribution_entropy | FLOAT               | Measure of resource distribution evenness |
| average_agent_health          | FLOAT               | Mean health across agents                 |
| average_agent_age             | INTEGER             | Mean age of agents                        |
| average_reward                | FLOAT               | Mean reward accumulated                   |
| combat_encounters             | INTEGER             | Number of combat interactions             |
| successful_attacks            | INTEGER             | Number of successful attacks              |
| resources_shared              | FLOAT               | Amount of resources shared                |
| genetic_diversity             | FLOAT               | Measure of genome variety (0-1)           |
| dominant_genome_ratio         | FLOAT               | Prevalence of most common genome (0-1)    |

### AgentActions Table

Records actions taken by agents during simulation.

| Column           | Type                | Description                  |
| ---------------- | ------------------- | ---------------------------- |
| action_id        | INTEGER PRIMARY KEY | Unique identifier for action |
| step_number      | INTEGER             | When action occurred         |
| agent_id         | INTEGER             | Agent that took action       |
| action_type      | STRING(20)          | Type of action taken         |
| action_target_id | INTEGER             | Target of action (if any)    |
| position_before  | STRING(32)          | Position before action       |
| position_after   | STRING(32)          | Position after action        |
| resources_before | FLOAT               | Resources before action      |
| resources_after  | FLOAT               | Resources after action       |
| reward           | FLOAT               | Reward received for action   |
| details          | STRING(1024)        | Additional action details    |

### HealthIncidents Table

Tracks changes in agent health status.

| Column        | Type                | Description                    |
| ------------- | ------------------- | ------------------------------ |
| incident_id   | INTEGER PRIMARY KEY | Unique identifier for incident |
| step_number   | INTEGER             | When incident occurred         |
| agent_id      | INTEGER             | Affected agent                 |
| health_before | FLOAT               | Health before incident         |
| health_after  | FLOAT               | Health after incident          |
| cause         | STRING(50)          | Cause of health change         |
| details       | STRING(512)         | Additional incident details    |

### SimulationConfig Table

Stores simulation configuration data.

| Column      | Type                | Description                  |
| ----------- | ------------------- | ---------------------------- |
| config_id   | INTEGER PRIMARY KEY | Unique identifier for config |
| timestamp   | INTEGER             | When config was created      |
| config_data | STRING(4096)        | JSON configuration data      |

### Simulations Table

Stores metadata about simulation runs.

| Column             | Type                | Description                      |
| ------------------ | ------------------- | -------------------------------- |
| simulation_id      | INTEGER PRIMARY KEY | Unique identifier for simulation |
| start_time         | DATETIME            | When simulation started          |
| end_time           | DATETIME            | When simulation ended            |
| status             | STRING(50)          | Current simulation status        |
| parameters         | JSON                | Simulation parameters            |
| results_summary    | JSON                | Summary of results               |
| simulation_db_path | STRING(255)         | Path to simulation database      |

## Relationships

- `AgentStates.agent_id` → `Agents.agent_id`: Links agent states to their agent records
- `AgentActions.agent_id` → `Agents.agent_id`: Links actions to agents
- `HealthIncidents.agent_id` → `Agents.agent_id`: Links health incidents to agents

## Indexes

The schema includes optimized indexes for common queries:

- Agent lookups by type and lifecycle events
- State queries by step number and agent
- Resource state queries by step and resource ID
- Action queries by step, agent, and type
- Health incident queries by step and agent

## Notes

- Uses SQLite as the backend database
- Implements foreign key constraints
- Includes indexes for performance optimization
- Supports concurrent access through session management
- Uses transaction safety with automatic rollback
