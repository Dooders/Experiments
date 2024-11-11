# Running Simulation Experiments

The experiment module provides functionality to run multiple simulation iterations with different parameters and analyze the results. This document explains how to use the experiment features.

## Overview

The experiment system allows you to:
- Run multiple iterations of simulations
- Test different parameter configurations
- Collect and analyze results
- Generate summary reports
- Compare outcomes across iterations

## Usage

### Basic Example

```python
from experiment import ExperimentRunner
from config import SimulationConfig

# Create base configuration
base_config = SimulationConfig(
    num_steps=1000,
    num_agents=100,
    # ... other configuration parameters
)

# Initialize experiment runner
runner = ExperimentRunner(
    base_config=base_config,
    experiment_name="my_experiment"
)

# Run multiple iterations
runner.run_iterations(num_iterations=5)

# Generate summary report
runner.generate_report()
```

### Parameter Variations

You can test different parameter configurations across iterations:

```python
# Define parameter variations
variations = [
    {"num_agents": 50},
    {"num_agents": 150},
    {"resource_rate": 2.0},
    {"system_threshold": 0.8},
]

# Run iterations with variations
runner.run_iterations(
    num_iterations=len(variations),
    config_variations=variations
)
```

## Results Analysis

The experiment runner generates two main result files:

1. `{experiment_name}_results.csv`: Detailed results for each iteration, including:
   - Final number of system and independent agents
   - Average resources per agent type
   - Timestamp
   - Configuration variation used
   - Iteration number

2. `{experiment_name}_summary.csv`: Statistical summary of results across all iterations, including:
   - Mean, standard deviation, min, max values
   - Quartile distributions
   - Count of successful iterations

## Logging

Each experiment maintains its own log file (`{experiment_name}.log`) that captures:
- Experiment progress
- Iteration status
- Error messages
- Configuration details

The log file can be found in the experiment's root directory.

## Best Practices

1. Use meaningful experiment names that reflect the purpose of the test
2. Start with a small number of iterations to validate configuration
3. Save base configurations for reproducibility
4. Document parameter variations used in experiments
5. Review logs for any warnings or errors before analyzing results

## Error Handling

The experiment runner includes built-in error handling to:
- Continue running if individual iterations fail
- Log error details for debugging
- Mark failed iterations in results

Failed iterations will be logged but won't prevent the completion of the overall experiment.
