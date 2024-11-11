"""
Script to run simulation experiments with different configurations.
"""

import logging
from pathlib import Path

from config import SimulationConfig
from experiment import ExperimentRunner

# Setup basic logging
logging.basicConfig(level=logging.INFO)


def run_resource_distribution_experiment():
    """Run experiment testing different initial resource distributions."""
    logging.info("Starting resource distribution experiment...")
    
    # Load base configuration
    base_config = SimulationConfig.from_yaml("config.yaml")
    
    # Create experiment runner
    experiment = ExperimentRunner(base_config, "resource_distribution_test")
    
    variations = [
        {"initial_resources": 100, "num_steps": 1000},
        {"initial_resources": 200, "num_steps": 1000},
        {"initial_resources": 300, "num_steps": 1000},
    ]
    
    try:
        # Run experiment with 3 iterations
        experiment.run_iterations(num_iterations=3, config_variations=variations)
        # Generate report
        experiment.generate_report()
    finally:
        # Ensure cleanup
        experiment.cleanup()  # You'll need to implement this method in ExperimentRunner
    
    logging.info("Resource distribution experiment completed")


def run_population_experiment():
    """Run experiment testing different initial population ratios."""
    logging.info("Starting population experiment...")
    
    base_config = SimulationConfig.from_yaml("config.yaml")
    experiment = ExperimentRunner(base_config, "population_ratio_test")
    
    variations = [
        {"initial_system_agents": 10, "initial_independent_agents": 40, "num_steps": 1000},
        {"initial_system_agents": 25, "initial_independent_agents": 25, "num_steps": 1000},
        {"initial_system_agents": 40, "initial_independent_agents": 10, "num_steps": 1000},
    ]
    
    try:
        experiment.run_iterations(num_iterations=3, config_variations=variations)
        experiment.generate_report()
    finally:
        # Ensure cleanup
        experiment.cleanup()
    
    logging.info("Population experiment completed")


def main():
    """Run selected experiments."""
    # Create experiments directory if it doesn't exist
    Path("experiments").mkdir(exist_ok=True)

    print("Starting experiments...")

    # Run experiments
    run_resource_distribution_experiment()
    run_population_experiment()

    print("Experiments completed! Check the experiments directory for results.")


if __name__ == "__main__":
    main()
