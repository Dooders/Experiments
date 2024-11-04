import itertools
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from config import SimulationConfig
from simulation import run_simulation, setup_logging


class BatchRunner:
    """
    Runs multiple simulations with varying parameters.

    This class handles:
    - Parameter variation management
    - Batch execution of simulations
    - Results collection and analysis
    """

    def __init__(self, base_config: SimulationConfig):
        """
        Initialize batch runner with base configuration.

        Parameters
        ----------
        base_config : SimulationConfig
            Base configuration to use for simulations
        """
        self.base_config = base_config
        self.parameter_variations = {}
        self.results = []

    def add_parameter_variation(self, parameter: str, values: List[Any]) -> None:
        """
        Add parameter variations to test.

        Parameters
        ----------
        parameter : str
            Name of parameter to vary
        values : List[Any]
            List of values to test for this parameter
        """
        self.parameter_variations[parameter] = values

    def run(self, experiment_name: str, num_steps: int = 1000) -> None:
        """
        Run batch of simulations with all parameter combinations.

        Parameters
        ----------
        experiment_name : str
            Name for this batch experiment
        num_steps : int
            Number of steps per simulation
        """
        # Setup logging for batch run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = f"batch_results/{experiment_name}_{timestamp}"
        os.makedirs(batch_dir, exist_ok=True)
        setup_logging(f"{batch_dir}/batch.log")

        # Generate all parameter combinations
        param_names = list(self.parameter_variations.keys())
        param_values = list(self.parameter_variations.values())
        combinations = list(itertools.product(*param_values))

        logging.info(f"Starting batch run with {len(combinations)} combinations")

        # Run simulation for each combination
        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))
            logging.info(f"Running combination {i+1}/{len(combinations)}: {params}")

            # Create configuration for this run
            config = self._create_config_variation(params)

            # Run simulation
            db_path = f"{batch_dir}/sim_{i}.db"
            try:
                environment = run_simulation(num_steps, config, db_path)
                self._collect_results(params, environment)
            except Exception as e:
                logging.error(f"Simulation failed for params {params}: {str(e)}")

        self._save_results(batch_dir)

    def _create_config_variation(self, params: Dict[str, Any]) -> SimulationConfig:
        """
        Create new configuration with specified parameter values.

        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of parameter values to use

        Returns
        -------
        SimulationConfig
            New configuration object with updated parameters
        """
        config = SimulationConfig.from_yaml(self.base_config.config_file)
        for param, value in params.items():
            setattr(config, param, value)
        return config

    def _collect_results(self, params: Dict[str, Any], environment: Any) -> None:
        """
        Collect results from completed simulation.

        Parameters
        ----------
        params : Dict[str, Any]
            Parameters used for this simulation
        environment : Environment
            Completed simulation environment
        """
        result = {
            **params,
            "final_agents": len([a for a in environment.agents if a.alive]),
            "total_resources": sum(r.amount for r in environment.resources),
            "average_resources_per_agent": (
                sum(a.resource_level for a in environment.agents if a.alive)
                / len([a for a in environment.agents if a.alive])
                if any(a.alive for a in environment.agents)
                else 0
            ),
        }
        self.results.append(result)

    def _save_results(self, batch_dir: str) -> None:
        """
        Save batch results to CSV file.

        Parameters
        ----------
        batch_dir : str
            Directory to save results in
        """
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(f"{batch_dir}/results.csv", index=False)
        logging.info(f"Results saved to {batch_dir}/results.csv")
