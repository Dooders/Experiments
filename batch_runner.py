import itertools
from typing import Any, Dict, List
import concurrent.futures
from pathlib import Path
import logging
from datetime import datetime

from config import SimulationConfig
from agents import main as run_simulation
from experiment_tracker import ExperimentTracker

class BatchRunner:
    def __init__(self, base_config: SimulationConfig):
        self.base_config = base_config
        self.parameter_variations = {}
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        self.experiment_tracker = ExperimentTracker()
        
    def add_parameter_variation(self, param_name: str, values: List[Any]):
        """Add a parameter to vary in the batch experiments."""
        self.parameter_variations[param_name] = values
        
    def _generate_configs(self) -> List[SimulationConfig]:
        """Generate all combinations of parameter variations."""
        param_names = list(self.parameter_variations.keys())
        param_values = list(self.parameter_variations.values())
        
        configs = []
        for values in itertools.product(*param_values):
            config_dict = self.base_config.to_dict()
            for name, value in zip(param_names, values):
                config_dict[name] = value
            configs.append(SimulationConfig(**config_dict))
            
        return configs
        
    def run(self, experiment_name: str, num_steps: int = 500, max_workers: int = None):
        """Run all parameter combinations in parallel."""
        configs = self._generate_configs()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        experiment_ids = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, config in enumerate(configs):
                db_path = self.results_dir / f'simulation_{timestamp}_{i}.db'
                
                # Register experiment
                exp_id = self.experiment_tracker.register_experiment(
                    name=f"{experiment_name}_{i}",
                    config=config.to_dict(),
                    db_path=str(db_path)
                )
                experiment_ids.append(exp_id)
                
                futures.append(
                    executor.submit(run_simulation, num_steps, config, str(db_path))
                )
            
            # Wait for all simulations to complete
            concurrent.futures.wait(futures)
            
        # Generate comparison report
        self.experiment_tracker.generate_comparison_report(
            experiment_ids,
            output_file=self.results_dir / f'comparison_report_{timestamp}.html'
        )

if __name__ == '__main__':
    # Example usage
    base_config = SimulationConfig.from_yaml('config.yaml')
    
    runner = BatchRunner(base_config)
    runner.add_parameter_variation('system_agents', [20, 30, 40])
    runner.add_parameter_variation('individual_agents', [20, 30, 40])
    
    runner.run(experiment_name='test_experiment', num_steps=500) 