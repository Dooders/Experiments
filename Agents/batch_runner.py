import itertools
from typing import Any, Dict, List
import concurrent.futures
from pathlib import Path
import logging
from datetime import datetime

from config import SimulationConfig
from agents import main as run_simulation

class BatchRunner:
    def __init__(self, base_config: SimulationConfig):
        self.base_config = base_config
        self.parameter_variations = {}
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
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
        
    def run(self, num_steps: int = 500, max_workers: int = None):
        """Run all parameter combinations in parallel."""
        configs = self._generate_configs()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, config in enumerate(configs):
                db_path = self.results_dir / f'simulation_{timestamp}_{i}.db'
                futures.append(
                    executor.submit(run_simulation, num_steps, config, str(db_path))
                )
            
            # Wait for all simulations to complete
            concurrent.futures.wait(futures)
            
        # Analyze results
        self._analyze_batch_results(timestamp)
        
    def _analyze_batch_results(self, timestamp: str):
        """Analyze and compare results from all simulations in the batch."""
        from analysis import SimulationAnalyzer
        
        # Create a combined report
        report_path = self.results_dir / f'batch_report_{timestamp}.html'
        with open(report_path, 'w') as f:
            f.write('<html><head><title>Batch Simulation Results</title></head><body>')
            f.write('<h1>Batch Simulation Results</h1>')
            
            # Analyze each simulation
            for db_file in self.results_dir.glob(f'simulation_{timestamp}_*.db'):
                analyzer = SimulationAnalyzer(str(db_file))
                
                f.write(f'<h2>Simulation: {db_file.name}</h2>')
                survival_rates = analyzer.calculate_survival_rates()
                efficiency_data = analyzer.analyze_resource_efficiency()
                
                # Add results to report
                f.write('<h3>Survival Rates</h3>')
                f.write('<table border="1">')
                f.write('<tr><th>Agent Type</th><th>Survival Rate</th></tr>')
                for agent_type, rate in survival_rates.items():
                    f.write(f'<tr><td>{agent_type}</td><td>{rate:.2%}</td></tr>')
                f.write('</table>')
                
                # Add efficiency statistics
                f.write('<h3>Resource Efficiency Statistics</h3>')
                f.write(f'<pre>{efficiency_data.describe().to_string()}</pre>')
                
            f.write('</body></html>')

if __name__ == '__main__':
    # Example usage
    base_config = SimulationConfig.from_yaml('config.yaml')
    
    runner = BatchRunner(base_config)
    runner.add_parameter_variation('system_agents', [20, 30, 40])
    runner.add_parameter_variation('individual_agents', [20, 30, 40])
    
    runner.run(num_steps=500) 