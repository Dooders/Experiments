import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ExperimentTracker:
    def __init__(self, experiments_dir: str = 'experiments'):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)
        self.metadata_file = self.experiments_dir / 'metadata.json'
        self._load_metadata()
        
    def _load_metadata(self):
        """Load or create experiment metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {'experiments': {}}
            self._save_metadata()
            
    def _save_metadata(self):
        """Save experiment metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
    def register_experiment(self, name: str, config: Dict[str, Any], db_path: str) -> str:
        """Register a new experiment run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{name}_{timestamp}"
        
        self.metadata['experiments'][experiment_id] = {
            'name': name,
            'timestamp': timestamp,
            'config': config,
            'db_path': db_path
        }
        self._save_metadata()
        return experiment_id
        
    def compare_experiments(self, experiment_ids: List[str], metrics: List[str] = None) -> pd.DataFrame:
        """Compare results from multiple experiments."""
        if metrics is None:
            metrics = ['total_agents', 'total_resources', 'average_agent_resources']
            
        results = []
        for exp_id in experiment_ids:
            exp_data = self.metadata['experiments'].get(exp_id)
            if exp_data is None:
                continue
                
            # Connect to experiment database
            conn = sqlite3.connect(exp_data['db_path'])
            
            # Query metrics
            query = f"""
                SELECT s.step_number, m.metric_name, m.metric_value
                FROM SimulationMetrics m
                JOIN SimulationSteps s ON s.step_id = m.step_id
                WHERE m.metric_name IN ({','.join('?' for _ in metrics)})
                ORDER BY s.step_number
            """
            
            df = pd.read_sql_query(query, conn, params=metrics)
            df['experiment_id'] = exp_id
            df['experiment_name'] = exp_data['name']
            results.append(df)
            
            conn.close()
            
        return pd.concat(results, ignore_index=True)
        
    def generate_comparison_report(self, experiment_ids: List[str], output_file: str = None):
        """Generate a detailed comparison report for selected experiments."""
        if output_file is None:
            output_file = self.experiments_dir / 'comparison_report.html'
            
        # Get comparison data
        df = self.compare_experiments(experiment_ids)
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Plot metrics over time for each experiment
        metrics = df['metric_name'].unique()
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
        
        for i, metric in enumerate(metrics):
            metric_data = df[df['metric_name'] == metric]
            sns.lineplot(data=metric_data, x='step_number', y='metric_value', 
                        hue='experiment_name', ax=axes[i])
            axes[i].set_title(f'{metric} Over Time')
            
        plt.tight_layout()
        plot_path = self.experiments_dir / 'comparison_plots.png'
        plt.savefig(plot_path)
        plt.close()
        
        # Generate HTML report
        html = f'''
        <html>
        <head>
            <title>Experiment Comparison Report</title>
            <style>
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid black; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Experiment Comparison Report</h1>
            
            <h2>Experiments Included</h2>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Timestamp</th>
                </tr>
                {''.join(f"<tr><td>{exp_id}</td><td>{self.metadata['experiments'][exp_id]['name']}</td><td>{self.metadata['experiments'][exp_id]['timestamp']}</td></tr>"
                        for exp_id in experiment_ids)}
            </table>
            
            <h2>Configuration Comparison</h2>
            {self._generate_config_comparison_table(experiment_ids)}
            
            <h2>Results Visualization</h2>
            <img src="{plot_path}" style="max-width: 100%;" />
            
            <h2>Statistical Summary</h2>
            {self._generate_statistical_summary(df)}
        </body>
        </html>
        '''
        
        with open(output_file, 'w') as f:
            f.write(html)
            
    def _generate_config_comparison_table(self, experiment_ids: List[str]) -> str:
        """Generate HTML table comparing configurations."""
        configs = {exp_id: self.metadata['experiments'][exp_id]['config'] 
                  for exp_id in experiment_ids}
        
        # Get all unique parameters
        all_params = set()
        for config in configs.values():
            all_params.update(config.keys())
            
        # Generate table
        html = '<table><tr><th>Parameter</th>'
        for exp_id in experiment_ids:
            html += f'<th>{self.metadata["experiments"][exp_id]["name"]}</th>'
        html += '</tr>'
        
        for param in sorted(all_params):
            html += f'<tr><td>{param}</td>'
            for exp_id in experiment_ids:
                value = configs[exp_id].get(param, '')
                html += f'<td>{value}</td>'
            html += '</tr>'
            
        html += '</table>'
        return html
        
    def _generate_statistical_summary(self, df: pd.DataFrame) -> str:
        """Generate statistical summary of results."""
        summary = df.groupby(['experiment_name', 'metric_name'])['metric_value'].describe()
        return f'<pre>{summary.to_string()}</pre>' 