import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from database import SimulationDatabase

class SimulationAnalyzer:
    def __init__(self, db_path: str = 'simulation_results.db'):
        self.db = SimulationDatabase(db_path)
        
    def calculate_survival_rates(self) -> Dict[str, float]:
        """Calculate survival rates for different agent types."""
        query = '''
            SELECT agent_type, 
                   COUNT(*) as total,
                   SUM(CASE WHEN death_time IS NULL THEN 1 ELSE 0 END) as survived
            FROM Agents
            GROUP BY agent_type
        '''
        self.db.cursor.execute(query)
        results = self.db.cursor.fetchall()
        
        return {
            agent_type: survived/total 
            for agent_type, total, survived in results
        }
        
    def analyze_resource_efficiency(self) -> pd.DataFrame:
        """Analyze resource usage efficiency over time."""
        history = self.db.get_historical_data()
        df = pd.DataFrame({
            'step': history['steps'],
            'resources': history['metrics']['total_resources'],
            'total_agents': history['metrics']['total_agents']
        })
        
        df['efficiency'] = df['total_agents'] / df['resources'].replace(0, np.nan)
        return df
        
    def generate_report(self, output_file: str = 'simulation_report.html'):
        """Generate an HTML report with analysis results."""
        survival_rates = self.calculate_survival_rates()
        efficiency_data = self.analyze_resource_efficiency()
        
        # Create plots
        plt.figure(figsize=(10, 6))
        plt.plot(efficiency_data['step'], efficiency_data['efficiency'])
        plt.title('Resource Efficiency Over Time')
        plt.savefig('efficiency_plot.png')
        
        # Generate HTML report
        html = f'''
        <html>
        <head><title>Simulation Analysis Report</title></head>
        <body>
            <h1>Simulation Analysis Report</h1>
            
            <h2>Survival Rates</h2>
            <table>
                <tr><th>Agent Type</th><th>Survival Rate</th></tr>
                {''.join(f"<tr><td>{k}</td><td>{v:.2%}</td></tr>" 
                        for k, v in survival_rates.items())}
            </table>
            
            <h2>Resource Efficiency</h2>
            <img src="efficiency_plot.png" />
            
            <h2>Summary Statistics</h2>
            <pre>{efficiency_data.describe().to_string()}</pre>
        </body>
        </html>
        '''
        
        with open(output_file, 'w') as f:
            f.write(html) 