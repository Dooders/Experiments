import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sqlalchemy import func, case, text

from database.database import SimulationDatabase, SimulationStep, AgentState, Agent


class SimulationAnalyzer:
    def __init__(self, db_path: str = "simulation_results.db"):
        self.db = SimulationDatabase(db_path)

    def calculate_survival_rates(self) -> pd.DataFrame:
        """Calculate survival rates for different agent types over time."""
        def _query(session):
            query = (
                session.query(
                    SimulationStep.step_number,
                    func.count(case([(Agent.agent_type == 'SystemAgent', 1)])).label('system_alive'),
                    func.count(case([(Agent.agent_type == 'IndependentAgent', 1)])).label('independent_alive')
                )
                .join(AgentState, SimulationStep.step_number == AgentState.step_number)
                .join(Agent, AgentState.agent_id == Agent.agent_id)
                .group_by(SimulationStep.step_number)
                .order_by(SimulationStep.step_number)
            )
            
            results = query.all()
            return pd.DataFrame(
                results, 
                columns=["step", "system_alive", "independent_alive"]
            )
            
        return self.db._execute_in_transaction(_query)

    def analyze_resource_distribution(self) -> pd.DataFrame:
        """Analyze resource accumulation and distribution patterns."""
        def _query(session):
            query = (
                session.query(
                    SimulationStep.step_number,
                    Agent.agent_type,
                    func.avg(AgentState.resource_level).label('avg_resources'),
                    func.min(AgentState.resource_level).label('min_resources'),
                    func.max(AgentState.resource_level).label('max_resources'),
                    func.count().label('agent_count')
                )
                .join(AgentState, SimulationStep.step_number == AgentState.step_number)
                .join(Agent, AgentState.agent_id == Agent.agent_id)
                .group_by(SimulationStep.step_number, Agent.agent_type)
                .order_by(SimulationStep.step_number)
            )
            
            results = query.all()
            return pd.DataFrame(
                results,
                columns=[
                    "step",
                    "agent_type",
                    "avg_resources",
                    "min_resources",
                    "max_resources",
                    "agent_count",
                ]
            )
            
        return self.db._execute_in_transaction(_query)

    def analyze_competitive_interactions(self) -> pd.DataFrame:
        """Analyze patterns in competitive interactions."""
        def _query(session):
            query = (
                session.query(
                    SimulationStep.step_number,
                    SimulationStep.combat_encounters.label('competitive_interactions')
                )
                .order_by(SimulationStep.step_number)
            )
            
            results = query.all()
            return pd.DataFrame(
                results, 
                columns=["step", "competitive_interactions"]
            )
            
        return self.db._execute_in_transaction(_query)

    def analyze_resource_efficiency(self) -> pd.DataFrame:
        """Analyze resource utilization efficiency over time."""
        def _query(session):
            query = (
                session.query(
                    SimulationStep.step_number,
                    SimulationStep.resource_efficiency.label('efficiency')
                )
                .order_by(SimulationStep.step_number)
            )
            
            results = query.all()
            return pd.DataFrame(
                results,
                columns=["step", "efficiency"]
            )
            
        return self.db._execute_in_transaction(_query)

    def analyze_allele_frequency_changes(self) -> pd.DataFrame:
        """Analyze allele frequency changes over generations."""
        def _query(session):
            query = (
                session.query(
                    Agent.generation,
                    Agent.allele_frequencies
                )
                .order_by(Agent.generation)
            )
            
            results = query.all()
            allele_frequency_changes = {}
            for generation, allele_frequencies in results:
                if generation not in allele_frequency_changes:
                    allele_frequency_changes[generation] = {}
                for allele, frequency in allele_frequencies.items():
                    if allele not in allele_frequency_changes[generation]:
                        allele_frequency_changes[generation][allele] = 0
                    allele_frequency_changes[generation][allele] += frequency
            
            return pd.DataFrame(allele_frequency_changes).T
            
        return self.db._execute_in_transaction(_query)

    def generate_report(self, output_file: str = "simulation_report.html"):
        """Generate an HTML report with analysis results."""
        survival_rates = self.calculate_survival_rates()
        efficiency_data = self.analyze_resource_efficiency()
        allele_frequency_changes = self.analyze_allele_frequency_changes()

        # Create plots
        plt.figure(figsize=(10, 6))
        plt.plot(efficiency_data["step"], efficiency_data["efficiency"])
        plt.title("Resource Efficiency Over Time")
        plt.savefig("efficiency_plot.png")

        plt.figure(figsize=(10, 6))
        allele_frequency_changes.plot(kind='bar', stacked=True)
        plt.title("Allele Frequency Changes Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Frequency")
        plt.savefig("allele_frequency_plot.png")

        # Generate HTML report
        html = f"""
        <html>
        <head><title>Simulation Analysis Report</title></head>
        <body>
            <h1>Simulation Analysis Report</h1>
            
            <h2>Survival Rates</h2>
            <table>
                <tr><th>Agent Type</th><th>Survival Rate</th></tr>
                {''.join(f"<tr><td>{k}</td><td>{v:.2%}</td></tr>" 
                        for k, v in survival_rates.iloc[-1].items())}
            </table>
            
            <h2>Resource Efficiency</h2>
            <img src="efficiency_plot.png" />
            
            <h2>Allele Frequency Changes</h2>
            <img src="allele_frequency_plot.png" />
            
            <h2>Summary Statistics</h2>
            <pre>{efficiency_data.describe().to_string()}</pre>
        </body>
        </html>
        """

        with open(output_file, "w") as f:
            f.write(html)

def analyze_simulation(simulation_data):
    """
    Analyze simulation data and return metrics.
    
    Args:
        simulation_data: Data from the simulation to analyze
        
    Returns:
        dict: Analysis results and metrics
    """
    # Add your analysis logic here
    results = {
        'metrics': {},
        'statistics': {},
        # Add other analysis results
    }
    return results
