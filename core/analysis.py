import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from core.database import SimulationDatabase, Agent, AgentState, SimulationStep
from sqlalchemy.orm import aliased
from sqlalchemy.sql import func


class SimulationAnalyzer:
    def __init__(self, db_path: str = "simulation_results.db"):
        self.db = SimulationDatabase(db_path)

    def calculate_survival_rates(self) -> pd.DataFrame:
        """Calculate survival rates for different agent types over time."""
        session = self.db.session
        AgentAlias = aliased(Agent)
        query = (
            session.query(
                SimulationStep.step_number,
                func.count(func.case([(Agent.agent_type == 'SystemAgent', 1)])).label('system_alive'),
                func.count(func.case([(Agent.agent_type == 'IndependentAgent', 1)])).label('independent_alive')
            )
            .join(AgentState, SimulationStep.step_number == AgentState.step_number)
            .join(Agent, AgentState.agent_id == Agent.agent_id)
            .group_by(SimulationStep.step_number)
            .order_by(SimulationStep.step_number)
        )
        results = query.all()
        return pd.DataFrame(results, columns=["step", "system_alive", "independent_alive"])

    def analyze_resource_distribution(self) -> pd.DataFrame:
        """Analyze resource accumulation and distribution patterns."""
        session = self.db.session
        query = (
            session.query(
                SimulationStep.step_number,
                Agent.agent_type,
                func.avg(AgentState.resource_level).label('avg_resources'),
                func.min(AgentState.resource_level).label('min_resources'),
                func.max(AgentState.resource_level).label('max_resources'),
                func.count(AgentState.agent_id).label('agent_count')
            )
            .join(AgentState, SimulationStep.step_number == AgentState.step_number)
            .join(Agent, AgentState.agent_id == Agent.agent_id)
            .filter(AgentState.current_health > 0)
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
            ],
        )

    def analyze_competitive_interactions(self) -> pd.DataFrame:
        """Analyze patterns in competitive interactions."""
        session = self.db.session
        query = (
            session.query(
                SimulationStep.step_number,
                SimulationStep.combat_encounters.label('competitive_interactions')
            )
            .order_by(SimulationStep.step_number)
        )
        results = query.all()
        return pd.DataFrame(results, columns=["step", "competitive_interactions"])

    def generate_report(self, output_file: str = "simulation_report.html"):
        """Generate an HTML report with analysis results."""
        survival_rates = self.calculate_survival_rates()
        resource_distribution = self.analyze_resource_distribution()
        competitive_interactions = self.analyze_competitive_interactions()

        # Create plots
        plt.figure(figsize=(10, 6))
        plt.plot(resource_distribution["step"], resource_distribution["avg_resources"])
        plt.title("Average Resource Distribution Over Time")
        plt.savefig("resource_distribution_plot.png")

        plt.figure(figsize=(10, 6))
        plt.plot(competitive_interactions["step"], competitive_interactions["competitive_interactions"])
        plt.title("Competitive Interactions Over Time")
        plt.savefig("competitive_interactions_plot.png")

        # Generate HTML report
        html = f"""
        <html>
        <head><title>Simulation Analysis Report</title></head>
        <body>
            <h1>Simulation Analysis Report</h1>
            
            <h2>Survival Rates</h2>
            <table>
                <tr><th>Step</th><th>System Agents Alive</th><th>Independent Agents Alive</th></tr>
                {''.join(f"<tr><td>{row['step']}</td><td>{row['system_alive']}</td><td>{row['independent_alive']}</td></tr>" 
                        for _, row in survival_rates.iterrows())}
            </table>
            
            <h2>Resource Distribution</h2>
            <img src="resource_distribution_plot.png" />
            
            <h2>Competitive Interactions</h2>
            <img src="competitive_interactions_plot.png" />
            
            <h2>Summary Statistics</h2>
            <pre>{resource_distribution.describe().to_string()}</pre>
        </body>
        </html>
        """

        with open(output_file, "w") as f:
            f.write(html)
