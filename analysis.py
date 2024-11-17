import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from database import SimulationDatabase


class SimulationAnalyzer:
    def __init__(self, db_path: str = "simulation_results.db"):
        self.db = SimulationDatabase(db_path)

    def calculate_survival_rates(self) -> Dict[str, float]:
        """Calculate survival rates for different agent types over time."""
        query = """
            SELECT s.step_number,
                   COUNT(CASE WHEN a.agent_type = 'SystemAgent' AND st.alive = 1 THEN 1 END) as system_alive,
                   COUNT(CASE WHEN a.agent_type = 'IndependentAgent' AND st.alive = 1 THEN 1 END) as independent_alive
            FROM SimulationSteps s
            LEFT JOIN AgentStates st ON s.step_id = st.step_id
            LEFT JOIN Agents a ON st.agent_id = a.agent_id
            GROUP BY s.step_number
            ORDER BY s.step_number
        """
        self.db.cursor.execute(query)
        results = self.db.cursor.fetchall()

        return pd.DataFrame(
            results, columns=["step", "system_alive", "independent_alive"]
        )

    def analyze_resource_distribution(self) -> pd.DataFrame:
        """Analyze resource accumulation and distribution patterns."""
        query = """
            SELECT s.step_number,
                   a.agent_type,
                   AVG(st.resource_level) as avg_resources,
                   MIN(st.resource_level) as min_resources,
                   MAX(st.resource_level) as max_resources,
                   COUNT(*) as agent_count
            FROM SimulationSteps s
            JOIN AgentStates st ON s.step_id = st.step_id
            JOIN Agents a ON st.agent_id = a.agent_id
            WHERE st.alive = 1
            GROUP BY s.step_number, a.agent_type
            ORDER BY s.step_number
        """
        self.db.cursor.execute(query)
        results = self.db.cursor.fetchall()

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
        query = """
            SELECT s.step_number,
                   m.metric_value as competitive_interactions
            FROM SimulationSteps s
            JOIN Metrics m ON s.step_id = m.step_id
            WHERE m.metric_name = 'competitive_interactions'
            ORDER BY s.step_number
        """
        self.db.cursor.execute(query)
        results = self.db.cursor.fetchall()

        return pd.DataFrame(results, columns=["step", "competitive_interactions"])

    def analyze_agent_actions(self) -> pd.DataFrame:
        """Analyze individual agent actions and their outcomes."""
        query = """
            SELECT step_number,
                   agent_id,
                   action_type,
                   action_target_id,
                   position_before,
                   position_after,
                   resources_before,
                   resources_after,
                   reward
            FROM AgentActions
            ORDER BY step_number, agent_id
        """
        self.db.cursor.execute(query)
        results = self.db.cursor.fetchall()

        return pd.DataFrame(
            results,
            columns=[
                "step_number",
                "agent_id",
                "action_type",
                "action_target_id",
                "position_before",
                "position_after",
                "resources_before",
                "resources_after",
                "reward",
            ],
        )

    def analyze_combat_events(self) -> pd.DataFrame:
        """Analyze combat interactions between agents."""
        query = """
            SELECT step_number,
                   attacker_id,
                   defender_id,
                   damage_dealt,
                   defender_health_before,
                   defender_health_after,
                   defender_died
            FROM CombatEvents
            ORDER BY step_number, attacker_id
        """
        self.db.cursor.execute(query)
        results = self.db.cursor.fetchall()

        return pd.DataFrame(
            results,
            columns=[
                "step_number",
                "attacker_id",
                "defender_id",
                "damage_dealt",
                "defender_health_before",
                "defender_health_after",
                "defender_died",
            ],
        )

    def analyze_sharing_events(self) -> pd.DataFrame:
        """Analyze resource sharing events between agents."""
        query = """
            SELECT step_number,
                   giver_id,
                   receiver_id,
                   amount_shared,
                   giver_resources_before,
                   receiver_resources_before,
                   cooperation_score
            FROM SharingEvents
            ORDER BY step_number, giver_id
        """
        self.db.cursor.execute(query)
        results = self.db.cursor.fetchall()

        return pd.DataFrame(
            results,
            columns=[
                "step_number",
                "giver_id",
                "receiver_id",
                "amount_shared",
                "giver_resources_before",
                "receiver_resources_before",
                "cooperation_score",
            ],
        )

    def analyze_learning_experiences(self) -> pd.DataFrame:
        """Analyze learning experiences and rewards."""
        query = """
            SELECT step_number,
                   agent_id,
                   module_type,
                   state_before,
                   action_taken,
                   reward,
                   state_after,
                   loss
            FROM LearningExperiences
            ORDER BY step_number, agent_id
        """
        self.db.cursor.execute(query)
        results = self.db.cursor.fetchall()

        return pd.DataFrame(
            results,
            columns=[
                "step_number",
                "agent_id",
                "module_type",
                "state_before",
                "action_taken",
                "reward",
                "state_after",
                "loss",
            ],
        )

    def generate_report(self, output_file: str = "simulation_report.html"):
        """Generate an HTML report with analysis results."""
        survival_rates = self.calculate_survival_rates()
        efficiency_data = self.analyze_resource_efficiency()

        # Create plots
        plt.figure(figsize=(10, 6))
        plt.plot(efficiency_data["step"], efficiency_data["efficiency"])
        plt.title("Resource Efficiency Over Time")
        plt.savefig("efficiency_plot.png")

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
                        for k, v in survival_rates.items())}
            </table>
            
            <h2>Resource Efficiency</h2>
            <img src="efficiency_plot.png" />
            
            <h2>Summary Statistics</h2>
            <pre>{efficiency_data.describe().to_string()}</pre>
        </body>
        </html>
        """

        with open(output_file, "w") as f:
            f.write(html)
