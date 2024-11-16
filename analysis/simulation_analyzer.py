import pandas as pd


class SimulationAnalyzer:
    def calculate_survival_rates(self) -> pd.DataFrame:
        """Calculate survival rates for each agent type over time."""
        query = """
        SELECT 
            step,
            SUM(CASE WHEN agent_type = 'SystemAgent' AND death_time IS NULL THEN 1 ELSE 0 END) as system_alive,
            SUM(CASE WHEN agent_type = 'IndependentAgent' AND death_time IS NULL THEN 1 ELSE 0 END) as independent_alive,
            SUM(CASE WHEN agent_type = 'ControlAgent' AND death_time IS NULL THEN 1 ELSE 0 END) as control_alive
        FROM agents
        GROUP BY step
        ORDER BY step
        """
        return pd.read_sql_query(query, self.conn)

    def analyze_resource_distribution(self) -> pd.DataFrame:
        """Analyze resource distribution across agent types."""
        query = """
        SELECT 
            step,
            agent_type,
            AVG(resource_level) as avg_resources,
            MIN(resource_level) as min_resources,
            MAX(resource_level) as max_resources
        FROM agents
        WHERE agent_type IN ('SystemAgent', 'IndependentAgent', 'ControlAgent')
        GROUP BY step, agent_type
        ORDER BY step, agent_type
        """
        return pd.read_sql_query(query, self.conn)

    def get_control_agent_stats(self) -> str:
        """Analyze Control Agent performance metrics."""
        query = """
        SELECT 
            AVG(CASE WHEN death_time IS NULL THEN 1 ELSE 0 END) as survival_rate,
            AVG(resource_level) as avg_resources,
            COUNT(*) as total_agents,
            AVG(JULIANDAY(death_time) - JULIANDAY(birth_time)) as avg_lifespan
        FROM Agents 
        WHERE agent_type = 'ControlAgent'
        """
        result = pd.read_sql_query(query, self.conn)

        return f"""
        Survival Rate: {result['survival_rate'][0]:.2%}
        Average Resources: {result['avg_resources'][0]:.2f}
        Total Agents: {result['total_agents'][0]}
        Average Lifespan: {result['avg_lifespan'][0]:.2f} steps
        """

    def analyze_population_balance(self) -> str:
        """Analyze the effectiveness of population balance."""
        query = """
        WITH AgentCounts AS (
            SELECT 
                step_number,
                system_agents::FLOAT / NULLIF(total_agents, 0) as system_ratio,
                independent_agents::FLOAT / NULLIF(total_agents, 0) as independent_ratio,
                control_agents::FLOAT / NULLIF(total_agents, 0) as control_ratio,
                total_resources,
                average_agent_resources
            FROM SimulationSteps
            WHERE total_agents > 0
        )
        SELECT 
            AVG(ABS(system_ratio - 0.33)) as system_deviation,
            AVG(ABS(independent_ratio - 0.33)) as independent_deviation,
            AVG(ABS(control_ratio - 0.34)) as control_deviation,
            AVG(total_resources) as avg_resources,
            AVG(average_agent_resources) as avg_agent_resources
        FROM AgentCounts
        """
        result = pd.read_sql_query(query, self.conn)

        return f"""
        Population Balance Analysis:
        System Agent Deviation: {result['system_deviation'][0]:.2%}
        Independent Agent Deviation: {result['independent_deviation'][0]:.2%}
        Control Agent Deviation: {result['control_deviation'][0]:.2%}
        Average System Resources: {result['avg_resources'][0]:.2f}
        Average Agent Resources: {result['avg_agent_resources'][0]:.2f}
        """
