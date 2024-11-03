import sqlite3
from datetime import datetime
import logging
from typing import Tuple, List, Dict, Any
import pandas as pd

class SimulationDatabase:
    def __init__(self, db_path: str = 'simulation_results.db'):
        """Initialize database connection and setup tables."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.setup_tables()

    def setup_tables(self):
        """Create necessary database tables if they don't exist."""
        self.cursor.executescript('''
            CREATE TABLE IF NOT EXISTS Agents (
                agent_id INTEGER PRIMARY KEY,
                birth_time INTEGER,
                death_time INTEGER,
                agent_type TEXT,
                initial_position_x REAL,
                initial_position_y REAL,
                initial_resources REAL
            );

            CREATE TABLE IF NOT EXISTS Resources (
                resource_id INTEGER PRIMARY KEY,
                initial_amount INTEGER,
                position_x REAL,
                position_y REAL
            );

            CREATE TABLE IF NOT EXISTS SimulationSteps (
                step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                step_number INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS AgentStates (
                state_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id INTEGER,
                step_id INTEGER,
                position_x REAL,
                position_y REAL,
                resource_level REAL,
                alive BOOLEAN,
                FOREIGN KEY(agent_id) REFERENCES Agents(agent_id),
                FOREIGN KEY(step_id) REFERENCES SimulationSteps(step_id)
            );

            CREATE TABLE IF NOT EXISTS ResourceStates (
                state_id INTEGER PRIMARY KEY AUTOINCREMENT,
                resource_id INTEGER,
                step_id INTEGER,
                amount INTEGER,
                FOREIGN KEY(resource_id) REFERENCES Resources(resource_id),
                FOREIGN KEY(step_id) REFERENCES SimulationSteps(step_id)
            );

            CREATE TABLE IF NOT EXISTS SimulationMetrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                step_id INTEGER,
                metric_name TEXT,
                metric_value REAL,
                FOREIGN KEY(step_id) REFERENCES SimulationSteps(step_id)
            );
        ''')
        self.conn.commit()

    def log_agent(self, agent_id: int, birth_time: int, agent_type: str, 
                  position: Tuple[float, float], initial_resources: float) -> None:
        """Log a new agent to the database."""
        self.cursor.execute('''
            INSERT INTO Agents (agent_id, birth_time, agent_type, 
                              initial_position_x, initial_position_y, initial_resources)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (agent_id, birth_time, agent_type, position[0], position[1], initial_resources))
        self.conn.commit()

    def log_resource(self, resource_id: int, initial_amount: int, 
                    position: Tuple[float, float]) -> None:
        """Log a new resource to the database."""
        self.cursor.execute('''
            INSERT INTO Resources (resource_id, initial_amount, position_x, position_y)
            VALUES (?, ?, ?, ?)
        ''', (resource_id, initial_amount, position[0], position[1]))
        self.conn.commit()

    def log_simulation_step(self, step_number: int, agents: List[Any], 
                          resources: List[Any], metrics: Dict[str, float]) -> None:
        """Log the current state of the simulation."""
        # Insert step
        self.cursor.execute('INSERT INTO SimulationSteps (step_number) VALUES (?)', 
                          (step_number,))
        step_id = self.cursor.lastrowid

        # Log agent states
        for agent in agents:
            if hasattr(agent, 'position'):  # Check if agent has required attributes
                self.cursor.execute('''
                    INSERT INTO AgentStates (agent_id, step_id, position_x, position_y, 
                                           resource_level, alive)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (agent.agent_id, step_id, agent.position[0], agent.position[1],
                     agent.resource_level, agent.alive))

        # Log resource states
        for resource in resources:
            self.cursor.execute('''
                INSERT INTO ResourceStates (resource_id, step_id, amount)
                VALUES (?, ?, ?)
            ''', (resource.resource_id, step_id, resource.amount))

        # Log metrics
        for metric_name, value in metrics.items():
            self.cursor.execute('''
                INSERT INTO SimulationMetrics (step_id, metric_name, metric_value)
                VALUES (?, ?, ?)
            ''', (step_id, metric_name, value))

        self.conn.commit()

    def update_agent_death(self, agent_id: int, death_time: int) -> None:
        """Update the death time of an agent."""
        self.cursor.execute('''
            UPDATE Agents 
            SET death_time = ? 
            WHERE agent_id = ?
        ''', (death_time, agent_id))
        self.conn.commit()

    def get_simulation_data(self, step_number: int) -> Dict[str, Any]:
        """Retrieve simulation data for a specific step."""
        # Get agent states
        self.cursor.execute('''
            SELECT a.agent_id, a.agent_type, s.position_x, s.position_y, 
                   s.resource_level, s.alive
            FROM AgentStates s
            JOIN Agents a ON a.agent_id = s.agent_id
            JOIN SimulationSteps st ON st.step_id = s.step_id
            WHERE st.step_number = ?
        ''', (step_number,))
        agent_states = self.cursor.fetchall()

        # Get resource states
        self.cursor.execute('''
            SELECT r.resource_id, s.amount, r.position_x, r.position_y
            FROM ResourceStates s
            JOIN Resources r ON r.resource_id = s.resource_id
            JOIN SimulationSteps st ON st.step_id = s.step_id
            WHERE st.step_number = ?
        ''', (step_number,))
        resource_states = self.cursor.fetchall()

        # Get metrics
        self.cursor.execute('''
            SELECT metric_name, metric_value
            FROM SimulationMetrics m
            JOIN SimulationSteps s ON s.step_id = m.step_id
            WHERE s.step_number = ?
        ''', (step_number,))
        metrics = dict(self.cursor.fetchall())

        return {
            'agent_states': agent_states,
            'resource_states': resource_states,
            'metrics': metrics
        }

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def get_historical_data(self, start_step: int = 0, end_step: int = None) -> Dict[str, List]:
        """Retrieve historical simulation data between start and end steps."""
        query = '''
            SELECT s.step_number, m.metric_name, m.metric_value
            FROM SimulationMetrics m
            JOIN SimulationSteps s ON s.step_id = m.step_id
            WHERE s.step_number >= ?
            {}
            ORDER BY s.step_number
        '''.format('AND s.step_number <= ?' if end_step is not None else '')
        
        params = [start_step]
        if end_step is not None:
            params.append(end_step)
            
        self.cursor.execute(query, params)
        results = self.cursor.fetchall()
        
        # Organize data by metric
        history = {}
        steps = []
        for step, metric_name, value in results:
            if step not in steps:
                steps.append(step)
            if metric_name not in history:
                history[metric_name] = []
            history[metric_name].append(value)
        
        return {'steps': steps, 'metrics': history} 

    def export_data(self, output_file: str = 'simulation_data.csv') -> None:
        """Export simulation data to a CSV file."""
        query = '''
            SELECT s.step_number, m.metric_name, m.metric_value
            FROM SimulationMetrics m
            JOIN SimulationSteps s ON s.step_id = m.step_id
            ORDER BY s.step_number, m.metric_name
        '''
        
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        # Convert to DataFrame and pivot
        df = pd.DataFrame(results, columns=['step', 'metric', 'value'])
        df_pivot = df.pivot(index='step', columns='metric', values='value')
        
        # Export to CSV
        df_pivot.to_csv(output_file)
        return df_pivot