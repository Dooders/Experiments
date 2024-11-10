import unittest
import os
import shutil
from pathlib import Path
import numpy as np

from config import SimulationConfig
from agents import Environment, SystemAgent, IndependentAgent, Resource
from database import SimulationDatabase

class TestSimulation(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path('test_data')
        self.test_dir.mkdir(exist_ok=True)
        self.config = SimulationConfig.from_yaml('tests/test_config.yaml')
        self.db_path = self.test_dir / 'test_simulation.db'
        
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_environment_initialization(self):
        """Test that environment is initialized correctly."""
        env = Environment(
            width=self.config.width,
            height=self.config.height,
            resource_distribution={"type": "random", "amount": self.config.initial_resources},
            db_path=str(self.db_path)
        )
        
        self.assertEqual(len(env.resources), self.config.initial_resources)
        self.assertEqual(len(env.agents), 0)
        self.assertEqual(env.width, self.config.width)
        self.assertEqual(env.height, self.config.height)

    def test_agent_creation(self):
        """Test that agents are created with correct attributes."""
        env = Environment(
            width=self.config.width,
            height=self.config.height,
            resource_distribution={"type": "random", "amount": self.config.initial_resources},
            db_path=str(self.db_path)
        )
        
        # Create one of each agent type
        system_agent = SystemAgent(0, (25, 25), self.config.initial_resource_level, env)
        independent_agent = IndependentAgent(1, (25, 25), self.config.initial_resource_level, env)
        
        self.assertTrue(system_agent.alive)
        self.assertTrue(independent_agent.alive)
        self.assertEqual(system_agent.resource_level, self.config.initial_resource_level)
        self.assertEqual(independent_agent.resource_level, self.config.initial_resource_level)

    def test_resource_consumption(self):
        """Test that resources are consumed correctly."""
        env = Environment(
            width=self.config.width,
            height=self.config.height,
            resource_distribution={"type": "random", "amount": 1},
            db_path=str(self.db_path)
        )
        
        # Place resource at known location
        resource = Resource(0, (25, 25), amount=10)
        env.resources = [resource]
        
        # Create agent near resource
        agent = SystemAgent(0, (25, 25), self.config.initial_resource_level, env)
        env.add_agent(agent)
        
        # Let agent gather resources
        agent.gather_resources()
        
        self.assertLess(resource.amount, 10)  # Resource should be depleted
        self.assertGreater(agent.resource_level, self.config.initial_resource_level)  # Agent should gain resources

    def test_agent_death(self):
        """Test that agents die when resources are depleted."""
        env = Environment(
            width=self.config.width,
            height=self.config.height,
            resource_distribution={"type": "random", "amount": 0},  # No resources
            db_path=str(self.db_path)
        )
        
        agent = SystemAgent(0, (25, 25), 1, env)  # Start with minimal resources
        env.add_agent(agent)
        
        # Run until agent dies
        for _ in range(20):
            if agent.alive:
                agent.act()
        
        self.assertFalse(agent.alive)

    def test_agent_reproduction(self):
        """Test that agents reproduce when conditions are met."""
        env = Environment(
            width=self.config.width,
            height=self.config.height,
            resource_distribution={"type": "random", "amount": self.config.initial_resources},
            db_path=str(self.db_path)
        )
        
        # Create agent with high resources
        agent = SystemAgent(0, (25, 25), 20, env)  # Plenty of resources to reproduce
        env.add_agent(agent)
        
        initial_agent_count = len(env.agents)
        agent.reproduce()
        
        self.assertGreater(len(env.agents), initial_agent_count)

    def test_database_logging(self):
        """Test that simulation state is correctly logged to database."""
        env = Environment(
            width=self.config.width,
            height=self.config.height,
            resource_distribution={"type": "random", "amount": self.config.initial_resources},
            db_path=str(self.db_path)
        )
        
        # Add some agents
        agent = SystemAgent(0, (25, 25), self.config.initial_resource_level, env)
        env.add_agent(agent)
        
        # Run a few steps
        for _ in range(5):
            env.update()
        
        # Check database contents
        db = SimulationDatabase(str(self.db_path))
        data = db.get_simulation_data(step_number=1)
        
        self.assertIsNotNone(data['agent_states'])
        self.assertIsNotNone(data['resource_states'])
        self.assertIsNotNone(data['metrics'])
        
        db.close()

    def test_resource_regeneration(self):
        """Test that resources regenerate correctly."""
        env = Environment(
            width=self.config.width,
            height=self.config.height,
            resource_distribution={"type": "random", "amount": 1},
            db_path=str(self.db_path)
        )
        
        # Set resource to low amount
        env.resources[0].amount = 1
        
        # Run multiple updates to allow regeneration
        initial_amount = env.resources[0].amount
        for _ in range(50):  # Give plenty of chances to regenerate
            env.update()
            
        self.assertGreater(env.resources[0].amount, initial_amount)

if __name__ == '__main__':
    unittest.main() 