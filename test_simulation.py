import unittest
import os
import shutil

from core.config import SimulationConfig
from agents import Environment, SystemAgent, IndependentAgent, Resource
from core.database import SimulationDatabase

class TestSimulation(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.test_dir = 'test_data'
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        self.config = SimulationConfig.from_yaml('tests/test_config.yaml')
        self.db_path = os.path.join(self.test_dir, 'test_simulation.db')
        
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    # ... rest of the test methods remain the same, just change any Path references to os.path ... 