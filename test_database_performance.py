import unittest
import time
import random
import os
from typing import List, Dict, Tuple
from core.database import SimulationDatabase

class TestDatabasePerformance(unittest.TestCase):
    """Performance test suite for SimulationDatabase."""
    
    def setUp(self):
        """Set up test database and test data."""
        self.test_db_path = f"test_performance_{time.time()}.db"  # Unique DB file for each test
        self.db = SimulationDatabase(self.test_db_path)
        self.num_agents = 1000
        self.num_steps = 100
        self.used_agent_ids = set()  # Track used agent IDs
        
    def tearDown(self):
        """Clean up test database."""
        try:
            self.db.close()
            if os.path.exists(self.test_db_path):
                os.remove(self.test_db_path)
        except Exception as e:
            print(f"Warning: Cleanup failed - {e}")
            
    def _generate_unique_agent_id(self) -> int:
        """Generate a unique agent ID."""
        while True:
            agent_id = random.randint(1, 100000)
            if agent_id not in self.used_agent_ids:
                self.used_agent_ids.add(agent_id)
                return agent_id
            
    def _generate_random_agent_data(self) -> Dict:
        """Generate random agent data for testing."""
        return {
            "agent_id": self._generate_unique_agent_id(),
            "birth_time": random.randint(0, 100),
            "agent_type": random.choice(["SystemAgent", "IndependentAgent", "ControlAgent"]),
            "position": (random.uniform(0, 100), random.uniform(0, 100)),
            "initial_resources": random.uniform(0, 100),
            "max_health": random.uniform(50, 100),
            "starvation_threshold": random.randint(10, 50),
            "genome_id": f"genome_{random.randint(1, 1000)}",
            "generation": random.randint(0, 10)
        }
        
    def _generate_random_state_data(self) -> Dict:
        """Generate random state data for testing."""
        return {
            "current_health": random.uniform(0, 100),
            "resource_level": random.uniform(0, 100),
            "position": (random.uniform(0, 100), random.uniform(0, 100)),
            "is_defending": random.choice([True, False]),
            "total_reward": random.uniform(-100, 100),
            "starvation_threshold": random.randint(10, 50)
        }
        
    def test_batch_agent_creation_performance(self):
        """Test performance of creating multiple agents in batch."""
        start_time = time.time()
        
        # Generate agent data
        agent_data_list = [self._generate_random_agent_data() for _ in range(self.num_agents)]
        
        # Use batch insert
        self.db.log_agents_batch(agent_data_list)
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert reasonable performance (adjust threshold as needed)
        self.assertLess(execution_time, 5.0, 
                       f"Batch agent creation took too long: {execution_time:.2f} seconds")
        
    def test_state_update_performance(self):
        """Test performance of updating agent states."""
        # Create test agent
        agent_data = self._generate_random_agent_data()
        self.db.log_agent(
            agent_data["agent_id"],
            agent_data["birth_time"],
            agent_data["agent_type"],
            agent_data["position"],
            agent_data["initial_resources"],
            agent_data["max_health"],
            agent_data["starvation_threshold"]
        )
        
        start_time = time.time()
        
        # Update state multiple times
        for step in range(self.num_steps):
            state_data = self._generate_random_state_data()
            self.db.update_agent_state(agent_data["agent_id"], step, state_data)
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert reasonable performance
        self.assertLess(execution_time, 2.0,
                       f"State updates took too long: {execution_time:.2f} seconds")
        
    def test_query_performance(self):
        """Test performance of various query operations."""
        # Create test agent
        agent_data = self._generate_random_agent_data()
        self.db.log_agent(
            agent_data["agent_id"],
            agent_data["birth_time"],
            agent_data["agent_type"],
            agent_data["position"],
            agent_data["initial_resources"],
            agent_data["max_health"],
            agent_data["starvation_threshold"]
        )
        
        # Add some state data
        for step in range(self.num_steps):
            state_data = self._generate_random_state_data()
            self.db.update_agent_state(agent_data["agent_id"], step, state_data)
        
        # Test various query operations
        start_time = time.time()
        
        # Test get_agent_data
        agent_info = self.db.get_agent_data(agent_data["agent_id"])
        self.assertIsNotNone(agent_info)
        
        # Test get_agent_types
        agent_types = self.db.get_agent_types()
        self.assertIsInstance(agent_types, list)
        
        # Test get_agent_behaviors
        behaviors = self.db.get_agent_behaviors()
        self.assertIsInstance(behaviors, list)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert reasonable query performance
        self.assertLess(execution_time, 1.0,
                       f"Query operations took too long: {execution_time:.2f} seconds")
        
    def test_concurrent_access_performance(self):
        """Test database performance under concurrent access."""
        import threading
        
        def worker(worker_id: int):
            """Worker function for concurrent testing."""
            for _ in range(10):
                agent_data = self._generate_random_agent_data()
                self.db.log_agent(
                    agent_data["agent_id"],
                    agent_data["birth_time"],
                    agent_data["agent_type"],
                    agent_data["position"],
                    agent_data["initial_resources"],
                    agent_data["max_health"],
                    agent_data["starvation_threshold"]
                )
                
        start_time = time.time()
        
        # Create and start multiple threads
        threads = []
        num_threads = 5
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert reasonable concurrent performance
        self.assertLess(execution_time, 3.0,
                       f"Concurrent operations took too long: {execution_time:.2f} seconds")
        
    def test_buffer_performance(self):
        """Test performance of buffer operations."""
        # First create some agents to satisfy foreign key constraints
        agent_ids = []
        for _ in range(10):
            agent_data = self._generate_random_agent_data()
            self.db.log_agent(
                agent_data["agent_id"],
                agent_data["birth_time"],
                agent_data["agent_type"],
                agent_data["position"],
                agent_data["initial_resources"],
                agent_data["max_health"],
                agent_data["starvation_threshold"]
            )
            agent_ids.append(agent_data["agent_id"])
            
        start_time = time.time()
        
        # Add multiple actions to buffer
        for _ in range(self.num_agents):
            self.db.log_agent_action(
                step_number=random.randint(0, 100),
                agent_id=random.choice(agent_ids),  # Use existing agent IDs
                action_type=random.choice(["move", "attack", "defend", "share"]),
                action_target_id=random.choice(agent_ids),  # Use existing agent IDs
                position_before=(random.uniform(0, 100), random.uniform(0, 100)),
                position_after=(random.uniform(0, 100), random.uniform(0, 100)),
                resources_before=random.uniform(0, 100),
                resources_after=random.uniform(0, 100),
                reward=random.uniform(-10, 10)
            )
            
        # Flush buffers
        self.db.flush_all_buffers()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert reasonable buffer performance
        self.assertLess(execution_time, 2.0,
                       f"Buffer operations took too long: {execution_time:.2f} seconds")
        
    def test_export_performance(self):
        """Test performance of data export operations."""
        # Create test data
        for _ in range(100):
            agent_data = self._generate_random_agent_data()
            self.db.log_agent(
                agent_data["agent_id"],
                agent_data["birth_time"],
                agent_data["agent_type"],
                agent_data["position"],
                agent_data["initial_resources"],
                agent_data["max_health"],
                agent_data["starvation_threshold"]
            )
            
        start_time = time.time()
        
        # Test different export formats
        formats = ["csv", "json"]  # Removed 'excel' as it requires additional dependencies
        for format_type in formats:
            self.db.export_data(
                f"test_export.{format_type}",
                format=format_type,
                data_types=["metrics", "agents", "resources", "actions"]
            )
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert reasonable export performance
        self.assertLess(execution_time, 5.0,
                       f"Export operations took too long: {execution_time:.2f} seconds")
        
        # Clean up export files
        for format_type in formats:
            if os.path.exists(f"test_export.{format_type}"):
                os.remove(f"test_export.{format_type}")

if __name__ == "__main__":
    unittest.main() 