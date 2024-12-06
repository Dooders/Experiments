import unittest
from unittest.mock import MagicMock, patch
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from database.base_repository import BaseRepository
from database.agent_repository import AgentRepository
from database.resource_repository import ResourceRepository
from database.learning_repository import LearningRepository
from database.population_repository import PopulationRepository
from database.simulation_repository import SimulationRepository
from database.models import Agent, ResourceState, LearningExperience, SimulationStep

class TestBaseRepository(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.session = MagicMock(spec=Session)
        self.db.Session.return_value = self.session
        self.model = Agent
        self.repository = BaseRepository(self.db, self.model)

    def test_add(self):
        entity = self.model()
        self.repository.add(entity)
        self.session.add.assert_called_once_with(entity)
        self.session.commit.assert_called_once()

    def test_get_by_id(self):
        entity_id = 1
        self.repository.get_by_id(entity_id)
        self.session.query(self.model).get.assert_called_once_with(entity_id)

    def test_update(self):
        entity = self.model()
        self.repository.update(entity)
        self.session.merge.assert_called_once_with(entity)
        self.session.commit.assert_called_once()

    def test_delete(self):
        entity = self.model()
        self.repository.delete(entity)
        self.session.delete.assert_called_once_with(entity)
        self.session.commit.assert_called_once()

    def test_execute_in_transaction(self):
        def func(session):
            return "result"

        result = self.repository._execute_in_transaction(func)
        self.assertEqual(result, "result")
        self.session.commit.assert_called_once()

    def test_execute_in_transaction_rollback_on_error(self):
        def func(session):
            raise SQLAlchemyError("error")

        with self.assertRaises(SQLAlchemyError):
            self.repository._execute_in_transaction(func)
        self.session.rollback.assert_called_once()

class TestAgentRepository(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.repository = AgentRepository(self.db)

    @patch('database.agent_repository.execute_query')
    def test_lifespan_statistics(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.lifespan_statistics()
        self.assertIsNotNone(result)

    @patch('database.agent_repository.execute_query')
    def test_survival_rates(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.survival_rates()
        self.assertIsNotNone(result)

    @patch('database.agent_repository.execute_query')
    def test_execute(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.execute()
        self.assertIsNotNone(result)

class TestResourceRepository(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.repository = ResourceRepository(self.db)

    @patch('database.resource_repository.execute_query')
    def test_resource_distribution(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.resource_distribution()
        self.assertIsNotNone(result)

    @patch('database.resource_repository.execute_query')
    def test_consumption_patterns(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.consumption_patterns()
        self.assertIsNotNone(result)

    @patch('database.resource_repository.execute_query')
    def test_resource_hotspots(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.resource_hotspots()
        self.assertIsNotNone(result)

    @patch('database.resource_repository.execute_query')
    def test_efficiency_metrics(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.efficiency_metrics()
        self.assertIsNotNone(result)

    @patch('database.resource_repository.execute_query')
    def test_execute(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.execute()
        self.assertIsNotNone(result)

class TestLearningRepository(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.repository = LearningRepository(self.db)

    @patch('database.learning_repository.execute_query')
    def test_learning_progress(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.learning_progress()
        self.assertIsNotNone(result)

    @patch('database.learning_repository.execute_query')
    def test_module_performance(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.module_performance()
        self.assertIsNotNone(result)

    @patch('database.learning_repository.execute_query')
    def test_agent_learning_stats(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.agent_learning_stats()
        self.assertIsNotNone(result)

    @patch('database.learning_repository.execute_query')
    def test_learning_efficiency(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.learning_efficiency()
        self.assertIsNotNone(result)

    @patch('database.learning_repository.execute_query')
    def test_execute(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.execute()
        self.assertIsNotNone(result)

class TestPopulationRepository(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.repository = PopulationRepository(self.db)

    @patch('database.population_repository.execute_query')
    def test_population_data(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.population_data()
        self.assertIsNotNone(result)

    @patch('database.population_repository.execute_query')
    def test_basic_population_statistics(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.basic_population_statistics()
        self.assertIsNotNone(result)

    @patch('database.population_repository.execute_query')
    def test_agent_type_distribution(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.agent_type_distribution()
        self.assertIsNotNone(result)

    @patch('database.population_repository.execute_query')
    def test_execute(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.execute()
        self.assertIsNotNone(result)

class TestSimulationRepository(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.repository = SimulationRepository(self.db)

    @patch('database.simulation_repository.execute_query')
    def test_agent_states(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.agent_states()
        self.assertIsNotNone(result)

    @patch('database.simulation_repository.execute_query')
    def test_resource_states(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.resource_states()
        self.assertIsNotNone(result)

    @patch('database.simulation_repository.execute_query')
    def test_simulation_state(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.simulation_state()
        self.assertIsNotNone(result)

    @patch('database.simulation_repository.execute_query')
    def test_execute(self, mock_execute_query):
        mock_execute_query.return_value = MagicMock()
        result = self.repository.execute()
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
