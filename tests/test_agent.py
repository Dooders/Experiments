import unittest
import torch
from agents.agent import Agent
from agents.modules.move_module import MoveModule
from agents.modules.attack_module import AttackModule

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.input_dim = 4
        self.move_output_dim = 4
        self.attack_output_dim = 5
        self.agent = Agent(self.input_dim, self.move_output_dim, self.attack_output_dim)

    def test_agent_initialization(self):
        self.assertIsInstance(self.agent.move_module, MoveModule)
        self.assertIsInstance(self.agent.attack_module, AttackModule)

    def test_decide_action(self):
        state = torch.randn(1, self.input_dim)
        action = self.agent.decide_action(state)
        self.assertIn(action, self.agent.move_module.action_space + self.agent.attack_module.action_space)

    def test_save_and_load_models(self):
        filepath = 'test_model.pth'
        self.agent.save_models(filepath)
        new_agent = Agent(self.input_dim, self.move_output_dim, self.attack_output_dim)
        new_agent.load_models(filepath)

        for param1, param2 in zip(self.agent.move_module.model.parameters(), new_agent.move_module.model.parameters()):
            self.assertTrue(torch.equal(param1, param2))

        for param1, param2 in zip(self.agent.attack_module.model.parameters(), new_agent.attack_module.model.parameters()):
            self.assertTrue(torch.equal(param1, param2))

if __name__ == '__main__':
    unittest.main()
