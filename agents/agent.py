import torch
from agents.modules.move_module import MoveModule
from agents.modules.attack_module import AttackModule

class Agent:
    def __init__(self, input_dim, move_output_dim, attack_output_dim):
        self.move_module = MoveModule(input_dim, move_output_dim)
        self.attack_module = AttackModule(input_dim, attack_output_dim)
        self.health = 1.0  # Example health value

    def decide_action(self, state):
        if self.should_attack(state):
            action = self.attack_module.select_action(state, self.health)
        else:
            action = self.move_module.select_action(state)
        return action

    def should_attack(self, state):
        # Logic to decide whether to attack
        return True  # Placeholder

    def train(self, state, action, reward, next_state, done):
        if action in self.attack_module.action_space:
            self.attack_module.train(state, action, reward, next_state, done)
        elif action in self.move_module.action_space:
            self.move_module.train(state, action, reward, next_state, done)

    def save_models(self, filepath):
        torch.save({
            'move_model_state_dict': self.move_module.model.state_dict(),
            'attack_model_state_dict': self.attack_module.model.state_dict(),
        }, filepath)

    def load_models(self, filepath):
        checkpoint = torch.load(filepath)
        self.move_module.model.load_state_dict(checkpoint['move_model_state_dict'])
        self.attack_module.model.load_state_dict(checkpoint['attack_model_state_dict'])
