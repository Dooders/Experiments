import torch
import torch.nn as nn
import torch.optim as optim

class AttackQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttackQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class AttackModule:
    def __init__(self, input_dim, output_dim, lr=0.001):
        self.model = AttackQNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.action_space = ['attack_up', 'attack_down', 'attack_left', 'attack_right', 'defend']

    def train(self, state, action, reward, next_state, done, gamma=0.99):
        self.model.train()
        state_action_values = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_state_values = self.model(next_state).max(1)[0]
        expected_state_action_values = reward + (gamma * next_state_values * (1 - done))

        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
