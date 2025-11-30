import gymnasium as gym
import torch
import torch.nn as nn
import os

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

env = gym.make("LunarLander-v3", render_mode="human")

q_net = QNetwork(8, 4)
q_net.load_state_dict(torch.load("lunarlander_dqn.pth"))
q_net.eval()

state, _ = env.reset()
for _ in range(1000):
    with torch.no_grad():
        action = q_net(torch.FloatTensor(state)).argmax().item()
    state, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        state, _ = env.reset()
env.close()