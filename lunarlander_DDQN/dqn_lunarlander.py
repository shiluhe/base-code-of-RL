# dqn_lunarlander_cpu.py
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# ----------------------------
# 1. 定义 Q 网络
# 网络层 和 前向传播逻辑
# ----------------------------
class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ----------------------------
# 2. DQN Agent（含 DDQN + Target Network）
# ----------------------------
class DQNAgent:
    def __init__(self, state_size, action_size, lr=5e-4):
        self.device = torch.device("cpu")  # 强制使用 CPU
        
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=100000)  # 大 buffer 提升稳定性
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.993
        self.epsilon_min = 0.01
        self.update_target_every = 1000
        self.step_count = 0
        
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.choice(np.arange(self.action_size))
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        # 转为 numpy array 再转 tensor
        states = np.array([e[0] for e in batch], dtype=np.float32)
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3] for e in batch], dtype=np.float32)
        dones = np.array([e[4] for e in batch])

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        # DDQN
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        target_q_values = rewards + (self.gamma * next_q_values * (~dones))
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ----------------------------
# 3. 训练主循环
# ----------------------------
def main():
    env = gym.make("LunarLander-v3")
    state_size = env.observation_space.shape[0]  # 8
    action_size = env.action_space.n            # 4
    agent = DQNAgent(state_size, action_size)
    
    scores = []
    EPISODES = 1000
    
    print("Starting DQN training (LunarLander-v3)...")
    
    for e in range(EPISODES):
        state, _ = env.reset()
        score = 0
        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if len(agent.memory) >= agent.batch_size:
                agent.replay()
                
            if done:
                break
        scores.append(score)
        print(f"Episode {e+1}/{EPISODES}, Score: {score:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        if (e + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"  -> Avg last 100: {avg_score:.2f}")
    
    # 保存模型
    torch.save(agent.q_network.state_dict(), "lunarlander_dqn.pth")
    
    # 画图
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('DDQN Training on LunarLander-v3')
    plt.grid(True)
    plt.savefig('training_curve.png', dpi=150)
    plt.show()
    env.close()

if __name__ == "__main__":
    main()