import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# ----------------------------
# 超参数
# ----------------------------
ENV_NAME = "LunarLander-v3"
SEED = 42
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.15
ENTROPY_COEF = 0.01
VF_COEF = 0.5
LR = 3e-4
BATCH_SIZE = 64
NUM_EPOCHS = 10
MAX_TIMESTEPS = 1000  # 每 rollout 最多步数（每个 episode 最多跑这么多步）
TOTAL_TIMESTEPS = 1_000_000
SAVE_MODEL_EVERY = 50_000

device = torch.device("cpu")

# 设置随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Actor-Critic 网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        self.fc_shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.fc_actor = nn.Linear(hidden_dim, action_dim)
        self.fc_critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        shared = self.fc_shared(x)
        logits = self.fc_actor(shared)
        value = self.fc_critic(shared)
        return logits, value

    def evaluate(self, state, action):
        # 用于计算 loss 时重新评估动作的 log_prob 和状态值
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.clip_eps = CLIP_EPS

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, value = self.policy(state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def update(self, memory):
        # memory: list of (s, a, lp, r, s', d, v)
        states = torch.FloatTensor([m[0] for m in memory]).to(device)
        actions = torch.LongTensor([m[1] for m in memory]).to(device)
        old_log_probs = torch.FloatTensor([m[2] for m in memory]).to(device)
        rewards = [m[3] for m in memory]
        dones = [m[5] for m in memory]
        values = [m[6] for m in memory]

        # 计算 GAE 优势估计
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            delta = rewards[i] + GAMMA * next_value * (1 - dones[i]) - values[i]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        advantages = torch.FloatTensor(advantages).to(device)
        returns = advantages + torch.FloatTensor(values).to(device)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 多轮优化
        for _ in range(NUM_EPOCHS):
            # 随机采样子批次
            indices = np.arange(len(memory))
            np.random.shuffle(indices)
            for start in range(0, len(memory), BATCH_SIZE):
                idx = indices[start:start + BATCH_SIZE]
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                # 重新评估
                log_probs, entropy, values = self.policy.evaluate(batch_states, batch_actions)
                ratios = torch.exp(log_probs - batch_old_log_probs)

                # PPO-Clip 损失
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values.squeeze(), batch_returns)
                entropy_loss = entropy.mean()

                loss = actor_loss + VF_COEF * critic_loss - ENTROPY_COEF * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


# Rollout Buffer（用于收集一个 rollout 的经验）
class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.buffer = []

    def push(self, state, action, log_prob, reward, next_state, done, value):
        self.buffer.append((state, action, log_prob, reward, next_state, done, value))

    def __len__(self):
        return len(self.buffer)

def main():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim)
    buffer = RolloutBuffer()

    total_timesteps = 0
    scores = []
    score_window = deque(maxlen=100)

    print(f"Starting PPO training on {ENV_NAME}...")

    while total_timesteps < TOTAL_TIMESTEPS:
        state, _ = env.reset()
        episode_score = 0
        step = 0
        buffer.reset()

        while step < MAX_TIMESTEPS:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, log_prob, reward, next_state, done, value)

            state = next_state
            episode_score += reward
            total_timesteps += 1
            step += 1

            if done:
                break

        # 更新策略
        agent.update(buffer.buffer)

        scores.append(episode_score)
        score_window.append(episode_score)

        # 打印训练进度
        if len(score_window) == 100:
            avg_score = np.mean(score_window)
            print(f"Timesteps: {total_timesteps}/{TOTAL_TIMESTEPS}, "
                  f"Last Score: {episode_score:.2f}, Avg(100): {avg_score:.2f}")

    # 保存最终模型
    torch.save(agent.policy.state_dict(), "ppo_lunarlander_final.pth")

    # 绘制训练曲线
    plt.figure(figsize=(12, 6))
    plt.plot(scores)
    plt.title("PPO Training - LunarLander-v3")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.grid(True)
    plt.savefig("ppo_training_curve.png", dpi=150)
    plt.show()

    env.close()

if __name__ == "__main__":
    main()