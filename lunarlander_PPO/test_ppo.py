import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# 与训练代码一致的 Actor-Critic 网络
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

# 测试函数
def test_model(model_path, env_name="LunarLander-v3", num_episodes=10, render=True, seed=42):
    # 设置随机种子（确保可复现）
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 创建环境
    if render:
        env = gym.make(env_name, render_mode="human")  # 显示窗口
    else:
        env = gym.make(env_name, render_mode="rgb_array")  # 无窗口（可录视频）
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(state_dim, action_dim).to(device)
    
    # 加载权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()  # 设置为评估模式（关闭 dropout 等，虽然这里没用）
    
    scores = []
    
    print(f"Testing model: {model_path}")
    print(f"Device: {device}")
    print(f"Running {num_episodes} episodes...\n")
    
    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed + ep)
        episode_score = 0
        step = 0
        
        while True:
            # 转为 tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, _ = model(state_tensor)
                # 方式1：采样动作（带策略随机性）
                # action = torch.distributions.Categorical(logits=logits).sample().item()
                
                # 方式2：选择概率最大的动作（确定性策略，推荐用于测试）
                action = torch.argmax(logits, dim=1).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_score += reward
            state = next_state
            step += 1
            
            if done:
                break
        
        scores.append(episode_score)
        print(f"Episode {ep+1}/{num_episodes} | Score: {episode_score:.2f}")
    
    env.close()
    
    # 打印统计结果
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    print("\n" + "="*50)
    print(f"Test Results over {num_episodes} episodes:")
    print(f"Average Score: {avg_score:.2f} ± {std_score:.2f}")
    print(f"Min: {np.min(scores):.2f} | Max: {np.max(scores):.2f}")
    print("="*50)
        
    return scores

# 主函数
if __name__ == "__main__":
    MODEL_PATH = "ppo_lunarlander_final.pth"
    
    # 运行测试
    scores = test_model(
        model_path=MODEL_PATH,
        num_episodes=10,      # 测试轮数
        render=True,          # 是否显示
        seed=42
    )