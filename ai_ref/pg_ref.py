import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# 环境参数
ENV_NAME = "CartPole-v1"
LEARNING_RATE = 0.01
GAMMA = 0.99
EPISODES = 1000
RENDER = False
SAVE_DIR = "./results"

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.output(x), dim=-1)

# 采样轨迹
def generate_episode(env, policy, device):
    state = env.reset()[0]
    log_probs = []
    rewards = []
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        action_probs = policy(state_tensor)

        action = np.random.choice(len(action_probs.detach().cpu().numpy()), 
                                 p=action_probs.detach().cpu().numpy())

        log_prob = torch.log(action_probs[action])
        log_probs.append(log_prob)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        rewards.append(reward)
        state = next_state

    return log_probs, rewards

# 计算折扣回报
def compute_returns(rewards, gamma=GAMMA):
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns

# 绘制奖励曲线
def plot_rewards(reward_history, save_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(reward_history, label="Total Reward per Episode", color="blue", alpha=0.7)

    # 滑动平均平滑曲线
    window = 50
    avg_rewards = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    plt.plot(range(window - 1, len(avg_rewards) + window - 1), avg_rewards, label="Moving Average (50 episodes)", color="red")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Policy Gradient Performance on CartPole-v1 (GPU)" if device == "cuda" else "CPU")
    plt.legend()
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/reward_curve.png")
    plt.show()

# 策略梯度训练
def train():
    env = gym.make(ENV_NAME,render_mode="human")
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    reward_history = []

    for episode in range(EPISODES):
        log_probs, rewards = generate_episode(env, policy, device)

        # 将回报计算迁移到 GPU 上
        returns = torch.tensor(compute_returns(rewards), device=device)
        
        # 标准化回报
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # 计算策略梯度损失
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss -= log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        reward_history.append(total_reward)

        # 每 50 回合输出一次信息
        if episode % 50 == 0:
            avg_reward = np.mean(reward_history[-50:])
            print(f"Episode {episode}/{EPISODES}, Reward: {total_reward:.2f}, Average Reward: {avg_reward:.2f}")

    env.close()

    # 保存模型权重
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(policy.state_dict(), f"{SAVE_DIR}/policy_model_gpu.pth")
    print("Model saved successfully!")

    # 绘制并保存曲线
    plot_rewards(reward_history, SAVE_DIR)

# 运行训练
if __name__ == "__main__":
    train()
