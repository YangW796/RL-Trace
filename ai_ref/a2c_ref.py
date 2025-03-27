import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# 参数配置
ENV_NAME = "CartPole-v1"
LEARNING_RATE = 0.001
GAMMA = 0.99
EPISODES = 1000
SAVE_DIR = "./results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Actor-Critic网络
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        
        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value

# 采样轨迹
def generate_episode(env, model):
    state = env.reset()[0]
    log_probs = []
    values = []
    rewards = []
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        
        # 动作分布和状态值
        action_probs, value = model(state_tensor)
        
        action = np.random.choice(len(action_probs.detach().cpu().numpy()), 
                                 p=action_probs.detach().cpu().numpy())

        log_prob = torch.log(action_probs[action])
        log_probs.append(log_prob)
        values.append(value)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        rewards.append(reward)
        state = next_state

    return log_probs, values, rewards

# 计算折扣回报和优势
def compute_advantages(rewards, values, gamma=GAMMA):
    returns = []
    advantages = []
    G = 0

    for reward, value in zip(reversed(rewards), reversed(values)):
        G = reward + gamma * G
        returns.insert(0, G)

        # Advantage = G - V(s)
        advantage = G - value.item()
        advantages.insert(0, advantage)

    returns = torch.tensor(returns, device=DEVICE)
    advantages = torch.tensor(advantages, device=DEVICE)

    return returns, advantages

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
    plt.title("A2C Performance on CartPole-v1 (GPU)" if DEVICE == "cuda" else "CPU")
    plt.legend()
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/a2c_reward_curve.png")
    plt.show()

# A2C 训练
def train():
    env = gym.make(ENV_NAME,render_mode="human")
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    reward_history = []

    for episode in range(EPISODES):
        log_probs, values, rewards = generate_episode(env, model)

        returns, advantages = compute_advantages(rewards, values)

        # 策略损失
        policy_loss = 0
        value_loss = 0

        for log_prob, value, G, advantage in zip(log_probs, values, returns, advantages):
            policy_loss -= log_prob * advantage
            value_loss += (G - value) ** 2

        # 优化
        optimizer.zero_grad()
        total_loss = policy_loss + 0.5 * value_loss
        total_loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        reward_history.append(total_reward)

        # 每50回合输出信息
        if episode % 50 == 0:
            avg_reward = np.mean(reward_history[-50:])
            print(f"Episode {episode}/{EPISODES}, Reward: {total_reward:.2f}, Avg: {avg_reward:.2f}")

    env.close()

    # 保存模型权重
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{SAVE_DIR}/a2c_model_gpu.pth")
    print("Model saved successfully!")

    # 绘制奖励曲线
    plot_rewards(reward_history, SAVE_DIR)

# 运行训练
if __name__ == "__main__":
    train()
