import os
import gym
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


ENV_NAME = "CartPole-v1"
LEARNING_RATE = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.99
EPISODES = 1000
SAVE_DIR = "./results"

class Actor(nn.Module):
    def __init__(self, state_t_dim,action_t_dim,hidden_dim=128):
        super(Actor, self).__init__()
        
        self.actor=nn.Sequential(
            nn.Linear(state_t_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,action_t_dim),
            nn.Softmax(dim=-1)
        )
        
    
    def forward(self, x):
        action_probs=self.actor(x)
        return action_probs

class Critic(nn.Module):
    def __init__(self, state_t_dim,hidden_dim=128):
        super(Critic, self).__init__()
        self.critic=nn.Sequential(
            nn.Linear(state_t_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
    def forward(self, x):
        v=self.critic(x)
        return v

# 采样轨迹    
def generate_episode(env, actor_model,v_model):
    state,info=env.reset()
    log_probs = []
    values = []
    rewards = []
    done=False
    while not done:
        state_tensor=torch.tensor(state,dtype=torch.float,device=DEVICE)
        
        actions_prob=actor_model(state_tensor)
        v=v_model(state_tensor)
        
        m = torch.distributions.Categorical(actions_prob)
        action = m.sample()
        #action表示采样到的动作索引
        log_prob = m.log_prob(action)
        log_probs.append(log_prob)
        values.append(v)
        action = action.item()
        next_state,reward,terminated,truncated,_=env.step(action)
        #truncated表示当前回合是否因时间限制或其他约束而提前终止。
        done=terminated or truncated
        rewards.append(reward)
        state=next_state
    
    return log_probs,values,rewards



def compute_advantages(rewards, values, gamma=GAMMA):
    advantages=[]
    returns=[]
    vt1=0
    for rt,vt in zip(reversed(rewards),reversed(values)):
        advantage=rt+gamma*vt1-vt
        advantages.insert(0,advantage)
        returns.insert(0,rt+gamma*vt1)
        vt1=vt
    advantages = torch.tensor(advantages, device=DEVICE)
    return returns,advantages

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

def train(actor_model,v_model):
    actor_optimizer=optim.Adam(actor_model.parameters(),lr=LEARNING_RATE)
    v_optimizer=optim.Adam(v_model.parameters(),lr=LEARNING_RATE)
    
    
    reward_history = []
    progress_bar = tqdm(range(EPISODES), desc="Training Episodes")

    for episode in progress_bar:
        log_probs,vs,rs=generate_episode(env,actor_model,v_model)
        
        returns,advantages=compute_advantages(rs,vs)
        
        policy_losses = 0
        value_losses = 0
        
        for log_prob,v,g,advantage in zip(log_probs,vs,returns,advantages):
            policy_losses -= log_prob*advantage
            value_losses +=  (g-v)**2#和公式不太一样
        
        actor_optimizer.zero_grad()
        policy_losses.backward()
        actor_optimizer.step()
            
        v_optimizer.zero_grad()
        value_losses.backward()
        v_optimizer.step()
        
        total_reward = sum(rs)
        reward_history.append(total_reward)
        
        progress_bar.set_postfix({
            "reward": total_reward,
            "avg_reward": np.mean(reward_history[-50:])  # 显示最近 50 个回合的平均奖励
        })
        
    env.close()
    plot_rewards(reward_history, SAVE_DIR)


if __name__ == "__main__":
    env = gym.make(ENV_NAME,render_mode="human")
    actor_model = Actor(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    v_model=Critic(env.observation_space.shape[0]).to(DEVICE)
    train(actor_model,v_model)