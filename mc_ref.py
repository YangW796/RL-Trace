import gym
import numpy as np
import matplotlib.pyplot as plt

def mc_control(env, num_episodes, gamma=1.0, epsilon=0.2):
    nS = env.observation_space.n  # 状态空间大小
    nA = env.action_space.n       # 动作空间大小
    
    # 使用数组存储 Q 值
    Q = np.zeros((nS, nA))
    returns_sum = np.zeros((nS, nA))
    returns_count = np.zeros((nS, nA))
    
    # 用于存储每个 episode 的回报
    episode_returns = []
    
    for i_episode in range(1, num_episodes + 1):
        if i_episode % 1000 == 0:
            print(f"Episode {i_episode}/{num_episodes}")
        
        # 生成一个 episode
        episode = []
        state_info = env.reset()  # state_info 是一个元组，例如 (0, {'prob': 1})
        state = state_info[0]     # 提取状态值
        for t in range(1000):
            # 使用 epsilon-贪婪策略选择动作
            action = epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, done,_,_ = env.step(action)  # Updated unpacking
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        
        # 计算回报并更新 Q 函数
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            state = int(state)  # 确保 state 是整数
            G = gamma * G + reward
            # 检查是否是首次访问 (state, action)
            if not any(x[0] == state and x[1] == action for x in episode[:t]):
                returns_sum[state, action] += G
                returns_count[state, action] += 1.0
                Q[state, action] = returns_sum[state, action] / returns_count[state, action]
        
        # 记录每个 episode 的回报
        episode_returns.append(G)
    
    # 将 Q 函数转换为确定性策略
    policy = np.argmax(Q, axis=1)
    
    return policy, Q, episode_returns

def epsilon_greedy_policy(Q, state, epsilon):
    state = int(state)  # 确保 state 是整数
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # 随机选择动作
    else:
        return np.argmax(Q[state])  # 选择 Q 值最大的动作

# 创建环境
env = gym.make('FrozenLake-v1')

# 训练智能体
num_episodes = 10000
policy, Q, episode_returns = mc_control(env, num_episodes)

# 绘制回报曲线
plt.plot(episode_returns)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Monte Carlo Control on FrozenLake-v1')
plt.show()

# 测试智能体
def test_policy(env, policy, num_episodes=100):
    total_rewards = 0.0
    for _ in range(num_episodes):
        state_info = env.reset()
        state = state_info[0]  # 提取状态值
        episode_reward = 0.0
        done = False
        while not done:
            action = policy[int(state)]  # 确保 state 是整数
            next_state_info, reward, done, _ ,_= env.step(action)
            state = next_state_info[0]  # 提取下一个状态值
            episode_reward += reward
        total_rewards += episode_reward
    return total_rewards / num_episodes

avg_reward = test_policy(env, policy)
print(f"Average reward over 100 episodes: {avg_reward}")

# 可视化智能体的行为
state_info = env.reset()
state = state_info[0]  # 提取状态值
done = False
while not done:
    action = policy[int(state)]  # 确保 state 是整数
    next_state_info, reward, done, _ = env.step(action)
    state = next_state_info[0]  # 提取下一个状态值
    env.render()
    if done:
        break
env.close()