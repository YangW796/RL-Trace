import gymnasium as gym

env = gym.make("CartPole-v1")
state = env.reset()
done = False

while not done:
    action = 0 if state[2] < 0 else 1  # 根据杆子角度简单决策
    next_state, reward, done, _ = env.step(action)
    state = next_state
    env.render()