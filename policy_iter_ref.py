import numpy as np
import gym

def policy_evaluation(env, policy, gamma=1.0, theta=1e-8):
    nS = env.observation_space.n
    V = np.zeros(nS)
    while True:
        delta = 0
        for s in range(nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, terminated in env.unwrapped.P[s][a]:
                    v += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

def policy_improvement(env, V, gamma=1.0):
    nS = env.observation_space.n
    nA = env.action_space.n
    policy = np.zeros([nS, nA]) / nA
    for s in range(nS):
        q = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, terminated,  in env.unwrapped.P[s][a]:
                q[a] += prob * (reward + gamma * V[next_state])
        best_a = np.argwhere(q == np.max(q)).flatten()
        policy[s] = np.sum([np.eye(nA)[i] for i in best_a], axis=0) / len(best_a)
    return policy

def policy_iteration(env, gamma=1.0, theta=1e-8):
    nS = env.observation_space.n
    nA = env.action_space.n
    policy = np.ones([nS, nA]) / nA
    while True:
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V, gamma)
        if np.allclose(new_policy, policy):
            break
        policy = new_policy
    return policy, V

def main():
    env = gym.make('FrozenLake-v1', is_slippery=False)  # 创建环境
    optimal_policy, optimal_value = policy_iteration(env)
    print("Optimal Policy:")
    print(optimal_policy)
    print("\nOptimal Value Function:")
    print(optimal_value)

if __name__ == "__main__":
    main()