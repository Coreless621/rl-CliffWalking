import gymnasium as gym
import numpy as np
from tqdm import tqdm

env = gym.make("CliffWalking-v0")

num_states = env.observation_space.n
num_actions = env.action_space.n

#hyperparamters

alpha = 0.8
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.1
episodes = 5000
decay = 0.9996

# Q values 

Q = np.zeros((num_states, num_actions))
total_reward = 0

for episode in tqdm(range(episodes), desc="Training Progress"):
    state, _ = env.reset()
    done = False

    while not done:
        if np.random.uniform(0,1) < epsilon:
            action = env.action_space.sample()

        else:
            action = np.argmax(Q[state, :])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Q update
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state

    # decaying epsilon
    if epsilon > min_epsilon:
        epsilon *= decay

average_reward = total_reward/episodes
print("training completed")
print(f"Average Reward: {average_reward}")
np.save("q_values", Q)
print(Q)