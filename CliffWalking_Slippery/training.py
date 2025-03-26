import gymnasium as gym
import numpy as np
from tqdm import tqdm

env = gym.make("CliffWalking-v0")
num_states = env.observation_space.n
num_actions = env.action_space.n

Q = np.zeros((num_states, num_actions))

#hyperparameters

alpha = 0.5
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.1
decay = 0.9996
episodes = 5000

for episode in tqdm(range(episodes)):
    state, _= env.reset()
    done = False
    while not done:
        if np.random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Q table update
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
        done = terminated or truncated # check if Cliffwalk is completed
        # decay epsilon
        if epsilon > min_epsilon:
            epsilon *= decay

print("training completed")
np.save("q_values.npy", Q)
print("q_values saved!")
print(Q)