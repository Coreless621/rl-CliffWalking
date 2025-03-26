import gymnasium as gym
import numpy as np

env = gym.make("CliffWalking-v0")
Q = np.load("q_values.npy")
total_reward = 0

for episode in range(5):
    done = False
    state, _ = env.reset()

    while not done:
        action = np.argmax(Q[state, :])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward +=reward
        state = next_state

print("finished testing")
print(f"Total Reward: {total_reward} out of 10 episodes.")