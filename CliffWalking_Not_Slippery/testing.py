import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
import numpy as np

env = gym.make("CliffWalking-v0", render_mode = "rgb_array")
env = RecordVideo(env, video_folder="CliffWalking-agent", name_prefix="eval", episode_trigger=lambda x: True)
Q = np.load("q_values.npy")
print(f"current q_values: {Q}")
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

env.close()
print("finished testing")
print(f"Total Reward: {total_reward} out of 10 episodes.")