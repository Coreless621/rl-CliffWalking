# ⛰️ CliffWalking-v0 – Q-Learning Agent

This project implements a **Q-learning agent** for the classic `CliffWalking-v0` environment from the Gymnasium library.  
The agent learns to navigate a dangerous cliff-filled grid to reach the goal without falling off the edge.

---

## 🌍 Environment Overview

- **Environment:** `CliffWalking-v0`
- **Grid layout:** 4 rows × 12 columns
- **States:** 48 discrete positions
- **Actions:** 4 discrete actions (left, down, right, up)
- **Reward structure:**
  - `-1` per step
  - `-100` if agent steps into the cliff
  - `0` for reaching the goal

---

## 🧠 Algorithm

- **Type:** Tabular Q-learning
- **Policy:** Epsilon-greedy with exponential decay
- **Update Rule:** Bellman equation (off-policy)
- **Goal:** Learn a path from start to goal avoiding the cliff

---

## ⚙️ Hyperparameters

- Learning rate   | `alpha = 0.8` 
- Discount factor | `gamma = 0.99` 
- Initial epsilon | `1.0` 
- Min epsilon     | `0.1` 
- Epsilon decay   | `0.9996` 
- Episodes        | `5000` 

---

## 📂 Project Structure

- `training.py`  | Trains the Q-learning agent and saves the resulting Q-table (`q_values.npy`) 
- `testing.py`   | Loads the Q-table and runs 5 test episodes using a greedy policy; records videos 
- `q_values.npy` | (Generated) Q-table as NumPy array after training 
- `CliffWalking-agent/` | (Generated) Folder containing recorded evaluation videos 

## Note
I used tqdm to create a training progress bar. It is not needed to train your agent. You can remove it if you want to.
