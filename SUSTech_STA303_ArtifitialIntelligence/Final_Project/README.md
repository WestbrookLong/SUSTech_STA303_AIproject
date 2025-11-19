<h3 align="center">
  <img src="assets/cartpole_icon_web.png" width="300">
</h3>

# 🧠 CartPole Reinforcement Learning Project

This project is part of the **STA303 Artificial Intelligence Course Project (2025–2026)**.  
It focuses on implementing, training, and analyzing reinforcement learning (RL) algorithms using the **CartPole-v1** environment provided by **Gymnasium**.

---

## 📘 Project Overview

Reinforcement Learning (RL) is a branch of machine learning where an agent learns to make decisions by interacting with an environment.  
In this project, you will apply RL techniques to the **CartPole** problem — a classic control task where the agent must balance a pole on a moving cart by applying forces to the left or right.

Through this assignment, you will:

1. **Understand the RL framework** (state, action, reward, policy, environment).  
2. **Implement and analyze RL algorithms** beyond the provided DQN baseline.  
3. **Study how hyperparameters influence learning performance.**  
4. **(Advanced)** Explore complex extensions such as Offline RL or Imitation Learning.

---

## 🧩 Environment Description: CartPole-v1

### 🔹 Task Description

A pole is attached by an un-actuated joint to a cart moving along a frictionless track.  
The agent must prevent the pole from falling by choosing at each step whether to push the cart **left** or **right**.

The episode ends if:
- The pole tilts more than **12° from vertical**, or  
- The cart moves more than **2.4 units** from the center, or  
- The maximum time step limit (default 500) is reached.

Each time step that the pole remains upright gives a **reward of +1**.  
The goal is to maximize the total cumulative reward per episode.

### 🔹 Observation and Action Spaces

| Element | Description | Type | Range |
|----------|--------------|------|--------|
| `x` | Cart position | Continuous | [-2.4, 2.4] |
| `x_dot` | Cart velocity | Continuous | (-∞, ∞) |
| `theta` | Pole angle (radians) | Continuous | [-0.21, 0.21] |
| `theta_dot` | Pole angular velocity | Continuous | (-∞, ∞) |
| **State** | `[x, x_dot, theta, theta_dot]` | 4D vector | — |
| **Actions** | 0 → push left, 1 → push right | Discrete | {0, 1} |

---

## ⚙️ Project Structure

```

cartpole_project/
│
├── agents/
│   ├── cartpole_dqn.py        # Provided DQN reference implementation
│   ├── cartpole_ppo.py        # (To be implemented by students)
│   └── cartpole_actorcritic.py # (To be implemented by students)
│
├── scores/
│   └── score_logger.py        # Handles logging and visualization
│
├── train.py                   # Main training script
├── requirements.txt            # Python dependencies
└── assets/                     # Optional images or results

````

### 🧠 `train.py`

The entry point for training.  
It sets up the environment, initializes the chosen agent, and runs the main training loop.  
Results (scores and plots) are automatically stored in `scores/`.

### ⚡ `agents/cartpole_dqn.py`

This file contains a **fully implemented DQN algorithm** that serves as a **reference baseline**.  
It demonstrates key RL components including:
- Q-value function approximation via neural networks  
- Epsilon-greedy exploration  
- Experience replay  
- Temporal Difference (TD) learning updates  

Students are **not required** to modify this file but should **study its structure** carefully to understand how RL algorithms interact with the environment.

### 📈 `scores/score_logger.py`

Handles:
- Tracking episode scores  
- Plotting learning curves (`scores.png`, `solved.png`)  
- Recording progress to `.csv` files  

---

## 🧪 How to Run

### 1️⃣ Setup Environment

```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate        # (Mac/Linux)
venv\Scripts\activate           # (Windows)

# Install dependencies
pip install -r requirements.txt
````

### 2️⃣ Run Training

```bash
python train.py
```

Training progress and score curves will be saved automatically under `scores/`.

---

## 🎓 Student Tasks

### **1. Understand the CartPole Environment**

* Observe how states, actions, and rewards interact.
* Understand what the agent learns — stabilizing the pole by selecting left/right actions.

### **2. Implement New RL Algorithms**

* **Do not modify the provided DQN baseline.**
* Instead, implement **two new RL algorithms** under the `agents/` folder, such as:

  * `cartpole_ppo.py` – Proximal Policy Optimization (Policy Gradient)
  * `cartpole_actorcritic.py` – Actor-Critic Method

Each implementation should:

* Define a neural network model (policy or value function)
* Implement the training loop and update logic
* Be runnable through `train.py` for fair comparison

### **3. Hyperparameter Exploration**

Analyze how hyperparameters affect your algorithm’s performance, including:

* Learning rate (`LEARNING_RATE`)
* Discount factor (`GAMMA`)
* Exploration parameters (for policy-based algorithms, entropy coefficients, etc.)
* Batch size and update frequency
* Network architecture (hidden layers, activation functions)

Include comparisons (e.g., learning curves, training speed, convergence behavior).

### **4. Report and Analysis**

Each student/team must submit a **concise report (≤ 8 pages)** covering:

* **Algorithm description and implementation ideas**
* **Training results and reward curves**
* **Hyperparameter sensitivity analysis**
* **Interpretation and insights**

👉 **Important:**
Analysis and discussion are key evaluation points — the focus is not only on getting high rewards,
but also on demonstrating understanding of *why* algorithms behave differently under different settings.

---

## ⚡ Advanced Challenge (Optional)

Students aiming for higher difficulty can extend their work with:

* **Offline Reinforcement Learning:** learning from pre-collected data instead of online interaction
* **Imitation Learning / Behavior Cloning:** training agents to mimic expert demonstrations

Relevant data and hints will be provided for these advanced topics.

---

## 🧪 Final Evaluation

Each team must submit:

1. **Code:** runnable and well-documented implementation (`agents/` + `train.py`).
2. **Trained Agent:** capable of performing 100 evaluation episodes.
3. **Report:** summarizing methods, results, and analysis.

During evaluation, your final trained agent will be tested for **100 episodes**.
If its **average reward exceeds a defined threshold**, the **performance component** will receive full credit.

Other grading components will assess:

* Code organization and reproducibility
* Algorithm correctness
* Depth of understanding (from the report)
* Quality of hyperparameter and result analysis

---

## 🔍 Recommended Extensions (Bonus)

* **Double DQN** – Reduce Q-value overestimation.
* **Dueling DQN** – Separate value and advantage learning.
* **Prioritized Replay** – Sample important transitions more frequently.
* **PPO / A2C / A3C** – Explore policy-based or hybrid actor-critic methods.

---

## 🧠 References

* Mnih et al., *Human-level Control through Deep Reinforcement Learning*, Nature 2015.
* Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv 2017.
* Van Hasselt et al., *Deep Reinforcement Learning with Double Q-Learning*, AAAI 2016.
* Wang et al., *Dueling Network Architectures for Deep Reinforcement Learning*, ICML 2016.
* [Gymnasium Documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
* [Deep RL Book: Hands-on RL Tutorial](https://hrl.boyuai.com/)

---

## ✍️ Authors & Acknowledgements

This project framework is adapted from Greg Surma’s [CartPole DQN repository](https://github.com/gsurma/cartpole)
and modified for **STA303: Artificial Intelligence** coursework at **SUSTech** (2025–2026).