# Width and Topology Optimization of Analog LDO Using Reinforcement Learning

## 📋 Overview

This project applies **Reinforcement Learning (RL)** to optimize transistor **widths** and **topologies** in an **analog Low-Dropout (LDO) voltage regulator**, focusing on minimizing power consumption and output voltage error.

The RL environment models LDO design as a **Markov Decision Process (MDP)** with a discrete action and state space. Two popular policy gradient methods — **Proximal Policy Optimization (PPO)** and **Advantage Actor-Critic (A2C)** — are used for training and evaluation.

All SPICE simulations are **precomputed** and stored in a lookup table for fast training without invoking SPICE at runtime.

---

## 📁 Repository Structure

```plaintext
.
├── environ.py                                                 # Custom Gymnasium-compatible RL environment
├── train_a2c.py                                               # Evaluation script for A2C-trained agent
├── train_ppo.py                                               # Evaluation script for PPO-trained agent
├── test_a2c.py                                                # Evaluation script for A2C-trained agent
├── test_ppo.py                                                # Evaluation script for PPO-trained agent
├── merged_simulation_results_0_159999_full.csv                # Precomputed SPICE results (160,000 configs)
├── best_rl_model_a2c.zip                                      # Best Trained A2C Model
├── best_rl_model_ppo.zip                                      # Best Trained PPO Model
├── README.md                                                  # This file
└── requirements.txt                                           # Python dependencies
```

---

## 🚀 How It Works

### 1. Environment Design

- **State Space (8-D):**
  - `w1` to `w4`: Transistor widths (1–10 µm)
  - `c1` to `c4`: Topology (1 = single, 2 = cascoded)

- **Action Space (16 actions):**
  - Increment/decrement each of the 8 parameters (width or config)

- **Reward Function:**

  ```python
  reward = -(error + power) * 1e6
  ```

### 2. Simulation Data

- `merged_simulation_results_0_159999_full.csv` contains SPICE simulation results for all 160,000 configurations.
- During training, these are indexed using an in-memory Python dictionary for **O(1)** lookup.

---

## 🔧 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required libraries include:
- `gymnasium`
- `stable-baselines3`
- `pandas`
- `numpy`
- `torch`
- `ltspice` (if needed for simulation validation)

### 2. Train RL Agents

> Note: This is already done if models are provided.

```bash
python train_a2c.py    # Optional - for A2C
python train_ppo.py    # Optional - for PPO
```


### 3. Evaluate Trained Models

```bash
python test_a2c.py     # Runs A2C policy from a default or user-defined state
python test_ppo.py     # Runs PPO policy and logs performance
```

---

## 📌 Key Highlights

- Fast training via SPICE result caching
- Discrete MDP modeling for analog design
- Reproducible results using fixed random seed

---

## 📚 References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [LTspice Simulator](https://www.analog.com/en/design-center/design-tools-and-calculators/ltspice-simulator.html)
- Razavi, B. "Design of Analog CMOS Integrated Circuits"

---

## 👨‍💻 Authors

- Komaragiri Sai Vishwanath Rohit  
- Ghantasala Sai Narayana Sujit  
- Akula Nishith  
