from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from environ import W1W2OptimizationEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn
import torch

# ✅ Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

# ✅ Create and wrap the environment beforehand
env = W1W2OptimizationEnv()
check_env(env, warn=True)

# ✅ Train the model with built-in MLP policy
model1 = A2C("MlpPolicy", env, verbose=1, device=device)

# ✅ Learn from the environment
model1.learn(total_timesteps=500_000)

# ✅ Save the trained model
model1.save("best_rl_model_a2c")

# ✅ Extract the best state and weights
best_indices = env.best_state
best_weights = [env.weight_values[i][idx] for i, idx in enumerate(best_indices)]

# ✅ Print results
print("\n✅ Best weights found by A2C:")
for i, w in enumerate(best_weights):
    print(f"w{i+1} = {w}")
print(f"🏆 Best Reward: {env.best_reward:.4f}")
