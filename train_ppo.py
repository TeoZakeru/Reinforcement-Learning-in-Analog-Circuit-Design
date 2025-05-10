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

# # ✅ Define custom policy (larger network to benefit from GPU)
# class CustomMLPPolicy(ActorCriticPolicy):
#     def __init__(self, *args, **kwargs):
#         super().__init__(
#             *args,
#             **kwargs,
#             net_arch=[dict(pi=[256, 256], vf=[256, 256])],
#             activation_fn=nn.ReLU
#         )

# ✅ Initialize your custom environment
env = W1W2OptimizationEnv()
check_env(env, warn=True)

# ✅ Instantiate A2C with GPU + custom policy
model = PPO('MlpPolicy', env, verbose=1, device=device)

# ✅ Train the model
model.learn(total_timesteps=500_000)

# ✅ Save the model
model.save("best_rl_model_ppo")

best_indices = env.best_state
best_weights = [env.weight_values[i][idx] for i, idx in enumerate(best_indices)]

print("\n✅ Best weights found by PPO:")
for i, w in enumerate(best_weights):
    print(f"w{i+1} = {w}")
print(f"🏆 Best Reward: {env.best_reward:.4f}")
