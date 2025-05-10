# from stable_baselines3 import PPO
# from W1W2OptimizationEnv import W1W2OptimizationEnv
# import numpy as np
# from stable_baselines3 import A2C

# env = W1W2OptimizationEnv()
# model = A2C.load("best_rl_model_a2c")

# def get_index(value, value_list):
#     return int(np.argmin([abs(v - value) for v in value_list]))

# def optimize_user_input(w1_input, w2_input, max_actions=100):
#     # convert to indices
#     w1_idx = get_index(w1_input, env.w1_values)
#     w2_idx = get_index(w2_input, env.w2_values)

#     # initialize env state
#     env.state = (w1_idx, w2_idx)
#     env.steps = 0
#     state = np.array(env.state, dtype=np.int32)

#     print(f"\n🔍 Starting from: w1={env.w1_values[w1_idx]}, w2={env.w2_values[w2_idx]}")

#     best_reward = -np.inf
#     best_w1 = best_w2 = None

#     for step in range(max_actions):
#         action, _ = model.predict(state)
#         state, reward, done, _, _ = env.step(action)

#         w1 = env.w1_values[state[0]]
#         w2 = env.w2_values[state[1]]
#         print(f"Step {step+1}: w1={w1}, w2={w2}, reward={reward:.6f}")

#         if reward > best_reward:
#             best_reward = reward
#             best_w1, best_w2 = w1, w2

#         if done:
#             break

#     print(f"\n✅ Best state: w1={best_w1}, w2={best_w2}, reward={best_reward:.6f}")
#     return best_w1, best_w2, best_reward

# # Example
# optimize_user_input(w1_input=1, w2_input=49)


from stable_baselines3 import PPO, A2C
from environ import W1W2OptimizationEnv
import numpy as np

env = W1W2OptimizationEnv()
# Load your pre-trained model
model = A2C.load("best_rl_model_a2c")

# Helper to find closest index in a discrete list
def get_index(value, value_list):
    return int(np.argmin([abs(v - value) for v in value_list]))

def optimize_user_input(inputs=[1,1,1,1,2,1,1,2], max_actions=500_000):
    idxs = [get_index(val, env.weight_values[i])
            for i, val in enumerate(inputs)]
    env.state = tuple(idxs)
    env.steps = 0
    state = np.array(env.state, dtype=np.int32)

    print(f"\n🔍 Starting from: " + ", ".join(
        f"w{i+1}={env.weight_values[i][idx]}" for i, idx in enumerate(idxs)
    ))

    best_reward = -np.inf
    best_weights = None

    for step in range(max_actions):
        action, _ = model.predict(state)
        state, reward, done, _, _ = env.step(action)
        current = [env.weight_values[i][idx]
                   for i, idx in enumerate(state)]

        print(f"Step {step+1}: " + ", ".join(
            f"w{i+1}={val}" for i, val in enumerate(current)
        ) + f", reward={reward:.6f}")

        if reward > best_reward:
            best_reward = reward
            best_weights = current.copy()
            best_step = step

        if done:
            break

    print(f"\n✅ Best state: " + ", ".join(
        f"w{i+1}={val}" for i, val in enumerate(best_weights)
    ) + f", reward={best_reward:.6f}, step={best_step}")
    return best_weights, best_reward

# Example usage
optimize_user_input()