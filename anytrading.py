import numpy as np
import gym
import gym_anytrading
import matplotlib.pyplot as plt

env = gym.make('stocks-v0')


print("env information:")
print("> shape:", env.shape)
print("> df.shape:", env.df.shape)
print("> prices.shape:", env.prices.shape)
print("> signal_features.shape:", env.signal_features.shape)
print("> max_possible_profit:", env.max_possible_profit())

env.reset()
env.render()
observation = env.reset()
while True:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    # env.render()
    if done:
        print("info:", info)
        break

plt.cla()
env.render_all()
plt.show()
# print()
# print("custom_env information:")
# print("> shape:", custom_env.shape)
# print("> df.shape:", env.df.shape)
# print("> prices.shape:", custom_env.prices.shape)
# print("> signal_features.shape:", custom_env.signal_features.shape)
# print("> max_possible_profit:", custom_env.max_possible_profit())