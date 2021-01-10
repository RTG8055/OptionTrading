import gym
import gym_anytrading
from gym_custom_trading import MyStockEnv
from gym_anytrading.envs import TradingEnv, StocksEnv, Actions, Positions 
from gym_anytrading.datasets import STOCKS_GOOGL
import matplotlib.pyplot as plt



env = MyStockEnv.MyStocksEnv(budget=10, df=STOCKS_GOOGL, window_size=30, frame_bound=(30, len(STOCKS_GOOGL)))

print(STOCKS_GOOGL)

observation = env.reset()
i=0
while True:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    # env.render()
    # print("----------Iteration {}-------------".format(i))
    # print("Observation:",observation[-2:,:2])
    # print("Action taken:", action)
    # print("Reward:", reward)
    # print("Done:", done)
    # print("info:", info)
    i+=1
    # if(i%12==0):
    	# break
    if done:
        print("info:", info)
        break

plt.cla()
env.render_all()
plt.show()