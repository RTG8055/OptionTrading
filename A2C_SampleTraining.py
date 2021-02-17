import gym
import gym_anytrading
from gym_custom_trading import MyStockEnv
from gym_anytrading.envs import TradingEnv, StocksEnv, Actions, Positions 
from gym_anytrading.datasets import STOCKS_GOOGL
import matplotlib.pyplot as plt

import quantstats as qs

from stable_baselines import A2C
from stable_baselines.common.vec_env import DummyVecEnv


df = gym_anytrading.datasets.STOCKS_GOOGL.copy()
window_size=30
start_index=window_size
end_index = len(df)

# env = MyStockEnv.MyStocksEnv(budget=10, df=df, window_size=window_size, frame_bound=(start_index, end_index))

# env_maker = lambda: gym.make(MyStockEnv.MyStocksEnv, budget=10, df=df, window_size=window_size, frame_bound=(start_index, end_index))

env_maker = lambda : MyStockEnv.MyStocksEnv(budget=10, df=df, window_size=window_size, frame_bound=(start_index, end_index))

env = DummyVecEnv([env_maker])

policy_kwargs = dict(net_arch=[64, 'lstm', dict(vf=[128, 128, 128], pi=[64, 64])])
model = A2C('MlpLstmPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
# model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1000)
model.save("a2c_googl_stocks")

print(STOCKS_GOOGL)

observation = env.reset()
i=0
while True:
    action, _states = model.predict(observation)
    # action = env.action_space.sample()
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


##Evaluation Metrics
qs.extend_pandas()

net_worth = pd.Series(env.history['total_profit'], index=df.index[start_index+1:end_index])
returns = net_worth.pct_change().iloc[1:]

qs.reports.full(returns)
qs.reports.html(returns, output='a2c_quantstats.html')