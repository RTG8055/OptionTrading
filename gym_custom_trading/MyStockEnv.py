import numpy as np
import pandas as pd
import datetime as dt
import gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv
# from gym_anytrading.datasets import STOCKS_GOOGL
import matplotlib.pyplot as plt
from gym import spaces
from gym.utils import seeding
from enum import Enum


# Calculating RSI without using loop
def RSI(df, n):
    '''
    "function to calculate RSI"
    rsi(df,14)
    '''
    delta = df["Adj Close"].diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[n-1]] = np.mean( u[:n]) # first value is average of gains
    u = u.drop(u.index[:(n-1)])
    d[d.index[n-1]] = np.mean( d[:n]) # first value is average of losses
    d = d.drop(d.index[:(n-1)])
    rs = u.ewm(com=n,min_periods=n).mean()/d.ewm(com=n,min_periods=n).mean()
    return 100 - 100 / (1+rs)


def MACD(DF, a, b, c):
    """
        function to calculate MACD
        typical values a = 12; b =26, c =9
        df = MACD(ohlcv, 12, 26, 9)
    """
    df = DF.copy()
    df["MA_Fast"]=df["Adj Close"].ewm(span=a,min_periods=a).mean()
    df["MA_Slow"]=df["Adj Close"].ewm(span=b,min_periods=b).mean()
    df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
    df["Signal"]=df["MACD"].ewm(span=c,min_periods=c).mean()
    df.fillna(0,inplace=True)
    return df

class NewActions(Enum):
    ## can buy, sell or hold ## TODO: buy sell multiple lots
    Sell = 0
    Hold = 1
    Buy = 2


class NewPositions(Enum):
    ##introduce budget concept, buy or sell only within budget, configurable lot size##
    Short = -1
    No_position = 0
    Long = 1



def _new_process_data(df,frame_bound,window_size):
    ## calculate the techincal indicators and adjust the window size accordingly ##
    '''
    df -  contains data of OHLC for a particular stock
    Close are the prices
    signal features are OHL and RSI, ADX, MACD, OBV
    '''

    MACD_a = 12
    MACD_b = 26
    MACD_c = 9
    RSI_n = 16
    prices = df.loc[:, 'Close'].to_numpy()

    prices[frame_bound[0] - window_size]  # validate index (TODO: Improve validation)
    prices = prices[frame_bound[0]-window_size:frame_bound[1]]

    diff = np.insert(np.diff(prices), 0, 0)
    macd = MACD(df, MACD_a, MACD_b, MACD_c)[['MA_Fast', 'MA_Slow', 'MACD', 'Signal']]
    rsi = np.insert(RSI(df, RSI_n).fillna(0).values,0,[0]*RSI_n)
    signal_features = np.column_stack((prices, diff, macd, rsi))

    return prices, signal_features

class MyStocksEnv(StocksEnv):

    def __init__(self,budget,**kwargs):
        '''
        budget and lot_size are relative terms, instead, removing lot_size
        Instead giving a budget of "n" implies the bot can take "n" positions at the same time
        budget increases with increase in profit and decreases with loss
        big losses should deplete the budget and impose extreme penalty
        '''
        super().__init__(**kwargs)
        # self.data = df
        self.budget = budget
        # self.lot_size = lot_size
        self.average_price = 0
        # self.average_sell = 0
        self.action_space = spaces.Discrete(len(NewActions))
        # self.prices, self.singal_features = self._process_data()


    def step(self, action):
        ''' 
        flawed mechanism, selling a long position does not mean it is a short position
        give negative reward if lost all budget before end of time.
        '''
        self._done=False
        self._current_tick+=1

        if(self._current_tick == self._end_tick):
            self._done=True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        # trade = False
        # if((action == NewActions.Buy.value and abs(self._position) < self.budget) or 
        # (action == NewActions.Sell.value and abs(self._position) > -self.budget)):
        #     trade=True

        # if trade:
        #     if action == NewActions.Buy.value:
        #         self._position +=1
        #         self.average_price = (self.average_price + self._current_tick) / self._position
        #     else:
        #         self._position -=1

        observation, info = self._update_parameters(action)

        return observation, step_reward, self._done, info


    def _update_parameters(self,action):
        trade = False
        if((action == NewActions.Buy.value and self._position < self.budget) or (action == NewActions.Sell.value and self._position > -self.budget)):
            trade=True

        if trade:
            current_price = self.prices[self._current_tick]
            if(action == NewActions.Buy.value):
                self._position +=1
                if(self._position>0):
                    self.average_price = ((self.average_price*(self._position-1)) + current_price) / abs(self._position)
            else:
                self._position -=1
                if(self._position <0):
                    self.average_price = abs(((self.average_price*(self._position+1))) + current_price) / abs(self._position)

            if(self._position == 0):
                self.average_price = 0
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            position = self._position
        )

        self._update_history(info)

        return observation, info


    def _process_data(self):
        return _new_process_data(self.df,self.frame_bound,self.window_size)

    def _calculate_reward(self, action):
        '''
        if end has reached, profit/loss is calculated.

        if any action, immediate reward/penalty for that action is given to the agent

        buying at low price gives reward
        selling at high price gives reward

        buying at high price gives penalty
        selling at high prices gives penalty

        accumlating losses greater than budget gives big penalty 

        TODO: decrease lot_size if loss more than 50%

        attempting wrong/impossible actions gives penalty

        '''
        step_reward = 0
        
        trade = False
        
        if((action == NewActions.Buy.value and self._position < self.budget) or 
        (action == NewActions.Sell.value and self._position > -self.budget)):
            # print("Action:",action)
            # print("position:",self._position)
            # print("budget:",self.budget)
            trade=True
        
        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.average_price
            if(self.average_price == 0):
                last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price
            if(action == NewActions.Buy.value and self._position >= 0):
                ''' implies averaging/entering long position
                        if bought at a lower price than before then
                            reward 
                        else 
                            penalty
                '''
                step_reward += price_diff
            elif(action == NewActions.Sell.value and self._position <= 0):
                ''' implies averaging/entering short position
                        if bought at a lower price than before then
                            penalty
                        else
                            reward
                '''
                step_reward += -price_diff

            '''
            realised profit/loss is more valuable than unrealised
            profit/loss while squaring off x3 (booking profit/loss in partial quantity effective multiplier - x2)
            '''
            step_reward += self._calculate_profit(action) * 3

        else:
            '''
            Taken an action which did not result in a trade => wrong action => penalty (0.1% of current market price)

            hold action any given time has little penalty (.05% of current market price)
            '''
            if action == NewActions.Hold.value:
                step_reward += -self.prices[self._current_tick] * 0.001
            else:
                step_reward += -self.prices[self._current_tick] * 0.01
        
        return step_reward


    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick -1
        self._position = NewPositions.No_position.value
        self._position_history = (self.window_size * [0]) + [self._position]
        self._total_reward = 0
        self._total_profit = 1
        self._first_rendering = True
        self.history = {}
        self.average_price = 0
        return self._get_observation()

    def _calculate_profit(self, action):
        '''
        if any position is squared off, corresponding profit/Loss is calculated
        '''
        trade = False
        if((action == NewActions.Buy.value and abs(self._position) < self.budget) or 
        (action == NewActions.Sell.value and abs(self._position) > -self.budget)):
            trade=True
        profit = 0
        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.average_price
            if(last_trade_price ==0):
                last_trade_price = self.prices[self._last_trade_tick]
            if(action == NewActions.Buy.value and self._position<0):
                profit = last_trade_price - current_price
            elif(action == NewActions.Sell.value and self._position > 0):
                profit = current_price - last_trade_price
            elif(self._done and self._position >0):
                profit = (current_price - last_trade_price) * abs(self._position)
            elif(self._done and self._position < 0 ):
                profit = (last_trade_price - current_price) * abs(self._position)

        return profit

    def _update_profit(self, action):
        profit = self._calculate_profit(action)
        self._total_profit +=profit


    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == NewPositions.Short:
                color = 'red'
            elif position == NewPositions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)


    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] <= NewPositions.Short.value:
                short_ticks.append(tick)
            elif self._position_history[i] >= NewPositions.Long.value:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError


# prices, signal_features = _new_process_data(df=STOCKS_GOOGL, window_size=30, frame_bound=(30, len(STOCKS_GOOGL)))
# env = MyStocksEnv(budget=10, df=STOCKS_GOOGL, window_size=30, frame_bound=(30, len(STOCKS_GOOGL)))

# print(STOCKS_GOOGL)