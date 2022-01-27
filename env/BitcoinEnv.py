import numpy as np
import tensorflow as tf

from gym import Env
from gym.spaces import Box

import random

class BitcoinTradingEnv(Env):

    def __init__(self, 
                    price_history,
                    start_money=1000.0,
                    start_btc=0.0,
                    price_memory_len=5
                ):
        super(BitcoinTradingEnv, self).__init__()

        assert(price_memory_len >= 1)
        assert(start_btc >= 0.0)
        assert(start_money >= 0)

        self.money = 0
        self.start_money = start_money
        """how much money at the beginnig"""

        self.btc = 0
        self.start_btc = start_btc
        """how much btc at the beginnig"""

        self.memory = price_memory_len
        self.observation_space = Box(low=0.0, high=100000.0, dtype=np.float32, shape=(1, self.memory))
        """price over last <<self.memory>> entries"""

        self.last_observation = None

        self.action_space = Box(low=-1.0, high=1.0, dtype=np.float32, shape=(1,))
        """1 means buy as much as possible, -1 mean sell as much as possible"""
        
        # could've designed a lazy loader or related
        # but it's easier, safer, and faster to just map all data in memory
        # (it's not that big anyways)
        self.price_history = price_history
        """all prices in a 1D array"""

        self.current_moment = self.memory - 1
        """clock"""

        #self.reward_range DEFAULT

    def reset(self):
        
        self.money = self.start_money
        self.btc = self.start_btc

        self.current_moment = self.memory - 1
        self.last_observation = np.array(self.price_history[:self.memory])

        self.total_balance = self.start_money + self.start_btc * self.price_history[self.current_moment]

        return self.last_observation

    def seed(self, seed=None):
        
        if seed is not None:
            
            random.seed(seed)
            tf.random.set_seed(seed)
            np.random.seed(seed)
    
    # (observation, reward, done, info)
    def step(self, action):

        observation = None
        reward = None
        info = {}   # unused

        # if action > 0, buy btc with self.money * action
        # if action < 0, sell btc with self.btc * action

        current_price = self.price_history[self.current_moment]

        if action > 0:
            
            self.btc += (self.money * action) / current_price
            self.money *= (1 - action)

        elif action < 0:
            
            self.money += self.btc * (-action) * current_price
            self.btc *= (1 - action)

        self.current_moment += 1
        observation = np.insert(self.last_observation[1:], self.memory - 1, self.price_history[self.current_moment])

        new_total_balance = self.btc * self.price_history[self.current_moment] + self.money
        reward = (10 * (new_total_balance - self.total_balance) / self.total_balance) ** 2

        self.total_balance = new_total_balance
        
        if self.current_moment == len(self.price_history) - 1:
            return observation, reward, True, info

        return observation, reward, False, info

    def render(self, mode="human"):
        print("not implemented, sorry :P")

    def close(self):
        return None
