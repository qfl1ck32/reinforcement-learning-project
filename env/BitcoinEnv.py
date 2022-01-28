import numpy as np
import tensorflow as tf

from gym import Env
from gym.spaces import Box

import random

class BitcoinTradingEnv(Env):

    def __init__(self, 
                    price_history,
                    start_money = 1000.0,
                    start_btc = 0.0,
                    memory = 5,
                    episode_len = 5000
                ):
        super(BitcoinTradingEnv, self).__init__()

        assert(memory >= 1)
        assert(start_btc >= 0.0)
        assert(start_money >= 0)

        self.episode_len = episode_len
        self.steps_todo = self.episode_len

        self.money = 0
        self.start_money = start_money
        """how much money at the beginnig"""

        self.btc = 0
        self.start_btc = start_btc
        """how much btc at the beginnig"""

        self.memory = memory
        self.observation_space = Box(low=0.0, high=100000.0, dtype=np.float32, shape=(1, self.memory))
        """price over last <<self.memory>> entries"""

        self.last_observation = None

        self.action_space = Box(low=-1.0, high=1.0, dtype=np.float32, shape=(1,))
        """1 means buy as much as possible, -1 mean sell as much as possible"""
        
        # could've designed a lazy loader or related
        # but it's easier, safer, and faster to just map all data in memory
        # (it's not that big anyways)

        self.price_history = price_history.astype(np.float32)
        """all prices in a 1D array"""

        self.price_history_deriv = np.zeros(shape=(price_history.shape[0] - 1,))
        """prices derivative wrt time"""

        for i in range(len(self.price_history_deriv)):
            self.price_history_deriv[i] = self.price_history[i + 1] / self.price_history[i]

        self.price_history_deriv = self.price_history_deriv.astype(np.float32)

        self.current_moment = self.memory
        """clock, starting from 0\n
            eg. self.memory = 3 =>\n
            current moment = 3, (initial) last observation = [p1 / p0, p2 / p1, p3 / p2]"""

    def reset(self):
        
        self.money = self.start_money
        self.btc = self.start_btc

        self.current_moment = random.choice(range(self.memory - 1, len(self.price_history) - self.episode_len))
        self.last_observation = self.price_history_deriv[self.current_moment - self.memory: self.current_moment]

        self.total_balance = self.start_money + self.start_btc * self.price_history[self.current_moment]
        self.steps_todo = self.episode_len

        return self.last_observation

    def seed(self, seed=None):
        
        if seed is not None:
            
            random.seed(seed)
            tf.random.set_seed(seed)
            np.random.seed(seed)
    
    # (observation, reward, done, info)
    def step(self, action):

        def _compute_reward(r0, r1, r):

            if -1 <= (r0 - r1) <= 1:
                return 2 * r

            r_med = (r0 + r1) / 2
            return r * (1 + (r - r_med) / ((r0 - r_med) if (r0 > r_med) else (r_med - r0)))

        observation = None
        reward = None
        info = {}   # unused

        # if action > 0, buy btc with self.money * action
        # if action < 0, sell btc with self.btc * (-action)

        current_price = self.price_history[self.current_moment]

        r_sell_all = np.log((self.money + self.btc * self.price_history[self.current_moment + 1]) / self.total_balance)
        r_buy_all = np.log(((self.btc + self.money / current_price) * self.price_history[self.current_moment + 1]) / self.total_balance)

        if action > 0:
            
            self.btc += (self.money * action) / current_price
            self.money *= (1 - action)

        elif action < 0:
            
            self.money += self.btc * (-action) * current_price
            self.btc *= 1 + action

        self.current_moment += 1
        observation = np.insert(self.last_observation[1:], self.memory - 1, self.price_history_deriv[self.current_moment - 1])

        new_total_balance = self.btc * self.price_history[self.current_moment] + self.money

        reward = np.log(new_total_balance / self.total_balance)
        reward = _compute_reward(r_sell_all, r_buy_all, reward)   

        self.total_balance = new_total_balance
        
        self.steps_todo -= 1
        if (self.steps_todo == 0) or (self.current_moment == len(self.price_history) - 1):
            return observation, reward, True, info

        return observation, reward, False, info

    def render(self, mode="human"):
        print("not implemented, sorry :P")

    def close(self):
        return None
