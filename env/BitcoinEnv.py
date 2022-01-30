import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

from gym import Env
from gym.spaces import Box

import random

from logger import logger

class BitcoinTradingEnv(Env):

    MIN_BALANCE = 100
    """Threshold for stopping an episode"""

    def __init__(self, 
                    price_history,
                    start_money = 1000.0,
                    start_btc = 0.0,
                    memory = 5,
                    episode_len = 5000,
                    use_snd_deriv = True,
                    stats4render = True,
                    control = True
                ):
        super(BitcoinTradingEnv, self).__init__()

        assert(memory >= 1)
        assert(start_btc >= 0.0)
        assert(start_money >= 0)
        assert(episode_len > memory)

        self.stats4render = stats4render
        """Must be true to be able to call .render(),
            but it consumes more memory"""

        self.control = control
        """Reserve fields for control stats\n
            (i.e. for each step, another random step is taken
            and the balance is updated in parralel;\n
            at the end, control stats ar plotted for comparison)"""

        if self.stats4render:

            self.balance_money_btc = []
            """List of tuples (total balance, money, btc)"""

            if self.control:
                self.balance_money_btc_control = []

        self.steps_todo = episode_len
        self.episode_len = episode_len
        """One episode length, in data entries"""

        self.money = 0
        self.start_money = start_money
        """how much money at the beginnig"""

        self.btc = 0
        self.start_btc = start_btc
        """how much btc at the beginnig"""

        if self.control:
            
            self.money_control = 0
            self.btc_control = 0
            self.total_balance_control = 0

        self.memory = memory
        self.observation_space = Box(low = 0.0, high = 100000.0, dtype = np.float32, shape = (1, self.memory))
        """price over last <<self.memory>> entries"""

        self.last_observation = None
        """last observation state"""

        self.action_space = Box(low = -1.0, high = 1.0, dtype = np.float32, shape = (1,))
        """between [-1, 1]\n
            eg. +0.7 means spend 70% of money that I have right now to buy btc\n
                -0.5 means sell 50% of the btc I currently have"""
        
        # could've designed a lazy loader or related
        # but it's easier, safer, and faster to just map all data in memory
        # (it's not that big anyways)

        self.price_history = price_history.astype(np.float32)
        """all prices in a 1D array"""

        self.use_snd_deriv = use_snd_deriv
        """whether to use second derivative instead of the first
            for the observation (state) space"""

        self.price_history_deriv = np.zeros(shape = (price_history.shape[0] - 1,))
        """prices derivative wrt time"""

        self.price_history_snd_deriv = np.zeros(shape = (price_history.shape[0] - 2,))
        """prices second derivative wrt time"""

        for i in range(len(self.price_history_deriv)):
            self.price_history_deriv[i] = self.price_history[i + 1] / self.price_history[i]

        self.price_history_deriv = self.price_history_deriv.astype(np.float32)

        for i in range(len(self.price_history_snd_deriv)):
            self.price_history_snd_deriv[i] = self.price_history_deriv[i + 1] / self.price_history_deriv[i]

        self.price_history_snd_deriv = self.price_history_snd_deriv.astype(np.float32)

        self.current_moment = self.memory
        """clock, starting from 0\n
            eg. self.memory = 3 =>
            current moment = 3,\n (fst deriv) last observation = [p1 / p0, p2 / p1, p3 / p2]\n
            (snd deriv) last observation = [d0 / d-1, d1 / d0, d2 / d1]\n"""

    def reset(self):
        
        self.money = self.start_money
        self.btc = self.start_btc

        assert(self.memory + 1 < len(self.price_history) - self.episode_len)
        self.current_moment = random.choice(range(self.memory + 1, len(self.price_history) - self.episode_len))
        
        if self.use_snd_deriv:
            self.last_observation = self.price_history_snd_deriv[self.current_moment - self.memory: self.current_moment]
        else:
            self.last_observation = self.price_history_deriv[self.current_moment - self.memory - 1: self.current_moment - 1]

        self.total_balance = self.start_money + self.start_btc * self.price_history[self.current_moment]
        self.steps_todo = self.episode_len

        if self.control:

            self.money_control = self.start_money
            self.btc_control = self.start_btc
            self.total_balance_control = self.start_money + self.start_btc * self.price_history[self.current_moment]

        if self.stats4render:
            self.balance_money_btc = [(self.total_balance, self.money, self.btc)]

            if self.control:
                self.balance_money_btc_control = [(self.total_balance_control, self.money_control, self.btc_control)]

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
        # if action < 0, sell btc with self.btc * (-action)

        current_price = self.price_history[self.current_moment]

        if action > 0:
            
            self.btc += (self.money * action) / current_price
            self.money *= (1 - action)

        elif action < 0:
            
            self.money += self.btc * (-action) * current_price
            self.btc *= 1 + action

        if self.total_balance < BitcoinTradingEnv.MIN_BALANCE:
            return observation, -10000000000, False, info

        self.current_moment += 1

        if self.use_snd_deriv:
            observation = np.insert(self.last_observation[1:], self.memory - 1, self.price_history_snd_deriv[self.current_moment - 2])
        else:
            observation = np.insert(self.last_observation[1:], self.memory - 1, self.price_history_deriv[self.current_moment - 1])

        new_total_balance = self.btc * self.price_history[self.current_moment] + self.money
        reward = np.log(new_total_balance / self.total_balance)

        self.total_balance = new_total_balance

        if self.control:
            """re-calculate everything for the control statistics"""

            random_action = random.random() * 2 - 1

            if random_action > 0:
            
                self.btc_control += (self.money_control * random_action) / current_price
                self.money_control *= (1 - random_action)

            elif random_action < 0:
                
                self.money_control += self.btc_control * (-random_action) * current_price
                self.btc_control *= 1 + random_action

            self.total_balance_control = self.btc_control * self.price_history[self.current_moment] + self.money_control

        if self.stats4render:

            self.balance_money_btc.append((self.total_balance, self.money, self.btc))

            if self.control:
                self.balance_money_btc_control.append((self.total_balance_control, self.money_control, self.btc_control))

        self.steps_todo -= 1
        if (self.steps_todo == 0) or (self.current_moment == len(self.price_history) - 1):
            return observation, reward, True, info

        return observation, reward, False, info

    def render(self, mode="human"):
        """Should contain:\n
            * A graph plotting all prices in the episode_len range\n
            * A graph plotting total balance, and subgraphs for btc and money\n"""

        if mode != "human":
            logger.warn("not implemented, sorry :P")
            return

        if self.stats4render is False:
            logger.warn("Must have stats4render set to true at the construction of the env to be able to show stats!")
            return 

        start_moment = self.current_moment + self.steps_todo - self.episode_len
        end_moment = self.current_moment

        if self.control:
            
            fig, (price_time, balance_time, balance_time_control) = plt.subplots(3)
            fig.subplots_adjust(hspace=0.6)

            price_time.plot(range(end_moment - start_moment + 1),
                            self.price_history[start_moment: end_moment + 1],
                            color = 'gold')

            price_time.set_title("Price evolution")
            price_time.set_xlabel("time")
            price_time.set_ylabel("$ / BTC")
            price_time.grid(True)

            balance_stats = {"Currency": [float(t[1]) for t in self.balance_money_btc],
                            "BTC (currency equivalent)": [float(t[0] - t[1]) for t in self.balance_money_btc]}
            

            balance_time.stackplot(range(end_moment - start_moment + 1),
                                    balance_stats.values(),
                                    labels = balance_stats.keys(),
                                    alpha = 0.8)

            balance_time.legend(loc = "upper left")
            balance_time.set_title("Balance evolution")
            balance_time.set_xlabel("time")
            balance_time.set_ylabel("$")
            balance_time.grid(True)

            balance_stats_control = {"Currency": [float(t[1]) for t in self.balance_money_btc_control],
                                    "BTC (currency equivalent)": [float(t[0] - t[1]) for t in self.balance_money_btc_control]}
            

            balance_time_control.stackplot(range(end_moment - start_moment + 1),
                                    balance_stats_control.values(),
                                    labels = balance_stats_control.keys(),
                                    alpha = 0.8)

            balance_time_control.legend(loc = "upper left")
            balance_time_control.set_title("Balance evolution (control)")
            balance_time_control.set_xlabel("time")
            balance_time_control.set_ylabel("$")
            balance_time_control.grid(True)

            plt.show()

        else:

            fig, (price_time, balance_time) = plt.subplots(2)
            fig.subplots_adjust(hspace=0.6)

            price_time.plot(range(end_moment - start_moment + 1),
                            self.price_history[start_moment: end_moment + 1],
                            color = 'gold')

            price_time.set_title("Price evolution")
            price_time.set_xlabel("time")
            price_time.set_ylabel("$ / BTC")
            price_time.grid(True)

            balance_stats = {"Currency": [float(t[1]) for t in self.balance_money_btc],
                            "BTC (currency equivalent)": [float(t[0] - t[1]) for t in self.balance_money_btc]}
            

            balance_time.stackplot(range(end_moment - start_moment + 1),
                                    balance_stats.values(),
                                    labels = balance_stats.keys(),
                                    alpha = 0.8)

            balance_time.legend(loc = "upper left")
            balance_time.set_title("Balance evolution")
            balance_time.set_xlabel("time")
            balance_time.set_ylabel("$")
            balance_time.grid(True)

            plt.show()

    def close(self):
        return None
