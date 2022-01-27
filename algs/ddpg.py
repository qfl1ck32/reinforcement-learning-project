# Author: Stanciu Andrei Calin
#
# design inspiration for classes and some default parameters taken from
# Unibuc - Introducere in Reinforcement Learning - lab 7
# 
# pseudocode and (some) documentation taken from 
# https://spinningup.openai.com/en/latest/algorithms/ddpg.html#deep-deterministic-policy-gradient

import numpy as np
from time import time
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.optimizers import *

import gym
from gym.spaces import * 

class BitcoinTradingEnv(gym.Env):

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

        return self.last_observation

    def seed(self, seed=None):
        pass
    
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
            
            self.money += self.btc * action * current_price
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

class Q(Model):

    def __init__(self, input_shape):
        super(Q, self).__init__()

        self.input_layer = InputLayer(input_shape = input_shape)

        self.hidden_layers = []
        self.hidden_layers.append(Dense(128), activation = "relu")

        self.output_layer = Dense(1, activation = "sigmoid")
    
    @tf.function
    def call(self, inputs):
        
        tmp = self.input_layer(inputs)

        for l in self.hidden_layers:
            tmp = l(tmp)

        return self.output_layer(tmp)

class Policy(Model):

    def __init__(self, input_shape):
        super(Policy, self).__init__()

        self.input_layer = InputLayer(input_shape = input_shape)

        self.hidden_layers = []
        self.hidden_layers.append(Dense(128), activation = "relu")

        self.output_layer = Dense(1, activation = "sigmoid")

    @tf.function
    def call(self, inputs):
        
        tmp = self.input_layer(inputs)

        for l in self.hidden_layers:
            tmp = l(tmp)

        return self.output_layer(tmp)

    def get_epsilon(self):
        pass
    
class DDPG_agent():

    def __init__(self,
                    data,
                    seed = None,
                    replay_buffer_len = 32000,
                    discount = 0.9,
                    batch_size = 1024,
                    q_lr = 0.001,
                    policy_lr = 0.001,
                    polyak = 0.9,
                    steps_until_sync = 10,
                    state_size = 5,
                    start_money = 1000,
                    start_btc = 1000
                    ):
        
        self.env = BitcoinTradingEnv(data, start_money, start_btc, state_size)

        if seed is not None:
            
            random.seed(seed)
            self.env.seed(seed)
            tf.random.set_seed(seed)
            np.random.seed(seed)

        self.q = Q(input_shape = (1 + state_size,))
        self.q_target = Q(input_shape = (1 + state_size,))

        self.policy = Policy(input_shape = (state_size,)) 
        self.policy_target = Policy(input_shape = (state_size,)) 

        self.q.build()
        self.q_target.build()

        self.policy.build()
        self.policy_target.build()
        
        self.replay_buffer = []
        self.replay_buffer_len = replay_buffer_len
        
        self.discount = discount
        self.polyak = polyak

        self.q_lr = q_lr
        self.policy_lr = policy_lr
        self.batch_size = batch_size

        self.steps_until_sync = steps_until_sync

    def get_action(self, state):
        pass

    def run_episode(self):
        
        state = self.env.reset()

        self.q_target.set_weights(self.q.get_weights())
        self.policy_target.set_weights(self.policy.get_weights())

        i = 1
        while True:

            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)

            if done:
                return self.env.total_balance

            self.replay_buffer.append((state, action, reward, next_state, 1 if done else 0))
            
            if i % self.steps_until_sync == 0:
                
                samples = random.sample(self.replay_buffer, self.replay_buffer_len)
                for sample in samples:

                    s, a, r, s_, d = sample

                    target_q_input = np.insert(s_, len(s_) - 1, self.policy_target(s_))
                    train_q_input = np.insert(s, len(s) - 1, a)

                    q_gt = r + self.discount * (1 - d) * self.q_target(target_q_input)
                    q_pred = self.q(train_q_input)

                    critic_loss = (q_gt - q_pred) ** 2
                    actor_loss = self.q(np.insert(s, len(s) - 1, self.policy(s)))

                    # TODO gradient descent 

                    q_target_w = self.q_target.get_weights()
                    q_train_w = self.q.get_weights()

                    q_target_w = self.polyak * q_target_w + (1 - self.polyak) * q_train_w
                    self.q_target.set_weights(q_target_w)

                    policy_target_w = self.policy_target.get_weights()
                    policy_train_w = self.policy.get_weights()

                    policy_target_w = self.polyak * policy_target_w + (1 - self.polyak) * policy_train_w
                    self.policy_target_w.set_weights(policy_target_w)                    

    def train(self, episodes = 1, save_model = True):
        
        for ep_idx in range(episodes):
            
            balance = self.run_episode()
            print(f"balance after episode {ep_idx}: {balance}")

        if save_model is True:

            tag = int(time())

            self.q_target.save(f"q_model_{tag}")
            self.policy_target.save(f"q_model_{tag}")

def run(data):
    """entry point"""   

    def _check_gap_frequency(data):

        fr = [0 for _ in range(1000000)]

        for i in range(data.shape[0] - 1):

            dif = data[i + 1][0] - data[i][0]

            if dif > 1:
                fr[int(dif)] += 1

        for i in range(len(fr)):
            if fr[i] > 0:
                print(f"{i}: {fr[i]}")

    #_check_gap_frequency(data)

    data_ = []
    for i in range(data.shape[0]):
        data_.append((data[i][2] + data[i][3]) / 2)

    data = np.array(data_)

    agent = DDPG_agent(data)
    agent.train()
