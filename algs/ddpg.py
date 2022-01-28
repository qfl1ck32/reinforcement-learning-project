# Author: Stanciu Andrei Calin
# Team: Rusu Andrei, Mitoi Stefan, (me)
#
# design inspiration for classes and some default parameters taken from
# Unibuc - Introducere in Reinforcement Learning - lab 7
# 
# pseudocode and (some) documentation taken from 
# https://spinningup.openai.com/en/latest/algorithms/ddpg.html#deep-deterministic-policy-gradient

import numpy as np

from logger.logger import logger

from time import time
from copy import deepcopy

import random
from numpy.random import normal

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *

from env.BitcoinEnv import BitcoinTradingEnv

DEBUG = True

class Q(Model):

    def __init__(self, input_shape):
        super(Q, self).__init__()

        self.input_layer = InputLayer(input_shape = input_shape)

        self.hidden_layers = []
        self.hidden_layers.append(Dense(128, activation = "relu"))
        self.hidden_layers.append(Dense(32, activation = "relu"))

        self.output_layer = Dense(1, activation = "linear")
    
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
        self.hidden_layers.append(Dense(128, activation = "relu"))
        self.hidden_layers.append(Dense(32, activation = "relu"))

        self.output_layer = Dense(1, activation = "tanh")

    @tf.function
    def call(self, inputs):

        tmp = self.input_layer(inputs)

        for l in self.hidden_layers:
            tmp = l(tmp)

        return self.output_layer(tmp)
    
class DDPG_agent():

    def __init__(self,
                    data,
                    seed = None,
                    episode_len = 10000,
                    noise_std = 0.1,
                    replay_buffer_len = 1024 * 16,
                    discount = 0.999,
                    batch_size = 128,
                    q_lr = 0.001,
                    policy_lr = 0.0001,
                    q_momentum = 0.9,
                    policy_momentum = 0.9,
                    polyak = 0.9,
                    steps_until_sync = 10,
                    state_size = 5,
                    start_money = 1000,
                    start_btc = 0.1,
                    relative_reward = True
                    ):

        self.env = BitcoinTradingEnv(data, start_money, start_btc, state_size, episode_len, relative_reward)
        self.env.seed(seed)

        self.episode_len = episode_len

        self.q = Q(input_shape = (1 + state_size,))
        self.q_target = Q(input_shape = (1 + state_size,))

        self.policy = Policy(input_shape = (state_size,))
        self.policy_target = Policy(input_shape = (state_size,))

        q_input_shape = tf.TensorShape([None, 1 + state_size])
        policy_input_shape = tf.TensorShape([None, state_size])

        self.q.build(input_shape = q_input_shape)
        self.q_target.build(input_shape = q_input_shape)

        self.policy.build(input_shape = policy_input_shape)
        self.policy_target.build(input_shape = policy_input_shape)
        
        self.replay_buffer = []
        self.replay_buffer_len = replay_buffer_len

        self.discount = discount
        self.polyak = polyak

        self.q_lr = q_lr
        self.policy_lr = policy_lr
        self.q_momentum = q_momentum
        self.policy_momentum = policy_momentum

        self.batch_size = batch_size

        self.steps_until_sync = steps_until_sync

        self.noise_std = noise_std

        self.q_optimizer = SGD(self.q_lr, self.q_momentum)
        self.policy_optimizer = SGD(self.policy_lr, self.policy_momentum)

    def get_action(self, state):

        if self.env.money < 0.1 or self.env.btc < 0.001:
            self.noise_std = 0.5
        
        policy_action = self.policy(np.array([state]))[0]
        return min(max(policy_action + normal(0, self.noise_std), np.array([-1])), np.array([1]))

    def run_episode(self):

        state = self.env.reset()

        self.q.set_weights(self.q_target.get_weights())
        self.policy.set_weights(self.policy_target.get_weights())

        step_idx = 0
        while True:

            if DEBUG:
                if step_idx % 100 == 0:
                    print(f"[Step {step_idx}] total balance {self.env.total_balance}, " +\
                            f"money {self.env.money}, btc {self.env.btc}, " + \
                            f"money/btc {self.env.price_history[self.env.current_moment]}")

            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)

            if done or step_idx >= self.episode_len:
                return self.env.total_balance

            self.replay_buffer.append((state, action, reward, next_state))
            
            if step_idx % self.steps_until_sync == 0 and len(self.replay_buffer) >= self.batch_size:

                samples = random.sample(self.replay_buffer[-self.replay_buffer_len:], self.batch_size)

                s_batch = tf.stack([sample[0] for sample in samples], axis=0)
                a_batch = tf.stack([sample[1] for sample in samples], axis=0)
                r_batch = tf.stack([sample[2] for sample in samples], axis=0)
                s_next_batch = tf.stack([sample[3] for sample in samples], axis=0)

                # train Q network

                with tf.GradientTape() as q_tape:

                    q_vars = self.q.trainable_variables
                    q_tape.watch(q_vars)

                    q_pred = self.q(tf.concat([s_batch, a_batch], axis=1))

                    a_next_batch = self.policy_target(s_next_batch)
                    q_gt = r_batch + self.discount * self.q_target(tf.concat([s_next_batch, a_next_batch], axis=1))

                    q_loss = (q_pred - q_gt) ** 2

                q_gradients = q_tape.gradient(q_loss, q_vars)
                self.q_optimizer.apply_gradients(zip(q_gradients, q_vars)) 

                # train Policy network

                with tf.GradientTape() as policy_tape:

                    policy_vars = self.policy.trainable_variables
                    policy_tape.watch(policy_vars)

                    a_pred_batch = self.policy(s_batch)
                    
                    policy_loss = -self.q(tf.concat([s_batch, a_pred_batch], axis=1))

                policy_gradients = policy_tape.gradient(policy_loss, policy_vars)
                self.policy_optimizer.apply_gradients(zip(policy_gradients, policy_vars))

                # polyak averaging

                q_target_w = self.q_target.get_weights()
                q_train_w = self.q.get_weights()

                assert(len(q_target_w) == len(q_train_w))

                for i in range(len(q_target_w)):
                    q_target_w[i] = self.polyak * q_target_w[i] + (1 - self.polyak) * q_train_w[i]

                self.q_target.set_weights(q_target_w)

                policy_target_w = self.policy_target.get_weights()
                policy_train_w = self.policy.get_weights()

                assert(len(policy_target_w) == len(policy_train_w))

                for i in range(len(policy_target_w)):
                    policy_target_w[i] = self.polyak * policy_target_w[i] + (1 - self.polyak) * policy_train_w[i]

                self.policy_target.set_weights(policy_target_w)    

            state = next_state
            step_idx += 1

    def train(self, episodes = 10, save_model = True):

        for ep_idx in range(episodes):

            balance = self.run_episode()
            logger.info(f"Balance after episode {ep_idx}: {balance}")

            if save_model is True:

                tag = int(time())

                self.q_target.save_weights(f"q_model_ep{ep_idx}_{tag}")
                self.policy_target.save_weights(f"policy_model_ep{ep_idx}_{tag}")

    def test(self, data):
        
        backup_ep_len = self.env.episode_len

        self.env.episode_len = len(data) - 10 # -10 as a precaution for improper indexes
        state = self.env.reset()

        step_idx = 0
        while True:

            if DEBUG:
                if step_idx % 100 == 0:
                    print(f"[Step {step_idx}] total balance {self.env.total_balance}, " +\
                            f"money {self.env.money}, btc {self.env.btc}, " + \
                            f"money/btc {self.env.price_history[self.env.current_moment]}")

            action = self.get_action(state)
            next_state, _, done, _ = self.env.step(action)

            if done or step_idx >= self.episode_len:

                self.env.episode_len = backup_ep_len
                return self.env.total_balance
            
            state = next_state
            step_idx += 1

    @staticmethod
    def gridsearch(data):
        
        f = open("gridsearch.txt", "w+")

        hyperparam_val =    {
                            "episode_len": [5000],
                            "noise_std": [0.05],
                            "replay_buffer_len": [1024 * 6],
                            "discount": [0.9997],
                            "batch_size": [256, 1024],
                            "q_lr": [0.001],
                            "policy_lr": [0.0001],
                            "q_momentum": [0.9],
                            "policy_momentum": [0.9],
                            "polyak": [0.8],
                            "steps_until_sync": [20],
                            "state_size": [3, 5, 10],
                            "start_money": [10000],
                            "start_btc": [0.1],
                            "relative_reward": [True, False]
                            }
        hyperparam_names = [name for name in hyperparam_val.keys()]

        def _get_hyperparam_seq(params):

            if len(params) == 0:
                yield {}
            
            else:

                for val in hyperparam_val[params[0]]:
                    for seq in _get_hyperparam_seq(params[1:]):

                        to_yield = {params[0]: val} 
                        to_yield.update(seq.copy())

                        yield to_yield

        for hyperparams in _get_hyperparam_seq(hyperparam_names):

            try:
            
                data_ = deepcopy(data)

                agent = DDPG_agent(data_, **hyperparams)
                agent.train(save_model = False)

                final_balance = agent.test(data_)

                f.write(f"parameters {hyperparams}, final_balance: {final_balance}")
                f.flush()

                agent.q_target.save_weights(f"q_model_balance_{int(final_balance)}")
                agent.policy_target.save_weights(f"policy_model_balance_{int(final_balance)}")

            except Exception as err:
                f.write(f"parameters {hyperparams}, ERROR: {err, err.args}")

        f.flush()
        f.close()

def run(data):
    """entry point"""

    def _check_gap_frequency(data):

        logger.info("gap length histogram...")

        fr = [0 for _ in range(100000)]

        for i in range(data.shape[0] - 1):

            dif = data[i + 1][0] - data[i][0]

            if dif > 60.0:
                fr[int(dif / 60)] += 1

        for i in range(len(fr)):
            if fr[i] > 0:
                logger.info(f"{i}: {fr[i]}")

    def _check_cont_frequency(data):

        logger.info("contigous region length histogram...")

        fr = [0 for _ in range(1000000)]

        wsize = 0
        for i in range(data.shape[0] - 1):

            dif = data[i + 1][0] - data[i][0]

            if dif > 60.0:
                fr[wsize] += 1
                wsize = 0

            else:
                wsize += 1

        fr[wsize] += 1

        for i in range(len(fr)):
            if fr[i] > 0:
                logger.info(f"{i}: {fr[i]}")

    #_check_gap_frequency(data)
    #_check_cont_frequency(data)

    # TODO remove after debug
    tf.debugging.enable_check_numerics()

    data_ = []
    for i in range(data.shape[0]):
        data_.append((data[i][2] + data[i][3]) / 2)

    data = np.array(data_)

    #DDPG_agent.gridsearch(data)

    agent = DDPG_agent(data, 
                        seed = 0,
                        episode_len = 5000,
                        noise_std = 0.1,
                        replay_buffer_len = 1024 * 4,
                        discount = 0.9997,
                        batch_size = 256,
                        q_lr = 0.001,
                        policy_lr = 0.0001,
                        q_momentum = 0.9,
                        policy_momentum = 0.9,
                        polyak = 0.8,
                        steps_until_sync = 20,
                        state_size = 5,
                        start_money = 10000,
                        start_btc = 0.1,
                        relative_reward = False
                    )
    agent.train()
