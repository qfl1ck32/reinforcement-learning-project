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
from numpy.random import normal

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *

from env.BitcoinEnv import BitcoinTradingEnv


class Q(Model):

    def __init__(self, input_shape):
        super(Q, self).__init__()

        self.input_layer = InputLayer(input_shape = input_shape)

        self.hidden_layers = []
        self.hidden_layers.append(Dense(128, activation = "relu"))

        self.output_layer = Dense(1, activation = "linear")
    
    @tf.function
    def call(self, inputs):

        tmp = self.input_layer(inputs)

        for l in self.hidden_layers:
            tmp = l(tmp)

        return self.output_layer(tmp)[0] # shape (1, 1) => (1,)

class Policy(Model):

    def __init__(self, input_shape):
        super(Policy, self).__init__()

        self.input_layer = InputLayer(input_shape = input_shape)

        self.hidden_layers = []
        self.hidden_layers.append(Dense(128, activation = "relu"))

        self.output_layer = Dense(1, activation = "tanh")

    @tf.function
    def call(self, inputs):

        tmp = self.input_layer(inputs)

        for l in self.hidden_layers:
            tmp = l(tmp)

        return self.output_layer(tmp)[0] # shape (1, 1) => (1,)
    
class DDPG_agent():

    def __init__(self,
                    data,
                    seed = None,
                    noise_std = 0.1,
                    replay_buffer_len = 1024 * 16,
                    discount = 0.98,
                    batch_size = 128,
                    q_lr = 0.001,
                    policy_lr = 0.0001,
                    q_momentum = 0.9,
                    policy_momentum = 0.9,
                    polyak = 0.9,
                    steps_until_sync = 10,
                    state_size = 5,
                    start_money = 1000,
                    start_btc = 1000
                    ):

        self.env = BitcoinTradingEnv(data, start_money, start_btc, state_size)
        self.env.seed(seed)

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
        
        policy_action = self.policy(np.array([state]))
        return min(max(policy_action + normal(0, self.noise_std), np.array([-1])), np.array([1]))

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
            
            if i % self.steps_until_sync == 0 and len(self.replay_buffer) >= self.batch_size:

                samples = random.sample(self.replay_buffer[-self.replay_buffer_len:], self.batch_size)

                # TODO remove after debug
                tf.debugging.enable_check_numerics()

                # s, a, r, s_, d = sample[0], sample[1], sample[2], sample[3], sample[4]

                s_batch = tf.stack([sample[0] for sample in samples], axis=0)
                a_batch = tf.stack([sample[1] for sample in samples], axis=0)
                r_batch = tf.stack([sample[2] for sample in samples], axis=0)
                s_next_batch = tf.stack([sample[3] for sample in samples], axis=0)

                print(a_batch.shape)
                print(r_batch.shape)

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

                print(f"OK {i}")

                # train Policy network

                # old
                '''for sample in samples:

                    s, a, r, s_, d = sample

                    # train (gradient descent) 

                    tf.debugging.enable_check_numerics()

                    with tf.GradientTape(watch_accessed_variables = True) as q_tape:

                        q_vars = self.q.trainable_variables
                        q_tape.watch(q_vars)

                        target_q_input = np.insert(s_, len(s_) - 1, self.policy_target(np.array([s_])))
                        train_q_input = np.insert(s, len(s) - 1, a)

                        q_gt = r + self.discount * (1 - d) * self.q_target(np.array([target_q_input]))
                        q_pred = self.q(np.array([train_q_input]))

                        q_loss = (q_gt - q_pred) ** 2

                    q_gradients = q_tape.gradient(q_loss, q_vars)
                    self.q_optimizer.apply_gradients(zip(q_gradients, q_vars))

                    print(q_gradients)
                    print("OK HERE =====================================================================")

                    # TODO adjust watch_accessed_variables ???
                    with tf.GradientTape(watch_accessed_variables = True) as policy_tape:

                        policy_vars = self.policy.trainable_variables
                        policy_tape.watch(policy_vars)

                        policy_input = np.insert(s, len(s) - 1, self.policy(np.array([s])))

                        policy_loss = -self.q(np.array([policy_input]))

                    policy_gradients = policy_tape.gradient(policy_loss, policy_vars)

                    print(policy_gradients)

                    self.policy_optimizer.apply_gradients(zip(policy_gradients, policy_vars))

                    # polyak averaging

                    q_target_w = self.q_target.get_weights()
                    q_train_w = self.q.get_weights()

                    q_target_w = self.polyak * q_target_w + (1 - self.polyak) * q_train_w
                    self.q_target.set_weights(q_target_w)

                    policy_target_w = self.policy_target.get_weights()
                    policy_train_w = self.policy.get_weights()

                    policy_target_w = self.polyak * policy_target_w + (1 - self.polyak) * policy_train_w
                    self.policy_target_w.set_weights(policy_target_w)        '''        

                q_target_w = self.q_target.get_weights()
                q_train_w = self.q.get_weights()

                q_target_w = self.polyak * q_target_w + (1 - self.polyak) * q_train_w
                self.q_target.set_weights(q_target_w)

                policy_target_w = self.policy_target.get_weights()
                policy_train_w = self.policy.get_weights()

                policy_target_w = self.polyak * policy_target_w + (1 - self.polyak) * policy_train_w
                self.policy_target_w.set_weights(policy_target_w)    

            i += 1

    def train(self, episodes = 1, save_model = True):

        for ep_idx in range(episodes):

            balance = self.run_episode()
            print(f"balance after episode {ep_idx}: {balance}")

        if save_model is True:

            tag = int(time())

            self.q_target.save(f"q_model_{tag}")
            self.policy_target.save(f"policy_model_{tag}")

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

    agent = DDPG_agent(data, seed=0)
    agent.train()
