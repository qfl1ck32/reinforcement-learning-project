from abc import ABC
from time import time
import numpy as np
import tensorflow as tf
import random

from keras import Model
from keras.optimizer_v2.gradient_descent import SGD
from numpy import stack, argmax, linspace
from numpy.random import normal
from tensorflow import expand_dims, TensorShape, GradientTape, one_hot, reduce_sum
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow import function as tf_function

from env.BitcoinEnv import BitcoinTradingEnv
from logger.logger import logger

DEBUG = True


class Q(Model, ABC):

    def __init__(self, input_shape, num_actions):
        super(Q, self).__init__()

        self.input_layer = InputLayer(input_shape=input_shape)

        self.hidden_layers = []
        self.hidden_layers.append(Dense(64, activation="relu"))
        self.hidden_layers.append(Dense(32, activation="relu"))

        self.output_layer = Dense(units=num_actions, activation="linear")

    @tf_function
    def call(self, inputs):
        tmp = self.input_layer(inputs)

        for layer in self.hidden_layers:
            tmp = layer(tmp)

        return self.output_layer(tmp)


class LinearScheduleEpsilon():
    def __init__(self, start_eps=1.0, final_eps=0.1,
                 pre_train_steps=10, final_eps_step=10000):
        self.start_eps = start_eps
        self.final_eps = final_eps
        self.pre_train_steps = pre_train_steps
        self.final_eps_step = final_eps_step
        self.decay_per_step = (self.start_eps - self.final_eps) \
                              / (self.final_eps_step - self.pre_train_steps)

    def get_value(self, step):
        if step <= self.pre_train_steps:
            return 1.0  # full exploration in the beginning
        else:
            epsilon = (1.0 - self.decay_per_step * (step - self.pre_train_steps))
            epsilon = max(self.final_eps, epsilon)
            return epsilon


class DQN:
    def __init__(self,
                 data,
                 seed=None,
                 episode_len=10000,
                 replay_buffer_len=1024 * 32,
                 discount=0.9,
                 batch_size=1024,
                 q_lr=0.001,
                 q_momentum=0.9,
                 steps_until_sync=10,
                 choose_action_frequency=25,
                 pre_train_steps=1,
                 train_frequency=1,
                 state_size=7,
                 action_size=7,
                 start_money=1000,
                 start_btc=0.1,
                 stats4render=True):

        self.env = BitcoinTradingEnv(data,
                                     start_money=start_money,
                                     start_btc=start_btc,
                                     memory=state_size,
                                     episode_len=episode_len,
                                     stats4render=stats4render,
                                     use_snd_deriv=False)
        self.env.seed(seed)

        self.episode_len = episode_len
        self.choose_action_frequency = choose_action_frequency
        self.pre_train_steps = pre_train_steps
        self.train_frequency = train_frequency

        assert (action_size > 2)
        self.actions = linspace(1, -1, num=action_size)

        self.num_actions = len(self.actions)
        self.q = Q(input_shape=(1 + state_size,),
                   num_actions=self.num_actions)
        self.q_target = Q(input_shape=(1 + state_size,),
                          num_actions=self.num_actions)

        q_input_shape = TensorShape([batch_size, state_size])

        self.q.build(input_shape=q_input_shape)
        self.q_target.build(input_shape=q_input_shape)

        self.replay_buffer = []
        self.replay_buffer_len = replay_buffer_len

        self.discount = discount

        self.q_lr = q_lr
        self.q_momentum = q_momentum

        self.batch_size = batch_size
        self.steps_until_sync = steps_until_sync

        self.loss_function = tf.keras.losses.MSE
        self.q_optimizer = SGD(self.q_lr, self.q_momentum)

        self.epsilon_scheduler = LinearScheduleEpsilon()
        self.total_steps = 0

    def get_action(self, states, epsilon):
        """between [-1, 1], but with a fixed 0.25 interval\n
            eg. +0.75 means spend 75% of money that I have right now to buy btc\n
                -0.5 means sell 50% of the btc I currently have"""
        sampled_value = random.random()

        if self.total_steps <= self.pre_train_steps or \
                sampled_value < epsilon or \
                len(self.replay_buffer) < self.batch_size:
            return np.random.choice(self.actions, size=[len(states), ])
        else:
            predict_q = self.q(states)
            actions = argmax(predict_q, axis=1)
            return actions

    def get_epsilon(self, step):
        return self.epsilon_scheduler.get_value(step=step)

    def run_episode(self):
        state = self.env.reset()

        self.q.set_weights(self.q_target.get_weights())

        step_idx = 0
        while True:

            if DEBUG:
                if step_idx % 100 == 0:
                    print(f"[Step {step_idx}] total balance {self.env.total_balance}, " + \
                          f"money {self.env.money}, btc {self.env.btc}, " + \
                          f"money/btc {self.env.price_history[self.env.current_moment]}")

            epsilon = self.get_epsilon(step_idx)

            if step_idx % self.choose_action_frequency == 0:
                actions = self.get_action(expand_dims(state, 0), epsilon)
                action = actions[0]

            next_state, reward, done, _ = self.env.step(action)

            if done or step_idx >= self.episode_len:
                self.total_steps += step_idx
                return self.env.total_balance

            self.replay_buffer.append((state, action, reward, next_state, done))

            if self.total_steps % self.steps_until_sync == 0:
                self.q_target.set_weights(self.q.get_weights())

            if self.total_steps > self.pre_train_steps and \
                    step_idx % self.steps_until_sync == 0 and \
                    len(self.replay_buffer) >= self.batch_size:
                samples = random.sample(
                    self.replay_buffer[-self.replay_buffer_len:],
                    self.batch_size)

                s_batch = stack([sample[0] for sample in samples], axis=0)
                a_batch = stack([sample[1] for sample in samples], axis=0)
                r_batch = stack([sample[2] for sample in samples], axis=0)
                s_next_batch = stack([sample[3] for sample in samples], axis=0)
                d_batch = stack([sample[4] for sample in samples], axis=0)

                # train Q network
                with GradientTape() as q_tape:
                    q_vars = self.q.trainable_variables
                    q_tape.watch(q_vars)

                    q_next_pred = self.q_target(s_next_batch)

                    a_next_batch = argmax(q_next_pred, axis=1)
                    a_next_one_hot = one_hot(indices=a_next_batch,
                                             depth=self.num_actions)

                    target_q = reduce_sum(a_next_one_hot * q_next_pred, axis=1)
                    target_q = r_batch + (1.0 -
                                          d_batch) * self.discount * target_q

                    q_pred = self.q(s_batch)

                    a_next_one_hot = one_hot(indices=a_batch,
                                             depth=self.num_actions)
                    predicted_q = reduce_sum(a_next_one_hot * q_pred, axis=1)

                    loss = self.loss_function(y_true=target_q,
                                              y_pred=predicted_q)

                q_gradients = q_tape.gradient(loss, q_vars)
                self.q_optimizer.apply_gradients(zip(q_gradients, q_vars))

            state = next_state
            step_idx += 1

    def train(self, episodes=10, save_model=True, render=True):

        for ep_idx in range(episodes):

            balance = self.run_episode()
            logger.info(f"Balance after episode {ep_idx}: {balance}")

            if render:
                self.env.render()

            if save_model is True:
                tag = int(time())

                self.q_target.save_weights(f"q_model_ep{ep_idx}_{tag}")

    def test(self, data):

        backup_ep_len = self.env.episode_len

        self.env.episode_len = len(
            data) - 10  # -10 as a precaution for improper indexes
        state = self.env.reset()

        step_idx = 0
        while True:

            if DEBUG:
                if step_idx % 100 == 0:
                    print(f"[Step {step_idx}] total balance {self.env.total_balance}, " + \
                          f"money {self.env.money}, btc {self.env.btc}, " + \
                          f"money/btc {self.env.price_history[self.env.current_moment]}")

            epsilon = self.get_epsilon(step_idx)
            action = self.get_action(state, epsilon)
            next_state, _, done, _ = self.env.step(action)

            if done or step_idx >= self.episode_len:
                self.env.episode_len = backup_ep_len
                return self.env.total_balance

            state = next_state
            step_idx += 1


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

    # _check_gap_frequency(data)
    # _check_cont_frequency(data)

    # TODO remove after debug
    tf.debugging.enable_check_numerics()

    data_ = []
    for i in range(data.shape[0]):
        data_.append((data[i][2] + data[i][3]) / 2)

    data = np.array(data_)

    agent = DQN(data)
    agent.train(episodes=10, save_model=False, render=True)
    agent.test(data)
