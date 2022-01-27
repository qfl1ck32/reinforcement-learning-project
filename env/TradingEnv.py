from abc import ABC, abstractmethod

from gym import Env
from gym.spaces import Discrete, Box
from gym.utils import seeding
from enums import DiscreteAction, Position
from numpy import ndarray, inf, float32
from matplotlib.pyplot import cla, plot, suptitle, pause, scatter

from typing import List, Tuple, Dict

class TradingEnv(Env, ABC):
    dataframe: ndarray
    window_size: int

    prices: ndarray
    signal_features: ndarray

    np_random: int

    shape: Tuple[int, int]

    _done: bool

    _start_tick: int
    _end_tick: int

    _current_tick: int
    _last_trade_tick: int

    _position: Position
    _position_history: List[Position]

    _total_reward: int
    _total_profit: int

    _history: Dict[str, list]  # total_reward: [int], total_profit: [int], position: [Position] am mancat sarmale

    _is_first_rendering: bool

    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, dataframe: ndarray, window_size: int):
        assert dataframe.ndim == 2

        self.dataframe = dataframe
        self.window_size = window_size

        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        self.seed()

        self.action_space = Discrete(len(DiscreteAction))
        self.observation_space = Box(
            low=-inf,
            high=inf,
            shape=self.shape,
            dtype=float32
        )

        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1

        self.reset()

    def _initialise_history(self):
        self._history = {key: [] for key in ["total_reward", "total_profit", "position"]}

    def reset(self):
        self._done = False

        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1

        self._position = Position.SHORT

        self._position_history = [Position.NONE] * self.window_size + [self._position]

        self._total_reward = 0
        self._total_profit = 1

        self._history = dict()

        self._is_first_rendering = True

        self._initialise_history()

    def step(self, action: DiscreteAction):
        self._current_tick += 1

        reward = self._calculate_reward(action)

        self._total_reward += reward

        self._update_profit(action)

        if self._should_trade(action):
            self._position = self._position.opposite()

            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)

        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position
        )

        self._update_history(info)

        if self._current_tick == self._end_tick:
            self._done = True

        return self._get_observation(), reward, self._done, info

    def _update_history(self, info: dict):
        for key, value in info.items():
            self._history[key].append(value)

    def seed(self, seed: int = None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def _should_trade(self, action: DiscreteAction):
        return action == DiscreteAction.BUY and self._position == Position.SHORT or action == DiscreteAction.SELL and self._position == Position.LONG

    def _get_observation(self):
        return self.signal_features[
            (self._current_tick - self.window_size) : self._current_tick
        ]

    def _scatter(self, position: Position, tick: int):
        scatter(
            x=tick,
            y=self.prices[tick],
            color='red' if position == Position.SHORT else 'green'
        )

    def render(self, mode='human'):
        if self._is_first_rendering:
            self._is_first_rendering = False

            cla()
            plot(self.prices)

            start_position = self._position_history[self._start_tick]

            self._scatter(start_position, self._start_tick)

        self._scatter(self._position, self._current_tick)

        suptitle(
            t=f"Total Reward: {self._total_reward}\n"
              f"Total Profit: {self._total_profit}"
        )

        pause(1e-2)

    @abstractmethod
    def _process_data(self):
        pass

    @abstractmethod
    def _update_profit(self, action: DiscreteAction):
        pass

    @abstractmethod
    def _calculate_reward(self, action: DiscreteAction):
        pass

    @abstractmethod
    def get_max_profit(self):
        pass
