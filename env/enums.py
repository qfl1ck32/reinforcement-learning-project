from enum import Enum


class Action(Enum):
    SELL = 0
    BUY = 1
    HOLD = 2


class Position(Enum):
    SHORT = 0
    LONG = 1

    NONE = 2

    def opposite(self):
        return self.LONG if self == self.SHORT else self.LONG
