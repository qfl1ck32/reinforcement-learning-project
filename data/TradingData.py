from datetime import datetime

import numpy as np


class TradingData:
    timestamp: int
    is_open: bool

    high: float
    low: float
    close: float

    volume_btc: float
    volume_currency: float

    weighted_price: float

    def __init__(self, timestamp: int, is_open: bool, high: float, low: float,
                 close: float, volume_btc: float, volume_currency: float, weighted_price: float):
        self.timestamp = timestamp
        self.is_open = is_open
        self.high = high
        self.low = low
        self.close = close
        self.volume_btc = volume_btc
        self.volume_currency = volume_currency
        self.weighted_price = weighted_price

    def __str__(self):
        return f"Timestamp: {datetime.fromtimestamp(self.timestamp)} | Open: {self.is_open} | High: {self.high} | Low: {self.low} | " \
               f"Close: {self.close} | Volume BTC: {self.volume_btc} | Volume currency: {self.volume_currency} | " \
               f"Weighted price: {self.weighted_price}"

    def return_numpy(self):
        return np.array([self.timestamp, self.high, self.low, self.close, self.volume_btc, self.volume_currency, self.weighted_price])
