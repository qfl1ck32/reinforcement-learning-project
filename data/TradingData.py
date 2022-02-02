from datetime import datetime

import numpy as np


class TradingData:
    timestamp: str
    is_open: bool

    high: float
    low: float
    close: float

    volume_btc: float
    volume_currency: float

    weighted_price: float

    def __init__(self, timestamp: str, is_open: bool, high: float, low: float,
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
        return f"Open: {self.is_open} | High: {self.high} | Low: {self.low} | " \
               f"Close: {self.close} | Volume BTC: {self.volume_btc} | Volume currency: {self.volume_currency} | " \
               f"Weighted price: {self.weighted_price}"

    def to_numpy(self):
        utc_time = datetime.strptime(self.timestamp, "%Y-%m-%d")

        epoch_time = (utc_time - datetime(1970, 1, 1)).total_seconds()

        ts = [epoch_time]
        ts.extend([float(x) for x in [self.high, self.low, self.close, self.volume_btc, self.volume_currency, self.weighted_price]])

        return np.array(ts, dtype="float32")
