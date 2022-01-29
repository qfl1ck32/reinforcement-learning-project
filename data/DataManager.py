import csv
from math import isnan
from typing import List

from constants import TRADING_DATA_FILE_DIRECTORY, START_YEAR_FOR_DATA, DATA_DIRECTORY, PROCESSED_DATA_FILENAME
from data.TradingData import TradingData

from datetime import datetime

import pickle
import os


def format_file_path(starting_with_year: int):
    return f"{DATA_DIRECTORY}/{PROCESSED_DATA_FILENAME}-{starting_with_year}.obj"


def exists_data(starting_with_year: int = START_YEAR_FOR_DATA):
    return os.path.exists(format_file_path(starting_with_year))


def generate_and_save_data(starting_with_year=START_YEAR_FOR_DATA, limit=float('+inf')):
    data: List[TradingData] = []

    year_to_timestamp = datetime(starting_with_year, 1, 1).timestamp()

    with open(TRADING_DATA_FILE_DIRECTORY, newline='') as file:
        reader = csv.reader(file, delimiter=',')

        reader.__next__()

        for line in reader:
            line = [float(x) for x in line]

            if any([isnan(x) for x in line]):
                continue

            [timestamp, is_open, high, low, close, volume_btc, volume_currency, weighted_price] = line

            if timestamp < year_to_timestamp:
                continue

            data.append(TradingData(timestamp, is_open, high, low, close, volume_btc, volume_currency, weighted_price))

            if len(data) == limit:
                break

    save_data(data, starting_with_year)

    return data


def save_data(data: List[TradingData], starting_with_year=START_YEAR_FOR_DATA):
    with open(format_file_path(starting_with_year), "wb") as file:
        pickle.dump(data, file)


def load_data(starting_with_year=START_YEAR_FOR_DATA) -> List[TradingData]:
    with open(format_file_path(starting_with_year), "rb") as file:
        return pickle.load(file)
