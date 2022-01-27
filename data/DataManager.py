import csv
from math import isnan
from typing import List

from constants import trading_data_file_directory, start_year_for_data, data_directory, processed_data_filename
from data.TradingData import TradingData

from datetime import datetime

import pickle
import os


def format_file_path(starting_with_year: int):
    return f"{data_directory}/{processed_data_filename}-{starting_with_year}.obj"


def exists_data(starting_with_year: int = start_year_for_data):
    return os.path.exists(format_file_path(starting_with_year))


def generate_and_save_data(starting_with_year=start_year_for_data, limit=float('+inf')):
    data: List[TradingData] = []

    year_to_timestamp = datetime(starting_with_year, 1, 1).timestamp()

    with open(trading_data_file_directory, newline='') as file:
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


def save_data(data: List[TradingData], starting_with_year=start_year_for_data):
    with open(format_file_path(starting_with_year), "wb") as file:
        pickle.dump(data, file)


def load_data(starting_with_year=start_year_for_data) -> List[TradingData]:
    with open(format_file_path(starting_with_year), "rb") as file:
        return pickle.load(file)
