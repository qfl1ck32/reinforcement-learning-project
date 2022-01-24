from logger.logger import logger
from data.DataManager import *


def get_data():
    if exists_data():
        logger.info("Data already generated, loading...")
        data = load_data()
        logger.info("Loaded.")
    else:
        logger.info("Generating and saving the data...")
        data = generate_and_save_data(limit=1e5)
        logger.info("Saved.")

    return data


def main():
    data = get_data()

    print(data[0])
    print(data[1])
    print(data[2])


if __name__ == '__main__':
    main()
