import numpy as np
from logger.logger import logger
from data.DataManager import *
from algs import dqn, ddpg


def get_data():

    if exists_data():
        logger.info("Data already generated, loading...")
        data = load_data()
        logger.info("Loaded.")
    else:
        logger.info("Generating and saving the data...")
        data = generate_and_save_data()
        logger.info("Saved.")

    return data


def get_numpy_data():
    data = get_data()

    for i in range(len(data)):
        data[i] = data[i].to_numpy()

    data = np.array(data)
    return data


def main_statistics():

    data = get_numpy_data()

    logger.info("running DDPG algorithm...")
    ddpg.run(data)

    statistics_data = generate_data_for_statistics()

    for symbol in statistics_data:
        data = statistics_data[symbol]

        for i in range(len(data)):
            data[i] = data[i].to_numpy()

        data = np.array(data)

        print(data)

        ddpg.run(data)

#
# def main_train_data():
#     data = get_numpy_data()
#
#     # logger.info("running DDPG algorithm...")
#     # ddpg.run(data)
#
#     # logger.info("running DQN algorithm...")
#     # dqn.run(data)
#     # # TODO MIT
#
#
#     logger.info("done")


if __name__ == '__main__':
    main_statistics()
