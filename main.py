import numpy as np
from logger.logger import logger
from data.DataManager import *
from algs import ddpg, dqn, actor_critic

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

def get_numpy_data():

    data = get_data()
    for i in range(len(data)):
        data[i] = data[i].return_numpy()

    data = np.array(data)
    return data
    
def main():

    data = get_numpy_data()

    print("running DDPG algorithm...")
    ddpg.run(data)

    print("running DQN algorithm...")
    dqn.run(data)
    # TODO RUS

    print("running actor-critic algorithm...")
    actor_critic.run(data)
    # TODO MIT

    print("done")

if __name__ == '__main__':
    main()
