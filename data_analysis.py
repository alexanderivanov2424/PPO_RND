import os
import csv
import configparser
import numpy as np
import matplotlib.pyplot as plt

#pandas and seaborn

def bin_data(data, digitized, mean_or_max=True):
    if mean_or_max:
        return [data[digitized == i].mean() for i in range(1, len(bins))]
    return [data[digitized == i].sum() for i in range(1, len(bins))]

path = os.getcwd()
for dir in os.listdir(path):
    if 'Montezuma_with_skills' in dir:
        folder = os.path.join(path,dir)
        config_file = os.path.join(folder, 'logs', 'config.conf')
        config = configparser.ConfigParser()
        config.optionxform=str
        config.read(config_file)
        run_params = str(config['DEFAULT']['LearningRate']) + "," + str(config['DEFAULT']['Entropy'])

        step_data_file = os.path.join(folder, 'step_data.csv')
        with open(step_data_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            rows = [row for row in reader][1:]
            X = np.array([row[3] for row in rows], dtype='float')
            rewards = np.array([row[4] for row in rows], dtype='float')

            # I = np.argsort(X)
            # X = X[I]
            # rewards = rewards[I]

            if len(X) == 0:
                continue

            bins = np.arange(0, np.max(X), 1000)
            digitized = np.digitize(X, bins)
            X = bin_data(X, digitized)
            rewards = bin_data(rewards, digitized, False)

            plt.plot(X, rewards, label=run_params)
plt.legend()
plt.show()
