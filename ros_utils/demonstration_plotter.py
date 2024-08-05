#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import pickle
import glob
import pandas as pd
import threading

# bag_path = "/home/avatar/git/avatar_behavior_cloning/bags/synced/pkls"
bag_path = "/home/avatar/Downloads/diffusion_data/light"


pkl_files = glob.glob(bag_path + "/*.pkl")

data = {
    'rdda_right_obs': [],
    'rdda_left_obs': [],
    'rdda_right_act': [],
    'rdda_left_act': [],
    'right_arm_pose': [],
    'right_operator_pose': [],
    'left_arm_pose': [],
    'left_operator_pose': [],
}

for index, pkl in enumerate(pkl_files):
    with open(pkl, "rb") as f:
        episode = pickle.load(f)
        data['rdda_right_obs'].append(episode['rdda_right_obs'])
        data['rdda_left_obs'].append(episode['rdda_left_obs'])
        data['rdda_right_act'].append(episode['rdda_right_act'])
        data['rdda_left_act'].append(episode['rdda_left_act'])
        data['right_arm_pose'].append(episode['right_arm_pose'])
        data['right_operator_pose'].append(episode['right_operator_pose'])
        data['left_arm_pose'].append(episode['left_arm_pose'])
        data['left_operator_pose'].append(episode['left_operator_pose'])
    
    print('Loaded', pkl)
    
    
def plot_episodes(data, field):
    """
    Plot the data in all episodes for the given field
    """
    axes = []
    for j in range(3):
        axes.append(plt.subplot(311 + j))
        for i in range(len(data[field])):
            axes[j].plot(np.array(data[field][i])[:, j], label='Episode ' + str(i) + ' ' + field + 'index ' + str(j))
        
        axes[j].legend()
        
    plt.show(block=False)

def plot_statistics(data, field):
    """
    Plot the mean and standard deviation of the data for the given field
    """
    # fill zeros to data
    plt.figure()
    
    max_len = max([len(x) for x in data[field]])
    for i in range(len(data[field])):
        if len(data[field][i]) < max_len:
            data[field][i] = np.concatenate((data[field][i], np.zeros((max_len - len(data[field][i]), np.array(data[field][i]).shape[-1]))))
    
    filled_data = np.array(data[field])    
    print("Filled data shape: ", filled_data.shape)
    axes = []
    for j in range(3):
        axes.append(plt.subplot(311 + j))
        mean = np.mean(filled_data, axis=0)
        std = np.std(filled_data, axis=0)
        axes[j].plot(mean[:, j], label='Mean ' + field + ' index ' + str(j))
        axes[j].fill_between(range(len(mean)), mean[:, j] - std[:, j], mean[:, j] + std[:, j], alpha=0.5)
        axes[j].legend()
    
    plt.show(block=False)


if __name__ == "__main__":
    field = 'rdda_right_obs'
    plot_episodes(data, field)
    plot_statistics(data, field)    
    plt.show()
    print('Done')
    