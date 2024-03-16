import torch
import numpy as np
import pickle

from training.vision_dataset_py import create_sample_indices, sample_sequence, get_data_stats, normalize_data, unnormalize_data
from training.vision_dataset_py import ImageHapticsDataset

# Load the dataset
dataset_path = 'complex_dummy_dataset.pkl'
pred_horizon = 10
obs_horizon = 10
action_horizon = 10
dataset = ImageHapticsDataset(dataset_path, pred_horizon, obs_horizon, action_horizon)

# Get the first sample
stats = dataset.stats

# print(stats)
print(dataset[0])
# batch = next(iter(dataset))
# print(batch['image'])
# print(batch['haptics'])
# print(batch['action'])
