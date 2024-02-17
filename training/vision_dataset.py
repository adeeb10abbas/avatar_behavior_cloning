import torch
import numpy as np
import pickle

# Helper functions remain similar
def create_sample_indices(episode_ends, sequence_length, pad_before=0, pad_after=0):
    indices = []
    for i, end in enumerate(episode_ends):
        start = episode_ends[i-1] if i > 0 else 0
        for idx in range(start - pad_before, end - sequence_length + pad_after + 1):
            buffer_start_idx = max(idx, start)
            buffer_end_idx = min(idx + sequence_length, end)
            sample_start_idx = buffer_start_idx - idx
            sample_end_idx = sequence_length - (buffer_end_idx - (idx + sequence_length))
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
    return np.array(indices)

def sample_sequence(train_data, sequence_length, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx):
    result = {}
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = np.zeros((sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
        if sample_start_idx > 0:
            data[:sample_start_idx] = sample[0]
        if sample_end_idx < sequence_length:
            data[sample_end_idx:] = sample[-1]
        data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

def get_data_stats(data):
    stats = {'min': np.min(data, axis=0), 'max': np.max(data, axis=0)}
    return stats

def normalize_data(data, stats):
    ndata = (data - stats['min']) / (stats['max'] - stats['min']) * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    data = (ndata + 1) / 2 * (stats['max'] - stats['min']) + stats['min']
    return data

class ImageHapticsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, pred_horizon, obs_horizon, action_horizon):
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)

        # Assuming the data structure is similar to what was described previously
        self.images = data['observations']['images']
        self.haptics = {'right': data['observations']['haptics']['right'],
                        'left': data['observations']['haptics']['left']}
        self.poses = {'right': data['actions']['pose']['right_arm_pose'],
                      'left': data['actions']['pose']['left_arm_pose']}
        self.gloves = {'right': data['actions']['joint_state']['right_glove'],
                       'left': data['actions']['joint_state']['left_glove']}

        episode_ends = [len(self.images)]  # Assuming a single episode for simplicity

        self.indices = create_sample_indices(episode_ends, pred_horizon, obs_horizon-1, action_horizon-1)
        # print(self.haptics[key].shape, self.poses[key].shape, self.gloves[key].shape)

        breakpoint()
        self.stats = {key: get_data_stats(np.concatenate((self.haptics[key], self.poses[key], self.gloves[key]), axis=1)) for key in ['right', 'left']}
        self.normalized_data = {key: normalize_data(np.concatenate((self.haptics[key], self.poses[key], self.gloves[key]), axis=1), self.stats[key]) for key in ['right', 'left']}

        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        
        sample = sample_sequence(self.normalized_data, self.pred_horizon, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)

        return {
            'image': self.images[sample_start_idx:sample_end_idx], 
            'haptics': {side: sample[side][:self.obs_horizon] for side in ['right', 'left']},
            'pose': {side: sample[side][:self.obs_horizon] for side in ['right', 'left']},
            'glove': {side: sample[side][self.obs_horizon:self.obs_horizon+self.action_horizon] for side in ['right', 'left']},
        }
