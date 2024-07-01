import pickle
import torch

import pickle

def load_and_inspect_pkl(file_path):
    try:
        with open(file_path, 'rb') as f:
            data_structure = pickle.load(f)

        # Iterate through each category ('observations' and 'actions')

        for i in data_structure.keys():
            print(i)
            print(len(data_structure[i]))
            print(data_structure[i][0].shape)

            print("-"*40)
    except Exception as e:
        pass
load_and_inspect_pkl('/app/processed_bottle_pick_data/torch_output_policy_aware/2024-06-10-15-58-34.pkl')

