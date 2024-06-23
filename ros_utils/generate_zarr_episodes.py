## Script that takes one bag file and generates a zarr for that entire bag file

# we take in a list of pkl file paths 
# we load them into memory
# we create zarr out of them 
# we save the zarr to a new directory

import os
import pickle
import zarr
import numpy as np
import torch
from diffusion_policy.common.replay_buffer import ReplayBuffer

NAME_REPLAY_BUFFER_OUT = "organized_data.pkl"

def main(input_pkl_file_path):
    # List of pkl files to process
    pkl_list = []
    for root, dirs, files in os.walk(input_pkl_file_path):
        for file in files:
            if file.endswith(".pkl"):
                pkl_list.append(os.path.join(root, file))
    replay_buffer = ReplayBuffer.create_from_path(NAME_REPLAY_BUFFER_OUT, mode="a")
    
    # Process each pkl file in the list
    for i_eps, episode in enumerate(pkl_list):
        print("Processing pkl file: %s" % episode)
        with open(episode, "rb") as f:
            data = pickle.load(f)
            # for key in data.keys():
            #     replay_buffer.add(key, data[key])

    
        print("Processed pkl file: %s" % episode)
        
if __name__=="__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: generate_zarr_episodes.py <input_pkl_file>")
        sys.exit(1)