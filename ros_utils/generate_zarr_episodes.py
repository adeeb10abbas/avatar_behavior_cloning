import os
import pickle
import zarr
import numpy as np
import torch
from diffusion_policy.common.replay_buffer import ReplayBuffer
from tqdm import tqdm
def main(input_pkl_file_path):
    # Determine the output directory and name for the Zarr file
    output_dir = os.path.dirname(input_pkl_file_path)
    zarr_name = os.path.basename(input_pkl_file_path) + "_replay_buffer.zarr"
    zarr_output_path = os.path.join(output_dir, zarr_name)
    print(zarr_output_path)
    # Create or open an existing Zarr file
    replay_buffer = ReplayBuffer.create_from_path(zarr_output_path, mode="a")

    # List all pkl files in the provided directory and subdirectories
    pkl_list = []
    for root, dirs, files in os.walk(input_pkl_file_path):
        for file in files:
            if file.endswith(".pkl"):
                pkl_list.append(os.path.join(root, file))

    # Process each pkl file
    for pkl_file in tqdm(pkl_list, desc="Processing pkl files"):
        print("Processing pkl file: %s" % pkl_file)
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        
        # Prepare data to be saved in Zarr
        data_to_save = {}
        for key, tensor_list in data.items():
            print(key, len(tensor_list))
            
            data_to_save[key] = torch.stack(tensor_list).numpy()
        # import pdb; pdb.set_trace()
        rdda_right_act = data_to_save["rdda_right_act"]
        right_smarty_arm_output = data_to_save["right_smarty_arm_output"]
        rdda_left_act = data_to_save["rdda_left_act"]
        left_smarty_arm_output = data_to_save["left_smarty_arm_output"]

        # Stack the arrays along the 0th dimension
        data_to_save["action"] = np.concatenate([rdda_right_act,
                                            right_smarty_arm_output,
                                            rdda_left_act,
                                            left_smarty_arm_output], axis=1)

        del data
        keys_to_delete = ["rdda_right_act", "right_smarty_arm", "rdda_left_act", "left_smarty_arm"]
        for key in keys_to_delete:
            if key in data_to_save:
                del data_to_save[key]

        # Add processed data to replay buffer
        replay_buffer.add_episode(data_to_save, compressors='disk')
        print("Added data from pkl file to Zarr: %s" % pkl_file)
        
if __name__=="__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: generate_zarr_episode.py <input_pkl_directory>")
        sys.exit(1)
    main(sys.argv[1])