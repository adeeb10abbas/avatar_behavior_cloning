import time
# from multiprocessing.managers import SharedMemoryManager
import click
import numpy as np
import torch
import dill
import hydra
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
import pickle

ckpt_path = "./weights/latest.ckpt"
pkl_path = "./eval_data/2024-07-20-12-10-38.pkl"
import matplotlib.pyplot as plt

def load_pkl_obs(pkl_path):
    print("Processing pkl file: %s" % pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
        # import pdb; pdb.set_trace()
    # Prepare data to be saved in Zarr
    data_to_save = {}
    for key, tensor_list in data.items():
        print(key, len(tensor_list))
        
        data_to_save[key] = torch.stack(tensor_list).numpy()
    # import pdb; pdb.set_trace()
    rdda_right_act = data_to_save["rdda_right_act"]
    right_operator_pose = data_to_save["right_operator_pose"]
    rdda_left_act = data_to_save["rdda_left_act"]
    left_operator_pose = data_to_save["left_operator_pose"]

    data_to_save["action"] = np.concatenate([rdda_right_act, # 6
                                        right_operator_pose, # 9
                                        rdda_left_act, # 6
                                        left_operator_pose # 9
                                        ], axis=1)

    obs_dict = {}
    obs_dict['usb_cam_left'] = data_to_save['usb_cam_left']
    obs_dict['usb_cam_right'] = data_to_save['usb_cam_right']
    obs_dict['usb_cam_table'] = data_to_save['usb_cam_table']
    obs_dict['rdda_left_obs'] = data_to_save['rdda_left_obs']
    obs_dict['rdda_right_obs'] = data_to_save['rdda_right_obs']
    obs_dict['left_arm_pose'] = data_to_save['left_arm_pose']
    obs_dict['right_arm_pose'] = data_to_save['right_arm_pose']
    obs_dict['action'] = data_to_save['action']
    return obs_dict

def get_obs_dict(loaded_pkl, index, size):
    """
    Extract a window of data from each key in the dictionary.
    
    Args:
        loaded_pkl (dict): Dictionary containing various time series or batched data.
        index (int): Start index from which to extract the data.
        size (int): Number of data points to extract from the start index.
    
    Returns:
        dict: A new dictionary with the same keys as `loaded_pkl` but containing only the window of data.
    """
    obs_dict = {}
    if index < 2:
        index = 2
    for key, data in loaded_pkl.items():
        if isinstance(data, np.ndarray) and data.ndim > 1:
            obs_dict[key] = data[index:index+size]
        else:
            raise ValueError("Data under key '{}' is not in the expected format.".format(key))
    return obs_dict

raw_dict = load_pkl_obs(pkl_path=pkl_path)



obs_dict_sub = get_obs_dict(raw_dict, 1, 2)
# import pdb; pdb.set_trace()
# load checkpoint
# ckpt_path = input
payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
cfg = payload['cfg']
cls = hydra.utils.get_class(cfg._target_)
workspace = cls(cfg)
workspace: BaseWorkspace
workspace.load_payload(payload, exclude_keys=None, include_keys=None)

# diffusion model
policy: BaseImagePolicy
policy = workspace.model
if cfg.training.use_ema:
    policy = workspace.ema_model

device = torch.device('cuda')
policy.eval().to(device)

# set inference params
policy.num_inference_steps = 16 # DDIM inference iterations
# policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
policy.n_action_steps = 4

inferred_actions = []
for i in range(1, len(raw_dict["action"])//policy.n_action_steps):
        obs_dict_sub = get_obs_dict(raw_dict, i*policy.n_action_steps, 2)
        obs_dict_torched = dict_apply(get_real_obs_dict(env_obs=obs_dict_sub, 
                                                        shape_meta=cfg.task.shape_meta), lambda x: torch.from_numpy(x).unsqueeze(0).to(device=device))
        result = policy.predict_action(obs_dict_torched)
        action = result["action"][0].detach().to("cpu").numpy()

        for j in [*action]:
            inferred_actions.append(j)

# Convert lists to NumPy arrays
inferred = np.array(inferred_actions)
ground_truth = np.array(raw_dict['action'])
# import pdb; pdb.set_trace()
# plt.cla()
#addition here - rdda_right_act (3)[pos] + right_arm_ee_pose(9) + rdda_left_act(3) [pos] + left_operator_ee_pose(9) 
plt.plot([i[3:12] for i in inferred], label='inferred_action')
plt.plot([i[3:12] for i in ground_truth[:]], label=' ground_truth_action')
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
# plt.show()
plt.savefig('saved_png.png')

# import pdb; pdb.set_trace()


