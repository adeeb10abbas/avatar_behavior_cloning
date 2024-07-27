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

ckpt_path = "/home/ali/weights/teacher_pos_wave/latest.ckpt"
pkl_path = "/home/ali/shared_volume/2024-07-20-12-08-18.pkl"
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
policy.num_inference_steps = 100 # DDIM inference iterations
# policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
policy.n_action_steps = 8


#     inferred_actions.append(action)
def analyze_and_plot_actions(pkl_path, ckpt_path):
    obs_ = load_pkl_obs(pkl_path)
    ground_truth_actions = obs_['action']

    inferred_actions = []
    for i in range(len(ground_truth_actions)):
        obs_dict_np = get_real_obs_dict(env_obs=obs_, shape_meta=cfg.task.shape_meta)
        obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
        result = policy.predict_action(obs_dict)
        action = result['action'][0].detach().to('cpu').numpy()  # Get first action
        inferred_actions.append(action)

    # Convert to numpy array and average over the last two dimensions
    inferred_actions = np.array(inferred_actions)
    return inferred_actions, ground_truth_actions
    # import pdb; pdb.set_trace()


# # Load the model
# payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
# cfg = payload['cfg']
# cls = hydra.utils.get_class(cfg._target_)
# workspace = cls(cfg)
# workspace.load_payload(payload)

# # Prepare the policy
# policy = workspace.model
# if cfg.training.use_ema:
#     policy = workspace.ema_model
# policy.eval().to(torch.device('cuda'))

# Analyze and plot
inferred_actions, ground_truth_actions = analyze_and_plot_actions(pkl_path, ckpt_path)

""""
we need to have 4 plots 
- 2 each arm 
- 2 each hand


    data_to_save["action"] = np.concatenate([rdda_right_act, # 6
                                        right_operator_pose, # 9
                                        rdda_left_act, # 6
                                        left_operator_pose # 9
                                        ], axis=1)
                                        
"""
right_arm_inferred = []
left_arm_inferred = []
right_hand_inferred = []
left_hand_inferred = []

right_arm_ground_truth = []
left_arm_ground_truth = []
right_hand_ground_truth = []
left_hand_ground_truth = []
import pdb; pdb.set_trace()
# inferred_actions = inferred_actions[0]
for i in range(len(inferred_actions)):
    left_arm_inferred.append(inferred_actions[i].squeeze()[:6][0])
    left_hand_inferred.append(inferred_actions[i].squeeze()[6:15][0])
    right_arm_inferred.append(inferred_actions[i].squeeze()[15:21][0])
    right_hand_inferred.append(inferred_actions[i].squeeze()[21:][0])
    
    
    left_arm_ground_truth.append(ground_truth_actions[i].squeeze()[:6][0])
    left_hand_ground_truth.append(ground_truth_actions[i].squeeze()[6:15][0])
    right_arm_ground_truth.append(ground_truth_actions[i].squeeze()[15:21][0])
    right_arm_ground_truth.append(ground_truth_actions[i].squeeze()[21:][0])
    
plt.plot(left_hand_inferred, label='Left Hand Inferred Action (pos (x))')
plt.plot(left_hand_ground_truth, label='Left Hand Ground Truth Action')
plt.legend()
plt.show()