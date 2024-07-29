import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from diffusion_policy.dataset.haptics_dataset import AvatarHapticsImageDataset
from diffusion_policy.common.replay_buffer import ReplayBuffer
import zarr
image_shape = [3, 480, 640]

shape_meta = {
    "obs": {
        "usb_cam_right": {
            "shape": image_shape,
            "type": "rgb"
        },
        "usb_cam_left": {
            "shape": image_shape,
            "type": "rgb"
        },
        "usb_cam_table": {
            "shape": image_shape,
            "type": "rgb"
        },
        "left_arm_pose": {
            "shape": [9],  # left robot EE pose
            "type": "low_dim"
        },
        "right_arm_pose": {
            "shape": [9],  # right robot EE pose
            "type": "low_dim"
        },
        "rdda_right_obs": {
            "shape": [9],  # pos_tensor, vel_tensor
            "type": "low_dim"
        },
        "rdda_left_obs": {
            "shape": [9],  # pos_tensor, vel_tensor
            "type": "low_dim"
        }
    },
    "action": {
        "shape": [30],
        # addition here - rdda_right_act (6)[wave, pos] + rdda_left_act(6) [wave, pos] + left_operator_ee_pose(9) 
        # + right_arm_ee_pose(9)
    }
}
dataset_path="/home/ali/shared_volume/bottle_pick/teacher_aware_pos_wave/_replay_buffer.zarr"

obs_keys = [
# "usb_cam_right",
# "usb_cam_left",
# "usb_cam_table",
"left_arm_pose",
"right_arm_pose",
"rdda_right_obs",
"rdda_left_obs"
]
all_keys = obs_keys + ["action"]

replay_buffer = ReplayBuffer.copy_from_path(
    zarr_path=dataset_path, 
    keys= all_keys,
    store=zarr.MemoryStore()
)
from diffusion_policy.common.normalize_util import (
    array_to_stats)

for key in all_keys:
    #Print all the stats
    stat = array_to_stats(replay_buffer[key])
    print(f"Key: {key} ----> {stat}")