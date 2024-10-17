import torch 

import torch
import rosbag
import pickle
from typing import Optional
import os
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Point, Quaternion, Twist, Pose
from cv_bridge import CvBridge
from rosgraph_msgs.msg import Clock
from functools import partial
import numpy as np

## Custom 
from rdda_interface.msg import RDDAPacket

from helpers import image_to_tensor, operator_arm_pose_to_tensor, rdda_packet_to_tensor, panda_arm_pose_to_tensor

def save_tensors_as_pickle(data_tensors, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(data_tensors, f)
    print(f"Data saved to {output_file}")

def preprocess_tensor(tensor, desired_len=256):
    """Reshape or flatten tensor to a desired length for concatenation."""
    original_len = tensor.numel()
    if original_len > desired_len:
        # Example: Flatten and take the first `desired_len` elements
        return tensor.view(-1)[:desired_len]
    elif original_len < desired_len:
        # Example: Pad with zeros
        return torch.cat([tensor.view(-1), torch.zeros(desired_len - original_len)], dim=0)
    else:
        return tensor.view(-1)

def concatenate_data(data_list, desired_len=256):
    """Concatenate a list of tensors, preprocessing each to match `desired_len`."""
    processed_tensors = [preprocess_tensor(tensor, desired_len) for tensor in data_list]
    return torch.stack(processed_tensors, dim=0)

# # Assuming other necessary imports and function definitions remain unchanged

def extract_and_organize_data_from_bag(bag_path, mode, output_file_path):
    assert mode in ["teacher_aware", "policy_aware"], "Mode must be either 'teacher_aware' or 'policy_aware'"
    data_structure = {"rdda_right_obs": [], "rdda_right_act": [], "rdda_left_obs": [], "rdda_left_act": [],}
    left_arm_pose_handler = partial(operator_arm_pose_to_tensor, side="left")
    right_arm_pose_handler = partial(operator_arm_pose_to_tensor, side="right")
    
    rdda_packet_to_tensor_teacher = partial(rdda_packet_to_tensor, mode=mode)

    topic_handlers = {
        "/left_cam/color/image_raw": image_to_tensor, # obs
        "/right_cam/color/image_raw": image_to_tensor, # obs
        "/table_cam/color/image_raw": image_to_tensor, # obs
        "/right_smarty_arm_output": right_arm_pose_handler, # obs + action (user)
        "/left_smarty_arm_output": left_arm_pose_handler, # obs + action (user)
        "/left_arm_pose": panda_arm_pose_to_tensor, # obs
        "/right_arm_pose": panda_arm_pose_to_tensor, # obs
        "/throttled_rdda_right_master_output": rdda_packet_to_tensor_teacher, # obs, act
        "/throttled_rdda_l_master_output": rdda_packet_to_tensor_teacher, # obs, act
    }

    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            if topic in topic_handlers:
                tensor = topic_handlers[topic](msg, t=t)
                if "throttled" in topic:
                    if mode == "teacher_aware": 
                        # Teacher Aware
                        obs_tensor = tensor[:3] # 3
                        action_tensor = tensor[3:] # 3
                    else: ## Policy Aware
                        obs_tensor = tensor[:6] # 6
                        action_tensor = tensor[6:] # 3

                    data_structure["rdda_left_act" if "rdda_l" in topic else "rdda_right_act"].append(action_tensor)
                    data_structure["rdda_left_obs" if "rdda_l" in topic else "rdda_right_obs"].append(obs_tensor)
                    continue          
                      
                if tensor is not None:
                    if "image" in topic:
                        topic_key = topic.split("/")[1]
                    elif "smarty" in topic:
                        topic_key = topic.split("/")[-1].replace("smarty_arm_output", "operator_pose")
                    else:
                        topic_key = topic.split("/")[-1]
                    if topic_key not in data_structure:
                        data_structure[topic_key] = []
                    data_structure[topic_key].append(tensor)

    # Saving the organized data without concatenation
    with open(output_file_path, 'wb') as f:
        pickle.dump(data_structure, f)
    print(f"Organized data saved to {output_file_path}")
    # print(f"Concatenated data and feature log extracted and saved to {output_file} and {log_file_path}, respectively.")

def main(input_file, mode, output_file):
    output_file_name = input_file.split("/")[-1].split(".")[0]
    output_file_path = os.path.join(output_file, f"{output_file_name}.pkl")
    extract_and_organize_data_from_bag(input_file, mode = mode, output_file_path = output_file_path)
    
if __name__=="__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: rosbags_to_torch.py <input_bag_file> <mode>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
