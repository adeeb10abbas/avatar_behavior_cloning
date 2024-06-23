import torch 
_path = "/app/experimental_throttled/rethrottled_out.bag"

import torch
import rosbag
import pickle
from typing import Optional

from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Point, Quaternion, Twist, Pose
from cv_bridge import CvBridge
from rosgraph_msgs.msg import Clock
from functools import partial
import numpy as np

## Custom 
from rdda_interface.msg import RDDAPacket

from helpers import image_to_tensor, arm_pose_to_tensor, rdda_packet_to_tensor

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

def extract_and_organize_data_from_bag(bag_path, mode, output_file="organized_data.pkl"):
    data_structure = {}
    left_arm_pose_handler = partial(arm_pose_to_tensor, side="left")
    right_arm_pose_handler = partial(arm_pose_to_tensor, side="right")
    rdda_packet_to_tensor_teacher = partial(rdda_packet_to_tensor, mode=mode)

    topic_handlers = {
        "/usb_cam_left/image_raw": image_to_tensor, # obs
        "/usb_cam_right/image_raw": image_to_tensor, # obs
        "/usb_cam_table/image_raw": image_to_tensor, # obs
        "/right_smarty_arm_output": right_arm_pose_handler, # obs + action (user)
        "/left_smarty_arm_output": left_arm_pose_handler, # obs + action (user)
        "/throttled_rdda_right_master_output": rdda_packet_to_tensor_teacher, # obs, act
        "/throttled_rdda_l_master_output": rdda_packet_to_tensor_teacher, # obs, act
    }

    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            if topic in topic_handlers:
                tensor = topic_handlers[topic](msg, t=t)
                if tensor is not None:
                    # category = "observations" if topic in ["/usb_cam_left/image_raw", "/usb_cam_right/image_raw", 
                    #                                       "/usb_cam_table/image_raw", "/throttled_rdda_right_master_output", 
                    #                                       "/throttled_rdda_l_master_output"] else "actions"
                    if "image" in topic:
                        topic_key = topic.split("/")[1]
                    else:
                        topic_key = topic.split("/")[-1]
                    if topic_key not in data_structure:
                        data_structure[topic_key] = []
                    data_structure[topic_key].append(tensor)

    # Saving the organized data without concatenation
    with open(output_file, 'wb') as f:
        pickle.dump(data_structure, f)
    print(f"Organized data saved to {output_file}")
    # print(f"Concatenated data and feature log extracted and saved to {output_file} and {log_file_path}, respectively.")

def main(input_file, mode):
    extract_and_organize_data_from_bag(input_file, mode = mode)
    
if __name__=="__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: rosbags_to_torch.py <input_bag_file> <mode>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])