import torch 
_path = "../hardware/test_trajs/2024-01-27-19-08-13.bag"

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

# Function to convert sensor_msgs/Image to a PyTorch tensor
def image_to_tensor(image_msg):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
    # Convert the CV image to a PyTorch tensor
    tensor_image = torch.tensor(np.array(cv_image), dtype=torch.float)
    return tensor_image

# def glove_msg_to_tensor(msg, side):    
#     # Convert position, velocity, and effort lists into PyTorch tensors
#     positions_tensor = torch.tensor(msg.position, dtype=torch.float32)
#     velocities_tensor = torch.tensor(msg.velocity, dtype=torch.float32)
#     efforts_tensor = torch.tensor(msg.effort, dtype=torch.float32)
    
#     # Concatenate the tensors along a new dimension to keep them separate
#     # Use torch.stack if you want to maintain the distinction between position, velocity, and effort
#     joint_state_tensor = torch.cat([positions_tensor, velocities_tensor, efforts_tensor], dim=0)
#     print(joint_state_tensor)
#     return joint_state_tensor

# right_glove_handler = partial(glove_msg_to_tensor, side="right")
# left_glove_handler = partial(glove_msg_to_tensor, side="left")

# def arm_pose_to_tensor(pose_msg, side):
#     # Assuming pose_msg is an instance of geometry_msgs/Pose
#     position_tensor = torch.tensor([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z], dtype=torch.float32)
#     orientation_tensor = torch.tensor([pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w], dtype=torch.float32)
#     pose_tensor = torch.cat([position_tensor, orientation_tensor], dim=0)
#     return pose_tensor
# left_arm_pose_handler = partial(arm_pose_to_tensor, side="left")
# right_arm_pose_handler = partial(arm_pose_to_tensor, side="right")

# def rdda_packet_to_tensor(rdda_packet):
#     # Convert the arrays from the RDDAPacket to PyTorch tensors
#     pos_tensor = torch.tensor(rdda_packet.pos, dtype=torch.float32)
#     vel_tensor = torch.tensor(rdda_packet.vel, dtype=torch.float32)
#     tau_tensor = torch.tensor(rdda_packet.tau, dtype=torch.float32)
#     wave_tensor = torch.tensor(rdda_packet.wave, dtype=torch.float32)
#     # wave_aux_tensor = torch.tensor(rdda_packet.wave_aux, dtype=torch.float32)
#     pressure_tensor = torch.tensor(rdda_packet.pressure, dtype=torch.float32)
    
#     # Concatenate all tensors into one tensor
#     all_tensors = torch.cat([pos_tensor, vel_tensor, tau_tensor, wave_tensor, pressure_tensor], dim=0)
    
#     return all_tensors
import torch
from functools import partial

def glove_msg_to_tensor(msg, side):
    positions_tensor = torch.tensor(msg.position, dtype=torch.float32)
    velocities_tensor = torch.tensor(msg.velocity, dtype=torch.float32)
    efforts_tensor = torch.tensor(msg.effort, dtype=torch.float32)
    
    print(f"{side}_glove_positions_tensor shape: {positions_tensor.shape}")
    print(f"{side}_glove_velocities_tensor shape: {velocities_tensor.shape}")
    print(f"{side}_glove_efforts_tensor shape: {efforts_tensor.shape}")

    # Concatenation
    joint_state_tensor = torch.cat([positions_tensor, velocities_tensor, efforts_tensor], dim=0)
    print(f"{side}_glove_joint_state_tensor after concatenation shape: {joint_state_tensor.shape}")
    
    return joint_state_tensor

right_glove_handler = partial(glove_msg_to_tensor, side="right")
left_glove_handler = partial(glove_msg_to_tensor, side="left")

def arm_pose_to_tensor(pose_msg, side):
    position_tensor = torch.tensor([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z], dtype=torch.float32)
    orientation_tensor = torch.tensor([pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w], dtype=torch.float32)
    
    print(f"{side}_arm_position_tensor shape: {position_tensor.shape}")
    print(f"{side}_arm_orientation_tensor shape: {orientation_tensor.shape}")

    # Concatenation
    pose_tensor = torch.cat([position_tensor, orientation_tensor], dim=0)
    print(f"{side}_arm_pose_tensor after concatenation shape: {pose_tensor.shape}")
    
    return pose_tensor

left_arm_pose_handler = partial(arm_pose_to_tensor, side="left")
right_arm_pose_handler = partial(arm_pose_to_tensor, side="right")

def rdda_packet_to_tensor(rdda_packet):
    pos_tensor = torch.tensor(rdda_packet.pos, dtype=torch.float32)
    vel_tensor = torch.tensor(rdda_packet.vel, dtype=torch.float32)
    tau_tensor = torch.tensor(rdda_packet.tau, dtype=torch.float32)
    wave_tensor = torch.tensor(rdda_packet.wave, dtype=torch.float32)
    pressure_tensor = torch.tensor(rdda_packet.pressure, dtype=torch.float32)
    
    print("rdda_pos_tensor shape:", pos_tensor.shape)
    print("rdda_vel_tensor shape:", vel_tensor.shape)
    print("rdda_tau_tensor shape:", tau_tensor.shape)
    print("rdda_wave_tensor shape:", wave_tensor.shape)
    print("rdda_pressure_tensor shape:", pressure_tensor.shape)

    # Concatenation
    all_tensors = torch.cat([pos_tensor, vel_tensor, tau_tensor, wave_tensor, pressure_tensor], dim=0)
    print("rdda_all_tensors after concatenation shape:", all_tensors.shape)
    
    return all_tensors

topic_handlers = {
    "/camera/color/image_raw": image_to_tensor, # obs
    "/right_arm_pose": right_arm_pose_handler, # action (user)
    "/left_arm_pose": left_arm_pose_handler, # action (user)
    "/right_glove_joint_states": right_glove_handler, # action (user) 
    "/left_glove_joint_states": left_glove_handler, # action (user)
    "/rdda_right_master_output": rdda_packet_to_tensor, # obs
    "/rdda_left_master_output": rdda_packet_to_tensor, # obs 
}
def save_tensors_as_pickle(data_tensors, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(data_tensors, f)
    print(f"Data saved to {output_file}")
import torch
import numpy as np

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

import torch
import rosbag
import pickle
from cv_bridge import CvBridge
import numpy as np

# Assuming other necessary imports and function definitions remain unchanged

def extract_and_concatenate_data_from_bag(bag_path, log_file_path):
    data_structure = {"observations": [], "actions": []}
    feature_log = {"observations": {}, "actions": {}}

    with rosbag.Bag(bag_path, 'r') as bag:
        observation_tensors = []
        action_tensors = []
        for topic, msg, t in bag.read_messages():
            if topic in topic_handlers:
                tensor = topic_handlers[topic](msg)
                if tensor is not None:
                    feature_key = topic.split("/")[-1]  # Simplified key for logging
                    if topic in ["/camera/color/image_raw", "/rdda_right_master_output", "/rdda_left_master_output"]:
                        observation_tensors.append(tensor)
                        feature_log["observations"][feature_key] = tensor.shape[0]
                    elif topic in ["/right_arm_pose", "/left_arm_pose", "/right_glove_joint_states", "/left_glove_joint_states"]:
                        action_tensors.append(tensor)
                        feature_log["actions"][feature_key] = tensor.shape[0]
        
        # Concatenate data with a placeholder desired length for each tensor
        if observation_tensors:
            data_structure["observations"] = concatenate_data(observation_tensors, desired_len=256)
        if action_tensors:
            data_structure["actions"] = concatenate_data(action_tensors, desired_len=256)

    # Save the concatenated data and the feature log
    with open(output_file, 'wb') as f:
        pickle.dump(data_structure, f)
    with open(log_file_path, 'w') as log_f:
        for category, features in feature_log.items():
            log_f.write(f"{category}:\n")
            for feature, dim in features.items():
                log_f.write(f"  - {feature}: {dim}\n")

    print(f"Concatenated data and feature log extracted and saved to {output_file} and {log_file_path}, respectively.")

# Main script adaptation
if __name__ == '__main__':
    bag_path = _path  # Ensure this path is correctly set
    output_file = 'concatenated_data.pkl'  # File to save the concatenated tensors
    log_file_path = 'feature_dimensions.log'  # File to save the log of feature dimensions

    extract_and_concatenate_data_from_bag(bag_path, log_file_path)
