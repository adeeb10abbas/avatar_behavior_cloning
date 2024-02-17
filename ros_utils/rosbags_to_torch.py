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

def glove_msg_to_tensor(msg, side):    
    # Convert position, velocity, and effort lists into PyTorch tensors
    positions_tensor = torch.tensor(msg.position, dtype=torch.float32)
    velocities_tensor = torch.tensor(msg.velocity, dtype=torch.float32)
    efforts_tensor = torch.tensor(msg.effort, dtype=torch.float32)
    
    # Concatenate the tensors along a new dimension to keep them separate
    # Use torch.stack if you want to maintain the distinction between position, velocity, and effort
    joint_state_tensor = torch.cat([positions_tensor, velocities_tensor, efforts_tensor], dim=0)
    print(joint_state_tensor)
    return joint_state_tensor

right_glove_handler = partial(glove_msg_to_tensor, side="right")
left_glove_handler = partial(glove_msg_to_tensor, side="left")

def arm_pose_to_tensor(pose_msg, side):
    # Assuming pose_msg is an instance of geometry_msgs/Pose
    position_tensor = torch.tensor([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z], dtype=torch.float32)
    orientation_tensor = torch.tensor([pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w], dtype=torch.float32)
    pose_tensor = torch.cat([position_tensor, orientation_tensor], dim=0)
    return pose_tensor
left_arm_pose_handler = partial(arm_pose_to_tensor, side="left")
right_arm_pose_handler = partial(arm_pose_to_tensor, side="right")

def rdda_packet_to_tensor(rdda_packet):
    # Convert the arrays from the RDDAPacket to PyTorch tensors
    pos_tensor = torch.tensor(rdda_packet.pos, dtype=torch.float32)
    vel_tensor = torch.tensor(rdda_packet.vel, dtype=torch.float32)
    tau_tensor = torch.tensor(rdda_packet.tau, dtype=torch.float32)
    wave_tensor = torch.tensor(rdda_packet.wave, dtype=torch.float32)
    wave_aux_tensor = torch.tensor(rdda_packet.wave_aux, dtype=torch.float32)
    pressure_tensor = torch.tensor(rdda_packet.pressure, dtype=torch.float32)
    
    # Concatenate all tensors into one tensor
    all_tensors = torch.cat([pos_tensor, vel_tensor, tau_tensor, wave_tensor, wave_aux_tensor, pressure_tensor], dim=0)
    
    return all_tensors



topic_handlers = {
    "/camera/color/image_raw": image_to_tensor, # obs
    "/right_arm_pose": right_arm_pose_handler, # action
    "/left_arm_pose": left_arm_pose_handler, # action
    "/right_glove_joint_states": right_glove_handler, # action 
    "/left_glove_joint_states": left_glove_handler, # action
    "/rdda_right_master_output": rdda_packet_to_tensor, # obs
    "/rdda_left_master_output": rdda_packet_to_tensor, # obs 
}
def save_tensors_as_pickle(data_tensors, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(data_tensors, f)
    print(f"Data saved to {output_file}")

# # Adjust this function for different ROS message types
def extract_data_from_bag(bag_path):
    data_tensors = {}
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            if topic in topic_handlers:
                handler_function = topic_handlers[topic]
                tensor = handler_function(msg)
                if tensor is not None:
                    if topic not in data_tensors:
                        data_tensors[topic] = []
                    data_tensors[topic].append(tensor)
            else:
                pass
    return data_tensors

# Main script
if __name__ == '__main__':
    bag_path = _path
    output_file = 'output_tensors.pkl'
    
    # Extract data from the ROS bag and convert to tensors
    data_tensors = extract_data_from_bag(bag_path)
    
    # Save the tensors to a pickle file
    save_tensors_as_pickle(data_tensors, output_file)
    print(f"Data extracted and saved to {output_file}")

    