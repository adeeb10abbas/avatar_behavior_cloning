import torch 
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
from avatar_msgs.msg import PTIPacket
import torch

# Function to convert sensor_msgs/Image to a PyTorch tensor
def image_to_tensor(image_msg, t: Optional[Clock] = None):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
    # Convert the CV image to a PyTorch tensor
    tensor_image = torch.tensor(np.array(cv_image), dtype=torch.float)
    return tensor_image


def panda_arm_pose_to_tensor(pose_msg, t):
    position_tensor = torch.tensor([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z], dtype=torch.float32)
    quat_tensor = torch.tensor([pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w], dtype=torch.float32)

    panda_arm_ee_pose_tensor = torch.cat([position_tensor, quat_tensor], dim=0)

    return panda_arm_ee_pose_tensor


def operator_arm_pose_to_tensor(pose_msg, side, t):
    position_tensor = torch.tensor([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z], dtype=torch.float32)
    angle_tensor = torch.tensor([pose_msg.angle.x, pose_msg.angle.y, pose_msg.angle.z], dtype=torch.float32)
    twist_tensor = torch.tensor([pose_msg.twist.linear.x, pose_msg.twist.linear.y, pose_msg.twist.linear.z,
                                 pose_msg.twist.angular.x, pose_msg.twist.angular.y, pose_msg.twist.angular.z], dtype=torch.float32)
    quat_tensor = torch.tensor([pose_msg.quat.x, pose_msg.quat.y, pose_msg.quat.z, pose_msg.quat.w], dtype=torch.float32)

    # Concatenation
    operator_ee_pose_tensor = torch.cat([position_tensor, quat_tensor], dim=0)

    return operator_ee_pose_tensor


def rdda_packet_to_tensor(rdda_packet: RDDAPacket, mode: str, t: Clock):
    pos_tensor = torch.tensor(rdda_packet.pos, dtype=torch.float32)
    pos_desired_tensor = torch.tensor(rdda_packet.pos_d, dtype=torch.float32)
    vel_tensor = torch.tensor(rdda_packet.vel, dtype=torch.float32)
    tau_tensor = torch.tensor(rdda_packet.tau, dtype=torch.float32)
    wave_tensor = torch.tensor(rdda_packet.wave, dtype=torch.float32)
    pressure_tensor = torch.tensor(rdda_packet.pressure, dtype=torch.float32)
    
    # Debugging prints to check tensor shapes
    print(f"pos_tensor shape: {pos_tensor.shape}")
    print(f"pos_desired_tensor shape: {pos_desired_tensor.shape}")
    print(f"vel_tensor shape: {vel_tensor.shape}")
    print(f"tau_tensor shape: {tau_tensor.shape}")
    print(f"wave_tensor shape: {wave_tensor.shape}")
    print(f"pressure_tensor shape: {pressure_tensor.shape}")

    obs_from_state = None
    action_stuff = None

    if mode == "teacher_aware":
        # We don't feed haptics to the model, it's only supposed to be implicitly learned
        obs_from_state = torch.cat([pos_tensor, vel_tensor, pos_desired_tensor], dim=0)
        action_stuff = torch.cat([pos_desired_tensor, wave_tensor], dim=0)
    elif mode == "policy_aware":
        obs_from_state = torch.cat([pos_tensor, vel_tensor, tau_tensor, pos_desired_tensor, pressure_tensor], dim=0)
        action_stuff = torch.cat([wave_tensor, pos_desired_tensor], dim=0)

    # assert obs_from_state is not None and action_stuff is not None

    # Ensure the tensors are the same shape before stacking
    print(f"obs_from_state shape: {obs_from_state.shape}")
    print(f"action_stuff shape: {action_stuff.shape}")

    # Adjusting the concatenation to be along the correct dimension
    all_tensors = torch.hstack([obs_from_state, action_stuff])
    
    return all_tensors


    # /app/processed_bottle_pick_data
    # /app/processed_bottle_pick_data/torch_output