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

# Function to convert sensor_msgs/Image to a PyTorch tensor
def image_to_tensor(image_msg, t: Optional[Clock] = None):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
    # Convert the CV image to a PyTorch tensor
    tensor_image = torch.tensor(np.array(cv_image), dtype=torch.float)
    return tensor_image

def arm_pose_to_tensor(pose_msg, side, t):
    position_tensor = torch.tensor([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z], dtype=torch.float32)
    angle_tensor = torch.tensor([pose_msg.angle.x, pose_msg.angle.y, pose_msg.angle.z], dtype=torch.float32)
    twist_tensor = torch.tensor([pose_msg.twist.linear.x, pose_msg.twist.linear.y, pose_msg.twist.linear.z,
                                 pose_msg.twist.angular.x, pose_msg.twist.angular.y, pose_msg.twist.angular.z], dtype=torch.float32)
    quat_tensor = torch.tensor([pose_msg.quat.x, pose_msg.quat.y, pose_msg.quat.z, pose_msg.quat.w], dtype=torch.float32)

    # print(f"{side}_arm_position_tensor shape: {position_tensor.shape}")
    # print(f"{side}_arm_angle_tensor shape: {angle_tensor.shape}")
    # print(f"{side}_arm_twist_tensor shape: {twist_tensor.shape}")
    # print(f"{side}_arm_quat_tensor shape: {quat_tensor.shape}")

    # Concatenation
    pose_tensor = torch.cat([position_tensor, quat_tensor], dim=0)
    # print(f"{side}_arm_pose_tensor after concatenation shape: {pose_tensor.shape}")
    # breakpoint()
    return pose_tensor

def rdda_packet_to_tensor(rdda_packet: RDDAPacket, mode: str, t: Clock):
    pos_tensor = torch.tensor(rdda_packet.pos, dtype=torch.float32)
    pos_desired_tensor = torch.tensor(rdda_packet.pos_d, dtype=torch.float32)
    vel_tensor = torch.tensor(rdda_packet.vel, dtype=torch.float32)
    tau_tensor = torch.tensor(rdda_packet.tau, dtype=torch.float32)
    wave_tensor = torch.tensor(rdda_packet.wave, dtype=torch.float32)
    pressure_tensor = torch.tensor(rdda_packet.pressure, dtype=torch.float32)

    if mode == "teacher":
        # We don't feed haptics to the model, it's only supposed to be implicitly learned
        obs_from_state = torch.cat([pos_tensor, vel_tensor, pos_desired_tensor], dim=0)
        action_stuff = torch.cat([pos_desired_tensor, wave_tensor, pos_desired_tensor], dim=0)
    elif mode == "policy":
        obs_from_state = torch.cat([pos_tensor, vel_tensor, tau_tensor, pos_desired_tensor, wave_tensor], dim=0)
        action_stuff = torch.cat([wave_tensor, pos_desired_tensor], dim=0)
    
    all_tensors = torch.vstack([obs_from_state, action_stuff])
    
    return all_tensors