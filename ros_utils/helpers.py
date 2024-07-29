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
from scipy.spatial.transform import Rotation
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

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
    # Position tensor
    position_tensor = torch.tensor([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z], dtype=torch.float32)
    
    # Quaternion tensor
    quat_tensor = torch.tensor([pose_msg.pose.orientation.w, pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z], dtype=torch.float32)
    
    tf = RotationTransformer(from_rep='quaternion', to_rep='rotation_6d')
    rot6d_tensor = torch.tensor(tf.forward(quat_tensor), dtype=torch.float32)
    
    # Concatenate position and rotation vector tensors
    panda_arm_ee_pose_tensor = torch.cat([position_tensor, rot6d_tensor], dim=0)
    print(f"panda_arm_ee_pose_tensor shape: {panda_arm_ee_pose_tensor.shape}")
    return panda_arm_ee_pose_tensor


def operator_arm_pose_to_tensor(pose_msg, side, t):
    # Position tensor
    position_tensor = torch.tensor([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z], dtype=torch.float32)
    
    # # Euler Angle tensor
    # angle_tensor = torch.tensor([pose_msg.angle.x, pose_msg.angle.y, pose_msg.angle.z], dtype=torch.float32)
    
    # # Twist tensor
    # twist_tensor = torch.tensor([pose_msg.twist.linear.x, pose_msg.twist.linear.y, pose_msg.twist.linear.z,
    #                              pose_msg.twist.angular.x, pose_msg.twist.angular.y, pose_msg.twist.angular.z], dtype=torch.float32)
    
    # Quaternion tensor
    quat_tensor = torch.tensor([pose_msg.quat.w, pose_msg.quat.x, pose_msg.quat.y, pose_msg.quat.z], dtype=torch.float32)
    
    # Convert quaternion to rotation vector (axis-angle representation)
    # rotation = Rotation.from_quat(quat_tensor.numpy()) 
    # rotvec = rotation.as_rotvec()
    # tf = RotationTransformer()
    # rot6d_tensor = torch.tensor(tf.forward(rotvec), dtype=torch.float32)
    tf = RotationTransformer(from_rep='quaternion', to_rep='rotation_6d')
    rot6d_tensor = torch.tensor(tf.forward(quat_tensor), dtype=torch.float32)
    
    # Concatenate position, rotation vector
    operator_ee_pose_tensor = torch.cat([position_tensor, rot6d_tensor], dim=0)
    print(f"operator_ee_pose_tensor shape: {operator_ee_pose_tensor.shape}")
    return operator_ee_pose_tensor


def rdda_packet_to_tensor(rdda_packet: RDDAPacket, mode: str, t: Clock):
    pos_tensor = torch.tensor(rdda_packet.pos, dtype=torch.float32)
    tau_tensor = torch.tensor(rdda_packet.tau, dtype=torch.float32)
    wave_tensor = torch.tensor(rdda_packet.wave, dtype=torch.float32)
    pressure_tensor = torch.tensor(rdda_packet.pressure, dtype=torch.float32)
    
    obs_from_state = None
    action_stuff = None

    if mode == "teacher_aware":
        # We don't feed haptics to the model, it's only supposed to be implicitly learned
        obs_from_state = torch.cat([pos_tensor], dim=0) # 3 DIM
        action_stuff = torch.cat([pos_tensor], dim=0) # 3 DIM

    elif mode == "policy_aware": # TODO: check the obs and action tensors of the gripper/glove here
        obs_from_state = torch.cat([pos_tensor, pressure_tensor], dim=0) # 6 DIM
        action_stuff = torch.cat([pos_tensor, wave_tensor], dim=0) # 6 DIM

    # assert obs_from_state is not None and action_stuff is not None
    # Ensure the tensors are the same shape before stacking
    print(f"obs_from_state shape: {obs_from_state.shape}")
    print(f"action_stuff shape: {action_stuff.shape}")

    # Adjusting the concatenation to be along the correct dimension
    all_tensors = torch.hstack([obs_from_state, action_stuff])
    
    return all_tensors


    # /app/processed_bottle_pick_data
    # /app/processed_bottle_pick_data/torch_output
