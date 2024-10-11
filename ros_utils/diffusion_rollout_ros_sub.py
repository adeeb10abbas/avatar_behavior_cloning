#!/usr/bin/env python

import rospy
from rdda_interface.msg import RDDAPacket
from avatar_msgs.msg import PTIPacket
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Quaternion
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import cv2
from collections import deque

from multiprocessing import Process, Manager

import torch
import hydra
import numpy as np
import dill
from torchvision import transforms
from typing import Tuple
import copy
import time

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.real_world.real_inference_util import get_real_obs_resolution, get_real_obs_dict
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from collections import deque

class SubscriberNode:
    def __init__(self, shared_obs_dict):
        rospy.init_node('observation_subscriber_node')

        self.images_obs_sub1 = message_filters.Subscriber("/left_cam/color/image_raw", Image)
        self.images_obs_sub2 = message_filters.Subscriber("/right_cam/color/image_raw", Image)
        self.images_obs_sub3 = message_filters.Subscriber("/table_cam/color/image_raw", Image)
        self.state_obs_left_gripper_sub = message_filters.Subscriber("/rdda_l_master_input", RDDAPacket)
        self.state_obs_right_gripper_sub = message_filters.Subscriber("/rdda_right_master_input", RDDAPacket)
        self.state_obs_left_arm_sub = message_filters.Subscriber("/pti_interface_left/pti_output", PTIPacket)
        self.state_obs_right_arm_sub = message_filters.Subscriber("/pti_interface_right/pti_output", PTIPacket)

        self.obs_dict = shared_obs_dict
        self.obs_buffer = deque(maxlen=2)  # Buffer to hold the last two observations
        
        obs_subs = [
            self.images_obs_sub1,
            self.images_obs_sub2,
            self.images_obs_sub3,
            self.state_obs_left_gripper_sub,
            self.state_obs_right_gripper_sub,
            self.state_obs_left_arm_sub,
            self.state_obs_right_arm_sub,
        ]
        self.ts = message_filters.ApproximateTimeSynchronizer(obs_subs, 10, slop=1.0)
        self.ts.registerCallback(self.callback)
        
        # self.run()

    def callback(self, img1, img2, img3, state1, state2, state3, state4):
        bridge = CvBridge()
        try:
            cv_image1 = bridge.imgmsg_to_cv2(img1, desired_encoding="passthrough")
            cv_image2 = bridge.imgmsg_to_cv2(img2, desired_encoding="passthrough")
            cv_image3 = bridge.imgmsg_to_cv2(img3, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        img_timestamp = (img1.header.stamp.to_sec() + img2.header.stamp.to_sec() + img3.header.stamp.to_sec()) / 3
        np_state1 = np.array(state1.pos)
        np_state2 = np.array(state2.pos)
        np_position3 = np.array([state3.position.x, state3.position.y, state3.position.z])
        np_position4 = np.array([state4.position.x, state4.position.y, state4.position.z])
        np_quat3 = np.array([state3.quat.w, state3.quat.x, state3.quat.y, state3.quat.z])
        np_quat4 = np.array([state4.quat.w, state4.quat.x, state4.quat.y, state4.quat.z])

        tf = RotationTransformer(from_rep='quaternion', to_rep='rotation_6d')
        np_state3 = np.concatenate((np_position3, tf.forward(np_quat3)))
        np_state4 = np.concatenate((np_position4, tf.forward(np_quat4)))

        observation = {
            'left_cam': cv_image1,
            'right_cam': cv_image2,
            'table_cam': cv_image3,
            'rdda_left_obs': np_state1,
            'rdda_right_obs': np_state2,
            'left_arm_pose': np_state3,
            'right_arm_pose': np_state4,
            'timestamp': img_timestamp
        }

        # Append the new observation to the buffer
        self.obs_buffer.append(observation)
        self.obs_dict.update(observation)

    def get_obs(self):
        # Returns the last two observations
        return self.obs_buffer

    def run(self):
        # rospy.spin()
        pass
