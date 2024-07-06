#!/usr/bin/env python

import rospy
from rdda_interface.msg import RDDAPacket
from avatar_msgs.msg import PTIPacket
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import cv2

import torch
import hydra
import numpy as np
import dill
from torchvision import transforms
from typing import Tuple

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply

class DiffusionROSInterface:
    def __init__(self, input, output):
        self.left_gripper_master_pub = rospy.Publisher("/rdda_left_master_output", RDDAPacket, queue_size=10)
        self.right_gripper_master_pub = rospy.Publisher("/rdda_right_master_output", RDDAPacket, queue_size=10)
        self.left_smarty_arm_pub = rospy.Publisher("/left_smarty_arm_output", PTIPacket, queue_size=10)
        self.right_smarty_arm_pub = rospy.Publisher("/right_smarty_arm_output", PTIPacket, queue_size=10)
        self.images_obs_sub1 = message_filters.Subscriber("/usb_cam_left/image_raw", Image)
        self.images_obs_sub2 = message_filters.Subscriber("/usb_cam_right/image_raw", Image)
        self.images_obs_sub3 = message_filters.Subscriber("/usb_cam_table/image_raw", Image)
        self.state_obs_left_gripper_sub = message_filters.Subscriber("/rdda_left_master_input", RDDAPacket)
        self.state_obs_right_gripper_sub = message_filters.Subscriber("/rdda_right_master_input", RDDAPacket)
        self.state_obs_left_arm_sub = message_filters.Subscriber("/left_smarty_arm_input", PTIPacket)
        self.state_obs_right_arm_sub = message_filters.Subscriber("/right_smarty_arm_input", PTIPacket)

        obs_subs = [self.images_obs_sub1, self.images_obs_sub2, self.images_obs_sub3, self.state_obs_left_gripper_sub, self.state_obs_right_gripper_sub, self.state_obs_left_arm_sub, self.state_obs_right_arm_sub]
        self.ts = message_filters.ApproximateTimeSynchronizer(obs_subs, 10)
        self.ts.registerCallback(self.image_callback)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load checkpoint
        ckpt_path = input
        payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
        self.cfg = payload['cfg']
        cls = hydra.utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # hacks for method-specific setup.
        action_offset = 0
        delta_action = False
        if 'diffusion' in self.cfg.name:
            # diffusion model
            self.policy: BaseImagePolicy
            self.policy = workspace.model
            if self.cfg.training.use_ema:
                self.policy = workspace.ema_model

            device = torch.device('cuda')
            self.policy.eval().to(device)

            # set inference params
            self.policy.num_inference_steps = 100 # DDIM inference iterations
            self.policy.n_action_steps = self.policy.horizon - self.policy.n_obs_steps + 1
        else:
            raise NotImplementedError(f"Unknown model type: {self.cfg.name}")


    def image_callback(self, img1, img2, img3, state1, state2, state3, state4):
        """
        Args:
            img1 (Image): left camera image
            img2 (Image): right camera image
            img3 (Image): table camera image
            state1 (RDDAPacket): rdda left gripper state
            state2 (RDDAPacket): rdda right gripper state
            state3 (PTIPacket): left arm state
            state4 (PTIPacket): right arm state
        """
        bridge = CvBridge()
        try:
            cv_image1 = bridge.imgmsg_to_cv2(img1, desired_encoding='passthrough')
            cv_image2 = bridge.imgmsg_to_cv2(img2, desired_encoding='passthrough')
            cv_image3 = bridge.imgmsg_to_cv2(img3, desired_encoding='passthrough')

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        self.obs_dict = {
            'usb_cam_left': cv_image1,
            'usb_cam_right': cv_image2,
            'usb_cam_table': cv_image3,
            'left_gripper_state': state1,
            'right_gripper_state': state2,
            'left_arm_state': state3,
            'right_arm_state': state4
        }


    def TensorToMsg(
        self, rdda_tensor1, rdda_tensor2, pti_tensor1, pti_tensor2
    ) -> Tuple[RDDAPacket, RDDAPacket, PTIPacket, PTIPacket]:
        """
        Convert tensor to ROS message
        """
        rdda_packet1 = RDDAPacket()
        rdda_packet2 = RDDAPacket()
        pti_packet1 = PTIPacket()
        pti_packet2 = PTIPacket()

        return rdda_packet1, rdda_packet2, pti_packet1, pti_packet2

    def main(self):
        # Feed the observation into the model
        while True:
            with torch.no_grad():
                # Do model inference here (TODO)
                self.policy.reset()
                obs_dict_np = get_real_obs_dict(
                    env_obs=self.obs_dict, shape_meta=self.cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
                result = self.policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 2
                del result

                rdda_tensor1, rdda_tensor2, pti_tensor1, pti_tensor2 = None, None, None, None
                rdda_packet1, rdda_packet2, pti_packet1, pti_packet2 = self.TensorToMsg(
                    rdda_tensor1, rdda_tensor2, pti_tensor1, pti_tensor2
                )

            # Publish the action
            self.left_gripper_master_pub.publish(rdda_packet1)
            self.right_gripper_master_pub.publish(rdda_packet2)
            self.left_smarty_arm_pub.publish(pti_packet1)
            self.right_smarty_arm_pub.publish(pti_packet2)


if __name__ == "__main__":
    rospy.init_node("diffusion_ros_interface")
    diffusion_ros_interface = DiffusionROSInterface()
    diffusion_ros_interface.main()
    rospy.spin()