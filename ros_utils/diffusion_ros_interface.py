#!/usr/bin/env python

import rospy
from rdda_interface.msg import RDDAPacket
from avatar_msgs.msg import PTIPacket
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Quaternion
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import cv2

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

        obs_subs = [
            self.images_obs_sub1,
            self.images_obs_sub2,
            self.images_obs_sub3,
            self.state_obs_left_gripper_sub,
            self.state_obs_right_gripper_sub,
            self.state_obs_left_arm_sub,
            self.state_obs_right_arm_sub,
        ]
        self.ts = message_filters.ApproximateTimeSynchronizer(obs_subs, 10)
        self.ts.registerCallback(self.obs_callback)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load checkpoint
        ckpt_path = input
        payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
        self.cfg = payload["cfg"]
        cls = hydra.utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # hacks for method-specific setup.
        self.frequency = 10
        self.dt = 1.0 / self.frequency
        self.steps_per_inference = 6

        if "diffusion" in self.cfg.name:
            # diffusion model
            self.policy: BaseImagePolicy
            self.policy = workspace.model
            if self.cfg.training.use_ema:
                self.policy = workspace.ema_model

            device = torch.device("cuda")
            self.policy.eval().to(device)

            # set inference params
            self.policy.num_inference_steps = 100  # DDIM inference iterations
            # (TODO: Double check this)
            self.policy.n_action_steps = self.policy.horizon - self.policy.n_obs_steps + 1
        else:
            raise NotImplementedError(f"Unknown model type: {self.cfg.name}")

    def obs_callback(self, img1, img2, img3, state1, state2, state3, state4):
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
            cv_image1 = bridge.imgmsg_to_cv2(img1, desired_encoding="passthrough")
            cv_image2 = bridge.imgmsg_to_cv2(img2, desired_encoding="passthrough")
            cv_image3 = bridge.imgmsg_to_cv2(img3, desired_encoding="passthrough")

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        img_timestamp = (img1.header.stamp.to_sec() + img2.header.stamp.to_sec() + img3.header.stamp.to_sec()) / 3
        state_timestamp = (
            state1.header.stamp.to_sec()
            + state2.header.stamp.to_sec()
            + state3.header.stamp.to_sec()
            + state4.header.stamp.to_sec()
        ) / 4
        rospy.loginfo(f"Image average timestamp: {img_timestamp}, State average timestamp: {state_timestamp}")

        np_state1 = np.concatenate((state1.wave, state1.pos_d))
        np_state2 = np.concatenate((state2.wave, state2.pos_d))
        assert np_state1.shape == (6,)
        assert np_state2.shape == (6,)
        
        # To use pytorch3d's conversion function, we need real part first quaternion
        np_position3 = np.array([state3.position.x, state3.position.y, state3.position.z])
        np_position4 = np.array([state4.position.x, state4.position.y, state4.position.z])
        np_quat3 = np.array([state3.quat.w, state3.quat.x, state3.quat.y, state3.quat.z])
        np_quat4 = np.array([state4.quat.w, state4.quat.x, state4.quat.y, state4.quat.z])
        
        tf = RotationTransformer(from_rep='quaternion', to_rep='rotation_6d')
        np_state3 = np.concatenate((np_position3, tf.forward(np_quat3)))
        np_state4 = np.concatenate((np_position4, tf.forward(np_quat4)))

        assert np_state3.shape == (9,)
        assert np_state4.shape == (9,)
        
        self.obs_dict = {
            "usb_cam_left": cv_image1,
            "usb_cam_right": cv_image2,
            "usb_cam_table": cv_image3,
            "left_gripper_state": np_state1,      # 6D 
            "right_gripper_state": np_state2,     # 6D
            "left_arm_state": np_state3,          # 9D
            "right_arm_state": np_state4,         # 9D
            "timestamp": img_timestamp,
        }

    def get_obs(self) -> dict:
        """
        A similar function as the env.get_obs in the orignial diffusion policy implementation.

        Returns:
            obs_dict (dict): a dictionary containing the synchronized observations.
        """
        # Since all the synchornization has been done by the filter, we can directly return the obs_dict
        obs_dict = copy.deepcopy(self.obs_dict)
        return obs_dict

    def publish_actions(self, action_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        """
        Publish the actions to the grippers and arms through ROS
        """
        assert len(action_tuple[0]) == len(action_tuple[1])
        assert len(action_tuple[1]) == len(action_tuple[2])
        assert len(action_tuple[2]) == len(action_tuple[3])

        def create_RDDAPacket(action):
            assert len(action) == 6
            packet = RDDAPacket()
            packet.wave = [action[0], action[1], action[2]]
            packet.pos_d = [action[3], action[4], action[5]]
            packet.timestamp = rospy.get_rostime()

            return packet

        def create_PTIPacket(packet, action):
            assert len(action) == 7
            packet = PTIPacket()
            packet.position.x = action[0]
            packet.position.y = action[1]
            packet.position.z = action[2]
            rotation_6d = action[3:]
            assert len(rotation_6d) == 6
            tf = RotationTransformer(from_rep='rotation_6d', to_rep='quaternion')
            quat = tf.forward(rotation_6d)
            packet.quat.w = quat[0]
            packet.quat.x = quat[1]
            packet.quat.y = quat[2]
            packet.quat.z = quat[3]

            packet.timestamp = rospy.get_rostime()

            return packet

        for step in range(len(action_tuple[0])):
            t = time.monotonic()
            left_gripper_packet = create_RDDAPacket(action_tuple[0][step])
            right_gripper_packet = create_RDDAPacket(action_tuple[1][step])
            left_arm_packet = create_PTIPacket(action_tuple[2][step])
            right_arm_packet = create_PTIPacket(action_tuple[3][step])

            self.left_gripper_master_pub.publish(left_gripper_packet)
            self.right_gripper_master_pub.publish(right_gripper_packet)
            self.left_smarty_arm_pub.publish(left_arm_packet)
            self.right_smarty_arm_pub.publish(right_arm_packet)
            print(f"Publishing time: {time.monotonic() - t} seconds")

            ## TODO Sleep??
            time.sleep(0.1)

    def parse_tensor_actions(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse the tensor action to the corresponding actions for the left gripper, right gripper, left arm and right arm
        """
        assert action.shape[-1] == 30
        left_gripper_action = action[:, 0:6]  # N x 6
        right_gripper_action = action[:, 6:12]  # N x 6
        left_arm_action = action[:, 12:21]  # N x 9
        right_arm_action = action[:, 21:30]  # N x 9
        return left_gripper_action, right_gripper_action, left_arm_action, right_arm_action

    def main(self):
        print("Warming up policy inference")
        obs = self.get_obs()
        with torch.no_grad():
            self.policy.reset()
            obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=self.cfg.task.shape_meta)
            obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
            result = self.policy.predict_action(obs_dict)
            action = result["action"][0].detach().to("cpu").numpy()
            assert action.shape[-1] == 2
            del result

        print("Ready!")
        # Feed the observation into the model
        try:
            # Don't know if we really need this (TODO)
            self.policy.reset()
            start_delay = 1.0
            eval_t_start = time.time() + start_delay
            t_start = time.monotonic() + start_delay
            # env.start_episode(eval_t_start)
            # wait for 1/30 sec to get the closest frame actually
            # reduces overall latency
            frame_latency = 1 / 30
            precise_wait(eval_t_start - frame_latency, time_func=time.time)
            print("Started!")
            iter_idx = 0
            while True:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + self.steps_per_inference) * self.dt

                # get obs
                print("get_obs")
                obs = self.get_obs()
                obs_timestamps = obs["timestamp"]
                print(f"Obs latency {time.time() - obs_timestamps[-1]}")
                with torch.no_grad():
                    s = time.monotonic()
                    self.policy.reset()
                    obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=self.cfg.task.shape_meta)
                    obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
                    result = self.policy.predict_action(obs_dict)
                    action = result["action"][0].detach().to("cpu").numpy()
                    print(f"Model inference time: {time.monotonic() - s} seconds")

                    # Timestamps check, if the action timestamp is in the past, skip it
                    action_offset = 0
                    action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset) * self.dt + obs_timestamps[-1]
                    action_exec_latency = 0.01
                    curr_time = time.time()
                    is_new = action_timestamps > (curr_time + action_exec_latency)
                    if np.sum(is_new) == 0:
                        # TODO: Not fully understand this part, skip it for now
                        # exceeded time budget, still do something
                        # this_target_poses = this_target_poses[[-1]]
                        # # schedule on next available step
                        next_step_idx = int(np.ceil((curr_time - eval_t_start) / self.dt))
                        action_timestamp = eval_t_start + (next_step_idx) * self.dt
                        print("Over budget", action_timestamp - curr_time)
                        # action_timestamps = np.array([action_timestamp])
                        continue
                    else:
                        action_commands = action[is_new]
                        action_timestamps = action_timestamps[is_new]

                    # Parse the tensor action (Need to double check this)
                    action_tuple = self.parse_tensor_actions(action_commands)

                    # Convert the numpy action to ROS message and publish
                    # TODO: Need to figure out how to publish a trajectory of actions (sync or async?)
                    self.publish_actions(action_tuple)

                    # wait for execution
                    precise_wait(t_cycle_end - frame_latency)
                    iter_idx += self.steps_per_inference
        except KeyboardInterrupt:
            print("Shutting down...")


if __name__ == "__main__":
    rospy.init_node("diffusion_ros_interface")
    diffusion_ros_interface = DiffusionROSInterface()
    diffusion_ros_interface.main()
    rospy.spin()
