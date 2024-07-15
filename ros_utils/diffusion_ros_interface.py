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

class SubscriberNode:
    def __init__(self, shared_obs_dict):
        rospy.init_node('observation_subscriber_node')

        self.images_obs_sub1 = message_filters.Subscriber("/usb_cam_left/image_raw", Image)
        self.images_obs_sub2 = message_filters.Subscriber("/usb_cam_right/image_raw", Image)
        self.images_obs_sub3 = message_filters.Subscriber("/usb_cam_table/image_raw", Image)
        self.state_obs_left_gripper_sub = message_filters.Subscriber("/rdda_l_master_input", RDDAPacket)
        self.state_obs_right_gripper_sub = message_filters.Subscriber("/rdda_right_master_input", RDDAPacket)
        self.state_obs_left_arm_sub = message_filters.Subscriber("/pti_interface_left/pti_output", PTIPacket)
        self.state_obs_right_arm_sub = message_filters.Subscriber("/pti_interface_right/pti_output", PTIPacket)

        self.obs_dict = shared_obs_dict
        obs_subs = [
            self.images_obs_sub1,
            self.images_obs_sub2,
            self.images_obs_sub3,
            self.state_obs_left_gripper_sub,
            self.state_obs_right_gripper_sub,
            self.state_obs_left_arm_sub,
            self.state_obs_right_arm_sub,
        ]
        self.ts = message_filters.ApproximateTimeSynchronizer(obs_subs, 10, slop=0.5)
        self.ts.registerCallback(self.callback)
        
        self.run()

    def callback(self, img1, img2, img3, state1, state2, state3, state4):
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
        # rospy.loginfo(f"Image average timestamp: {img_timestamp}, State average timestamp: {state_timestamp}")

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
        
        self.obs_dict['usb_cam_left'] = cv_image1
        self.obs_dict['usb_cam_right'] = cv_image2
        self.obs_dict['usb_cam_table'] = cv_image3
        self.obs_dict['rdda_left_obs'] = np_state1
        self.obs_dict['rdda_right_obs'] = np_state2
        self.obs_dict['left_arm_pose'] = np_state3
        self.obs_dict['right_arm_pose'] = np_state4
        self.obs_dict['timestamp'] = img_timestamp
        
        # rospy.loginfo("Observations synchronized!")
    
    def run(self):
        rospy.spin()

class DiffusionROSInterface:
    def __init__(self, input, shared_obs_dict, fake_data=False):
        rospy.init_node("diffusion_ros_interface")
        self.left_gripper_master_pub = rospy.Publisher("/rdda_l_master_output_", RDDAPacket, queue_size=10)
        self.right_gripper_master_pub = rospy.Publisher("/rdda_right_master_output_", RDDAPacket, queue_size=10)
        self.left_smarty_arm_pub = rospy.Publisher("/left_smarty_arm_output_", PTIPacket, queue_size=10)
        self.right_smarty_arm_pub = rospy.Publisher("/right_smarty_arm_output_", PTIPacket, queue_size=10)
        self.obs_dict = shared_obs_dict
        self.obs_history = {
            'usb_cam_left': deque(maxlen=10),
            'usb_cam_right': deque(maxlen=10),
            'usb_cam_table': deque(maxlen=10),
            'rdda_left_obs': deque(maxlen=10),
            'rdda_right_obs': deque(maxlen=10),
            'left_arm_pose': deque(maxlen=10),
            'right_arm_pose': deque(maxlen=10),
            'timestamp': deque(maxlen=10),
        }
        self.fake_data = fake_data

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load checkpoint
        ckpt_path = input
        payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
        self.cfg = payload["cfg"]
        print(self.cfg)
        cls = hydra.utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # hacks for method-specific setup.
        self.frequency = 10
        self.dt = 1.0 / self.frequency
        self.steps_per_inference = self.cfg['n_action_steps']

        if "diffusion" in self.cfg.name:
            # diffusion model
            self.policy: BaseImagePolicy
            self.policy = workspace.model
            if self.cfg.training.use_ema:
                self.policy = workspace.ema_model

            device = torch.device("cuda")
            self.policy.eval().to(device)
            rospy.loginfo("Policy evaluated")

            # set inference params
            # self.policy.num_inference_steps = 100  # DDIM inference iterations
            self.policy.n_action_steps = self.policy.horizon - self.policy.n_obs_steps + 1
        else:
            raise NotImplementedError(f"Unknown model type: {self.cfg.name}")
        
        rospy.loginfo("Model Loaded!")
        self.obs_ready = False
        self.main()

    def get_obs(self) -> dict:
        """
        A similar function as the env.get_obs in the orignial diffusion policy implementation.

        Returns:
            obs_dict (dict): a dictionary containing the synchronized observations.
        """
        # Since all the synchornization has been done by the filter, we can directly return the obs_dict
        t = time.monotonic()
        while (len(self.obs_dict) == 0 and not self.fake_data):
            try:
                if time.monotonic() - t > 2:
                    rospy.logerr("Timeout, no observations received")
                    exit()
                rospy.loginfo("Waiting for observations...")
                time.sleep(0.5)
            except KeyboardInterrupt:
                rospy.loginfo("Shutting down...")
                exit()
            
        t = time.monotonic()
        obs_dict = copy.deepcopy(self.obs_dict)
        print(f"Get obs elapsed: {time.monotonic() - t} seconds")

        if self.fake_data == False:
            self.obs_history['usb_cam_left'].append(obs_dict['usb_cam_left'])
            self.obs_history['usb_cam_right'].append(obs_dict['usb_cam_right'])
            self.obs_history['usb_cam_table'].append(obs_dict['usb_cam_table'])
            self.obs_history['rdda_left_obs'].append(obs_dict['rdda_left_obs'])
            self.obs_history['rdda_right_obs'].append(obs_dict["rdda_left_obs"])
            self.obs_history['left_arm_pose'].append(obs_dict['left_arm_pose'])
            self.obs_history['right_arm_pose'].append(obs_dict['right_arm_pose'])
            self.obs_history['timestamp'].append(obs_dict['timestamp'])
        else:
            self.obs_history['usb_cam_left'].append(np.random.rand(480, 640 ,3))
            self.obs_history['usb_cam_right'].append(np.random.rand(480, 640 ,3))
            self.obs_history['usb_cam_table'].append(np.random.rand(480, 640, 3))
            self.obs_history['rdda_left_obs'].append(np.random.rand(6))
            self.obs_history['rdda_right_obs'].append(np.random.rand(6))
            self.obs_history['left_arm_pose'].append(np.random.rand(9))
            self.obs_history['right_arm_pose'].append(np.random.rand(9))
            self.obs_history['timestamp'].append(time.time())
            rospy.logwarn("Using fake data...")
                
        
        obs_dict = dict_apply(self.obs_history, lambda x: np.array(x))
        # print(obs_dict['usb_cam_left'].shape)
        # print(obs_dict['left_arm_pose'].shape)
        
        # Pop the pulled observations
        # for key, item in self.obs_history.items():
        #     if len(item) > 1:
        #         self.obs_history[key].popleft()
        
        return obs_dict

    def interpolate_action(self, action_low_freq:np.ndarray, target_freq: int) -> np.ndarray:
        """
        Interpolate the low frequency actions to match the frequency of the robot control
        """
        # print("Input low frequency action shape: ", action_low_freq.shape)
        if action_low_freq.shape[0] < 2:
            print("Action length less than 2, no need to interpolate")
            return action_low_freq
        
        scale = target_freq // self.frequency
        assert scale > 0
        # print("Scale: ", scale)
        # Linear interpolation
        interpolated_action = np.zeros(((len(action_low_freq)-1) * scale + 1, action_low_freq.shape[-1]))
        for i in range(len(action_low_freq) - 1):
            interpolated_action[i*scale:i*scale+scale,:] = np.linspace(action_low_freq[i], action_low_freq[i+1], scale+1)[:-1]
        
        interpolated_action[-1] = action_low_freq[-1]
        # import pdb; pdb.set_trace()
        
        if np.any(interpolated_action[0] != action_low_freq[0]):
            print("First element not equal")
            print(interpolated_action)
            print(action_low_freq)
            exit()
        
        if np.any(interpolated_action[-1] != action_low_freq[-1]):
            print("Last element not equal")
            print(interpolated_action)
            print(action_low_freq)
            exit()
        assert np.all(interpolated_action[0] == action_low_freq[0])
        assert np.all(interpolated_action[-1] == action_low_freq[-1])
        # print("==============")
        # print(interpolated_action)
        # print("==============")
        # print(action_low_freq)
        # print("Interpolated action shape: ", interpolated_action.shape)
        
        return interpolated_action
    
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
            packet.header.stamp = rospy.get_rostime()

            return packet

        def create_PTIPacket(action):
            assert len(action) == 9
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

            packet.header.stamp = rospy.get_rostime()

            return packet

        action_publish_rate = 20
               
        left_gripper_action = self.interpolate_action(action_tuple[0], action_publish_rate)
        right_gripper_action = self.interpolate_action(action_tuple[1], action_publish_rate)
        left_arm_action = self.interpolate_action(action_tuple[2], action_publish_rate)
        right_arm_action = self.interpolate_action(action_tuple[3], action_publish_rate)
        
        for step in range(len(left_gripper_action)):
            t = time.monotonic()
            left_gripper_packet = create_RDDAPacket(left_gripper_action[step])
            right_gripper_packet = create_RDDAPacket(right_gripper_action[step])
            left_arm_packet = create_PTIPacket(left_arm_action[step])
            right_arm_packet = create_PTIPacket(right_arm_action[step])

            self.left_gripper_master_pub.publish(left_gripper_packet)
            self.right_gripper_master_pub.publish(right_gripper_packet)
            self.left_smarty_arm_pub.publish(left_arm_packet)
            self.right_smarty_arm_pub.publish(right_arm_packet)
            
            elapsed = time.monotonic() - t
            # print(f"Publishing elapsed: {elapsed} seconds")
            if (1.0/action_publish_rate - elapsed) > 0:
                time.sleep(1.0/action_publish_rate - elapsed)
            else:
                print("Publishing time exceeds the time budget")
        
        print("All actions published successfully!")

    def parse_tensor_actions(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse the tensor action to the corresponding actions for the left gripper, right gripper, left arm and right arm
        """
        # TODO: Double check the order
        assert action.shape[-1] == 30
        right_gripper_action = action[:, 0:6]  # N x 6
        left_gripper_action = action[:, 6:12]  # N x 6
        left_arm_action = action[:, 12:21]  # N x 9
        right_arm_action = action[:, 21:30]  # N x 9
        return left_gripper_action, right_gripper_action, left_arm_action, right_arm_action

    def main(self):
        print("Warming up policy inference")
        for i in range(self.policy.n_obs_steps):
            obs = self.get_obs()
        
        with torch.no_grad():
            self.policy.reset()
            print(obs['usb_cam_left'].shape)
            obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=self.cfg.task.shape_meta)
            
            obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
            print("In warming")
            print(obs_dict['left_arm_pose'].shape)
            # import pdb; pdb.set_trace()
            result = self.policy.predict_action(obs_dict)
            action = result["action"][0].detach().to("cpu").numpy()
            assert action.shape[-1] == 30
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
                obs = self.get_obs()
                if len(obs['timestamp']) < self.policy.n_obs_steps:
                    continue
                obs_timestamps = obs["timestamp"]
                print(f"Obs latency {time.time() - obs_timestamps[-1]}")
                with torch.no_grad():
                    self.policy.reset()
                    obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=self.cfg.task.shape_meta)
                    obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
                    
                    s = time.monotonic()
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
    manager = Manager()
    shared_obs_dict = manager.dict()

    subscriber_process = Process(target=SubscriberNode, args=(shared_obs_dict,))
    subscriber_process.start()
    
    diffusion_process = Process(target=DiffusionROSInterface, args=("../weights/epoch=0100-train_loss=0.012-001.ckpt", shared_obs_dict,))
    diffusion_process.start()
    
    subscriber_process.join()
    diffusion_process.join()
