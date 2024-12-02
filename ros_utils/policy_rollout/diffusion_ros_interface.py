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

from policy_wrapper import PolicyWrapper, ZarrPolicyWrapper
from shared_obs_dict_node import SubscriberNode
        
class DiffusionROSInterface:
    def __init__(self, ckpt_path, shared_obs_dict, fake_data=False, zarr_replay=True):
        rospy.init_node("diffusion_ros_interface")
        self.left_gripper_master_pub = rospy.Publisher("/_rdda_l_master_output", RDDAPacket, queue_size=10)
        self.right_gripper_master_pub = rospy.Publisher("/_rdda_right_master_output", RDDAPacket, queue_size=10)
        self.left_smarty_arm_pub = rospy.Publisher("/_left_smarty_arm_output", PTIPacket, queue_size=10)
        self.right_smarty_arm_pub = rospy.Publisher("/_right_smarty_arm_output", PTIPacket, queue_size=10)
        self.obs_dict = shared_obs_dict
        self.obs_history = {
            'left_cam': deque(maxlen=10),
            'right_cam': deque(maxlen=10),
            'table_cam': deque(maxlen=10),
            'rdda_left_obs': deque(maxlen=10),
            'rdda_right_obs': deque(maxlen=10),
            'left_arm_pose': deque(maxlen=10),
            'right_arm_pose': deque(maxlen=10),
            'timestamp': deque(maxlen=10),
        }
        self.fake_data = fake_data
        
        rospy.loginfo("Model Loaded!")
        self.obs_ready = False
        self.zarr_only = zarr_replay
        if zarr_replay:
            self.policy = ZarrPolicyWrapper(zarr_path="/app/avatar_behavior_cloning/eval/weights/_replay_buffer.zarr", ckpt_path=ckpt_path)
            rospy.loginfo("Streaming the data from zarr replay buffer")
        else:
            self.policy = PolicyWrapper(ckpt_path)
            
        ## alleged hacks 
        self.frequency = 10
        self.dt = 1.0 / self.frequency
        
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
                    # exit()
                # rospy.loginfo("Waiting for observations...")
                # time.sleep(0.5)
            except KeyboardInterrupt:
                rospy.loginfo("Shutting down...")
                exit()
            
        t = time.monotonic()
        obs_dict = copy.deepcopy(self.obs_dict)
        print(f"Get obs elapsed: {time.monotonic() - t} seconds")

        if self.fake_data == False:
            self.obs_history['left_cam'].append(obs_dict['left_cam'])
            self.obs_history['right_cam'].append(obs_dict['right_cam'])
            self.obs_history['table_cam'].append(obs_dict['table_cam'])
            self.obs_history['rdda_left_obs'].append(obs_dict['rdda_left_obs'])
            self.obs_history['rdda_right_obs'].append(obs_dict["rdda_left_obs"])
            self.obs_history['left_arm_pose'].append(obs_dict['left_arm_pose'])
            self.obs_history['right_arm_pose'].append(obs_dict['right_arm_pose'])
            self.obs_history['timestamp'].append(obs_dict['timestamp'])
        else:
            self.obs_history['left_cam'].append(np.random.rand(480, 640 ,3))
            self.obs_history['right_cam'].append(np.random.rand(480, 640 ,3))
            self.obs_history['table_cam'].append(np.random.rand(480, 640, 3))
            self.obs_history['rdda_left_obs'].append(np.random.rand(3))
            self.obs_history['rdda_right_obs'].append(np.random.rand(3))
            self.obs_history['left_arm_pose'].append(np.random.rand(9))
            self.obs_history['right_arm_pose'].append(np.random.rand(9))
            self.obs_history['timestamp'].append(time.time())
            rospy.logwarn("Using fake data...")
                
        
        obs_dict = dict_apply(self.obs_history, lambda x: np.array(x))
        # print(obs_dict['/left_cam/color/image_raw'].shape)
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
        
        
        delta = 1e-5

        if not np.allclose(interpolated_action[0], action_low_freq[0], atol=delta):
            rospy.logerr("First element not within delta")
            rospy.logerr(f"Interpolated action: {interpolated_action[0]}")
            rospy.logerr(f"Original action: {action_low_freq[0]}")
            raise ValueError("First element of interpolated action does not match the original action within delta")

        if not np.allclose(interpolated_action[-1], action_low_freq[-1], atol=delta):
            rospy.logerr("Last element not within delta")
            rospy.logerr(f"Interpolated action: {interpolated_action[-1]}")
            rospy.logerr(f"Original action: {action_low_freq[-1]}")
            raise ValueError("Last element of interpolated action does not match the original action within delta")
        
        return interpolated_action
    
    def publish_actions(self, action_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        """
        Publish the actions to the grippers and arms through ROS
        """
        assert len(action_tuple[0]) == len(action_tuple[1])
        assert len(action_tuple[1]) == len(action_tuple[2])
        assert len(action_tuple[2]) == len(action_tuple[3])

        def create_RDDAPacket(action):
            assert len(action) == 3
            packet = RDDAPacket()
            packet.pos = [action[0], action[1], action[2]] 
            # ^ Get the action from the tensor
            # packet.wave = [action[0], action[1], action[2]]
            # packet.pos_d = [action[3], action[4], action[5]]
            packet.header.stamp = rospy.get_rostime()
            # assert(len(packet.wave) == 3)
            # assert(len(packet.pos_d) == 3)

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

        action_publish_rate = 100
               
        #print("shape of left gripper action: ", action_tuple[0].shape)
        left_gripper_action = self.interpolate_action(action_tuple[0], action_publish_rate)
        #print("interpolated len of left gripper action: ", len(left_gripper_action))
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
            if (1.0/action_publish_rate - elapsed) > 0:
                time.sleep(1.0/action_publish_rate - elapsed)
            else:
                print("Publishing time exceeds the time budget")
        
        #print("All actions published successfully!")

    def parse_tensor_actions(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse the tensor action to the corresponding actions for the left gripper, right gripper, left arm and right arm
        """
        assert action.shape[-1] == 24

        # print("Action shape: ", action.shape, action.shape[-1])
        right_gripper_action = action[:, 0:3]  # N x 3
        right_arm_action = action[:, 3:12]  # N x 9
        
        left_gripper_action = action[:, 12:15]  # N x 3
        left_arm_action = action[:, 15:]  # N x 9
        return left_gripper_action, right_gripper_action, left_arm_action, right_arm_action

    def main(self):
        # print(f"Publishing elapsed: {elapsed} seconds")
        # print("Warming up policy inference")
        for i in range(2):
            obs = self.get_obs()

        # Warm up the policy
        self.policy.warm_it_up(obs)
        
        # print("Ready!")
        # Feed the observation into the model
        try:
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
                t_cycle_end = t_start + (iter_idx + self.policy.steps_per_inference) * self.dt

                # get obs
                obs = self.get_obs()
                if len(obs['timestamp']) < self.policy.n_obs_steps:
                    continue
                obs_timestamps = obs["timestamp"]
                print(f"Obs latency {time.time() - obs_timestamps[-1]}")
                with torch.no_grad():
                    action = self.policy.run_inference(obs)
                    # Timestamps check, if the action timestamp is in the past, skip it
                    action_offset = 0
                    action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset) * self.dt + obs_timestamps[-1]
                    action_exec_latency = 0.01
                    # make the logic so that when zarr it takes the obs_timestamps
                    # curr_time = time.time()
                    curr_time = obs_timestamps[-1]
                    is_new = action_timestamps > (curr_time + action_exec_latency)
                    if np.sum(is_new) == 0:
                        # TODO: Not fully understand this part, skip it for now
                        # exceeded time budget, still do something
                        # TODO: Ask Eric what does this even mean lmao 
                        # (i can't seem to tell what rui was going for here)
                        # -----------
                        
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
#                    print("Publishing actions...")
#                    print("Action timestamps: ", action_timestamps[-1])

                    self.publish_actions(action_tuple)

                    # wait for execution
                    if not self.zarr_only:
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += self.policy.steps_per_inference
                    
        except KeyboardInterrupt:
            print("Shutting down...")

def subscriber_node_process(shared_obs_dict):
    subscriber_node = SubscriberNode(shared_obs_dict)
    subscriber_node.run()

if __name__ == "__main__":
    manager = Manager()
    shared_obs_dict = manager.dict()

    subscriber_process = Process(target=subscriber_node_process, args=(shared_obs_dict,))
    subscriber_process.start()
    
    diffusion_process = Process(target=DiffusionROSInterface, args=("/app/avatar_behavior_cloning/eval/weights/epoch=0990-train_loss=0.000.ckpt", shared_obs_dict, False))
    diffusion_process.start()
    subscriber_process.join()
    diffusion_process.join()