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
from collections import defaultdict

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.real_world.real_inference_util import get_real_obs_resolution, get_real_obs_dict
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_rollout_ros_sub import SubscriberNode

class DiffusionROSInterface:
    def __init__(self, input, subscriber, fake_data=False):
        # rospy.init_node("diffusion_ros_interface")
        self.left_gripper_master_pub = rospy.Publisher("/rdda_l_master_output_", RDDAPacket, queue_size=10)
        self.right_gripper_master_pub = rospy.Publisher("/rdda_right_master_output_", RDDAPacket, queue_size=10)
        self.left_smarty_arm_pub = rospy.Publisher("/left_smarty_arm_output", PTIPacket, queue_size=10)
        self.right_smarty_arm_pub = rospy.Publisher("/right_smarty_arm_output", PTIPacket, queue_size=10)
        self.obs_dict = None
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
        self.frequency = 7
        self.dt = 1.0 / self.frequency
        self.steps_per_inference = self.cfg['n_action_steps']

        self.policy: BaseImagePolicy
        self.policy = workspace.model
        if self.cfg.training.use_ema:
            self.policy = workspace.ema_model

        device = torch.device("cuda")
        self.policy.eval().to(device)
        rospy.loginfo("Policy evaluated")

        # set inference params
        self.policy.num_inference_steps = 16  # DDIM inference iterations
        self.policy.n_action_steps = self.policy.horizon - self.policy.n_obs_steps + 1
        
            # self.policy.n_action_steps = 1
            # self.policy.horizon - self.policy.n_obs_steps + 1
            # self.policy.n_action_s = self.cfg

        rospy.loginfo("Model Loaded!")
        self.obs_ready = False
        self.subscriber = subscriber
        self.main()

    def get_obs(self) -> dict:
        """
        A similar function as the env.get_obs in the orignial diffusion policy implementation.

        Returns:
            obs_dict (dict): a dictionary containing the synchronized observations.
        """
        # Since all the synchornization has been done by the filter, we can directly return the obs_dict
        t = time.monotonic()
        # while (len(self.obs_dict) == 0 and not self.fake_data):
        #     try:
        #         if time.monotonic() - t > 2:
        #             rospy.logerr("Timeout, no observations received")
        #             exit()
        #         rospy.loginfo("Waiting for observations...")
        #         time.sleep(0.5)
        #     except KeyboardInterrupt:
        #         rospy.loginfo("Shutting down...")
        #         exit()
            
        t = time.monotonic()
        self.obs_dict = self.subscriber.get_obs()
        self.obs_dict = dict_apply(self.obs_dict, lambda x: np.array(x))

        # obs_dict = dict_apply(copy.deepcopy(self.subscriber.get_obs()), lambda x: x)
        print(f"Get obs elapsed: {time.monotonic() - t} seconds")

        # if self.fake_data == False:
        #     self.obs_history['left_cam'].append(obs_dict['left_cam'])
        #     self.obs_history['right_cam'].append(obs_dict['right_cam'])
        #     self.obs_history['table_cam'].append(obs_dict['table_cam'])
        #     self.obs_history['rdda_left_obs'].append(obs_dict['rdda_left_obs'])
        #     self.obs_history['rdda_right_obs'].append(obs_dict["rdda_left_obs"])
        #     self.obs_history['left_arm_pose'].append(obs_dict['left_arm_pose'])
        #     self.obs_history['right_arm_pose'].append(obs_dict['right_arm_pose'])
        #     self.obs_history['timestamp'].append(obs_dict['timestamp'])
        # else:
        #     self.obs_history['left_cam'].append(np.random.rand(480, 640 ,3))
        #     self.obs_history['right_cam'].append(np.random.rand(480, 640 ,3))
        #     self.obs_history['table_cam'].append(np.random.rand(480, 640, 3))
        #     self.obs_history['rdda_left_obs'].append(np.random.rand(3))
        #     self.obs_history['rdda_right_obs'].append(np.random.rand(3))
        #     self.obs_history['left_arm_pose'].append(np.random.rand(9))
        #     self.obs_history['right_arm_pose'].append(np.random.rand(9))
        #     self.obs_history['timestamp'].append(time.time())
        #     rospy.logwarn("Using fake data...")
                
        
        # obs_dict = dict_apply(self.obs_history, lambda x: np.array(x))
        # print(obs_dict['/left_cam/color/image_raw'].shape)
        # print(obs_dict['left_arm_pose'].shape)
        
        # Pop the pulled observations
        # del self.obs_history
        # self.obs_history = {
        #     'left_cam': deque(maxlen=10),
        #     'right_cam': deque(maxlen=10),
        #     'table_cam': deque(maxlen=10),
        #     'rdda_left_obs': deque(maxlen=10),
        #     'rdda_right_obs': deque(maxlen=10),
        #     'left_arm_pose': deque(maxlen=10),
        #     'right_arm_pose': deque(maxlen=10),
        #     'timestamp': deque(maxlen=10),
        # }
        return self.obs_dict

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
               
        print("shape of left gripper action: ", action_tuple[0].shape)
        left_gripper_action = self.interpolate_action(action_tuple[0], action_publish_rate)
        print("interpolated len of left gripper action: ", len(left_gripper_action))
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
        assert action.shape[-1] == 18
        # print(action)
        # print(self.obs_dict)
        # zeros = np.zeros((13, 3)) #only because it's an ablation rollout atm. Not commanding the grippers
        # action = np.hstack((zeros, action[:, :9], zeros, action[:, 9:]))
        
        zeros = np.zeros((action.shape[0], 3))
        action = np.hstack((zeros, action[:, :9], zeros, action[:, 9:]))


        print("Action shape: ", action.shape, action.shape[-1])
        # import pdb; pdb.set_trace()i
        right_gripper_action = action[:, 0:3]  # N x 3
        right_arm_action = action[:, 3:12]  # N x 9
        
        left_gripper_action = action[:, 12:15]  # N x 3
        left_arm_action = action[:, 15:]  # N x 9
        return left_gripper_action, right_gripper_action, left_arm_action, right_arm_action

    def main(self):
        print("Warming up policy inference")
        for i in range(self.policy.n_obs_steps):
            obs = self.get_obs()
        
        with torch.no_grad():
            self.policy.reset()
            # print(obs['/left_cam/color/image_raw'].shape)
            obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=self.cfg.task.shape_meta)
            
            obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
            print("In warming")
            # print(obs_dict['left_arm_pose'].shape)
            # import pdb; pdb.set_trace()
            result = self.policy.predict_action(obs_dict)
            action = result["action"][0].detach().to("cpu").numpy()
            assert action.shape[-1] == 18
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
                # obs = self.subscriber.get_obs()[0]
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
                    action_exec_latency = 0.0
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

    # subscriber_process = Process(target=SubscriberNode, args=(shared_obs_dict,))
    # subscriber_process.start()
    # shared_obs_dict = defaultdict(lambda: None)
    subscriber = SubscriberNode(shared_obs_dict)
    
    # for i in range(100):
    #     time.sleep(2)
    #     print(shared_obs_dict)
    DiffusionROSInterface("/home/ali/Downloads/latest_table_only.ckpt", subscriber, False)
    # diffusion_process = Process(target=DiffusionROSInterface, args=("/home/ali/Downloads/latest_table_only.ckpt", shared_obs_dict, False))
    # diffusion_process.start()
    
    # subscriber_process.join()
    # diffusion_process.join()
