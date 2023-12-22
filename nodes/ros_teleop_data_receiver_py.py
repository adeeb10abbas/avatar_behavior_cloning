from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import numpy as np
import rclpy

class RosTeleopDataReceiver(Node):
    def __init__(self, side: str):
        super().__init__('ros_teleop_data_receiver')
        self.side = side
        assert self.side in ['left', 'right'], "Side must be either 'left' or 'right'"
        self.desired_finger_pos = None
        self.desired_eff_pose = None
        print(f"Waiting for {self.side} hand data...")
        joint_state_topic = f'/{self.side}_glove_joint_states'
        arm_pose_topic = f'/{self.side}_arm_pose'
        
        self.subscription_joint_state = self.create_subscription(
            JointState,
            joint_state_topic,
            self.finger_state_callback,
            100)
        self.subscription_eff_pose = self.create_subscription(
            Pose,
            arm_pose_topic,
            self.eff_pose_callback,
            100)
        print(f"Subscribed to {joint_state_topic} and {arm_pose_topic}")
        
        self.subscription_joint_state  # prevent unused variable warning
        self.subscription_eff_pose  # prevent unused variable warning

    def finger_state_callback(self, msg):
        self.desired_finger_pos = -1.0 * np.array([msg.position[1], msg.position[2], msg.position[0]])

    def eff_pose_callback(self, msg):
        self.desired_eff_pose = np.array([msg.position.x, msg.position.y, msg.position.z,
                                          msg.orientation.x, msg.orientation.y, msg.orientation.z,
                                          msg.orientation.w])
    
    def get_finger_pos(self, mode="real"):
        if mode == "simulated":
            # Define simulated finger positions here
            simulated_finger_pos = np.array([-0.5, 0.5, -0.5])  # Example simulated values
            noise_mean = 0.0
            noise_std = 0.25  # Adjust the standard deviation as needed

            # Add Gaussian noise to the simulated finger positions
            noisy_finger_pos = simulated_finger_pos + np.random.normal(noise_mean, noise_std, size=simulated_finger_pos.shape)

            return noisy_finger_pos
        return self.desired_finger_pos
    
    def get_eff_pose(self, mode="real"):
        if mode == "simulated":
            # Define simulated effector pose here
            simulated_eff_pose = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])  # Example simulated values
            noise_mean = 0.0
            noise_std = 0.1  # Adjust the standard deviation as needed

            # Add Gaussian noise to the simulated effector pose
            noisy_eff_pose = simulated_eff_pose + np.random.normal(noise_mean, noise_std, size=simulated_eff_pose.shape)

            return noisy_eff_pose
        return self.desired_eff_pose