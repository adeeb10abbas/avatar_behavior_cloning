import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import numpy as np

class RosTeleopDataReceiver(Node):
    def __init__(self):
        super().__init__('ros_teleop_data_receiver')
        self.desired_finger_pos = None
        self.desired_eff_pose = None
        
        self.subscription_joint_state = self.create_subscription(
            JointState,
            '/right_glove_joint_states',
            self.finger_state_callback,
            10)
        self.subscription_eff_pose = self.create_subscription(
            Pose,
            '/right_arm_pose',
            self.eff_pose_callback,
            10)
        
        self.subscription_joint_state  # prevent unused variable warning
        self.subscription_eff_pose  # prevent unused variable warning

    def finger_state_callback(self, msg):
        self.desired_finger_pos = -1.0 * np.array([msg.position[1], msg.position[2], msg.position[0]])
        # You can add additional processing here

    def eff_pose_callback(self, msg):
        self.desired_eff_pose = np.array([msg.position.x, msg.position.y, msg.position.z,
                                          msg.orientation.x, msg.orientation.y, msg.orientation.z,
                                          msg.orientation.w])
                                          