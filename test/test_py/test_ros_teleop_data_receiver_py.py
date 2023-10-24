# test_ros_data_receiver.py

import sys
import unittest
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import numpy as np

from avatar_behavior_cloning.nodes.ros_teleop_data_receiver_py import RosTeleopDataReceiver

class TestRosTeleopDataReceiver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize the ROS context for the test node
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        # Shutdown the ROS context
        rclpy.shutdown()

    def setUp(self):
        self.node = RosTeleopDataReceiver()

    def tearDown(self):
        self.node.destroy_node()

    def test_finger_state_callback(self):
        # Mock a JointState message
        test_msg = JointState()
        test_msg.position = [0.1, 0.2, 0.3]

        # Call the callback manually
        self.node.finger_state_callback(test_msg)

        # Assert the desired position has been updated
        self.assertIsNotNone(self.node.desired_finger_pos)
        np.testing.assert_array_almost_equal(
            self.node.desired_finger_pos, 
            -1.0 * np.array([test_msg.position[1], test_msg.position[2], test_msg.position[0]])
        )

    def test_eff_pose_callback(self):
        # Mock a Pose message
        test_msg = Pose()
        test_msg.position.x = 1.0
        test_msg.position.y = 2.0
        test_msg.position.z = 3.0
        test_msg.orientation.x = 0.0
        test_msg.orientation.y = 0.0
        test_msg.orientation.z = 0.0
        test_msg.orientation.w = 1.0

        # Call the callback manually
        self.node.eff_pose_callback(test_msg)

        # Assert the desired pose has been updated
        self.assertIsNotNone(self.node.desired_eff_pose)
        np.testing.assert_array_almost_equal(
            self.node.desired_eff_pose, 
            np.array([
                test_msg.position.x, test_msg.position.y, test_msg.position.z,
                test_msg.orientation.x, test_msg.orientation.y, test_msg.orientation.z,
                test_msg.orientation.w
            ])
        )

if __name__ == '__main__':
    unittest.main()
