import unittest
from unittest.mock import Mock
import numpy as np
from avatar_behavior_cloning.nodes.ros_teleop_data_receiver_py import RosTeleopDataReceiver

class TestRosTeleopDataReceiver(unittest.TestCase):

    def setUp(self):
        # Setup for each test
        self.right_node = RosTeleopDataReceiver('right')
        self.left_node = RosTeleopDataReceiver('left')

    def test_initialization(self):
        self.assertEqual(self.right_node.side, 'right', "Right side initialization failed")
        self.assertEqual(self.left_node.side, 'left', "Left side initialization failed")

    def test_finger_state_callback(self):
        mock_msg = Mock()
        mock_msg.position = [1.0, 2.0, 3.0]

        self.right_node.finger_state_callback(mock_msg)
        np.testing.assert_array_almost_equal(self.right_node.desired_finger_pos, np.array([-2.0, -3.0, -1.0]))

        self.left_node.finger_state_callback(mock_msg)
        np.testing.assert_array_almost_equal(self.left_node.desired_finger_pos, np.array([-2.0, -3.0, -1.0]))

    def test_eff_pose_callback(self):
        mock_msg = Mock()
        mock_msg.position.x, mock_msg.position.y, mock_msg.position.z = 1.0, 2.0, 3.0
        mock_msg.orientation.x, mock_msg.orientation.y, mock_msg.orientation.z, mock_msg.orientation.w = 0.1, 0.2, 0.3, 0.4

        self.right_node.eff_pose_callback(mock_msg)
        np.testing.assert_array_almost_equal(self.right_node.desired_eff_pose, np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4]))

        self.left_node.eff_pose_callback(mock_msg)
        np.testing.assert_array_almost_equal(self.left_node.desired_eff_pose, np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4]))

def main():
    unittest.main()
