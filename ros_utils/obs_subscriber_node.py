#!/usr/bin/env python

import rospy
from rdda_interface.msg import RDDAPacket
from avatar_msgs.msg import PTIPacket
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Quaternion
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from multiprocessing import Process, Manager


#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import threading
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class SubscriberNode:
    def __init__(self, shared_obs_dict):
        rospy.init_node('observation_subscriber_node')
        self.shared_obs_dict = shared_obs_dict

        # Initialize storage for the latest messages
        self.img1 = None
        self.img2 = None
        self.img3 = None
        self.state1 = None
        self.state2 = None
        self.state3 = None
        self.state4 = None

        self.lock = threading.Lock()  # For thread safety

        # Create subscribers with their callbacks
        rospy.Subscriber("/left_cam/color/image_raw", Image, self.image1_callback)
        rospy.Subscriber("/right_cam/color/image_raw", Image, self.image2_callback)
        rospy.Subscriber("table_cam/color/image_raw", Image, self.image3_callback)
        rospy.Subscriber("/rdda_l_master_input", RDDAPacket, self.state1_callback)
        rospy.Subscriber("/rdda_right_master_input", RDDAPacket, self.state2_callback)
        rospy.Subscriber("/pti_interface_left/pti_output", PTIPacket, self.state3_callback)
        rospy.Subscriber("/pti_interface_right/pti_output", PTIPacket, self.state4_callback)

        # Start a timer to check for messages periodically
        rospy.Timer(rospy.Duration(0.1), self.check_and_process_messages)

    def image1_callback(self, msg):
        with self.lock:
            self.img1 = msg

    def image2_callback(self, msg):
        with self.lock:
            self.img2 = msg

    def image3_callback(self, msg):
        with self.lock:
            self.img3 = msg

    def state1_callback(self, msg):
        with self.lock:
            self.state1 = msg

    def state2_callback(self, msg):
        with self.lock:
            self.state2 = msg

    def state3_callback(self, msg):
        with self.lock:
            self.state3 = msg

    def state4_callback(self, msg):
        with self.lock:
            self.state4 = msg

    def check_and_process_messages(self, event):
        with self.lock:
            if all([self.img1, self.img2, self.img3, self.state1, self.state2, self.state3, self.state4]):
                # Get the timestamps
                img_times = [self.img1.header.stamp.to_sec(), self.img2.header.stamp.to_sec(), self.img3.header.stamp.to_sec()]
                state_times = [
                    self.state1.header.stamp.to_sec(),
                    self.state2.header.stamp.to_sec(),
                    self.state3.header.stamp.to_sec(),
                    self.state4.header.stamp.to_sec(),
                ]
                all_times = img_times + state_times
                max_time = max(all_times)
                min_time = min(all_times)
                time_diff = max_time - min_time

                # Check if messages are synchronized within 0.5 seconds
                if time_diff < 0.5:
                    self.process_messages()
                    # Reset messages after processing
                    self.img1 = self.img2 = self.img3 = None
                    self.state1 = self.state2 = self.state3 = self.state4 = None
                # else:
                #     rospy.logwarn("Messages not synchronized within 0.5 seconds. Skipping processing.")

    def process_messages(self):
        bridge = CvBridge()
        try:
            cv_image1 = bridge.imgmsg_to_cv2(self.img1, desired_encoding="passthrough")
            cv_image2 = bridge.imgmsg_to_cv2(self.img2, desired_encoding="passthrough")
            cv_image3 = bridge.imgmsg_to_cv2(self.img3, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        img_timestamp = (
            self.img1.header.stamp.to_sec()
            + self.img2.header.stamp.to_sec()
            + self.img3.header.stamp.to_sec()
        ) / 3
        state_timestamp = (
            self.state1.header.stamp.to_sec()
            + self.state2.header.stamp.to_sec()
            + self.state3.header.stamp.to_sec()
            + self.state4.header.stamp.to_sec()
        ) / 4

        np_state1 = np.array(self.state1.pos)
        np_state2 = np.array(self.state2.pos)
        assert np_state1.shape == (3,)
        assert np_state2.shape == (3,)

        np_position3 = np.array([
            self.state3.position.x,
            self.state3.position.y,
            self.state3.position.z,
        ])
        np_position4 = np.array([
            self.state4.position.x,
            self.state4.position.y,
            self.state4.position.z,
        ])
        np_quat3 = np.array([
            self.state3.quat.w,
            self.state3.quat.x,
            self.state3.quat.y,
            self.state3.quat.z,
        ])
        np_quat4 = np.array([
            self.state4.quat.w,
            self.state4.quat.x,
            self.state4.quat.y,
            self.state4.quat.z,
        ])

        tf = RotationTransformer(from_rep='quaternion', to_rep='rotation_6d')
        np_state3 = np.concatenate((np_position3, tf.forward(np_quat3)))
        np_state4 = np.concatenate((np_position4, tf.forward(np_quat4)))

        assert np_state3.shape == (9,)
        assert np_state4.shape == (9,)

        self.shared_obs_dict['left_cam'] = cv_image1
        self.shared_obs_dict['right_cam'] = cv_image2
        self.shared_obs_dict['table_cam'] = cv_image3
        self.shared_obs_dict['rdda_left_obs'] = np_state1
        self.shared_obs_dict['rdda_right_obs'] = np_state2
        self.shared_obs_dict['left_arm_pose'] = np_state3
        self.shared_obs_dict['right_arm_pose'] = np_state4
        self.shared_obs_dict['timestamp'] = img_timestamp

        # rospy.loginfo("Data processed and stored in shared_obs_dict.")

# if __name__ == '__main__':
#     rospy.init_node('observation_subscriber_node', anonymous=True)
#     from multiprocessing import Manager
#     manager = Manager()
#     shared_obs_dict = manager.dict()

#     node = SubscriberNode(shared_obs_dict)
#     rospy.spin()