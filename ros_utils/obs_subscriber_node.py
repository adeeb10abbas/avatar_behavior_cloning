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


class SubscriberNode:
    def __init__(self, shared_obs_dict):
        rospy.init_node('observation_subscriber_node', anonymous=True)

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
        
        print("!!!!")


if __name__ == '__main__':
    manager = Manager()
    shared_obs_dict = manager.dict()
    
    process = Process(target=SubscriberNode, args=(shared_obs_dict,))
    process.start()
    process.join()