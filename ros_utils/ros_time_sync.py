import message_filters
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Point, Quaternion, Twist, Pose
from cv_bridge import CvBridge
from rosgraph_msgs.msg import Clock
from functools import partial
import numpy as np
import rospy
import torch

def sync_time_callback(image_msg, joint_state_msg, clock_msg):
    # Convert image messages to torch tensors
    bridge = CvBridge()
    image_left = bridge.imgmsg_to_cv2(image_msg[0], desired_encoding="passthrough")
    image_right = bridge.imgmsg_to_cv2(image_msg[1], desired_encoding="passthrough")
    image_table = bridge.imgmsg_to_cv2(image_msg[2], desired_encoding="passthrough")
    image_left_tensor = torch.from_numpy(image_left)
    image_right_tensor = torch.from_numpy(image_right)
    image_table_tensor = torch.from_numpy(image_table)

    # Convert joint state message to torch tensor
    joint_state_tensor = torch.tensor(joint_state_msg.position)

    # Get the timestamp from the clock message
    timestamp = clock_msg.clock

    # Save all three tensors with the timestamp
    # TODO: Implement your saving logic here

    # Example: Save tensors to a file with the timestamp as the filename
    filename = f"{timestamp}.pt"
    torch.save((image_left_tensor, image_right_tensor, image_table_tensor, joint_state_tensor), filename)
topic_handlers = {
    "/usb_cam_left/image_raw": Image, # obs
    "/usb_cam_right/image_raw": Image, # obs
    "/usb_cam_table/image_raw": Image, # obs
    "/right_arm_pose": Pose, # action (user)
    "/left_arm_pose": Pose, # action (user)
    # "/right_glove_joint_states": , # action (user) 
    # "/left_glove_joint_states": left_glove_handler, # action (user)
    # "/rdda_right_master_output": rdda_packet_to_tensor, # obs
    # "/rdda_l_master_output": rdda_packet_to_tensor, # obs 
}

image_sub_left = message_filters.Subscriber("/usb_cam_left/image_raw", Image)
image_sub_right = message_filters.Subscriber("/usb_cam_right/image_raw", Image)
image_sub_table = message_filters.Subscriber("/usb_cam_table/image_raw", Image)
right_arm_pose_sub = message_filters.Subscriber("/right_arm_pose", Pose)
left_arm_pose_sub = message_filters.Subscriber("/left_arm_pose", Pose)


ts = message_filters.TimeSynchronizer([image_sub_left, image_sub_right, image_sub_table, right_arm_pose_sub, left_arm_pose_sub], 10)

ts.registerCallback(sync_time_callback)
rospy.spin()
