# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import sys
import time
import copy
import math
import cv2

from avatar_behavior_cloning.utils.geometry_utils import *
from transforms3d.quaternions import *
from transforms3d.euler import *
from transforms3d.axangles import *

import matplotlib.pyplot as plt
import numpy as np

from collections import deque

joystick_lowpass_filter = deque([np.zeros(6)], maxlen=30)
vr_has_init = False
rec_action_pose = None
vr_frame_origin = None
oculus_reader = None

def get_keyboard_input():
    finger_state = ""

    action = np.zeros(6)
    trans_scale = 0.01
    rot_scale = 0.05
    cv2.imshow("robot_teleop", np.zeros((64, 64, 3), dtype=np.uint8))
    k = cv2.waitKey(1)

    if k == ord("w"):
        action[0] = trans_scale
    elif k == ord("s"):
        action[0] = -trans_scale
    elif k == ord("a"):
        action[1] = trans_scale
    elif k == ord("d"):
        action[1] = -trans_scale
    elif k == ord("q"):
        action[2] = trans_scale
    elif k == ord("e"):
        action[2] = -trans_scale

    elif k == ord("i"):
        action[3] = rot_scale
    elif k == ord("k"):
        action[3] = -rot_scale
    elif k == ord("j"):
        action[4] = rot_scale
    elif k == ord("l"):
        action[4] = -rot_scale
    elif k == ord("u"):
        action[5] = rot_scale
    elif k == ord("o"):
        action[5] = -rot_scale

    elif k == ord("["):
        finger_state = "thumb_up"
    elif k == ord("]"):
        finger_state = "thumb_down"
    elif k == ord(";"):
        finger_state = "thumb_open"
    elif k == ord("'"):
        finger_state = "thumb_close"
    elif k == ord("."):
        finger_state = "index_open"
    elif k == ord("/"):
        finger_state = "index_close"
    return action, finger_state


def get_mouse_input(action, robot_id):
    global joystick_lowpass_filter
    event = spacenav.poll()
    finger_state = ""
    action = np.zeros_like(action)
    trans_scale = 0.02
    rot_scale = 0.05
    normalize_scale = 512
    deadband = 0.1

    if event is spacenav.MotionEvent:
        if event.x != -2147483648:
            action[0] = event.z / normalize_scale
            action[1] = -event.x / normalize_scale
            action[2] = event.y / normalize_scale
            action[3] = event.rz / normalize_scale
            action[4] = -event.rx / normalize_scale
            action[5] = event.ry / normalize_scale

            if (np.abs(action[:3]) > deadband).sum() == 0:
                action[:3] = 0
            if (np.abs(action[3:]) > deadband).sum() == 0:
                action[3:] = 0

            action[:3] *= trans_scale
            action[3:] *= rot_scale

            spacenav_prev_action[:] = action
            # print("action:", action)

        if hasattr(event, "pressed"):
            if event.pressed:
                finger_state = "close"
    print("action: ", joystick_lowpass_filter)
    joystick_lowpass_filter.append(action)
    action = np.mean(list(joystick_lowpass_filter), axis=0)
    return action, finger_state


def scale_action_pose(pose, factor=1.0):
    # scale both translation and action
    scaled_pose = pose.copy()
    scaled_pose[:3, 3] = pose[:3, 3] * factor
    axangle = mat2axangle(scaled_pose[:3, :3])
    scaled_pose[:3, :3] = axangle2mat(axangle[0], axangle[1] * factor)
    return scaled_pose


def constrain_action_pose_new(pose):
    # constraint to have no rotation motion
    t_thre = 0.03
    r_thre = 0.3
    trans = pose[:3, 3]
    euler = np.array(mat2euler(pose[:3, :3]))

    t_scale = min(1, t_thre / np.max(np.abs(trans)))
    r_scale = min(1, r_thre / np.max(np.abs(euler)))

    euler = euler * r_scale
    trans = trans * t_scale
    scaled_pose = pose.copy()
    scaled_pose[:3, :3] = euler2mat(*euler)
    scaled_pose[:3, 3] = trans

    return scaled_pose


def constrain_action_pose(pose):
    # constraint to have no rotation motion
    scaled_pose = pose.copy()
    scaled_pose[:3, 3] = pose[:3, 3]
    euler = mat2euler(scaled_pose[:3, :3])
    scaled_pose[:3, :3] = euler2mat(0, 0, 0)  # euler[0] euler[1]
    return scaled_pose


# :)
# base camera frame has x left, y up and z backward
# robot base frame has x forward, y left, and z up
# end effector frame has x up, y right, z forward
# oculus headset frame has x right, y up, z back


# represent robot base frame in the VR camera frame, or robot base to VR
offset_pose_from_vr_to_base = np.eye(4)
offset_pose_from_vr_to_base[:3, :3] = (rotY(np.pi / 2)[:3, :3] @ rotX(np.pi / 2)[:3, :3]).T
offset_pose_from_vr_to_base[:3, 3] = [0.0, -0.8, 0]
# approximately [0.1, -0.0, 0.05]


def get_vr_to_human_frame():
    offset_pose_from_base_to_table = np.eye(4)
    offset_pose_from_base_to_table[:3, 3] = [0.5, 0.0, 0.2]  #   [0.5, -0.8, 0.2]
    return offset_pose_from_base_to_table


def get_vr_to_local_control_frame(pose):
    offset_pose_vr = np.eye(4)
    offset_pose_vr[:3, :3] = rotY(np.pi)[:3, :3] @ pose[:3, :3] @ offset_pose_from_vr_to_base[:3, :3] 
    offset_pose_vr[:3, 3] = offset_pose_from_vr_to_base[:3, 3] + pose[:3, 3]
    return offset_pose_vr


def get_vr_input():
    """Transform oculus motion to actual end effector motion
    note (1) put VR to be at a good place visible and check the drift
    (2) the headset location needs to be fixed because it is used to transform w.r.t base
    """

    global vr_has_init, vr_frame_origin, rec_action_pose, oculus_reader
    ctrl_frame_transform = se3_inverse(rotY(-np.pi / 2) @ rotZ(np.pi / 2) )
    if not oculus_reader:
        try:
            from misc.reader import OculusReader
            oculus_reader = OculusReader()
        except:
            print("please install oculus reader first")
            pass

    if not vr_has_init:
        """initialize the joystick frame, the origin is at the headset frame
        x right, y up, z back
        """
        print("VR Init, click rightTrig")
        while True:
            time.sleep(0.01)
            transformations, buttons = oculus_reader.get_transformations_and_buttons()

            if "rightTrig" not in buttons:
                continue

            if buttons["rightTrig"][0] > 0.1:
                rec_action_pose = transformations["r"]
                vr_frame_origin = se3_inverse(transformations["r"])
                vr_has_init = True
                break
        print("finish VR init")

    transformations, buttons = oculus_reader.get_transformations_and_buttons()
    if "r" in transformations:
        rec_action_pose = transformations["r"]

    # joystick frame w.r.t the headset frame, X_VJ
    # the local rotation is the same as the robot base frame
    # J has x forward, y left, and z up (while holding with hole forward)
    abs_action_pose = rec_action_pose @ rotZ(np.pi / 2) @ rotY(-np.pi / 2) @ rotZ(np.pi)

    # joystick frame with respect to robot base X_BJ
    offset_pose_vr = get_vr_to_local_control_frame(get_vr_to_human_frame())
    abs_action_pose = offset_pose_vr @ abs_action_pose @ rotX(-np.pi)

    # joystick frame with respect to the previous joystick frame X_JJ' (local transform)
    action_pose = se3_inverse(vr_frame_origin).dot(rec_action_pose)

    # similarity transform the local frame X_JJ' to follow real-world motion
    # The transformed motion would now follow panda end effector local frame with
    # J' has x up, y right, and z forward
    action_pose = se3_inverse(ctrl_frame_transform).dot(action_pose).dot(ctrl_frame_transform)

    # update the joystick origin to be the previous frame
    vr_frame_origin = rec_action_pose

    # limit the size of the relative action pose
    action_pose = constrain_action_pose(action_pose)

    # misc control on fingers, reset
    finger_state = ""
    if buttons["B"]:
        finger_state = "close"
    if buttons["A"]:
        finger_state = "open"

    # use pose at the headset origin otherwise joystick origin
    command_goal_pose = buttons["rightTrig"][0] > 0.1

    global_action_pose = np.zeros(6)
    global_action_pose[:3] = abs_action_pose[:3, 3]
    global_action_pose[3:] = mat2euler(abs_action_pose[:3, :3])

    return global_action_pose, finger_state
