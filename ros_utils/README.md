Hosts standalone ros1 noetic utilities used in this project. 

List of topics we're interested in - 

/camera/color/image_raw : sensor_msgs/Image


/clock : rosgraph_msgs/Clock
/pti_interface_right/joint_states : sensor_msgs/JointState

/pti_interface_right/panda_info : franka_control/PInfo 
int32 slow_catching_index
float64 external_load_mass

/pti_interface_right/pti_output : franka_control/PTIPacket
float64[] wave
geometry_msgs/Point position
geometry_msgs/Point angle
geometry_msgs/Twist twist
geometry_msgs/Quaternion quat
float64[] test

float64 local_stamp
float64 remote_stamp

float64[] est_ext_force
float64 robot_translation_mass
geometry_msgs/Point position_d


/rdda_right_master_input : rdda_interface/RDDAPacket
/rdda_right_master_output : rdda_interface/RDDAPacket
float64[] pos
float64[] vel
float64[] tau
float64[] wave
float64[] wave_aux
float64[] pressure
int32[] contact_flag
float64[] test
int16 error_signal
float64 local_stamp
float64 remote_stamp
float64 time_delay
float64[] delay_energy_reservior
float64[] pos_d
float64[] energy
float64[] ct

/right_arm_pose : geometry_msgs/Pose
/right_glove_joint_states : sensor_msgs/JointState
/right_gripper_joint_states : sensor_msgs/JointState

/right_smarty_arm_output : smarty_arm_interface/PTIPacket
float64[] wave
geometry_msgs/Point position
geometry_msgs/Point angle
geometry_msgs/Twist twist
geometry_msgs/Quaternion quat
float64[] test

float64 local_stamp
float64 remote_stamp

float64[] est_ext_force
float64 robot_translation_mass
geometry_msgs/Point position_d

observations 
        - images
        - haptics   - right - [pos_tensor, vel_tensor, tau_tensor, wave_tensor, pressure_tensor]
                    - left  - [pos_tensor, vel_tensor, tau_tensor, wave_tensor, pressure_tensor]
actions 
        - pose (franka is controlled via cartesian coordinates) (teleop side)
            --left_arm_pose
            --right_arm_pose
        - joint_state
            --left_glove
            --right_glove

structure 
```
+-- root (pickle file)
    |
    +-- 'observations' (Tensor)
    |   |
    |   +-- [Batch Size, Feature Dimension]
    |       |
    |       +-- Features concatenated from:
    |           |
    |           +-- Image data (flattened or feature extracted)
    |           +-- Haptic data from right glove
    |           +-- Haptic data from left glove
    |
    +-- 'actions' (Tensor)
        |
        +-- [Batch Size, Action Dimension]
            |
            +-- Actions concatenated from:
                |
                +-- Pose from right arm
                +-- Pose from left arm
                +-- joint state data from right glove
                +-- joint state data from left glove
```


Camera setup - 
-  right wrist - 108222250646
    ```
    roslaunch realsense2_camera rs_camera.launch camera:=cam_right_wrist serial_no:=108222250646
    ```
-  left wrist - 105322250285
    ```
    roslaunch realsense2_camera rs_camera.launch camera:=cam_left_wrist serial_no:=105322250285
    ```