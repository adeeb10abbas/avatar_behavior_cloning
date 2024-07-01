Hosts standalone ros1 noetic utilities used in this project. 
- the breakup of the ingested topics is mentioned in detail in this [doc](https://docs.google.com/document/d/10PGkL1rPhBMk4Fhg2gB8kr6wMFb5UPtAzT4CoX8u3q0/edit#heading=h.9mkckrvffoqm)

Camera setup - 
To launch all the cameras - `rosrun` need to fill out 
-  right wrist - 108222250646
    ```
    roslaunch realsense2_camera rs_camera.launch camera:=cam_right_wrist serial_no:=108222250646
    ```
-  left wrist - 105322250285
    ```
    roslaunch realsense2_camera rs_camera.launch camera:=cam_left_wrist serial_no:=105322250285
    ```
