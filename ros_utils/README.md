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

Other relevant - 
Helpful for zarr state action generation stuff  


```
rdda_stuff contains the following - 

pos_tensor shape: torch.Size([3])
pos_desired_tensor shape: torch.Size([3])
vel_tensor shape: torch.Size([3])
tau_tensor shape: torch.Size([3])
wave_tensor shape: torch.Size([3])
pressure_tensor shape: torch.Size([3])

obs_from_state shape: torch.Size([9])
action_stuff shape: torch.Size([6])


if mode == "teacher_aware":
    # We don't feed haptics to the model, it's only supposed to be implicitly learned
    obs_from_state = torch.cat([pos_tensor, vel_tensor, pos_desired_tensor], dim=0)
    action_stuff = torch.cat([pos_desired_tensor, wave_tensor], dim=0)
elif mode == "policy_aware":
    obs_from_state = torch.cat([pos_tensor, vel_tensor, tau_tensor, pos_desired_tensor, wave_tensor, pressure_tensor], dim=0)
    action_stuff = torch.cat([wave_tensor, pos_desired_tensor], dim=0)

all_tensors = torch.hstack([obs_from_state, action_stuff])
```
