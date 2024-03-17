import h5py
import numpy as np

def condense_hd5(file_path):
    with h5py.File(file_path, 'r') as file:
            left_image_raw = file['observations/observations/usb_cam_left_image_raw'][:]
            right_image_raw = file['observations/observations/usb_cam_right_image_raw'][:]
            table_image_raw = file['observations/observations/usb_cam_table_image_raw'][:]
            
            # Haptics data consolidation 
            left_rdda_raw = file['observations/observations/rdda_l_master_output'][:]
            right_rddb_raw = file['observations/observations/rdda_right_master_output'][:]
            combine_haptics = np.concatenate((left_rdda_raw, right_rddb_raw), axis=0)
            print(combine_haptics.shape)
            # Initialize the new combined array
            combined_array_image = np.zeros((3, 3, 480, 640), dtype=left_image_raw.dtype)
            
            # Assign the datasets to the new array
            combined_array_image[0, :, :, :] = np.transpose(left_image_raw, (2, 0, 1))
            combined_array_image[1, :, :, :] = np.transpose(right_image_raw, (2, 0, 1))
            combined_array_image[2, :, :, :] = np.transpose(table_image_raw, (2, 0, 1))
            
            ## Actions side of it 
            right_arm_pose = file['actions/actions/right_arm_pose'][:]
            right_glove_joint_states = file['actions/actions/right_glove_joint_states'][:]
            left_arm_pose = file['actions/actions/left_arm_pose'][:]
            left_glove_joint_states = file['actions/actions/left_glove_joint_states'][:]
            combine_actions = np.concatenate((right_arm_pose, right_glove_joint_states, left_arm_pose, left_glove_joint_states), axis=0)
            print(combine_actions.shape)
            print(combine_haptics.shape)
            print(combine_actions.shape)
            print(combined_array_image.shape)
            # print(combined_array_image)
            # Save them combined arrays to a new file
            # with h5py.File('structured_data_cleaned.h5', 'w') as f:
            #     f.create_dataset('images', data=combined_array_image)
            #     f.create_dataset('actions', data=combine_actions)
            #     f.create_dataset('haptics', data=combine_haptics[:combine_actions.shape[0]])
            #     f.close()

condense_hd5('/home/adeebabbas/isolated/avatar_behavior_cloning/ros_utils/structured_data.h5')

# import h5py
# import numpy as np
# import zarr
# import os

# def condense_hd5_to_zarr(file_path, zarr_path):
#     with h5py.File(file_path, 'r') as file:
#         # Image data
#         left_image_raw = file['observations/observations/usb_cam_left_image_raw'][:]
#         right_image_raw = file['observations/observations/usb_cam_right_image_raw'][:]
#         table_image_raw = file['observations/observations/usb_cam_table_image_raw'][:]

#         # Haptics data consolidation
#         left_rdda_raw = file['observations/observations/rdda_l_master_output'][:]
#         right_rddb_raw = file['observations/observations/rdda_right_master_output'][:]
#         combine_haptics = np.concatenate((left_rdda_raw, right_rddb_raw), axis=0)
        
#         # Initialize the new combined array
#         combined_array_image = np.zeros((3, 3, 480, 640), dtype=left_image_raw.dtype)
#         # Assign the datasets to the new array
#         combined_array_image[0, :, :, :] = np.transpose(left_image_raw, (2, 0, 1))
#         combined_array_image[1, :, :, :] = np.transpose(right_image_raw, (2, 0, 1))
#         combined_array_image[2, :, :, :] = np.transpose(table_image_raw, (2, 0, 1))

#         # Actions data
#         right_arm_pose = file['actions/actions/right_arm_pose'][:]
#         right_glove_joint_states = file['actions/actions/right_glove_joint_states'][:]
#         left_arm_pose = file['actions/actions/left_arm_pose'][:]
#         left_glove_joint_states = file['actions/actions/left_glove_joint_states'][:]
#         combine_actions = np.concatenate((right_arm_pose, right_glove_joint_states, left_arm_pose, left_glove_joint_states), axis=0)

#     # Ensure the target directory exists
#     os.makedirs(zarr_path, exist_ok=True)
    
#     # Save the combined arrays to a new Zarr group within a single file
#     zarr_group = zarr.open_group(zarr_path, mode='w')
#     zarr_group.create_dataset('images', data=combined_array_image, chunks=True, compressor=zarr.Blosc(cname='zstd', clevel=3))
#     zarr_group.create_dataset('actions', data=combine_actions, chunks=True, compressor=zarr.Blosc(cname='zstd', clevel=3))
#     zarr_group.create_dataset('haptics', data=combine_haptics, chunks=True, compressor=zarr.Blosc(cname='zstd', clevel=3))

#     # Attach metadata directly to the root group
#     zarr_group.attrs['creation_date'] = 'YYYY-MM-DD'
#     zarr_group.attrs['description'] = 'This is a comprehensive dataset containing images, actions, and haptics.'
    
#     # Add more metadata as needed

# condense_hd5_to_zarr('/home/adeebabbas/isolated/avatar_behavior_cloning/ros_utils/structured_data.h5', '/home/adeebabbas/isolated/avatar_behavior_cloning/ros_utils/structured_data.zarr')