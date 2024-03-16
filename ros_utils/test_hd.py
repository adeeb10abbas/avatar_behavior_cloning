import h5py

def print_hdf5_contents_shapes_and_counts(file_path):
    with h5py.File(file_path, 'r') as f:
        def print_name_shape_and_count(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Calculate the count as the product of the dataset's shape dimensions
                count = obj.size  # Alternatively, use np.prod(obj.shape) for older h5py versions
                print(f"Dataset: {name}, Shape: {obj.shape}, Count: {count}")
            else:
                print(f"Group: {name}")
        f.visititems(print_name_shape_and_count)

# Example usage:
file_path = '/home/adeebabbas/isolated/avatar_behavior_cloning/ros_utils/structured_data_cleaned.h5'
print_hdf5_contents_shapes_and_counts(file_path)
