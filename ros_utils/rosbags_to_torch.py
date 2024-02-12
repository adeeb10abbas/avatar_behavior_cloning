import torch 
_path = "hardware/test_trajs/2024-01-27-19-08-13.bag"

import torch
import rosbag
import pickle
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

# Function to convert sensor_msgs/Image to a PyTorch tensor
def image_to_tensor(image_msg):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
    # Convert the CV image to a PyTorch tensor
    tensor_image = torch.tensor(np.array(cv_image), dtype=torch.float)
    return tensor_image


# Adjust this function for different ROS message types
def extract_data_from_bag(bag_path):
    data_tensors = []
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            if isinstance(msg, Image):  # Adjust this condition based on your data type
                tensor = image_to_tensor(msg)
                data_tensors.append(tensor)
    return data_tensors
    
# Main script
if __name__ == '__main__':
    bag_path = 'path_to_your_rosbag.bag'
    output_file = 'output_tensors.pkl'
    
    # Extract data from the ROS bag and convert to tensors
    data_tensors = extract_data_from_bag(bag_path)
    
    # Save the tensors to a pickle file
    save_tensors_as_pickle(data_tensors, output_file)
    print(f"Data extracted and saved to {output_file}")
