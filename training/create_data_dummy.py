import pickle
import numpy as np
import torch

def generate_gaussian_noise(iterations, shape):
    """
    Generate Gaussian noise for a given shape, repeated over a specified number of iterations.
    """
    return [np.random.normal(size=shape) for _ in range(iterations)]

def create_complex_dummy_data():
    iterations = 100
    # Define shapes for each type of data
    image_shape = (224, 224, 3)
    haptic_shape = (6,)
    pose_shape = (7,)
    glove_shape = (5,)
    
    data = {
        'observations': {
            'images': generate_gaussian_noise(iterations, image_shape),
            'haptics': {
                'right': generate_gaussian_noise(iterations, haptic_shape),
                'left': generate_gaussian_noise(iterations, haptic_shape),
            },
        },
        'actions': {
            'pose': {
                'right_arm_pose': generate_gaussian_noise(iterations, pose_shape),
                'left_arm_pose': generate_gaussian_noise(iterations, pose_shape),
            },
            'joint_state': {
                'right_glove': generate_gaussian_noise(iterations, glove_shape),
                'left_glove': generate_gaussian_noise(iterations, glove_shape),
            },
        }
    }

    # Convert data to PyTorch tensors
    for key in data['observations']:
        if key == 'images':
            data['observations'][key] = [torch.tensor(item, dtype=torch.float32) for item in data['observations'][key]]
        else:  # For haptics
            for side in data['observations'][key]:
                data['observations'][key][side] = [torch.tensor(item, dtype=torch.float32) for item in data['observations'][key][side]]
    
    for key in data['actions']:
        for action in data['actions'][key]:
            data['actions'][key][action] = [torch.tensor(item, dtype=torch.float32) for item in data['actions'][key][action]]
    
    # Save the data to a .pkl file
    with open('complex_dummy_dataset.pkl', 'wb') as f:
        pickle.dump(data, f)

create_complex_dummy_data()

