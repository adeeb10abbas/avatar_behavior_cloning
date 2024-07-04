import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from diffusion_policy.dataset.haptics_dataset import AvatarHapticsImageDataset

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch

class TestAvatarHapticsImageDataset(unittest.TestCase):

    @patch('diffusion_policy.common.replay_buffer.ReplayBuffer')
    @patch('diffusion_policy.common.sampler.SequenceSampler')
    def test_get_item(self, MockSequenceSampler, MockReplayBuffer):
        # Mocking the ReplayBuffer
        mock_replay_buffer = MagicMock()
        mock_replay_buffer.n_episodes = 10
        mock_replay_buffer.__getitem__.side_effect = lambda key: {
            "usb_cam_right": np.random.rand(10, 640, 480, 3),
            "usb_cam_left": np.random.rand(10, 640, 480, 3),
            "usb_cam_table": np.random.rand(10, 640, 480, 3),
            "left_smarty_arm_output": np.random.rand(10, 7),
            "right_smarty_arm_output": np.random.rand(10, 7),
            "rdda_right_obs": np.random.rand(10, 9),
            "rdda_left_obs": np.random.rand(10, 9),
            "action": np.random.rand(10, 26)
        }[key]
        
        MockReplayBuffer.copy_from_path.return_value = mock_replay_buffer
        
        # Mocking the SequenceSampler
        mock_sampler = MagicMock()
        mock_sampler.sample_sequence.return_value = {
            "usb_cam_right": np.random.rand(10, 640, 480, 3),
            "usb_cam_left": np.random.rand(10, 640, 480, 3),
            "usb_cam_table": np.random.rand(10, 640, 480, 3),
            "left_smarty_arm_output": np.random.rand(10, 7),
            "right_smarty_arm_output": np.random.rand(10, 7),
            "rdda_right_obs": np.random.rand(10, 9),
            "rdda_left_obs": np.random.rand(10, 9),
            "action": np.random.rand(10, 26)
        }
        
        MockSequenceSampler.return_value = mock_sampler
        
        # Define the shape_meta as given
        shape_meta = {
            "obs": {
                "usb_cam_right": {"shape": [3, 640, 480], "type": "rgb"},
                "usb_cam_left": {"shape": [3, 640, 480], "type": "rgb"},
                "usb_cam_table": {"shape": [3, 640, 480], "type": "rgb"},
                "left_smarty_arm_output": {"shape": [7], "type": "low_dim"},
                "right_smarty_arm_output": {"shape": [7], "type": "low_dim"},
                "rdda_right_obs": {"shape": [9], "type": "low_dim"},
                "rdda_left_obs": {"shape": [9], "type": "low_dim"}
            },
            "action": {"shape": [26]}
        }

        # Instantiate the dataset
        dataset = AvatarHapticsImageDataset(
            dataset_path="/home/ali/shared_volume/processed_bottle_pick_data/finer/teacher_aware_out/_replay_buffer.zarr",
            shape_meta=shape_meta,
            horizon=10,
            n_obs_steps=5,
            pad_before=2,
            pad_after=2,
            abs_action=False,
            seed=42,
            val_ratio=0.1,
            include_gripper_action=True
        )

        # Retrieve an item
        item = dataset[0]

        # Assertions
        self.assertIn("obs", item)
        self.assertIn("action", item)

        obs = item["obs"]
        # self.assertEqual(set(obs.keys()), set(shape_meta["obs"].keys()))

        # Check the shape and type of one of the rgb observations
        self.assertEqual(obs["usb_cam_right"].shape, (5, 3, 480, 640))
        self.assertEqual(obs["usb_cam_right"].dtype, torch.float32)

        # Check the shape and type of one of the low_dim observations
        self.assertEqual(obs["left_smarty_arm_output"].shape, (5, 7))
        self.assertEqual(obs["left_smarty_arm_output"].dtype, torch.float32)

        # Check the shape and type of the action
        self.assertEqual(item["action"].shape, (10, 26))
        self.assertEqual(item["action"].dtype, torch.float32)

if __name__ == '__main__':
    unittest.main()
