import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from diffusion_policy.dataset.haptics_dataset import AvatarHapticsImageDataset

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
            "left_arm_pose": np.random.rand(10, 7),
            "right_arm_pose": np.random.rand(10, 7),
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
            "left_arm_pose": np.random.rand(10, 7),
            "right_arm_pose": np.random.rand(10, 7),
            "rdda_right_obs": np.random.rand(10, 9),
            "rdda_left_obs": np.random.rand(10, 9),
            "action": np.random.rand(10, 26)
        }
        
        MockSequenceSampler.return_value = mock_sampler
        
        shape_meta = {
            "obs": {
                "usb_cam_right": {"shape": [3, 640, 480], "type": "rgb"},
                "usb_cam_left": {"shape": [3, 640, 480], "type": "rgb"},
                "usb_cam_table": {"shape": [3, 640, 480], "type": "rgb"},
                "left_arm_pose": {"shape": [7], "type": "low_dim"},
                "right_arm_pose": {"shape": [7], "type": "low_dim"},
                "rdda_right_obs": {"shape": [9], "type": "low_dim"},
                "rdda_left_obs": {"shape": [9], "type": "low_dim"}
            },
            "action": {"shape": [26]}
        }

        dataset = AvatarHapticsImageDataset(
            dataset_path="/home/ali/shared_volume/bottle_pick/teacher_aware_pkl/_replay_buffer.zarr",
            shape_meta=shape_meta,
            horizon=8,
            n_obs_steps=2,
            pad_before=2,
            pad_after=2,
            abs_action=False,
            seed=42,
            val_ratio=0.1,
            include_gripper_action=True
        )

        item = dataset[10]
        
        self.assertIn("obs", item)
        self.assertIn("action", item)

        obs = item["obs"]
        # Check for NaN values in observations
        for key, value in obs.items():
            self.assertFalse(torch.isnan(value).any(), f"NaN values found in {key}")

        # Check for NaN values in actions
        import pdb; pdb.set_trace()
        self.assertFalse(torch.isnan(item["action"]).any(), "NaN values found in action")

        # Check the shape and type of the action
        self.assertEqual(item["action"].shape, (10, 26))
        self.assertEqual(item["action"].dtype, torch.float32)

if __name__ == '__main__':
    unittest.main()
