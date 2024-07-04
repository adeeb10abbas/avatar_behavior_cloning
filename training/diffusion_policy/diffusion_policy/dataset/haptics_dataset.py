# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import concurrent.futures
import copy
import multiprocessing

import h5py
import numpy as np
import torch
import zarr
from diffusion_policy.codecs.imagecodecs_numcodecs import (
    Jpeg2k,
    register_codecs,
)
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
    robomimic_abs_action_only_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import (
    BaseImageDataset,
    BaseLowdimDataset,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
# from diffusion_policy.model.common.rotation_transformer import (
#     RotationTransformer,
# )
from threadpoolctl import threadpool_limits
from tqdm import tqdm

register_codecs()


class AvatarHapticsImageDataset(BaseImageDataset):
    """
    A dataset class for low-dimensional & image data.

    This class inherits from `BaseImageDataset`, specifically designed
    for the IiwaWSGManipulation environment. It facilitates the creation of
    training and validation datasets, normalization, and action retrieval.
    """

    def __init__(
        self,
        dataset_path: str,
        shape_meta: dict[str, dict],
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: int | None = None,
        abs_action: bool = False,
        rotation_rep: str = "rotation_6d",  # ignored when abs_action=False
        seed: int = 42,
        val_ratio: float = 0.0,
        include_gripper_action: bool = True,
    ) -> None:
        """
        Initialize the dataset object, load data, and prepare the training dataset.

        Args:
        ----
            dataset_path: Path to the htf5 file containing the recorded episodes.
            shape_meta: Dictionary describing the action and observation space
                and dimensions.
            horizon: Length of the sequence of observations and actions to be sampled.
            pad_before: How many steps to pad before each sampled sequence.
            pad_after: How many steps to pad after each sampled sequence.
            n_obs_steps: The number of observation used for computing actions.
                Used in the sampler to sample correct length observations.
            abs_action: Whether we are using absolute actions or relative.
            rotation_rep: The method used for representing the rotation portion
                of the robot actions.
            seed: Seed for random number generator for reproducibility.
            val_ratio: The ratio of the total samples to use for validation.
                If 0, no validation set will be created.
            include_gripper_action: Whether to include the gripper action in
                the dataset.
        """
        # rotation_transformer = RotationTransformer(
        #     from_rep="quaternion", to_rep=rotation_rep
        # )
        self.obs_keys = [
        "usb_cam_right",
        "usb_cam_left",
        "usb_cam_table",
        "left_smarty_arm_output",
        "right_smarty_arm_output",
        "rdda_right_obs",
        "rdda_left_obs"
        ]
        self.all_keys = self.obs_keys.append("action")
        replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path=dataset_path, 
            keys= self.all_keys,
            store=zarr.MemoryStore()
        )

        rgb_keys = list()
        lowdim_keys = list()

        obs_shape_meta = shape_meta["obs"]

        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)
            else:
                raise ValueError(f"Unsupported key type f{key}.")

        key_first_k = dict()
        if n_obs_steps is not None:
            for key in self.obs_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self) -> BaseLowdimDataset:
        """
        Create and return a validation dataset.

        The validation set has the same underlying data using
        the complement of the training mask.

        Returns
        -------
            val_set: A new BaseLowdimDataset instance representing the validation data.
        """
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """
        Compute and return a normalizer object based on the replay buffer data.

        Args:
        ----
            mode: The mode used to fit the normalizer.
            **kwargs: Additional keyword arguments to pass to the normalizer's fit
                method.

        Returns:
        -------
            normalizer: The fitted normalizer.
        """
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer["action"])
        if self.abs_action:
            this_normalizer = robomimic_abs_action_only_normalizer_from_stat(
                stat
            )
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer["action"] = this_normalizer

        # Setup normalizers for the observation keys. Supported keys include
        # keys ending with "_translation" and "_rotation" which define a SE(3)
        # pose of a robot or object, "wsg" which is the gripper width, and
        # "spatial_force_at_ee"/"ee_forces" which describe the forces on
        # the robot's end-effector.
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            if key.endswith("_translation"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("_orientation") or key.endswith("_rotation") or "smarty" in key:
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key == "wsg":
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key in ["spatial_force_at_ee", "ee_forces"] or "rdda" in key:
                # TODO: Test to see if the values are properly normalized
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f"Unsupported key {key}")
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self) -> int:
        """Return the number of items of sampler container."""
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a specific item."""
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()

        for key in self.obs_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = (
                np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32)
                / 255.0
            )
            # T,C,H,W
            # del data[key]
        
        print(obs_dict.keys())
        print("---"*1000)
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            # del data[key]

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }
        return torch_data
