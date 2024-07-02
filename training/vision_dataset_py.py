import copy

import numpy.typing as npt
import numpy as np
import torch
import wandb
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
)
from diffusion_policy.dataset.base_dataset import (
    BaseImageDataset,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from typing_extensions import Unpack

class ImageHapticsDataset(BaseImageDataset):
    def __init__(
        self,
        dataset_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        obs_keys: list[str] = ["images", "haptics"],
        action_key: str = "actions", 
        seed: int = 42,
        val_ratio: float = 0.0,
        use_wandb_artifact: bool = False,
        artifact_name: str = None,
        artifact_tag: str = None,
        ):
        super().__init__()
    
        self.replay_buffer = ReplayBuffer.copy_from_path(
            dataset_path, keys=obs_keys + [action_key]
        )

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        self.obs_keys = obs_keys
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.sampler = SequenceSampler(
                replay_buffer=self.replay_buffer,
                sequence_length=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=train_mask,
            )

    def get_validation_dataset(self):
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
    
    def get_normalizer(self, mode='limits', **kwargs):
        data = {}
        for key in self.obs_keys:
            data[key] = self.replay_buffer[key]
        data["action"] = self.replay_buffer[self.action_key]
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer
    
    def __len__(self) -> int:
        return len(self.sampler)
    
    def _sample_to_data(self, sample):
        obs_dict = {}
        for key in self.obs_keys:
            obs_dict[key] = sample[key]
        action = sample[self.action_key]  # T, D_a

        data = {
            'observation': obs_dict, # T
            'action': action # T, 
        }
        return data
    
    def __getitem__(self, idx: int):
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, np.array)
    