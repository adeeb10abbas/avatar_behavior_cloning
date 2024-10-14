import copy
import time
# from multiprocessing.managers import SharedMemoryManager
import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import dill
import hydra
import tqdm
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from diffusion_policy.common import debug
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
import pickle


ckpt_path = "./training/diffusion_policy/data/outputs/2024.10.11/14.03.56_train_diffusion_unet_hybrid_haptic_image_teacher_aware/checkpoints/latest.ckpt"


@debug.iex
def main():
    print("Load checkpoint + cfg")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    cfg.policy.num_inference_steps = 8

    print("Load dataset")
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    print("Construct")
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # diffusion model
    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    torch.set_grad_enabled(False)

    normalizer = copy.deepcopy(policy.normalizer)

    device = torch.device('cuda')
    policy.eval().to(device)

    # TODO: Plot normalized values.

    count = len(dataset)
    # count = 33

    print("get all obs")
    obs_traj = {}
    for i in tqdm.trange(1, count, policy.n_obs_steps):
        batch = dataset[i]
        obs_dict = batch["obs"]
        obs_dict = normalizer.normalize(obs_dict)
        for k, v in obs_dict.items():
            if k not in obs_traj:
                obs_traj[k] = []
            obs_traj[k].extend(list(v.numpy()))

    # Convert to numpy arrays.
    obs_traj = {k: np.array(v) for k, v in obs_traj.items()}
    for field, value in obs_traj.items():
        _, D = value.shape
        indices = np.arange(D)
        label = [f"x[{x}]" for x in indices]

        plt.figure()
        plt.xlabel("obs index")
        plt.title(f"{field}: used for inference")
        plt.plot(value)

        # plt.show()
        file = f'saved_png_obs_{field}.png'
        plt.savefig(file)
        print(f"saved {file}")

    act_start = policy.n_obs_steps - 1
    act_end = act_start + policy.n_action_steps
    act_slice = slice(act_start, act_end)

    print("get gt vs inferred act")
    gt_actions = []
    inferred_actions = []

    act_normalizer = normalizer['action']

    # why is this so slow?!
    for i in tqdm.trange(1, count, policy.n_action_steps):
        batch = dataset[i]
        obs_dict, gt_action = batch["obs"], batch["action"]
        obs_dict_torched = dict_apply(
            obs_dict, lambda x: x.cuda().unsqueeze(0)
        )
        result = policy.predict_action(obs_dict_torched)
        # Use full action, and downselect.
        action = result["action_pred"][0].cpu()

        action = act_normalizer.normalize(action)
        gt_action = act_normalizer.normalize(gt_action)

        action = action.numpy()
        gt_action = gt_action.numpy()
        assert len(action) == len(gt_action)

        action = action[act_slice]
        gt_action = gt_action[act_slice]

        for action_j, gt_action_j in zip(action, gt_action):
            inferred_actions.append(action_j)
            gt_actions.append(gt_action_j)

    # Convert lists to NumPy arrays

    inferred = np.array(inferred_actions)
    ground_truth = np.array(gt_actions)

    act_fields = {
        "rdda_right_act": slice(0, 3),
        "right_arm_ee_pose": slice(3, 12),
        "rdda_left_act": slice(12, 15),
        "left_arm_ee_pose": slice(15, 24),
    }
    all_indices = np.arange(24)

    for field, indices in act_fields.items():
        label = [f"x[{x}]" for x in all_indices[indices]]

        plt.figure()
        plt.title(f"{field}: gt vs inferred")
        plt.plot([i[indices] for i in ground_truth], label=label, linewidth=2)
        plt.gca().set_prop_cycle(None)
        plt.plot([i[indices] for i in inferred], linestyle="--")
        plt.legend()

        # plt.show()
        file = f'saved_png_act_{field}.png'
        plt.savefig(file)
        print(f"saved {file}")

        # import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
