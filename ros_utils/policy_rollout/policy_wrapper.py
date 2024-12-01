
import torch
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import hydra
import dill
# import rospy
from omegaconf import OmegaConf
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.real_world.real_inference_util import get_real_obs_resolution, get_real_obs_dict
import numpy as np

class BasePolicyWrapper:
    def __init__(self) -> None:
        pass
    
    def warm_it_up(self, obs_dict):
        """
        raw obs_dict. We cast it here to torch
        """
        pass
    
    def torchify_obs(self, obs_dict):
        obs_dict_np = get_real_obs_dict(env_obs=obs_dict, shape_meta=self.cfg.task.shape_meta)
        obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
        return obs_dict
    
    def run_inference(self, obs_dict):
        pass
        
import rospy
class ZarrPolicyWrapper(BasePolicyWrapper):
    def __init__(self, zarr_path, ckpt_path) -> None:
        self.payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
        self.cfg_dataset = self.payload["cfg"].task.dataset
        self.cfg_dataset.dataset_path = zarr_path
        OmegaConf.set_struct(self.cfg_dataset.shape_meta.obs, False)
        self.cfg_dataset.shape_meta.obs["timestamp"] = {"shape": [1], "type": "low_dim"}
        OmegaConf.set_struct(self.cfg_dataset.shape_meta.obs, True)
        
        #Extract all the relevant params out of it
        self.num_inference_steps = self.payload["cfg"].policy.num_inference_steps
        ## TODO: make sure num_inference_steps is the same as steps_per_inference
        self.horizon = self.payload["cfg"].policy.horizon
        self.n_obs_steps = self.payload["cfg"].policy.n_obs_steps
        self.n_action_steps = self.payload["cfg"].policy.n_action_steps
        self.steps_per_inference = self.n_action_steps
        
        self.starting_policy_timestamp = None
        self.dataset = hydra.utils.instantiate(self.cfg_dataset)

    def reset(self, first_timestamp):
        self.starting_policy_timestamp = first_timestamp
        self.starting_timestamp = float(self.dataset[0]["obs"]["timestamp"][0]) + self.starting_policy_timestamp
    
    
    def get_actions_from_zarr(self, current_timestamp):
        """
        Note:
        Check if we're past the zarr action length
        If we are we start repeating the actions because we're running the bag in a loop
        """
        
        if current_timestamp >= len(self.dataset):
            current_timestamp = current_timestamp % len(self.dataset)
            
        actions = self.dataset[current_timestamp]["action"][:self.n_action_steps]
        return actions
    
    def run_inference(self, obs_dict):
        if self.starting_policy_timestamp is None:
            self.reset(obs_dict["timestamp"][0])
        
        current_timestamp = float(obs_dict["timestamp"][0]) - self.starting_policy_timestamp
        action = self.get_actions_from_zarr(int(current_timestamp))
        
        return action

# ckpt_path = "/app/avatar_behavior_cloning/eval/weights/epoch=0990-train_loss=0.000.ckpt"
# policy = ZarrPolicyWrapper(zarr_path="/app/avatar_behavior_cloning/eval/weights/_replay_buffer.zarr", ckpt_path=ckpt_path)
class PolicyWrapper(BasePolicyWrapper):
    def __init__(self, ckpt_path) -> None:
        self.ckpt_path = ckpt_path        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load checkpoint
        payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
        self.cfg = payload["cfg"]
        # print(self.cfg)
        cls = hydra.utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # hacks for method-specific setup.
        self.frequency = 10
        self.dt = 1.0 / self.frequency
        self.steps_per_inference = self.cfg['n_action_steps']
        # self.steps_per_inference = 4
        self.policy: BaseImagePolicy
        self.policy = workspace.model
        if self.cfg.training.use_ema:
            self.policy = workspace.ema_model

        device = torch.device("cuda")
        self.policy.eval().to(device)
        # rospy.loginfo("Policy evaluated")

        # set inference params
        self.policy.num_inference_steps = 16  # DDIM inference iterations
        self.policy.n_action_steps = self.cfg['n_action_steps']

    def torchify_obs(self, obs_dict):
        obs_dict_np = get_real_obs_dict(env_obs=obs_dict, shape_meta=self.cfg.task.shape_meta)
        obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
        return obs_dict
    
    def warm_it_up(self, obs_dict):
        """
        raw obs_dict. We cast it here to torch
        """
        obs_dict = self.torchify_obs(obs_dict)
        with torch.no_grad():
            result = self.policy.predict_action(obs_dict)
            action = result["action"][0].detach().to("cpu").numpy()
            assert action.shape[-1] == 24
            del result
        print("Warm up done! Ready for roll!")
    
    def run_inference(self, obs_dict):
        obs_dict = self.torchify_obs(obs_dict)
        result = self.policy.predict_action(obs_dict)
        action = result["action"][0].detach().to("cpu").numpy()
        return action
    