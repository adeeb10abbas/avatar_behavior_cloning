
import torch
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import hydra
import dill
# import rospy
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.real_world.real_inference_util import get_real_obs_resolution, get_real_obs_dict

import zarr

class BasePolicyWrapper:
    def __init__(self) -> None:
        pass
    
    def warm_it_up(self, obs_dict):
        """
        raw obs_dict. We cast it here to torch
        """
        pass
    
    def run_inference(self, obs_dict):
        pass
        
from omegaconf import OmegaConf,open_dict

class ZarrPolicyWrapper(BasePolicyWrapper):
    def __init__(self, zarr_path, ckpt_path) -> None:
        payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
        self.cfg = payload["cfg"].task.dataset
        self.cfg.dataset_path = zarr_path
        with open_dict(self.cfg):
            pass
            # self.cfg.shape_meta.obs["timestamp"] = {"shape": [1], "type": "low_dim"}
            
        # self.cfg.shape_meta.obs["timestamp"] = {"shape": [1], "type": "low_dim"}
        print(self.cfg)
        self.dataset = hydra.utils.instantiate(self.cfg)
        import pdb; pdb.set_trace()
        # rospy.loginfo("Policy evaluated")

    # def reset(self):
    #     self.starting_timestamp = self.dataset
    #     self.starting_policy_timestamp = 
    
    def warm_it_up(self, obs_dict):
        """
        raw obs_dict. We cast it here to torch
        """
        pass
    
    def run_inference(self, obs_dict):
        obs_time = obs_dict["timestamp"]
        

zarr_path_ = "/home/ali/avatar_recordings/dumb_lift/isolated_zarr_playback/_generated_replay_buffer.zarr"
ckpt_path_ = "/home/ali/avatar/avatar_behavior_cloning/eval/epoch=0990-train_loss=0.000.ckpt"

zarr_policy = ZarrPolicyWrapper(zarr_path=zarr_path_, ckpt_path=ckpt_path_)

class PolicyWrapper:
    def __init__(self, ckpt_path) -> None:
        self.ckpt_path = ckpt_path        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load checkpoint
        payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
        self.cfg = payload["cfg"]
        print(self.cfg)
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
        rospy.loginfo("Policy evaluated")

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
        