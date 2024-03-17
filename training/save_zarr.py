import torch 
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer

FILE = "utils/test.pkl"
SAVE_PATH = "utils/test.zarr"
data = torch.load(FILE)

# data = data.cpu().detach().numpy()

replay_buffer = ReplayBuffer.create_empty_zarr()

episode = {
    "images": np.asarray(data["images"].cpu().detach().numpy()),
    "haptics": np.asarray(data["haptics"].cpu().detach().numpy()),
    "actions": np.asarray(data["actions"].cpu().detach().numpy())  
}


replay_buffer.add_episode(episode)

replay_buffer.save_to_path(str(SAVE_PATH))