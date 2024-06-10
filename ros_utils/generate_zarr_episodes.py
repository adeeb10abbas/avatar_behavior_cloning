## Script that takes one bag file and generates a zarr for that entire bag file

# we take in a list of pkl file paths 
# we load them into memory
# we create zarr out of them 
# we save the zarr to a new directory

import os
import pickle
import zarr
import numpy as np
import torch

PATH_REPLAY_BUFFER = "organized_data.pkl"


