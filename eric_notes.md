```sh
cd here
source ./eric_setup.sh
python ./scripts/minimize_data.py
python ros_utils/generate_zarr_episodes.py data/pkl/test/2024-07-26-21-15-44.min.pkl
python ros_utils/generate_zarr_episodes.py data/pkl/test

d training/diffusion_policy/
python train.py --config-name=eric_overfit
```
