```sh
cd ./avatar_behavior_cloning

# generate small overfit dataset
source ./eric_setup.sh
python ./scripts/minimize_data.py
python ros_utils/generate_zarr_episodes.py data/pkl/test

# train
cd training/diffusion_policy/
python train.py --config-name=eric_overfit

# evaluate
python eval/eval_policy.py
```
