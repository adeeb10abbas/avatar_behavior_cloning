export CUDA_VISIBLE_DEVICES=0,1,2,3
ray init
ray start --head --num-gpus=4
export HYDRA_FULL_ERROR=1; python3 ray_train_multirun.py --config-dir diffusion_policy/config --config-name train_haptic_image_teacher_aware
