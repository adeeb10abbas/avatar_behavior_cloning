#!/bin/bash

conda-setup
conda activate robodiff
export PYTHONPATH=${PWD}/training/diffusion_policy:${PYTHONPATH}
