#!/bin/bash

# Array of the bags, modify this to add more bags
first_args=(
    "2024-06-10-16-03-28.bag"
    "2024-06-10-16-03-44.bag"
    "2024-06-13-11-16-26.bag"
)

# Array of outputs, modify this to add more outputs
second_args=(
    "test1.bag"
    "test2.bag"
    "test3.bag"
)

num_args=${#first_args[@]}

# Loop through the arrays and run the command, this will run each command in background
for ((i=0; i<num_args; i++)); do
    ./time_sync_ros_avatar.py "${first_args[i]}" "${second_args[i]}" &
done
