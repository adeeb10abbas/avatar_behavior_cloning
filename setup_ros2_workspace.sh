#!/bin/bash

# Define the workspace directory
WORKSPACE_DIR=~/avatar_ws
SRC_DIR=$WORKSPACE_DIR/src

# Create a new ROS2 workspace
mkdir -p $SRC_DIR
cd $WORKSPACE_DIR

# Clone the repositories
git clone https://github.com/adeeb10abbas/rdda_avatar $SRC_DIR/rdda_avatar
git clone https://github.com/adeeb10abbas/rdda_interface_types $SRC_DIR/rdda_interface_types
git clone https://github.com/adeeb10abbas/smarty_arm_interface $SRC_DIR/smarty_arm_interface
git clone https://github.com/adeeb10abbas/smarty_arm_msg $SRC_DIR/smarty_arm_msg

# Source the ROS2 environment
source /opt/ros/humble/setup.bash

# Build the workspace with colcon
colcon build --symlink-install

# Source the local workspace setup file
source $WORKSPACE_DIR/install/setup.bash

echo "[INFO] Smarty Arm and RDDA Gripper - ROS2 Workspace setup and build complete."

