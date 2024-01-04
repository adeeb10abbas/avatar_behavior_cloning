#!/bin/bash
# Installs all the relevant drivers needed for the teleop-operator arm and hand control. 

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to check if a command exists
command_exists() {
    type "$1" &> /dev/null ;
}

# Ensure necessary tools are installed
for cmd in git cmake make; do
    if ! command_exists $cmd; then
        echo "Error: $cmd is not installed." >&2
        exit 1
    fi
done

# Function to clone and build a repository in /tmp
clone_and_build() {
    local repo_url=$1
    local install_prefix=${2:-}
    local tmp_dir="/tmp/$(basename $repo_url .git)"

    # Create a temporary directory and clone the repository there
    mkdir -p $tmp_dir
    git clone $repo_url $tmp_dir

    cd $tmp_dir
    mkdir -p build
    cd build

    if [ -n "$install_prefix" ]; then
        cmake .. -DCMAKE_INSTALL_PREFIX=$install_prefix
    else
        cmake ..
    fi

    make

    if [ -n "$install_prefix" ]; then
        sudo make install
    fi

    # Clean up: Optionally delete the tmp directory after installation
    # rm -rf $tmp_dir

    # Go back to the original directory
    cd ../../
}

echo "Installing SOEM..."
clone_and_build "https://github.com/OpenEtherCATsociety/SOEM" "/usr/local"

echo "Installing RDDA..."
clone_and_build "https://github.com/RoboticsCollaborative/RDDA.git"

echo "Installation completed successfully."
