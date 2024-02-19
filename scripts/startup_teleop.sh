#!/bin/bash

source /opt/ros/noetic/setup.bash
source /home/ali/avatar/ws/devel/setup.bash

# Function to start a process and echo its PID
start_process() {
    command=$1
    name=$2

    # Start command in the background and get its PID
    eval $command &> "${name}.log" &
    PID=$!
    echo "$name started with PID: $PID"
    echo $PID > "${name}_pid.txt"
}

# Function to check if a process is still running
check_process_alive() {
    name=$1
    PID=$(cat "${name}_pid.txt" 2>/dev/null)

    if [ -z "$PID" ]; then
        echo "$name PID not found."
        return
    fi

    if kill -0 $PID 2>/dev/null; then
        echo "$name ($PID) is still running."
    else
        echo "$name ($PID) is not running."
    fi
}

# Start processes
start_process "sudo rdda_master left_glove" "left_glove"
start_process "roslaunch rdda_interface rdda_interface.launch rdda_type:=left_glove" "lgi"
start_process "sudo rdda_master right_glove" "right_glove"
start_process "roslaunch rdda_interface rdda_interface.launch rdda_type:=right_glove" "rgi"
start_process "sudo smarty_arm_control l" "left_arm"
start_process "roslaunch smarty_arm_interface smarty_arm_interface.launch smarty_arm_type:=l" "lsai"
start_process "sudo smarty_arm_control r" "right_arm"
start_process "roslaunch smarty_arm_interface smarty_arm_interface.launch smarty_arm_type:=r" "rsai"

# Periodically check if each process is alive
while true; do
    check_process_alive "left_glove"
    check_process_alive "lgi"
    check_process_alive "right_glove"
    check_process_alive "rgi"
    check_process_alive "left_arm"
    check_process_alive "lsai"
    check_process_alive "right_arm"
    check_process_alive "rsai"
    sleep 60  # Check every 60 seconds
done

