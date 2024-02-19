#!/bin/bash

source /opt/ros/noetic/setup.bash
source /home/ali/avatar/ws/devel/setup.bash

# Start process function
start_process() {
    command=$1
    name=$2

    # Use a conditional approach for commands that need `sudo` and `rosrun` or `roslaunch`
    if [[ $command == sudo* ]]; then
        # Run command with sudo. Adjust if your setup doesn't require sudo for these operations
        eval $command &> "${name}.log" &
    elif [[ $command == roslaunch* ]] || [[ $command == rosrun* ]]; then
        # Use bash -c to ensure environment variables are correctly used
        bash -c "$command" &> "${name}.log" &
    else
        eval $command &> "${name}.log" &
    fi
    
    PID=$!
    echo "$name started with PID: $PID"
    echo $PID > "${name}_pid.txt"
}

# Check if process is still running
check_process_alive() {
    name=$1
    if [ ! -f "${name}_pid.txt" ]; then
        echo "$name PID file not found."
        return
    fi

    PID=$(cat "${name}_pid.txt")
    if kill -0 $PID 2>/dev/null; then
        echo "$name ($PID) is still running."
    else
        echo "$name ($PID) is not running."
    fi
}

# Commands based on aliases
start_process "sudo ~/$avatar/ws/src/RDDA/build/rdda_slave left_gripper" "lgripper"
start_process "sudo ~/$avatar/ws/src/RDDA/build/rdda_slave right_gripper" "rgripper"

start_process "roslaunch rdda_interface rdda_interface.launch rdda_type:=left_gripper" "lgripper_interface"
start_process "roslaunch rdda_interface rdda_interface.launch rdda_type:=right_gripper" "rgripper_interface"

start_process "rosrun franka_control avatar_panda left" "lpanda"
start_process "rosrun franka_control avatar_panda right" "rpanda"

start_process "rosrun franka_lock_unlock __init__.py 10.103.1.14 admin Boston1234 -u -c -p -l" "lfranka"
start_process "rosrun franka_lock_unlock __init__.py 10.103.1.12 admin Boston1234 -u -c -p -l" "rfranka"

# Periodically check if each process is alive
while true; do
    check_process_alive "lgripper"
    check_process_alive "rgripper"
    check_process_alive "lgripper_interface"
    check_process_alive "rgripper_interface"
    check_process_alive "lpanda"
    check_process_alive "rpanda"
    check_process_alive "lfranka"
    check_process_alive "rfranka"
    sleep 60  # Check every 60 seconds
done

