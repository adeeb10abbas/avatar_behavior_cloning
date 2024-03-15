import os
import signal
import subprocess
import time
import tempfile

# Define a temporary folder for logs
temp_dir = tempfile.mkdtemp(prefix="logs_")
print(f"Logs will be stored in: {temp_dir}")

processes = {}  # Store process information


def start_process(command, name):
    """Starts a process in the background and stores its log in the temporary folder."""
    log_file = os.path.join(temp_dir, f"{name}.log")
    with open(log_file, "w") as logfile:
        process = subprocess.Popen(command, shell=True, stdout=logfile, stderr=subprocess.STDOUT)
        processes[name] = {
            "process": process,
        }
        print(f"{name} started with PID: {process.pid}")


def check_process_alive(name):
    """Checks if a process is still running."""
    info = processes.get(name)
    if info:
        try:
            os.kill(info["process"].pid, 0)  # Send a null signal to check if process exists
            print(f"{name} ({info['process'].pid}) is still running.")
        except OSError:
            print(f"{name} ({info['process'].pid}) is not running.")
            try:
                info["process"].terminate()  # Attempt graceful termination
            except OSError:
                pass  # Process already terminated
            del processes[name]
    else:
        print(f"{name} PID not found.")


def kill_processes():
    """Kills all started processes and removes the temporary folder."""
    print("Terminating all processes...")
    for info in processes.values():
        try:
            info["process"].kill()
            print(f"Killing {info['process'].pid} ({info['process'].args})")
        except OSError:
            pass  # Process already terminated
    exit(0)  # Exit the script successfully

    # Remove the temporary folder after killing processes (optional)
    shutil.rmtree(temp_dir, ignore_errors=True)


def signal_handler(sig, frame):
    """Handles SIGINT (Ctrl+C) and SIGTERM signals."""
    kill_processes()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Start processes
start_process("sudo ~/avatar/ws/src/RDDA/build/rdda_master left_glove", "left_glove")
start_process("sudo ~/avatar/ws/src/RDDA/build/rdda_master right_glove", "right_glove")
start_process("sudo smarty_arm_control l", "left_arm")
start_process("sudo smarty_arm_control r", "right_arm")

# time.sleep(5)  # Wait for the arms to start
print("Press enter to start teleop")
input("Waiting...")

start_process("roslaunch rdda_interface rdda_interface.launch rdda_type:=left_glove", "lgi")
start_process("roslaunch rdda_interface rdda_interface.launch rdda_type:=right_glove", "rgi")

start_process("roslaunch smarty_arm_interface smarty_arm_interface.launch smarty_arm_type:=l", "lsai")
start_process("roslaunch smarty_arm_interface smarty_arm_interface.launch smarty_arm_type:=r", "rsai")

# Periodically check if each process is alive
while True:
    for name in processes:
        check_process_alive(name)
    time.sleep(60)  # Check every 60 seconds

