import rclpy
from ros_teleop_data_receiver_py import RosTeleopDataReceiver  

def main():
    # Initialize ROS 2
    rclpy.init(args=None)

    # Create instances of RosTeleopDataReceiver for left and right sides
    left_receiver = RosTeleopDataReceiver(side='left')
    right_receiver = RosTeleopDataReceiver(side='right')

    # Create an executor to manage the nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(left_receiver)
    executor.add_node(right_receiver)

    # Use a separate thread for the executor
    import threading
    executor_thread = threading.Thread(target=executor.spin)
    executor_thread.start()

    try:
        while rclpy.ok():
            # Example: print the simulated data every 5 seconds
            print("\nGetting simulated finger positions and effector poses:")
            print("Left Finger Position (Simulated):", left_receiver.get_finger_pos(mode="simulated"))
            print("Right Finger Position (Simulated):", right_receiver.get_finger_pos(mode="simulated"))
            print("Left Effector Pose (Simulated):", left_receiver.get_eff_pose(mode="simulated"))
            print("Right Effector Pose (Simulated):", right_receiver.get_eff_pose(mode="simulated"))

            rclpy.spin_once(left_receiver, timeout_sec=0.1)
            rclpy.spin_once(right_receiver, timeout_sec=0.1)

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Shutdown ROS 2
        executor.shutdown()
        left_receiver.destroy_node()
        right_receiver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
