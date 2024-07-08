import sys
import os
import multiprocessing
import time_sync_ros_nodes 

def main(input_bag_path, output_bag_path):
    # Check if the input bag file exists
    if not os.path.exists(input_bag_path):
        print("Input bag file does not exist: %s" % input_bag_path)

    # Check if the output bag file exists
    assert os.path.exists(output_bag_path) == True

    # get the list of bags to process 
    bag_list = []
    for root, dirs, files in os.walk(input_bag_path):
        for file in files:
            if file.endswith(".bag"):
                bag_list.append(os.path.join(root, file))
    
    def process_bag_file(bag_file, output_bag_path):
        print("Processing bag file: %s" % bag_file)
        out_bag_file = os.path.join(output_bag_path, os.path.basename(bag_file))
        time_sync_ros_nodes.main(bag_file, out_bag_file)
        print("Processed bag file: %s" % bag_file)

    processes = []
    for bag_file in bag_list:
        process = multiprocessing.Process(target=process_bag_file, args=(bag_file, output_bag_path))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print("All bag files have been processed.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: sync_runner.py <input_bag_path> <output_bag_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
