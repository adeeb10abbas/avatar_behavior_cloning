import ros_sync_time
import sys
import os
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
    print(bag_list)
    
    # Process each bag file in the list 
    for bag_file in bag_list:
        print("Processing bag file: %s" % bag_file)
        out_bag_file = os.path.join(output_bag_path, os.path.basename(bag_file))
        ros_sync_time.main(bag_file, out_bag_file)
        print("Processed bag file: %s" % bag_file)
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: ros_sync_time.py <input_bag_file> <output_bag_file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
