#!/bin/bash

# Function to process files
process_file() {
    local input_file=$1
    local mode=$2
    local output_directory=$3

    echo "Processing $input_file in $mode mode..."
    python3 rosbags_to_torch.py "$input_file" "$mode" "$output_directory"

}

# Check the number of arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_directory> <mode> <output_directory>"
    exit 1
fi

input_directory=$1
mode=$2
output_directory=$3

if [ ! -d "$output_directory" ]; then
    mkdir -p "$output_directory"
fi

if [ ! -d "$input_directory" ]; then
    echo "Error: Input directory '$input_directory' does not exist."
    exit 1
fi

# Loop through each file in the input directory
for input_file in "$input_directory"/*; do
    # Call process_file for each file
    if [[ $input_file == *.bag ]]; then
        process_file "$input_file" "$mode" "$output_directory"
    else
        echo "Skipping $input_file as it does not have a .bag extension"
    fi
done