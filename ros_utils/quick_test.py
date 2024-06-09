import pickle
import torch

import pickle

def load_and_inspect_pkl(file_path):
    try:
        with open(file_path, 'rb') as f:
            data_structure = pickle.load(f)

        # Iterate through each category ('observations' and 'actions')
        for category, topics in data_structure.items():
            print(f"Category: {category}")
            # Iterate through each topic within the category
            for topic, tensors in topics.items():
                print(f"  Topic: {topic}")
                # Check if there are tensors stored under the topic
                if tensors:
                    print(f"    Total number of tensors: {len(tensors)}")
                    # Print the dimensions of the first tensor for size reference
                    print(f"    Tensor dimensions (example): {tensors[0].shape}")
                else:
                    print("    No data available for this topic.")
            print()  # Newline for better readability between categories
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred while loading or processing the file: {e}")

# Replace 'your_data_file.pkl' with the actual path to your pickle file
load_and_inspect_pkl('organized_data.pkl')

