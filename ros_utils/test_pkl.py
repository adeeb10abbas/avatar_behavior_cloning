import torch
import pickle

def list_topics_and_counts_from_pickle(pickle_file_path):
    # Load the pickle file
    with open(pickle_file_path, 'rb') as f:
        data_tensors = pickle.load(f)
    
    # Print the keys of the dictionary
    print("Keys in the pickle file:")
    for key in data_tensors.keys():
        print(data_tensors[key])
        breakpoint()
    # Print the shapes of the tensors
    print("Shapes of the tensors:")
    for tensor in data_tensors.values():
        print(tensor)
        breakpoint()
    # Initialize a dictionary to hold the count of tensors for observations and actions
    category_counts = {}

    # Directly count the number of items (tensors) for observations and actions
    for category, tensor in data_tensors.items():
        # Assuming each category is now a single tensor, not a dict of subcategories
        # The 'count' is now the shape of the first dimension of each tensor
        count = tensor.shape[0] if isinstance(tensor, torch.Tensor) else len(tensor)
        category_counts[category] = count

    return category_counts

# Example usage
pickle_file_path = 'data.pkl'
topic_counts = list_topics_and_counts_from_pickle(pickle_file_path)
print("Categories and counts in the pickle file:")
for topic, count in topic_counts.items():
    print(f"{topic}: {count} items")
    # shapes 
    # print(tensor.shape)
    # print(tensor.shape[0])

