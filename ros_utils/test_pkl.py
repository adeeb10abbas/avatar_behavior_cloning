# import pickle

# def list_topics_from_pickle(pickle_file_path):
#     # Load the pickle file
#     with open(pickle_file_path, 'rb') as f:
#         data_tensors = pickle.load(f)
    
#     # Get the list of topic names, which are the keys of the dictionary
#     topic_names = list(data_tensors.keys())
    
#     return topic_names

# # Example usage
# pickle_file_path = 'output_tensors.pkl'
# topic_names = list_topics_from_pickle(pickle_file_path)
# print("Topics in the pickle file:")
# for topic in topic_names:
#     print(topic)

import pickle

def list_topics_and_counts_from_pickle(pickle_file_path):
    # Load the pickle file
    with open(pickle_file_path, 'rb') as f:
        data_tensors = pickle.load(f)
    
    # Initialize a dictionary to hold the count of fields for each topic
    topic_counts = {}
    
    # Iterate over the dictionary to count the number of tensors for each topic
    for topic, tensors in data_tensors.items():
        # Assuming each entry under a topic is a list of tensors
        count = len(tensors) if isinstance(tensors, list) else 1
        topic_counts[topic] = count
    
    return topic_counts

# Example usage
pickle_file_path = 'output_tensors.pkl'
topic_counts = list_topics_and_counts_from_pickle(pickle_file_path)
print("Topics and counts in the pickle file:")
for topic, count in topic_counts.items():
    print(f"{topic}: {count} items")
