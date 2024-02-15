import pickle

def list_topics_from_pickle(pickle_file_path):
    # Load the pickle file
    with open(pickle_file_path, 'rb') as f:
        data_tensors = pickle.load(f)
    
    # Get the list of topic names, which are the keys of the dictionary
    topic_names = list(data_tensors.keys())
    
    return topic_names

# Example usage
pickle_file_path = 'output_tensors.pkl'
topic_names = list_topics_from_pickle(pickle_file_path)
print("Topics in the pickle file:")
for topic in topic_names:
    print(topic)

