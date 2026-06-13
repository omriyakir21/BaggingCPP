import pickle

def save_as_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def load_as_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data
