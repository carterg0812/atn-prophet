import pickle

def save_model(model, filename):
    """Saves the trained model as a pickle file."""
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    """Loads a saved pickle model."""
    with open(filename, 'rb') as file:
        return pickle.load(file)
