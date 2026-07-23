import pickle

class Sequential:
    def __init__(self, layers=None):
        # If layers aren't provided, start with an empty list
        self.layers = layers if layers is not None else []

    def forward(self, X):
        """Passes the data forward through every layer in order."""
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dZ):
        """Passes the error backward through every layer in reverse order."""
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)
        return dZ

    def update_parameters(self, optimizer):
        """Tells each layer to update its weights based on the optimizer."""
        for layer in self.layers:
            # Only update layers that actually have weights (skip ReLULayer)
            if hasattr(layer, 'weights'):
                optimizer.update(layer)

    def save(self, filename):
        """Saves the entire model architecture and trained weights to a file."""
        with open(filename, 'wb') as file:
            pickle.dump(self.layers, file)
        print(f"Model successfully saved to {filename}")

    def load(self, filename):
        """Loads trained weights and architecture from a file."""
        with open(filename, 'rb') as file:
            self.layers = pickle.load(file)
        print(f"Model successfully loaded from {filename}")
