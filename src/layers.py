import numpy as np

class DenseLayer:
    def __init__(self, input_dim, output_dim):
        """
        Initializes weights and biases.
        - input_dim: Number of incoming features (e.g., 784 pixels)
        - output_dim: Number of neurons in this layer (e.g., 128)
        """
        # He/Xavier initialization: Keep small numbers so gradients don't explode
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.biases = np.zeros((1, output_dim))
        
        # Cache variables needed for backpropagation
        self.inputs = None
        self.dweights = None
        self.dbiases = None

    def forward(self, inputs):
        """
        Forward pass: Z = X * W + b
        """
        self.inputs = inputs  # Save inputs for backprop
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, dZ):
        """
        Backward pass: Compute gradients dW, db, and dX
        - dZ: Gradient of loss with respect to output Z
        """
        # Gradients of parameters
        self.dweights = np.dot(self.inputs.T, dZ)
        self.dbiases = np.sum(dZ, axis=0, keepdims=True)
        
        # Gradient with respect to input (to pass back to previous layer)
        dinputs = np.dot(dZ, self.weights.T)
        return dinputs


class ReLULayer:
    def __init__(self):
        self.inputs = None

    def forward(self, inputs):
        """
        Forward pass: A = max(0, Z)
        """
        self.inputs = inputs  # Save inputs for backprop
        return np.maximum(0, inputs)

    def backward(self, dA):
        """
        Backward pass: Pass gradient through only where inputs > 0
        """
        dinputs = dA.copy()
        dinputs[self.inputs <= 0] = 0
        return dinputs
