import numpy as np

class OptimizerSGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        """
        Updates the weights and biases of a given layer using standard Gradient Descent.
        """
        # If the layer doesn't have weights (like ReLU), we just skip it
        if not hasattr(layer, 'weights'):
            return
        
        # Update weights and biases
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases


class OptimizerAdam:
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def update_params(self, layer):
        """
        Updates parameters using the Adam algorithm (Adaptive Moment Estimation).
        It uses moving averages of the gradients to smooth out the learning process.
        """
        if not hasattr(layer, 'weights'):
            return

        # If layer doesn't have cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum and cache for weights
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * (layer.dweights ** 2)
        
        # Update weights
        layer.weights -= self.learning_rate * layer.weight_momentums / (np.sqrt(layer.weight_cache) + self.epsilon)

        # Update momentum and cache for biases
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * (layer.dbiases ** 2)
        
        # Update biases
        layer.biases -= self.learning_rate * layer.bias_momentums / (np.sqrt(layer.bias_cache) + self.epsilon)
