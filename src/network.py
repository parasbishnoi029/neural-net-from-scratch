
class Sequential:
    def __init__(self, layers):
        """
        Initializes the network with a list of layers.
        Example: [DenseLayer(784, 128), ReLULayer(), DenseLayer(128, 10)]
        """
        self.layers = layers

    def forward(self, X):
        """
        Passes the input data X through all layers sequentially.
        """
        # The output of the first layer becomes the input to the next
        current_output = X
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output

    def backward(self, dZ):
        """
        Passes the gradient backward through all layers in reverse order.
        """
        # We start with the gradient from the loss function and go backward
        current_gradient = dZ
        for layer in reversed(self.layers):
            current_gradient = layer.backward(current_gradient)
        return current_gradient

    def update_parameters(self, optimizer):
        """
        Calls the optimizer to update weights and biases for every layer.
        """
        for layer in self.layers:
            optimizer.update_params(layer)
