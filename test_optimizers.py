import numpy as np
from src.layers import DenseLayer
from src.optimizers import OptimizerSGD

# 1. Create a tiny dense layer (2 inputs, 1 output)
dense = DenseLayer(input_dim=2, output_dim=1)

# Manually set weights to 1.0 so we can easily see them change
dense.weights = np.array([[1.0], [1.0]])
dense.biases = np.array([[0.0]])

print("--- BEFORE OPTIMIZER ---")
print("Weights:\n", dense.weights)

# 2. Simulate a backward pass (pretend the network calculated these gradients)
dense.dweights = np.array([[0.5], [-0.2]])
dense.dbiases = np.array([[0.1]])

# 3. Create the SGD Optimizer with a learning rate of 0.1
optimizer = OptimizerSGD(learning_rate=0.1)

# 4. Update the layer's parameters
optimizer.update_params(dense)

print("\n--- AFTER OPTIMIZER (Weights should change!) ---")
print("Weights:\n", dense.weights)
