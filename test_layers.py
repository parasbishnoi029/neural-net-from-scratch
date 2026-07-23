import numpy as np
from src.layers import DenseLayer, ReLULayer

# 1. Create dummy batch: 3 samples, each with 4 features
X = np.array([
    [1.0, 2.0, -1.0, 0.5],
    [0.5, -3.0, 2.0, 1.1],
    [0.0, 1.0, 1.5, -2.0]
])

print("--- 1. INPUT MATRIX (Shape: 3x4) ---")
print(X)

# 2. Forward pass through DenseLayer (4 features -> 2 neurons)
dense = DenseLayer(input_dim=4, output_dim=2)
Z = dense.forward(X)
print("\n--- 2. DENSE FORWARD OUTPUT Z (Shape: 3x2) ---")
print(Z)

# 3. Forward pass through ReLULayer
relu = ReLULayer()
A = relu.forward(Z)
print("\n--- 3. RELU FORWARD OUTPUT A (Shape: 3x2) ---")
print(A)

# 4. Simulate a dummy gradient coming back from loss (Shape: 3x2)
dummy_dA = np.ones_like(A)

# 5. Backward pass through ReLU then Dense
dZ = relu.backward(dummy_dA)
dX = dense.backward(dZ)

print("\n--- 4. BACKWARD SUCCESSFUL! ---")
print(f"Weight gradient shape (dW): {dense.dweights.shape} (Expected: 4x2)")
print(f"Bias gradient shape (db):   {dense.dbiases.shape} (Expected: 1x2)")
print(f"Input gradient shape (dX):  {dX.shape} (Expected: 3x4)")
