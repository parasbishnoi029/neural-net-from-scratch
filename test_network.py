import numpy as np
from src.layers import DenseLayer, ReLULayer
from src.losses import SoftmaxCategoricalCrossEntropy
from src.optimizers import OptimizerSGD
from src.network import Sequential

# 1. Create a tiny dataset: 2 samples, 3 features each
X = np.array([
    [0.5, -1.0, 0.1],
    [0.0,  2.0, -0.5]
])
# True labels (Sample 1 is Class 0, Sample 2 is Class 1)
y_true = np.array([0, 1]) 

# 2. Build the Network Architecture
# 3 inputs -> 4 hidden neurons -> 2 output classes
model = Sequential([
    DenseLayer(input_dim=3, output_dim=4),
    ReLULayer(),
    DenseLayer(input_dim=4, output_dim=2)
])

# 3. Initialize Loss and Optimizer
loss_function = SoftmaxCategoricalCrossEntropy()
optimizer = OptimizerSGD(learning_rate=0.1)

# --- TRAINING STEP (1 EPOCH) ---

# 4. Forward Pass (Data goes through network)
raw_outputs = model.forward(X)

# 5. Calculate Loss and Probabilities
loss = loss_function.forward(raw_outputs, y_true)
print("--- PREDICTIONS (Probabilities) ---")
print(loss_function.probabilities)
print(f"Initial Loss: {loss:.4f}")

# 6. Backward Pass (Calculate Gradients)
dZ = loss_function.backward()  # Gradient from loss
model.backward(dZ)             # Pass gradient backward through the network

# 7. Update Weights
model.update_parameters(optimizer)
print("\n--- NETWORK SUCCESSFULLY UPDATED! ---")
