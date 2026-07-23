import numpy as np
from src.losses import SoftmaxCategoricalCrossEntropy

# 1. Dummy raw output scores (logits) for 2 samples across 3 classes
# Sample 1 score is high for Class 2; Sample 2 score is high for Class 0
logits = np.array([
    [2.0, 1.0, 0.1],
    [0.5, 3.0, 0.2]
])

# 2. True target class labels (Sample 1 -> class 0, Sample 2 -> class 1)
targets = np.array([0, 1])

# 3. Instantiate Loss function
loss_fn = SoftmaxCategoricalCrossEntropy()
loss_value = loss_fn.forward(logits, targets)

print("--- SOFTMAX PROBABILITIES ---")
print(loss_fn.probabilities)
print(f"\nCalculated Loss: {loss_value:.4f}")

# 4. Backward pass
dZ = loss_fn.backward()
print("\n--- LOSS GRADIENT dZ (P - Y) ---")
print(dZ)
