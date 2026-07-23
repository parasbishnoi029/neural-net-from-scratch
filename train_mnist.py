import numpy as np
from sklearn.datasets import fetch_openml

from src.layers import DenseLayer, ReLULayer
from src.losses import SoftmaxCategoricalCrossEntropy
from src.optimizers import OptimizerAdam
from src.network import Sequential

print("Downloading MNIST dataset (this may take a minute)...")
# Using the fixed parser configuration to avoid Pandas errors
mnist = fetch_openml('mnist_784', version=1, cache=True, parser='liac-arff', as_frame=False)

# Prepare the data
X, y = mnist["data"], mnist["target"].astype(int)
X = X / 255.0  # Normalize pixel brightness to be between 0.0 and 1.0

# Define the "Brain"
model = Sequential([
    DenseLayer(784, 128),
    ReLULayer(),
    DenseLayer(128, 10)
])

# Initialize the loss function and optimizer
loss_fn = SoftmaxCategoricalCrossEntropy()
optimizer = OptimizerAdam(learning_rate=0.01)

# Training settings
epochs = 20
batch_size = 256
n_samples = X.shape[0]

print("\nStarting Training Loop...")
for epoch in range(epochs):
    # Shuffle the dataset so the network doesn't memorize the order
    permutation = np.random.permutation(n_samples)
    X_shuffled = X[permutation]
    y_shuffled = y[permutation]
    
    correct_predictions = 0
    total_loss = 0
    
    # Train in batches rather than one image at a time
    for i in range(0, n_samples, batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        
        # 1. Forward Pass (Make a guess)
        logits = model.forward(X_batch)
        
        # 2. Calculate Error (Score the guess)
        loss = loss_fn.forward(logits, y_batch)
        total_loss += loss * X_batch.shape[0]
        
        # Track accuracy
        predictions = np.argmax(loss_fn.probabilities, axis=1)
        correct_predictions += np.sum(predictions == y_batch)
        
        # 3 & 4. Backward Pass and Update Weights (Learn from the mistake)
        model.backward(loss_fn.backward())
        model.update_parameters(optimizer)
        
    # Print the results at the end of each epoch
    accuracy = (correct_predictions / n_samples) * 100
    avg_loss = total_loss / n_samples
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

print("\nTraining Complete!")

# ==========================================
# NEW: Save the fully trained model to a file
# ==========================================
model.save("mnist_model.pkl")
