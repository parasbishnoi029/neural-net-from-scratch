import numpy as np
from sklearn.datasets import fetch_openml
from src.layers import DenseLayer, ReLULayer
from src.losses import SoftmaxCategoricalCrossEntropy
from src.optimizers import OptimizerAdam
from src.network import Sequential

print("Downloading MNIST dataset (this may take a minute)...")
# 1. Fetch data
# To this (adding as_frame=False):
mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto', as_frame=False)
X, y = mnist["data"].to_numpy(), mnist["target"].to_numpy().astype(int)

# 2. Preprocess Data
# Normalize pixel values from 0-255 to 0.0-1.0 to help the network learn faster
X = X / 255.0 

# 3. Build the Neural Network Architecture
# 784 pixels -> 128 hidden neurons -> ReLU -> 10 output classes (digits 0-9)
model = Sequential([
    DenseLayer(input_dim=784, output_dim=128),
    ReLULayer(),
    DenseLayer(input_dim=128, output_dim=10)
])

# 4. Initialize Loss and Optimizer (Using Adam for faster convergence)
loss_fn = SoftmaxCategoricalCrossEntropy()
optimizer = OptimizerAdam(learning_rate=0.01)

# 5. Training Loop Setup
epochs = 20
batch_size = 256
n_samples = X.shape[0]

print("\nStarting Training Loop...")
for epoch in range(epochs):
    # Shuffle the dataset before every epoch to prevent memorization
    permutation = np.random.permutation(n_samples)
    X_shuffled = X[permutation]
    y_shuffled = y[permutation]
    
    epoch_loss = 0
    correct_predictions = 0
    
    # Process the data in mini-batches
    for i in range(0, n_samples, batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        
        # --- FORWARD PASS ---
        logits = model.forward(X_batch)
        batch_loss = loss_fn.forward(logits, y_batch)
        epoch_loss += batch_loss
        
        # Calculate Accuracy for this batch
        predictions = np.argmax(loss_fn.probabilities, axis=1)
        correct_predictions += np.sum(predictions == y_batch)
        
        # --- BACKWARD PASS ---
        dZ = loss_fn.backward()
        model.backward(dZ)
        
        # --- OPTIMIZE ---
        model.update_parameters(optimizer)
        
    # Print metrics at the end of every epoch
    avg_loss = epoch_loss / (n_samples / batch_size)
    accuracy = (correct_predictions / n_samples) * 100
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

print("\nTraining Complete!")
