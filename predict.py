import numpy as np
from sklearn.datasets import fetch_openml
from src.network import Sequential 

def print_ascii_digit(image_array):
    """Converts a 28x28 numpy array into terminal ASCII art."""
    image_2d = image_array.reshape(28, 28)
    for row in image_2d:
        # Use @@ for dark pixels, .. for light pixels, and spaces for background
        line = "".join(["@@" if pixel > 0.5 else ".." if pixel > 0.1 else "  " for pixel in row])
        print(line)

# 1. Load the Pre-Trained Brain
print("1. Loading trained model from file...")
model = Sequential()
model.load("mnist_model.pkl")

# 2. Fetch data (just to get a test image)
print("\n2. Fetching MNIST dataset for a test image...")
mnist = fetch_openml('mnist_784', version=1, cache=True, parser='liac-arff', as_frame=False)
X, y = mnist["data"] / 255.0, mnist["target"].astype(int)

# 3. Pick a random image from the 70,000 available
random_idx = np.random.randint(0, X.shape[0])
test_image = X[random_idx]
true_label = y[random_idx]

# 4. Display the image in the terminal
print("\n3. Visualizing Test Image:")
print("-" * 56)
print_ascii_digit(test_image)
print("-" * 56)

# 5. Make the Prediction (Instantaneous!)
print("\n4. Asking the model to predict...")
# Reshape the 1D array into a 2D batch of 1 (1 row, 784 columns)
logits = model.forward(test_image.reshape(1, 784))
predicted_label = np.argmax(logits)

# 6. The Verdict
print(f"\n--- RESULTS ---")
print(f"True Label: {true_label}")
print(f"Model Predicted: {predicted_label}")

if true_label == predicted_label:
    print("✅ Success! The model recognized the handwriting perfectly.")
else:
    print("❌ Oops! The model made a mistake on this one.")
