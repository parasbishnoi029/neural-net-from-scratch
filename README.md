# Neural Network Framework from Scratch

A lightweight, modular deep learning framework built entirely in Python and NumPy. This project implements core deep learning components (layers, activations, optimizers, and loss functions) from first principles, without relying on external libraries like PyTorch or TensorFlow.

## 🚀 Project Overview

The goal of this project was to understand the underlying mathematics and matrix calculus of neural networks by building an engine from scratch. The framework successfully trains a Multi-Layer Perceptron (MLP) on the MNIST dataset, achieving **~97% accuracy**.

## 🧠 Features

*   **Modular Architecture:** Designed with a PyTorch-like `Sequential` container for easy layer stacking.
*   **Custom Layers:** 
    *   Fully Connected (Dense) Layer
    *   ReLU Activation
*   **Loss Functions:** 
    *   Softmax Activation combined with Categorical Cross-Entropy (for numerical stability).
*   **Optimizers:** 
    *   Stochastic Gradient Descent (SGD)
    *   Adam (Adaptive Moment Estimation)

## 📐 The Mathematics

This framework explicitly calculates all forward and backward passes using matrix algebra. 

**Dense Layer Forward Pass:**
$Z = X \cdot W + b$

**Backpropagation Gradients (Chain Rule):**
*   Weights: $\frac{\partial L}{\partial W} = X^T \cdot dZ$
*   Biases: $\frac{\partial L}{\partial b} = \sum dZ$
*   Inputs: $\frac{\partial L}{\partial X} = dZ \cdot W^T$

## ⚡ Try it Live in Google Colab

You can test this custom framework directly in your browser without installing anything locally. 

**Step 1:** Open a new [Google Colab Notebook](https://colab.research.google.com/).  
**Step 2:** Paste this into the first cell and run it to download the engine and train the model:

```python
# Clone the repository and move into the folder
!git clone [https://github.com/parasbishnoi029/neural-net-from-scratch.git](https://github.com/parasbishnoi029/neural-net-from-scratch.git)
%cd neural-net-from-scratch

# Run the full training loop
!python train_mnist.py
#or 
# Clone the repository and move into the folder
!git clone https://github.com/parasbishnoi029/neural-net-from-scratch.git
%cd neural-net-from-scratch

# Run the full training loop
!python train_mnist.py
