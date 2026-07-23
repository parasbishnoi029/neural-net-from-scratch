import numpy as np

class SoftmaxCategoricalCrossEntropy:
    """
    Combines Softmax activation and Categorical Cross-Entropy loss
    for numerical stability and simplified gradient computation.
    """
    def __init__(self):
        self.probabilities = None
        self.targets = None

    def forward(self, logits, targets):
        """
        logits: Raw predictions from the last dense layer (shape: N x C)
        targets: True class labels as integer indices (shape: N,) or one-hot encoded (shape: N x C)
        """
        # 1. Numerical stability trick: subtract max per row
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        
        # 2. Compute Softmax probabilities
        exp_scores = np.exp(shifted_logits)
        self.probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        batch_size = logits.shape[0]
        
        # 3. Convert integer targets to one-hot encoding if necessary
        if targets.ndim == 1:
            self.targets = np.zeros_like(self.probabilities)
            self.targets[np.arange(batch_size), targets] = 1.0
        else:
            self.targets = targets

        # 4. Compute Categorical Cross-Entropy loss (clip to avoid log(0))
        clipped_probs = np.clip(self.probabilities, 1e-15, 1.0 - 1e-15)
        loss = -np.sum(self.targets * np.log(clipped_probs)) / batch_size
        return loss

    def backward(self):
        """
        Backward pass: Derivative dZ = P - Y
        Normalized by batch size for stable gradient updates.
        """
        batch_size = self.probabilities.shape[0]
        dZ = (self.probabilities - self.targets) / batch_size
        return dZ
