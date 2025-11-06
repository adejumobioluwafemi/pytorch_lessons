import numpy as np
import torch
import torch.nn as nn

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def cross_entropy(y, y_hat):
    loss = -np.sum(y * np.log(y_hat))
    return loss # / float(y_hat.shape[0])

logit_numpy_good = np.array([2.0, 0.7, 0.1])
logit_numpy_bad = np.array([0.7, 2.0, 0.1])
outputs_numpy_good = softmax(logit_numpy_good)
outputs_numpy_bad = softmax(logit_numpy_bad)
print(f"softmax numpy good: {outputs_numpy_good}")
print(f"softmax numpy bad: {outputs_numpy_bad}")


logit_torch = torch.tensor([2.0, 0.7, 0.1])
outputs_torch = torch.softmax(logit_torch, dim=0)
print(f"softmax torch: {outputs_torch}")
# y must be one hot encoded
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
Y = np.array([1, 0, 0])

# y_hat has probabilities
loss_good = cross_entropy(Y, outputs_numpy_good)
loss_bad = cross_entropy(Y, outputs_numpy_bad)
print(f'Loss good: {loss_good}')
print(f'Loss bad: {loss_bad}')

# ==============================
# PyTorch version
# ==============================

# For PyTorch CrossEntropyLoss, we need:
# - Input: raw logits (no softmax applied), shape (batch_size, num_classes)
# - Target: class indices (NOT one-hot), shape (batch_size,)

# Single sample with 3 classes
logit_torch_good = torch.tensor([[2.0, 0.7, 0.1]])  # shape: (1, 3) - batch_size=1, num_classes=3
logit_torch_bad = torch.tensor([[0.7, 2.0, 0.1]])   # shape: (1, 3)
Y_torch = torch.tensor([0])  # shape: (1,) - class index 0 (NOT one-hot)

loss_fn = nn.CrossEntropyLoss()

l1_good = loss_fn(logit_torch_good, Y_torch)
l1_bad = loss_fn(logit_torch_bad, Y_torch)

print(f'Loss good (torch): {l1_good.item():.4f}')
print(f'Loss bad (torch): {l1_bad.item():.4f}')

print(f"\nVerification - losses should be similar:")
print(f"Numpy good: {loss_good:.4f}, PyTorch good: {l1_good.item():.4f}")
print(f"Numpy bad: {loss_bad:.4f}, PyTorch bad: {l1_bad.item():.4f}")


# ========
# Multiple samples
# ========

Y = torch.tensor([2,0,1])

# nsamples x nclasses = 3x3
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 0.1, 3.0]])

l1 = loss_fn(Y_pred_good, Y)
l2 = loss_fn(Y_pred_bad, Y)

_, prediction1 = torch.max(Y_pred_good, 1)
_, prediction2 = torch.max(Y_pred_bad, 1)
print(prediction1)
print(prediction2)