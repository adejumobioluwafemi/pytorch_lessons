# step 1
# - implement linear regression: manually
# - prediction: manually
# - gradients computation: manually
# - Loss computation: manually
# - parameter updates: manually

import numpy as np

# simple linear regression

# f = w * x
# f = 2 * w

X = np.array([1,2,3,4,6,7,8], dtype=np.float32)
Y = np.array([2,4,6,8,12,14,16], dtype=np.float32)

# initialise weight
w = 0.0

# model prediction
def forward(x):
    return w*x

# loss
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

# gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N 2 (x (w*x - y))
def gradient(x, y, y_pred):
    return (2 * np.dot(x, (y_pred-y))) / len(x)

def gradient2(x, y, y_pred):
    errors = y_pred - y
    gradients = 2 * x * errors
    return np.mean(gradients)

print(f"Prediction before training: f(5) = {forward(5):.3f}")

learning_rate = 0.01
n_iters = 50

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    l = loss(Y, y_pred)

    #gradient
    #dw = gradient(X, Y, y_pred)

    dw = gradient2(X, Y, y_pred)

    # update weights
    w -= learning_rate * dw

    if epoch % 10 == 0:
        print(f"epoch {epoch+1}: w = {w:.3f}, loss= {l:.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")