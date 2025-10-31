# step 2 
# - prediction: manually
# - gradients computation: Autograd
# - Loss computation: manually
# - parameter updates: manually

# step 3 
# - prediction: manually
# - gradients computation: Autograd
# - Loss computation: PyTorch Loss
# - parameter updates: PyTorch Optimizer

# step 4 
# - prediction: PyTorch Model
# - gradients computation: Autograd
# - Loss computation: PyTorch Loss
# - parameter updates: PyTorch Optimizer

# pipeline
# 1. Design model (input size, output size, forward pass)
# 2. Construct loss and optimizer
# 3. Training loop
#   - Forward pass: compute prediction
#   - Backward pass: gradients
#   - Update weights

import torch
import torch.nn as nn
# simple linear regression

# f = w * x
# f = 2 * w

X = torch.tensor([[1],[2],[3],[4],[6],[7],[8]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8],[12],[14],[16]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(f"n_samples==> {n_samples}, n_features==> {n_features}")

input_size = n_features
output_size = n_features

#model = nn.Linear(in_features=input_size, out_features=output_size)

# custom model
class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()

        # layers

        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)
print(f"Prediction before training: f(5) = {float(model(X_test)):.3f}")

learning_rate = 0.01
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_iters = 100
for epoch in range(n_iters):
    # prediction 
    y_pred = model(X)

    l = loss(Y, y_pred)

    #gradient
    l.backward() 

    # update weights
    optimizer.step()

    optimizer.zero_grad()

    if epoch % 10==9:
    # with torch.no_grad():
    #     w_value = model.weight.item()
    #     b_value = model.bias.item()
        [w,b] = model.parameters()
        w_value = w[0][0].item()
        b_value = b.item()
        
        print(f"epoch {epoch+1}: w = {w_value:.3f}, b = {b_value:.3f}, loss = {l:.8f}")

print(f"Prediction after training: f(5) = {float(model(X_test)):.3f}")