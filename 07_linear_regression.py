# pipeline
# 0. prepare data
# 1. Design model (input size, output size, forward pass)
# 2. Construct loss and optimizer
# 3. Training loop
#   - Forward pass: compute prediction
#   - Backward pass: gradients
#   - Update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# data prep
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0],1)

print(f"y.shape ==> {y.shape}")

# the model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()

        self.lin = nn.Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, x):
        return self.lin(x)

n_samples, n_features = X.shape
input_size = n_features
output_size = 1 
model = LinearRegression(input_size, output_size)

lr = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

num_epochs = 100
for epoch in range(num_epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1)%10==0:
        print(f'epoch: {epoch+1}, loss= {loss.item():.4f}')


predicted = model(X).detach().numpy()

plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()