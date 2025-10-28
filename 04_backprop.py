import torch

x = torch.tensor(1.0)  # feature
y = torch.tensor(2.0) # ground truth

w = torch.tensor(1.0, requires_grad=True) # initialise weights

# forward pass and compute loss
y_hat = w*x
loss = (y_hat - y)**2

print("loss ==> ", loss)

# backward pass ==> compute grad
loss.backward()

### update weights
### next forward and backward pass