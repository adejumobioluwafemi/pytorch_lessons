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


def compare_optimizers(X, y, epochs=200):
    optimizers = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam,
        'RMSprop': torch.optim.RMSprop,
        'Adagrad': torch.optim.Adagrad
    }

    results = {}

    for opt_name, opt_class in optimizers.items():
        print(f"\nTraining with {opt_name}...")

        # Reset model for each optimizer
        model = LinearRegression(input_size, output_size)
        criterion = nn.MSELoss()

        if opt_name == 'SGD':
            optimizer = opt_class(model.parameters(), lr=0.1)
        else:
            optimizer = opt_class(model.parameters(), lr=0.01)

        losses = []

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        results[opt_name] = losses
        print(f"{opt_name} final loss: {losses[-1]:.4f}")

    return results


optimizer_results = compare_optimizers(X, Y)

print("\n" + "="*50)
print("OPTIMIZER PERFORMANCE SUMMARY:")
for opt_name, losses in optimizer_results.items():
    print(f"{opt_name:8} | Initial: {losses[0]:.4f} | Final: {losses[-1]:.4f} | Improvement: {losses[0]-losses[-1]:.4f}")