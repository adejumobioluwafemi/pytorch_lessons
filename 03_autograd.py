import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
x = torch.randn(3, requires_grad=True, device=device)
print(x)

y = x+2
print(y)

z = y*y*2
z = z.mean() # grad can be implicitly created only for scalar outputs

print("z", z)

# the last operation before taking derivation should be a scalar value, 
# else give a vector to the backward()
z.backward() #dz/dx 
print("x.grad", x.grad) # 

# preventing gradient history
# Option 1: x.requires_grad_(False)
x.requires_grad_(False)
print(x)

# Option 2: x.detach() ==> which creates a new tensor with the grad_fn tracking
#x_without_grad = x.detach()
#print(x_without_grad)

# Option 3: with torch.no_grad():
#with torch.no_grad():
#    y = x+2
#    print(y)

# sample training loop

weights = torch.ones(4, requires_grad=True)

for epoch in range(4):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)

    weights.grad.zero_()