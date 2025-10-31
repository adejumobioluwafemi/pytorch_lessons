import torch 

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

x = torch.tensor([1.0, 2.0, 3.0]).to(device)
result = x * 2.0
print("result on mps", result)

print("result on cpu",result.cpu())


