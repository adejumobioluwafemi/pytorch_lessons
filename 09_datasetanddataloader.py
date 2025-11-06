"""
Terms
----
epoch = 1 forward and backward pass of all training samples

batch_size = number of training samples in one forward & backward pass

number of iterations = number of passes, each pass using [batch_size] number of samples

e.g. 100 samples, batch_size=20 --> 100/20 = 5 iterations for 1 epoch
"""

import torch 
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):

    def __init__(self, filepath="./data/wine/wine.csv"):
       # data loading
       xy = np.loadtxt(filepath, delimiter=",", skiprows=1, dtype=np.float32)
       self.x = torch.from_numpy(xy[:, 1:])
       self.y = torch.from_numpy(xy[:, [0]])
       self.n_samples = xy.shape[0]
       
    def __getitem__(self, index):
        # get an instance of the data
        return self.x[index, :], self.y[index, :]
    def __len__(self):
        # len(dataset)
        return self.n_samples
        
if __name__ == '__main__':
    wine_dataset = WineDataset()
    first_data = wine_dataset[0]
    features, labels = first_data
    print(f"features ==> {features}")
    print(f"labels ==> {labels}")

    dataloader = DataLoader(dataset=wine_dataset, batch_size=4, shuffle=True, num_workers=0)
    dataiter = iter(dataloader)
    data = next(dataiter)
    features, labels = data
    print(f"features from dataloader ==> {features}")
    print(f"labels from dataloader ==> {labels}")
    
    # training loop example
    num_epochs = 2
    total_samples = len(wine_dataset)
    n_iterations = math.ceil(total_samples / 4)
    print(f"Total samples: {total_samples}, Iterations per epoch: {n_iterations}")

    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(dataloader):
            # Forward, backward, update steps would go here
            if (i+1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_iterations}], Inputs {inputs.shape}')