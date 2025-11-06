"""
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the Dataset

complete list of built-in transforms:
https://docs.pytorch.org/vision/0.9/transforms.html

On images
---------
CenterCrop, Grayscale, Pad, RandomAffine, Scale
RandomCrop, RandomHorizontalFlip, RandomRotation, Resize

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Coversion
---------
ToPILImage: from tensor or ndarray
ToTensor: from numpy.ndarray or PILImage

Generic
-------
Use Lambda

Custom
------
Write own class

Compose multiple transforms
---------------------------
composed = torchvision.transforms.Compose([
                                            ReScale(256),
                                            RandomCrop(224)
                                            ])

torchvision.transforms.ReScale(256)
torchvision.transforms.ToTensor()
"""

import torch 
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):

    def __init__(self, transform=None, filepath="./data/wine/wine.csv"):
       # data loading
       xy = np.loadtxt(filepath, delimiter=",", skiprows=1, dtype=np.float32)
       self.x = xy[:, 1:]
       self.y = xy[:, [0]]
       self.n_samples = xy.shape[0]

       self.mean, self.std = self.compute_dataset_statistics()
       
       self.transform = transform
       
    def __getitem__(self, index):
        # get an instance of the data
        sample = self.x[index, :], self.y[index, :]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        # len(dataset)
        return self.n_samples
    
    def compute_dataset_statistics(self):
        """Compute mean and std for standardization across entire dataset"""
        return np.mean(self.x, axis=0), np.std(self.x, axis=0)


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    

class Standardize:
    """Standardize features to have mean=0 and std=1"""
    def __init__(self, mean=None, std=None):
        # Allow pre-computed statistics for consistent transformation
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        inputs, targets = sample
        
        if self.mean is None or self.std is None:
            # Compute statistics from current data
            self.mean = np.mean(inputs, axis=0)
            self.std = np.std(inputs, axis=0)
        
        # Avoid division by zero
        self.std = np.where(self.std == 0, 1, self.std)
        
        standardized_inputs = (inputs - self.mean) / self.std
        return standardized_inputs, targets


if __name__ == '__main__':
    dataset = WineDataset(transform=None)
    
    composed = torchvision.transforms.Compose([
        Standardize(mean=dataset.mean, std=dataset.std),
        ToTensor()
    ])
    
    # Recreate dataset with the transform
    dataset = WineDataset(transform=composed)
    
    # Test the transformed data
    features, labels = dataset[0]
    print(f"Transformed features shape: {features.shape}")
    print(f"Transformed features type: {features.dtype}")
    print(f"First few standardized features: {features[:5]}")
    print(f"Label: {labels}")
    
    # Verify standardization
    dataloader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False)
    all_features, all_labels = next(iter(dataloader))
    print(f"\nVerification - All features mean: {all_features.mean():.6f}")
    print(f"Verification - All features std: {all_features.std():.6f}")