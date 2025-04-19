import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset


PROJECT_DIR = os.path.dirname(os.getcwd())


def get_mnist_pipeline(batch_size: int=32, val_split: float=0.2, exclude: set=None):
    """
    Creates PyTorch DataLoaders for the MNIST dataset with preprocessing.
    Args:
        batch_size (int): The number of samples per batch.
        val_split (float): Proportion of training data to use for validation.

    Returns:
        tuple: A tuple containing the training DataLoader and the testing DataLoader.
    """
    # Define the transformations to be applied to the images
    transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(2/28, 2/28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # Load the MNIST training dataset
    train_set = torchvision.datasets.MNIST(root=f'{PROJECT_DIR}/data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.MNIST(root=f'{PROJECT_DIR}/data', train=False, download=True, transform=transform_eval)
    # class exclusion
    if exclude is not None:
        # Convert train_set.targets to a tensor if it's not already
        if not isinstance(train_set.targets, torch.Tensor):
            train_set.targets = torch.tensor(train_set.targets)
        
        # Convert test_set.targets to a tensor if it's not already
        if not isinstance(test_set.targets, torch.Tensor):
            test_set.targets = torch.tensor(test_set.targets)
        
        # Create masks for exclusion
        train_mask = torch.tensor([target.item() not in exclude for target in train_set.targets])
        test_mask = torch.tensor([target.item() not in exclude for target in test_set.targets])
        
        # Create indices lists
        train_idx = torch.nonzero(train_mask).squeeze().tolist()
        test_idx = torch.nonzero(test_mask).squeeze().tolist()
        
        # Apply exclusion
        train_set = Subset(train_set, train_idx)
        test_set = Subset(test_set, test_idx)
    # split train validation
    idx = list(range(len(test_set)))
    idx = np.random.permutation(idx)
    split = int(val_split * len(test_set))
    train_idx, val_idx = idx[split:], idx[:split]
    test_subset = Subset(test_set, train_idx)
    val_subset = Subset(test_set, val_idx)
    # loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_mnist_pipeline(batch_size=64)
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        if i == 0:
            print("Shape of training batch inputs:", inputs.shape)
            print("Shape of training batch labels:", labels.shape)
            break
    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        if i == 0:
            print("Shape of validation batch inputs:", inputs.shape)
            print("Shape of validation batch labels:", labels.shape)
            break
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        if i == 0:
            print("Shape of testing batch inputs:", inputs.shape)
            print("Shape of testing batch labels:", labels.shape)
            break
    print("MNIST dataset loaded into PyTorch DataLoaders.")