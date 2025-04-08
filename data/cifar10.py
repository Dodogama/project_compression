import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset

PROJECT_DIR = os.path.dirname(os.getcwd())


def get_cifar10_pipeline(batch_size: int=32, val_split: float=0.2):
    """
    Creates PyTorch DataLoaders for the CIFAR-10 dataset with preprocessing.

    Args:
        batch_size (int): The number of samples per batch.
        val_split (float): Proportion of training data to use for validation.

    Returns:
        tuple: A tuple containing the training DataLoader and the testing DataLoader.
    """
    # Define the transformations to be applied to the images
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # Load the CIFAR-10 training dataset
    train_set = torchvision.datasets.CIFAR10(root=f'{PROJECT_DIR}/data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=f'{PROJECT_DIR}/data', train=False, download=True, transform=transform)
    # split train validation
    idx = list(range(len(train_set)))
    idx = np.random.permutation(idx)
    split = int(val_split * len(train_set))
    train_idx, val_idx = idx[split:], idx[:split]
    train_subset = Subset(train_set, train_idx)
    val_subset = Subset(train_set, val_idx)
    # loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_cifar10_pipeline(batch_size=64)
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
    print("CIFAR-10 dataset loaded into PyTorch DataLoaders.")