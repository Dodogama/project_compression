import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_cifar10_pipeline(batch_size=32):
    """
    Creates PyTorch DataLoaders for the CIFAR-10 dataset with preprocessing.

    Args:
        batch_size (int): The number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple: A tuple containing the training DataLoader and the testing DataLoader.
    """
    # Define the transformations to be applied to the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Load the CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # Load the CIFAR-10 testing dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader


if __name__ == '__main__':
    train_loader, test_loader = get_cifar10_pipeline(batch_size=64, num_workers=4)
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        if i == 0:
            print("Shape of training batch inputs:", inputs.shape)
            print("Shape of training batch labels:", labels.shape)
            break
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        if i == 0:
            print("Shape of testing batch inputs:", inputs.shape)
            print("Shape of testing batch labels:", labels.shape)
            break
    print("CIFAR-10 dataset loaded into PyTorch DataLoaders.")