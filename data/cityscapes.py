import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from torchvision.datasets import Cityscapes

PROJECT_DIR = os.path.dirname(os.getcwd())


def get_cityscapes_pipeline(batch_size: int = 4, crop_size=(256, 512), val_split: float = 0.1):
    """
    Creates PyTorch DataLoaders for the Cityscapes dataset with preprocessing.
    Args:
        batch_size (int): Batch size for data loaders.
        crop_size (tuple): Crop size for training/validation.
        val_split (float): Portion of training data used for validation.
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Define common transformations
    common_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2869, 0.3251, 0.2839], std=[0.1761, 0.1804, 0.1775])  # Cityscapes mean/std
    ])

    # Define image and target transformations
    def target_transform(target):
        return np.array(target)

    train_transform = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        *common_transform.transforms  # Apply normalization after augmentation
    ])

    # Load datasets
    city_root = os.path.join(PROJECT_DIR, "data/cityscapes")

    train_set = Cityscapes(
        root=city_root,
        split='train',
        mode='fine',
        target_type='semantic',
        transform=train_transform,
        target_transform=target_transform
    )

    val_set = Cityscapes(
        root=city_root,
        split='val',
        mode='fine',
        target_type='semantic',
        transform=common_transform,
        target_transform=target_transform
    )

    test_set = Cityscapes(
        root=city_root,
        split='test',
        mode='fine',
        target_type='semantic',
        transform=common_transform,
        target_transform=target_transform
    )

    # Optional: shuffle and split train into train/val (if you want to override default val split)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_cityscapes_pipeline(batch_size=2)

    for name, loader in zip(["train", "val", "test"], [train_loader, val_loader, test_loader]):
        for i, data in enumerate(loader, 0):
            images, masks = data
            print(f"{name.capitalize()} - Image batch shape: {images.shape}")
            print(f"{name.capitalize()} - Mask batch shape: {masks.shape}")
            break

    print("Cityscapes dataset loaded into PyTorch DataLoaders.")
