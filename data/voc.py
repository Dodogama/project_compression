import os
import numpy as np
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torchvision.transforms.functional as TF
import random

PROJECT_DIR = os.path.dirname(os.getcwd())

class SegmentationJointTransform:
    def __init__(self, resize=(256, 256), rotation=10, hflip_prob=0.5, color_jitter=True):
        self.resize = resize
        self.rotation = rotation
        self.hflip_prob = hflip_prob
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2) if color_jitter else None

    def __call__(self, image, mask):
        # Resize
        image = TF.resize(image, self.resize, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.resize, interpolation=TF.InterpolationMode.NEAREST)

        # Random horizontal flip
        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random rotation
        angle = random.uniform(-self.rotation, self.rotation)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Color jitter (image only)
        if self.color_jitter:
            image = self.color_jitter(image)

        # Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.pil_to_tensor(mask).squeeze(0).long()  # [H, W] with class labels

        return image, mask

class VOCSegmentationWithTransform(VOCSegmentation):
    def __init__(self, root, year, image_set, transform=None, download=False):
        super().__init__(root=root, year=year, image_set=image_set, download=download)
        self.transform = transform

    def __getitem__(self, index):
        image, target = super().__getitem__(index)

        if self.transform:
            image, target = self.transform(image, target)

        return image, target
    
def get_voc_pipeline(batch_size: int=32):
    """
    Can update to return number of classes as well from train_set
    """
    # transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    target_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.PILToTensor(),
    ])
    # Load VOC dataset
    train_set = VOCSegmentation(root=f'{PROJECT_DIR}/data', year='2012', image_set='train', download=True,
                                 transform=transform, target_transform=target_transform)
    val_set = VOCSegmentation(root=f'{PROJECT_DIR}/data', year='2012', image_set='val', download=True,
                               transform=transform, target_transform=target_transform)
    # loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
    
def get_voc_pipeline_test(batch_size: int=32):
    """
    Can update to return number of classes as well from train_set
    """
    # transforms
    joint_transform = SegmentationJointTransform()
    train_set = VOCSegmentationWithTransform(
        root=f'{PROJECT_DIR}/data', year='2012', image_set='train',
        transform=joint_transform, download=True
    )

    val_set = VOCSegmentationWithTransform(
        root=f'{PROJECT_DIR}/data', year='2012', image_set='val',
        transform=joint_transform, download=True
    )
    # loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_voc_pipeline(batch_size=64)
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
    
    
