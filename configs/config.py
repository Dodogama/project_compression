from dataclasses import dataclass
import torch.nn as nn
import torch.optim as optim
import torch

@dataclass
class Config:
    model: nn.Module
    criterion: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    device: str
    epochs: int
    lr: float
    metrics: dict
    path: str
    
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        test_loader,
        device,
        epochs,
        lr,
        path,
        **kwargs
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.path = path
        self.metrics = {'train_loss': [], 'val_loss': []}
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __post_init__(self):
        # instantiate these
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

    def clear_metrics(self):
        for key in self.metrics:
            self.metrics[key] = []

    def train_attr(self):
        return {
            'train_loader': self.train_loader,
            'model': self.model,
            'criterion': self.criterion,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'device': self.device
        }
    
    def val_attr(self):
        return {
            'val_loader': self.val_loader,
            'model': self.model,
            'criterion': self.criterion,
            'device': self.device
        }
    
    def test_attr(self):
        return {
            'val_loader': self.test_loader,
            'model': self.model,
            'criterion': self.criterion,
            'device': self.device
        }