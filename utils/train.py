import sys
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .losses import Accuracy, MIoU
import numpy as np


def train_model(train_loader: DataLoader, model: nn.Module, criterion: nn.Module, optimizer: nn.Module,
                scheduler: nn.Module=None, device: str='cpu') -> list:
    """
    Train the model for one epoch.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for the training data.
        optimizer: Optimizer for updating model parameters.
        criterion: Loss function.
        device: Device to run the training on ('cpu' or 'cuda').

    Returns:
        list: Collection of train losses.
    """
    model.train()
    epoch_losses = []
    for inputs, targets in tqdm.tqdm(train_loader, desc='training...', file=sys.stdout):
        inputs = inputs.to(device)
        targets = targets.to(device)
        preds = model(inputs)
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss.item())
    if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()
    return np.mean(epoch_losses)


def evaluate_model(val_loader: DataLoader, model: nn.Module, criterion: nn.Module, device: str='cpu') -> list:
    """
    Evaluate the model on validation data.

    Args:
        model: The PyTorch model to evaluate.
        val_loader: DataLoader for the validation data.
        criterion: Loss function.
        device: Device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        list: Collection of metrics.
    """
    model.eval()
    epoch_metrics = []
    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(val_loader, desc='evaluating...', file=sys.stdout):
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = model(inputs)
            loss = criterion(preds, targets)
            epoch_metrics.append(loss.item())
    return np.mean(epoch_metrics)


def train_val(train_loader, val_loader, model, criterion, optimizer, scheduler, 
              device='cpu', aux_metrics={}, path="./temp.pth", patience=50, epochs=50):
    metrics = {"train_loss": [], "accuracy": []}
    for k in aux_metrics.keys():
        metrics[k] = []

    best_val_acc = 0
    counter = 0
    for epoch in range(epochs):
        metrics['train_loss'].append(train_model(train_loader, model, criterion, optimizer, scheduler, device))
        metrics['accuracy'].append(evaluate_model(val_loader, model, Accuracy(), device))
        for k, v in aux_metrics.items():
            metrics[k].append(evaluate_model(val_loader, model, v, device))
        if metrics['accuracy'][-1] >= best_val_acc:
            best_val_acc = metrics['accuracy'][-1]
            counter = 0
            print(f"Epoch {epoch+1}: New best accuracy: {best_val_acc:.4f} saving model...")
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            state.update({k: metrics[k][-1] for k in metrics.keys()})
            torch.save(state, path)
        else:
            counter += 1
        if counter >= patience:
            print(f"Epoch {epoch+1}: Early stop triggered.")
            break
    return metrics


def train_val_seg(train_loader, val_loader, model, criterion, optimizer, scheduler, device, aux_metrics, path,
                  patience=50, epochs=50):
    metrics = {"train_loss": [], "val_loss": []}
    for k in aux_metrics.keys():
        metrics[k] = []

    best_val_acc = 0
    counter = 0

    for epoch in range(epochs):
        train_loss = train_model(train_loader, model, criterion, optimizer, scheduler, device)
        val_loss = evaluate_model(val_loader, model, criterion, device)
        metrics['train_loss'].append(np.mean(train_loss))
        metrics['val_loss'].append(np.mean(val_loss))
        for k, v in aux_metrics.items():
            stat = evaluate_model(val_loader, model, v, device)
            metrics[k].append(np.mean(stat))
        if metrics['accuracy'][-1] >= best_val_acc:
            best_val_acc = metrics['accuracy'][-1]
            counter = 0
            print(f"Epoch {epoch+1}: New best accuracy: {best_val_acc:.4f} saving model...")
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            state.update({k: metrics[k][-1] for k in metrics.keys()})
            torch.save(state, path)
        else:
            counter += 1
        if counter >= patience:
            print(f"Epoch {epoch+1}: Early stop triggered.")
            break
    return metrics