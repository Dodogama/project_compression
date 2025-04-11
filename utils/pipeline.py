import sys
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from configs.config import Config
import itertools
from .losses import DistillationLoss, Accuracy


def grid_search_cv(hparams_grid):
    param_names = list(hparams_grid.keys())
    param_values = list(hparams_grid.values())
    all_combinations = list(itertools.product(*param_values))

    for combo in all_combinations:
        current_hparams = dict(zip(param_names, combo))
        print(current_hparams)


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
        elif scheduler:
            scheduler.step()
    return epoch_losses


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
    return epoch_metrics


def train_val(train_loader, val_loader, model, criterion, optimizer, scheduler, device, aux_metrics, path):
    metrics = {"train_loss": [], "val_loss": []}
    for k in aux_metrics.keys():
        metrics[k] = []
    try:
        best_val_acc = torch.load(path)['accuracy']
    except Exception:
        best_val_acc = 0
    patience = 10
    counter = 0
    epochs = 200

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
            print(f"Epoch {epoch+1}: New best accuracy: {metrics['accuracy'][-1]:.4f} saving model...")
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': metrics['val_loss'][-1],
                'accuracy': metrics['accuracy'][-1]
            }
            torch.save(state, path)
        else:
            counter += 1
        if counter >= patience:
            print(f"Epoch {epoch+1}: Early stop triggered.")
            break
    return metrics


def _train_val(cfg: Config):    
    """
    Trains and Validates Model using a Config dataclass
    Args:
    Return:
    """
    try:
        best_val_loss = torch.load(cfg.path)['val_loss']
    except Exception:
        best_val_loss = float('inf')
    patience = 10
    counter = 0
    for epoch in range(cfg.epochs):
        # dump config + add device
        train_loss = train_model(**cfg.train_attr)
        val_loss = evaluate_model(**cfg.val_attr)
        # pop metrics from config
        cfg.metrics['train_loss'].append(np.mean(train_loss))
        cfg.metrics['val_loss'].append(np.mean(val_loss))
        if cfg.metrics['val_loss'][-1] < best_val_loss:
            best_val_loss = cfg.metrics['val_loss'][-1]
            counter = 0
            print(f"Epoch {epoch+1}: New best val loss: {best_val_loss:.4f}, saving model...")
            state = {
                'epoch': epoch,
                'state_dict': cfg.model.state_dict(),
                'optimizer': cfg.optimizer.state_dict(),
                'val_loss': best_val_loss
            }
            torch.save(state, cfg.path)
        else:
            counter += 1
        if counter >= patience:
            print(f"Epoch {epoch+1}: Early stop triggered.")
            break


def distill_model(train_loader: DataLoader, student: nn.Module, teacher: nn.Module, 
                  criterion: nn.Module, optimizer: nn.Module,
                  scheduler: nn.Module=None, device: str='cpu') -> list:
    """
    Train the student model for one epoch using knowledge distillation.

    Args:
        train_loader: DataLoader for the training data.
        student_model: The PyTorch student model to train.
        teacher_model: The PyTorch teacher model (should be in eval mode).
        optimizer: Optimizer for updating student model parameters.
        criterion: Distillation loss function (e.g., DistillationLoss).
        device: Device to run the training on ('cpu' or 'cuda').

    Returns:
        list: Collection of train losses.
    """
    student.train()
    teacher.eval()  # Ensure teacher model is in evaluation mode
    epoch_losses = []
    for inputs, targets in tqdm.tqdm(train_loader, desc='distilling...', file=sys.stdout):
        inputs = inputs.to(device)
        targets = targets.to(device)
        # preds
        student_preds = student(inputs)
        with torch.no_grad():
            teacher_preds = teacher(inputs)
        # distillation loss
        loss = criterion(student_preds, teacher_preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss.item())
        elif scheduler:
            scheduler.step()
    return epoch_losses


def distill(train_loader, val_loader, student, teacher, loss, criterion, optimizer, scheduler, device, aux_metrics, path):
    metrics = {"train_loss": [], "val_loss": []}
    for k in aux_metrics.keys():
        metrics[k] = []
    try:
        best_val_acc = torch.load(path)['accuracy']
    except Exception:
        best_val_acc = 0
    patience = 10
    counter = 0
    epochs = 200

    distill_loss = DistillationLoss(T=T)
    for epoch in range(epochs):
        train_loss = distill_model(train_loader, student, teacher, distill_loss, optimizer, scheduler, device)
        val_loss = evaluate_model(val_loader, student, criterion, device)
        metrics['train_loss'].append(np.mean(train_loss))
        metrics['val_loss'].append(np.mean(val_loss))
        for k, v in aux_metrics.items():
            stat = evaluate_model(val_loader, student, v, device)
            metrics[k].append(np.mean(stat))
        if metrics['accuracy'][-1] >= best_val_acc:
            best_val_acc = metrics['accuracy'][-1]
            counter = 0
            print(f"Epoch {epoch+1}: New best accuracy: {metrics['accuracy'][-1]:.4f} saving model...")
            state = {
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': metrics['val_loss'][-1],
                'accuracy': metrics['accuracy'][-1]
            }
            torch.save(state, path)
        else:
            counter += 1
        if counter >= patience:
            print(f"Epoch {epoch+1}: Early stop triggered.")
            break
    return metrics


if __name__ == '__main__':
    hparams_grid = {
        'learning_rate': [1e-3, 1e-4],
        'batch_size': [32],
        'optimizer': ['Adam', 'RMSprop'],
        'num_epochs': [5]
    }
    grid_search_cv(hparams_grid)