import sys
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from .losses import DistillationLoss, Accuracy
from .train import evaluate_model


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
    if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()
    return np.mean(epoch_losses)


def distill(train_loader, val_loader, student, teacher, optimizer, scheduler,
            T=5, a=0.5, device='cpu', aux_metrics={}, path="./temp.pth", patience=50, epochs=50):
    metrics = {"train_loss": [], "accuracy": []}
    for k in aux_metrics.keys():
        metrics[k] = []
    best_val_acc = 0.
    counter = 0

    for epoch in range(epochs):
        metrics['train_loss'].append(distill_model(train_loader, student, teacher, 
                                                   DistillationLoss(T=T, alpha=a), optimizer, scheduler, device))
        metrics['accuracy'].append(evaluate_model(val_loader, student, Accuracy(), device))
        for k, v in aux_metrics.items():
            metrics[k].append(evaluate_model(val_loader, student, v, device))
        if metrics['accuracy'][-1] >= best_val_acc:
            best_val_acc = metrics['accuracy'][-1]
            counter = 0
            print(f"Epoch {epoch+1}: New best accuracy: {best_val_acc:.4f} saving model...")
            state = {
                'epoch': epoch,
                'state_dict': student.state_dict(),
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


def distill_self():
    pass


def distill_seg(train_loader, val_loader, student, teacher, T, criterion, optimizer, scheduler, device, aux_metrics, path,
            patience=50, epochs=50):
    metrics = {"train_loss": [], "val_loss": []}
    for k in aux_metrics.keys():
        metrics[k] = []
    best_val_acc = 0.
    counter = 0

    distill_loss = DistillationLoss(T=T)
    for epoch in range(epochs):
        train_loss = distill_model(train_loader, student, teacher, distill_loss, optimizer, scheduler, device)
        val_loss = evaluate_model(val_loader, student, criterion, device)
        metrics['train_loss'].append(np.mean(train_loss))
        metrics['val_loss'].append(np.mean(val_loss))
        for k, v in aux_metrics.items():
            stat = evaluate_model(val_loader, student, v, device)
            metrics[k].append(np.mean(stat))
        if metrics['val_loss'][-1] >= best_val_acc:
            best_val_acc = metrics['val_loss'][-1]
            counter = 0
            print(f"Epoch {epoch+1}: New best val loss: {metrics['val_loss'][-1]:.4f} saving model...")
            state = {
                'epoch': epoch,
                'state_dict': student.state_dict(),
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