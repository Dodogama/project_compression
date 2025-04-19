import torch
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score

class DistillationLoss(nn.Module):
    def __init__(self, T=1.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels):
        return distillation_loss(student_logits, teacher_logits, labels, self.T, self.alpha)


def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    """
    Compute the knowledge distillation loss.
    
    Args:
        student_logits: Logits from the student model
        teacher_logits: Logits from the teacher model
        labels: True labels
        T: Temperature for softening probability distributions
        alpha: Weight for the distillation loss vs. standard cross-entropy loss
    
    Returns:
        Combined loss
    """
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    soft_prob = F.log_softmax(student_logits / T, dim=1)
    distillation = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (T * T)
    standard_loss = F.cross_entropy(student_logits, labels)
    return alpha * distillation + (1 - alpha) * standard_loss

class RMSELoss(nn.Module):
    """Custom RMSE loss."""
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred: torch.tensor, y_true: torch.tensor):
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

class SMAPELoss(nn.Module):
    """Custom SMAPE loss."""
    def __init__(self):
        super(SMAPELoss, self).__init__()

    def forward(self, y_pred: torch.tensor, y_true: torch.tensor):
        denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2
        smape_value = torch.mean(torch.abs(y_pred - y_true) / (denominator + 1e-8)) * 100
        return smape_value
    
class Accuracy(nn.Module):
    """Accuracy metric."""
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, y_pred: torch.tensor, y_true: torch.tensor):
        class_pred = torch.argmax(y_pred, dim=1)
        c = (class_pred == y_true).sum().item()
        n = y_true.size(0)
        return torch.tensor(c / n)
    
class PrecisionRecallF1(nn.Module):
    """
    Calculates Precision, Recall, and F1-Score for multi-class classification.

    Args:
        num_classes (int): The number of classes.
        average (str): Averaging method for the metrics ('macro', 'micro', 'weighted', None).
                       Defaults to 'macro'.
    """
    def __init__(self, num_classes: int, average: str = 'macro'):
        super().__init__()
        self.num_classes = num_classes
        self.average = average
        self.precision_metric = MulticlassPrecision(num_classes=num_classes, average=average)
        self.recall_metric = MulticlassRecall(num_classes=num_classes, average=average)
        self.f1_metric = MulticlassF1Score(num_classes=num_classes, average=average)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates precision, recall, and f1-score.

        Args:
            preds (torch.Tensor): Model predictions (logits or probabilities).
            target (torch.Tensor): True labels (integers).

        Returns:
            tuple: A tuple containing (precision, recall, f1-score) as torch.Tensor.
                   If average is None, these will be tensors of shape (num_classes,).
                   Otherwise, they will be scalar tensors.
        """
        self.precision_metric.update(preds, target)
        self.recall_metric.update(preds, target)
        self.f1_metric.update(preds, target)
        return self.precision_metric.compute(), self.recall_metric.compute(), self.f1_metric.compute()

    def reset(self):
        """Resets the internal state of the metrics."""
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()

class MIoU(nn.Module):
    def __init__(self, num_classes=21, ignore_index=255):
        super(MIoU, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
        
    def reset(self):
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)
        
    def forward(self, outputs, targets):
        """
        Args:
            outputs: [B, C, H, W] tensor of raw model outputs (logits)
            targets: [B, H, W] tensor of ground truth labels
        Returns:
            mean IoU score (scalar)
        """
        # Convert logits to predictions
        preds = torch.argmax(outputs, dim=1)  # [B, H, W]
        
        batch_size = outputs.size(0)
        mean_iou = 0.0
        
        for b in range(batch_size):
            current_pred = preds[b]  # [H, W]
            current_target = targets[b]  # [H, W]
            
            # Create mask for valid pixels (not ignore_index)
            valid_mask = current_target != self.ignore_index
            
            # Calculate per-class IoU
            class_iou_sum = 0.0
            valid_classes = 0
            
            for cls in range(self.num_classes):
                pred_mask = (current_pred == cls) & valid_mask
                target_mask = (current_target == cls) & valid_mask
                
                intersection = torch.logical_and(pred_mask, target_mask).sum().item()
                union = torch.logical_or(pred_mask, target_mask).sum().item()
                
                # Only count classes that appear in the ground truth
                if union > 0:
                    class_iou_sum += intersection / union
                    valid_classes += 1
            
            # Average IoU across valid classes for this image
            if valid_classes > 0:
                mean_iou += class_iou_sum / valid_classes
        
        # Average across batch
        return torch.tensor(mean_iou / batch_size) if batch_size > 0 else torch.tensor(0.0)
    
    # Added for compatibility with previous evaluation function
    def compute(self):
        # For this implementation compute() is not needed since forward() calculates directly
        # But we keep it for interface compatibility
        return self.intersection.sum() / self.union.sum() if self.union.sum() > 0 else 0.0
    
    # Added for compatibility with previous training loop
    def update(self, outputs, targets):
        # Calculate batch IoU and accumulate stats
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            
            for cls in range(self.num_classes):
                valid_mask = targets != self.ignore_index
                pred_mask = (preds == cls) & valid_mask
                target_mask = (targets == cls) & valid_mask
                
                intersection = torch.logical_and(pred_mask, target_mask).sum().item()
                union = torch.logical_or(pred_mask, target_mask).sum().item()
                
                self.intersection[cls] += intersection
                self.union[cls] += union
