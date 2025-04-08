import torch
import torch.nn.functional as F
import torch.nn as nn

class DistillationLoss(nn.Module):
    def __init__(self, T=1.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.T = T
        self.alpha = alpha

    @staticmethod
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
        # Softmax with temperature for soft targets
        soft_targets = F.softmax(teacher_logits / T, dim=1)
        soft_prob = F.log_softmax(student_logits / T, dim=1)
        # Calculate the distillation loss (soft targets)
        distillation = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (T * T)
        # Calculate the standard cross-entropy loss (hard targets)
        standard_loss = F.cross_entropy(student_logits, labels)
        # Return the weighted sum
        return alpha * distillation + (1 - alpha) * standard_loss

    def forward(self, student_logits, teacher_logits, labels):
        return self.distillation_loss(student_logits, teacher_logits, labels, self.T, self.alpha)


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