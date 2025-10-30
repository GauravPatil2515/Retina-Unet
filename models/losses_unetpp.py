"""
Loss Functions and Metrics for U-Net++
Implements:
- Dice Coefficient
- Dice Loss
- BCE + Dice Combined Loss
- Multi-output weighted loss (Deep Supervision)
- Evaluation Metrics (Dice, Accuracy, Sensitivity, Specificity, AUC)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    Formula: 1 - (2 * intersection + smooth) / (sum(y_true) + sum(y_pred) + smooth)
    Works with logits by applying sigmoid internally
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits [B, 1, H, W]
            target: Ground truth binary mask [B, 1, H, W]
        """
        # Apply sigmoid to logits
        pred = torch.sigmoid(pred)
        
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss
    Formula: BCE(pred, target) + Dice_Loss(pred, target)
    
    Uses BCEWithLogitsLoss for numerical stability and autocast compatibility
    
    Benefits:
    - BCE handles pixel-wise classification
    - Dice handles overlap/segmentation quality
    """
    def __init__(self, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()  # Changed from BCELoss
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return bce_loss + dice_loss


class DeepSupervisionLoss(nn.Module):
    """
    Multi-output weighted loss for deep supervision
    
    Weights: [0.25, 0.25, 0.25, 1.0] for outputs 1-4
    Total Loss = 0.25*L1 + 0.25*L2 + 0.25*L3 + 1.0*L4
    
    The final output (output 4) has the highest weight
    """
    def __init__(self, weights=[0.25, 0.25, 0.25, 1.0]):
        super(DeepSupervisionLoss, self).__init__()
        self.weights = weights
        self.loss_fn = BCEDiceLoss()
    
    def forward(self, outputs, target):
        """
        Args:
            outputs: Tuple of 4 predictions from deep supervision heads
            target: Ground truth mask [B, 1, H, W]
        """
        total_loss = 0
        for i, (output, weight) in enumerate(zip(outputs, self.weights)):
            loss = self.loss_fn(output, target)
            total_loss += weight * loss
        
        return total_loss


def dice_coefficient(pred, target, smooth=1e-6):
    """
    Calculate Dice coefficient (F1 score)
    Range: [0, 1] where 1 = perfect overlap
    
    Args:
        pred: Predicted logits [B, 1, H, W]
        target: Ground truth binary mask [B, 1, H, W]
    """
    # Apply sigmoid to logits
    pred = torch.sigmoid(pred)
    
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculate comprehensive metrics:
    - Dice Coefficient
    - Accuracy
    - Sensitivity (True Positive Rate)
    - Specificity (True Negative Rate)
    - AUC (Area Under ROC Curve)
    
    Args:
        pred: Predicted logits [B, 1, H, W]
        target: Ground truth binary mask [B, 1, H, W]
        threshold: Threshold for binarizing predictions (default: 0.5)
    
    Returns:
        Dictionary with all metrics
    """
    # Apply sigmoid to logits
    pred = torch.sigmoid(pred)
    
    # Convert to numpy
    pred_np = pred.cpu().detach().numpy().flatten()
    target_np = target.cpu().detach().numpy().flatten()
    
    # Binarize predictions
    pred_binary = (pred_np > threshold).astype(np.float32)
    
    # Calculate confusion matrix elements
    TP = np.sum((pred_binary == 1) & (target_np == 1))
    TN = np.sum((pred_binary == 0) & (target_np == 0))
    FP = np.sum((pred_binary == 1) & (target_np == 0))
    FN = np.sum((pred_binary == 0) & (target_np == 1))
    
    # Calculate metrics
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    sensitivity = TP / (TP + FN + 1e-6)  # Recall, TPR
    specificity = TN / (TN + FP + 1e-6)  # TNR
    
    # Calculate AUC
    try:
        auc = roc_auc_score(target_np, pred_np)
    except:
        auc = 0.0
    
    return {
        'dice': dice,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc': auc,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }


class MetricsTracker:
    """
    Track metrics across epochs and batches
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.dice_scores = []
        self.losses = []
    
    def update(self, loss, dice):
        self.losses.append(loss)
        self.dice_scores.append(dice)
    
    def get_average(self):
        return {
            'loss': np.mean(self.losses) if self.losses else 0,
            'dice': np.mean(self.dice_scores) if self.dice_scores else 0
        }


if __name__ == "__main__":
    print("Testing Loss Functions and Metrics...")
    
    # Create dummy data
    batch_size = 4
    pred = torch.sigmoid(torch.randn(batch_size, 1, 128, 128))
    target = torch.randint(0, 2, (batch_size, 1, 128, 128)).float()
    
    print(f"\nInput shapes:")
    print(f"  Prediction: {pred.shape}")
    print(f"  Target: {target.shape}")
    
    # Test Dice Loss
    dice_loss = DiceLoss()
    loss = dice_loss(pred, target)
    print(f"\nDice Loss: {loss.item():.4f}")
    
    # Test BCE+Dice Loss
    bce_dice_loss = BCEDiceLoss()
    loss = bce_dice_loss(pred, target)
    print(f"BCE+Dice Loss: {loss.item():.4f}")
    
    # Test Deep Supervision Loss
    outputs = (pred, pred, pred, pred)  # 4 outputs
    ds_loss = DeepSupervisionLoss()
    loss = ds_loss(outputs, target)
    print(f"Deep Supervision Loss: {loss.item():.4f}")
    
    # Test Dice Coefficient
    dice = dice_coefficient(pred, target)
    print(f"\nDice Coefficient: {dice:.4f}")
    
    # Test comprehensive metrics
    metrics = calculate_metrics(pred, target)
    print(f"\nComprehensive Metrics:")
    print(f"  Dice: {metrics['dice']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  TP: {metrics['TP']}, TN: {metrics['TN']}, FP: {metrics['FP']}, FN: {metrics['FN']}")
    
    print("\n[OK] Loss functions and metrics test complete!")
