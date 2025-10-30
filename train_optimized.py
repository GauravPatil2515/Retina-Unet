"""
OPTIMIZED TRAINING SCRIPT - Maximum Performance
Includes all best practices for retina vessel segmentation:
- Data augmentation
- Combined loss function
- Cosine learning rate schedule
- Mixed precision training
- Gradient clipping
- Advanced metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import os
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import time

from dataloader import ImageDataset, PATH
from unet import Unet
from config_optimized import *

# =============================================================================
# DATA AUGMENTATION
# =============================================================================
class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset with augmentation"""
    def __init__(self, base_dataset, is_train=True):
        self.base_dataset = base_dataset
        self.is_train = is_train
        
        # Augmentation transforms
        if is_train and USE_AUGMENTATION:
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(AUG_ROTATION),
                transforms.RandomHorizontalFlip(AUG_HORIZONTAL_FLIP),
                transforms.RandomVerticalFlip(AUG_VERTICAL_FLIP),
                transforms.ColorJitter(
                    brightness=AUG_BRIGHTNESS,
                    contrast=AUG_CONTRAST
                ),
            ])
        else:
            self.img_transform = None
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, mask = self.base_dataset[idx]
        
        # Apply same transform to both image and mask
        if self.img_transform is not None:
            # Set same seed for image and mask
            seed = torch.randint(0, 2**32, (1,)).item()
            
            torch.manual_seed(seed)
            img = self.img_transform(img)
            
            # Apply geometric transforms to mask (not color)
            torch.manual_seed(seed)
            mask_transform = transforms.Compose([
                transforms.RandomRotation(AUG_ROTATION),
                transforms.RandomHorizontalFlip(AUG_HORIZONTAL_FLIP),
                transforms.RandomVerticalFlip(AUG_VERTICAL_FLIP),
            ])
            mask = mask_transform(mask.unsqueeze(0)).squeeze(0)
        
        return img, mask

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================
class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)[:, 1, :, :]
        target = target.float()
        
        intersection = (pred * target).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    """Combined CrossEntropy + Dice Loss"""
    def __init__(self, ce_weight=0.5, dice_weight=0.5, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target.long())
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice

# =============================================================================
# METRICS
# =============================================================================
def calculate_metrics(pred, target):
    """Calculate comprehensive metrics"""
    pred = torch.softmax(pred, dim=1)[:, 1, :, :] > PREDICTION_THRESHOLD
    target = target.bool()
    
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    tp = (pred_flat & target_flat).sum().float()
    fp = (pred_flat & ~target_flat).sum().float()
    tn = (~pred_flat & ~target_flat).sum().float()
    fn = (~pred_flat & target_flat).sum().float()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'dice': dice.item(),
        'iou': iou.item()
    }

# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_metrics = {'dice': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{EPOCHS} [Train]')
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if USE_AMP:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks.long())
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if GRADIENT_CLIPPING:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks.long())
            loss.backward()
            
            if GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            
            optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            metrics = calculate_metrics(outputs, masks)
        
        epoch_loss += loss.item()
        for key in epoch_metrics:
            epoch_metrics[key] += metrics[key]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{metrics["dice"]:.4f}'
        })
    
    # Average metrics
    num_batches = len(loader)
    epoch_loss /= num_batches
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches
    
    return epoch_loss, epoch_metrics

def validate_epoch(model, loader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    epoch_loss = 0
    epoch_metrics = {'dice': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{EPOCHS} [Val]  ')
    
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            if USE_AMP:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks.long())
            else:
                outputs = model(images)
                loss = criterion(outputs, masks.long())
            
            metrics = calculate_metrics(outputs, masks)
            
            epoch_loss += loss.item()
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{metrics["dice"]:.4f}'
            })
    
    num_batches = len(loader)
    epoch_loss /= num_batches
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches
    
    return epoch_loss, epoch_metrics

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================
def train():
    """Main training function"""
    
    # Print configuration
    print_config()
    
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    
    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Device
    device = torch.device(DEVICE)
    
    # Load dataset
    print("📂 Loading dataset...")
    full_dataset = ImageDataset(folder_path=PATH)
    
    # Split dataset
    val_size = int(VALIDATION_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    # Wrap with augmentation
    train_dataset = AugmentedDataset(train_dataset, is_train=True)
    val_dataset = AugmentedDataset(val_dataset, is_train=False)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    # Model
    print("\n🏗️  Building model...")
    model = Unet(IN_CHANNELS, OUT_CHANNELS).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model parameters: {num_params:,}")
    
    # Loss function
    if LOSS_TYPE == "crossentropy":
        if USE_CLASS_WEIGHTS:
            class_weights = torch.tensor([BACKGROUND_WEIGHT, VESSEL_WEIGHT]).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
    elif LOSS_TYPE == "dice":
        criterion = DiceLoss()
    else:  # combined
        if USE_CLASS_WEIGHTS:
            class_weights = torch.tensor([BACKGROUND_WEIGHT, VESSEL_WEIGHT]).to(device)
        else:
            class_weights = None
        criterion = CombinedLoss(CE_WEIGHT, DICE_WEIGHT, class_weights)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    if LR_SCHEDULER_TYPE == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=LR_MIN)
    elif LR_SCHEDULER_TYPE == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=LR_FACTOR,
            patience=LR_PATIENCE
        )
    else:
        scheduler = None
    
    # Mixed precision scaler
    scaler = GradScaler() if USE_AMP else None
    
    # Tensorboard
    if USE_TENSORBOARD:
        writer = SummaryWriter(log_dir=LOG_DIR)
    
    # Training tracking
    best_val_dice = 0
    epochs_no_improve = 0
    
    print("\n🚀 Starting training...\n")
    start_time = time.time()
    
    # Training loop
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        if scheduler is not None:
            if LR_SCHEDULER_TYPE == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{EPOCHS} Summary ({epoch_time:.1f}s)")
        print(f"{'='*70}")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"\nTrain - Loss: {train_loss:.4f} | Dice: {train_metrics['dice']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Loss: {val_loss:.4f} | Dice: {val_metrics['dice']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f}")
        
        # Tensorboard logging
        if USE_TENSORBOARD:
            writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalars('Dice', {'train': train_metrics['dice'], 'val': val_metrics['dice']}, epoch)
            writer.add_scalars('Accuracy', {'train': train_metrics['accuracy'], 'val': val_metrics['accuracy']}, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save checkpoint
        if epoch % SAVE_CHECKPOINT_EVERY == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_dice': val_metrics['dice']
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            epochs_no_improve = 0
            
            if KEEP_BEST_MODEL:
                best_model_path = os.path.join(MODEL_DIR, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"⭐ New best model! Dice: {best_val_dice:.4f} → Saved to {best_model_path}")
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if EARLY_STOPPING and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n⚠️  Early stopping triggered! No improvement for {EARLY_STOPPING_PATIENCE} epochs.")
            break
    
    # Save final model
    if KEEP_LAST_MODEL:
        last_model_path = os.path.join(MODEL_DIR, 'last_model.pth')
        torch.save(model.state_dict(), last_model_path)
        print(f"\n💾 Final model saved: {last_model_path}")
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"{'='*70}\n")
    
    if USE_TENSORBOARD:
        writer.close()

if __name__ == "__main__":
    train()
