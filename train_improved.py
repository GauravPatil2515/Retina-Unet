"""
Improved Training Script for Retina U-Net Segmentation
Features:
- Validation split
- Learning rate scheduler
- Early stopping
- Tensorboard logging
- Better metrics tracking
- Progress visualization
"""

import torch
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from utils import *
from dataloader import ImageDataset, PATH
from unet import Unet
from config import *

from tqdm import tqdm
import numpy as np

class DiceLoss(torch.nn.Module):
    """Dice Loss for segmentation tasks"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)[:, 1, :, :]  # Get vessel probability
        target = target.float()
        
        intersection = (pred * target).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(torch.nn.Module):
    """Combination of CrossEntropy and Dice Loss"""
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice

def calculate_metrics(pred, target):
    """Calculate accuracy, precision, recall, F1, and Dice score"""
    with torch.no_grad():
        # Convert predictions to class indices
        pred_class = torch.argmax(pred, dim=1)
        
        # Flatten tensors
        pred_flat = pred_class.flatten().cpu().numpy()
        target_flat = target.flatten().cpu().numpy()
        
        # Calculate metrics
        tp = ((pred_flat == 1) & (target_flat == 1)).sum()
        tn = ((pred_flat == 0) & (target_flat == 0)).sum()
        fp = ((pred_flat == 1) & (target_flat == 0)).sum()
        fn = ((pred_flat == 0) & (target_flat == 1)).sum()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Dice coefficient
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'dice': dice
        }

def validate(model, val_loader, loss_fn, device):
    """Validation loop"""
    model.eval()
    total_loss = 0
    all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'dice': []}
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            
            # Calculate metrics
            metrics = calculate_metrics(pred, y)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
    
    avg_loss = total_loss / len(val_loader)
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    
    return avg_loss, avg_metrics

def save_predictions(model, dataloader, device, epoch, save_dir):
    """Save sample predictions for visualization"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        # Get first batch
        x, y = next(iter(dataloader))
        x = x.to(device)
        
        pred = model(x)
        pred_class = torch.argmax(pred, dim=1).float().unsqueeze(1)
        
        # Save images
        for i in range(min(4, x.size(0))):  # Save up to 4 samples
            save_image(x[i].cpu(), os.path.join(save_dir, f"epoch_{epoch}_input_{i}.png"))
            save_image(pred_class[i].cpu(), os.path.join(save_dir, f"epoch_{epoch}_pred_{i}.png"))
            save_image(y[i].unsqueeze(0).float().cpu(), os.path.join(save_dir, f"epoch_{epoch}_target_{i}.png"))

def train():
    """Main training function"""
    
    # Print configuration
    print_config()
    
    # Initialize Tensorboard
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    # Load dataset
    print("📂 Loading dataset...")
    full_dataset = ImageDataset(path=os.path.join(PATH, "train"))
    
    # Split into train and validation
    val_size = int(VALIDATION_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    print(f"📊 Dataset split:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        pin_memory=True, 
        num_workers=NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        pin_memory=True, 
        num_workers=NUM_WORKERS
    )
    
    # Initialize model
    print("🏗️  Building model...")
    model = Unet(INPUT_CHANNELS, OUTPUT_CHANNELS).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {trainable_params:,} (total: {total_params:,})")
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        amsgrad=AMSGRAD
    )
    
    # Initialize loss function
    if LOSS_TYPE == "cross_entropy":
        loss_fn = torch.nn.CrossEntropyLoss()
    elif LOSS_TYPE == "dice":
        loss_fn = DiceLoss()
    elif LOSS_TYPE == "combined":
        loss_fn = CombinedLoss()
    else:
        raise ValueError(f"Unknown loss type: {LOSS_TYPE}")
    
    print(f"📉 Loss function: {LOSS_TYPE}")
    
    # Initialize learning rate scheduler
    if USE_LR_SCHEDULER:
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=LR_SCHEDULER_FACTOR, 
            patience=LR_SCHEDULER_PATIENCE
        )
        print(f"📅 Learning rate scheduler enabled (patience={LR_SCHEDULER_PATIENCE})")
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_val_dice = 0.0
    avg_val_dice = 0.0
    patience_counter = 0
    
    print("\n🚀 Starting training...\n")
    
    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        epoch_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'dice': []}
        
        # Progress bar for batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            # Forward pass
            pred = model(x)
            loss = loss_fn(pred, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            metrics = calculate_metrics(pred, y)
            for key in epoch_metrics:
                epoch_metrics[key].append(metrics[key])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{metrics['dice']:.4f}"
            })
        
        # Calculate average training metrics
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        
        # Validation
        if epoch % VALIDATE_EVERY == 0:
            avg_val_loss, avg_val_metrics = validate(model, val_loader, loss_fn, DEVICE)
            avg_val_dice = avg_val_metrics['dice']
            
            print(f"\n📊 Epoch {epoch} Results:")
            print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"   Train Dice: {avg_train_metrics['dice']:.4f} | Val Dice: {avg_val_dice:.4f}")
            print(f"   Val Accuracy: {avg_val_metrics['accuracy']:.4f}")
            print(f"   Val F1: {avg_val_metrics['f1']:.4f}\n")
            
            # Tensorboard logging
            writer.add_scalars('Loss', {
                'train': avg_train_loss,
                'validation': avg_val_loss
            }, epoch)
            
            writer.add_scalars('Dice', {
                'train': avg_train_metrics['dice'],
                'validation': avg_val_metrics['dice']
            }, epoch)
            
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Learning rate scheduling
            if USE_LR_SCHEDULER:
                scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_dice': avg_val_metrics['dice'],
                }, os.path.join(MODEL_DIR, "best_model.pth"))
                
                print(f"💾 Saved new best model! (Dice: {best_val_dice:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if USE_EARLY_STOPPING and patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n⏹️  Early stopping triggered after {epoch} epochs")
                print(f"   Best validation Dice: {best_val_dice:.4f}")
                break
        
        # Save checkpoint
        if epoch % SAVE_CHECKPOINT_EVERY == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save prediction visualizations
        if epoch % LOG_IMAGES_EVERY == 0:
            save_predictions(model, val_loader, DEVICE, epoch, RESULT_DIR)
        
        # Save latest model
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "last_model.pth"))
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n✅ Training completed!")
    print(f"📊 Best validation Dice score: {best_val_dice:.4f}")
    print(f"📁 Models saved in: {MODEL_DIR}")
    print(f"📁 Logs saved in: {LOG_DIR}")
    print(f"\n💡 View training progress with: tensorboard --logdir={LOG_DIR}")
    
    writer.close()

if __name__ == "__main__":
    train()
