"""
U-Net++ Training Script
Complete implementation with all features from Kaggle notebook:
- Nested U-Net architecture with deep supervision
- Patch-based training with filtering
- Combined BCE + Dice loss
- Multi-output weighted loss
- Comprehensive metrics tracking
- Learning rate scheduling
- Early stopping
- Model checkpointing
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import custom modules
from unet_plus_plus import UNetPlusPlus, count_parameters
from dataloader_unetpp import create_data_loaders
from losses_unetpp import DeepSupervisionLoss, BCEDiceLoss, dice_coefficient, calculate_metrics, MetricsTracker


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration - based on Kaggle notebook settings"""
    
    # Paths
    DATA_ROOT = os.path.join(os.getcwd(), "Retina")
    TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "image")
    TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "train", "mask")
    TEST_IMG_DIR = os.path.join(DATA_ROOT, "test", "image")
    TEST_MASK_DIR = os.path.join(DATA_ROOT, "test", "mask")
    CHECKPOINT_DIR = "checkpoints_unetpp"
    
    # Model
    IN_CHANNELS = 3
    OUT_CHANNELS = 1
    DEEP_SUPERVISION = True
    
    # Training
    EPOCHS = 60
    BATCH_SIZE = 8  # Reduced from 16 for 6GB GPU
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Data
    PATCH_SIZE = 128
    STRIDE = 64  # 50% overlap
    NUM_WORKERS = 2  # Reduced from 4 to save memory
    
    # Augmentation
    AUGMENT = True
    FILTER_PATCHES = True
    
    # Training features
    MIXED_PRECISION = True
    GRADIENT_CLIP = 1.0
    ACCUMULATION_STEPS = 2  # Accumulate gradients to simulate batch_size=16
    
    # Callbacks
    SAVE_BEST_ONLY = True
    EARLY_STOPPING_PATIENCE = 10
    LR_PATIENCE = 5
    LR_FACTOR = 0.1
    MIN_LR = 1e-6
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_config():
    """Print training configuration"""
    print("\n" + "="*80)
    print("U-NET++ TRAINING CONFIGURATION")
    print("="*80)
    print(f"\nDevice: {Config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"\nModel Architecture:")
    print(f"  Type: U-Net++ (Nested U-Net)")
    print(f"  Input Channels: {Config.IN_CHANNELS}")
    print(f"  Output Channels: {Config.OUT_CHANNELS}")
    print(f"  Deep Supervision: {Config.DEEP_SUPERVISION}")
    
    print(f"\nTraining Parameters:")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Gradient Accumulation: {Config.ACCUMULATION_STEPS} steps")
    print(f"  Effective Batch Size: {Config.BATCH_SIZE * Config.ACCUMULATION_STEPS}")
    print(f"  Learning Rate: {Config.LEARNING_RATE}")
    print(f"  Weight Decay: {Config.WEIGHT_DECAY}")
    print(f"  Mixed Precision: {Config.MIXED_PRECISION}")
    
    print(f"\nData Configuration:")
    print(f"  Patch Size: {Config.PATCH_SIZE}x{Config.PATCH_SIZE}")
    print(f"  Stride: {Config.STRIDE} (overlap: {(1 - Config.STRIDE/Config.PATCH_SIZE)*100:.0f}%)")
    print(f"  Augmentation: {Config.AUGMENT}")
    print(f"  Patch Filtering: {Config.FILTER_PATCHES}")
    
    print(f"\nCallbacks:")
    print(f"  Early Stopping Patience: {Config.EARLY_STOPPING_PATIENCE}")
    print(f"  LR Reduce Patience: {Config.LR_PATIENCE}")
    print(f"  LR Reduce Factor: {Config.LR_FACTOR}")
    print(f"  Min LR: {Config.MIN_LR}")
    
    print("="*80 + "\n")


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    metrics = MetricsTracker()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{Config.EPOCHS} [TRAIN]')
    
    optimizer.zero_grad()
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Mixed precision training
        if Config.MIXED_PRECISION:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss = loss / Config.ACCUMULATION_STEPS  # Scale loss for accumulation
            
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % Config.ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss = loss / Config.ACCUMULATION_STEPS
            loss.backward()
            
            if (batch_idx + 1) % Config.ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
                optimizer.step()
                optimizer.zero_grad()
        
        # Calculate dice on final output
        if isinstance(outputs, tuple):
            final_output = outputs[-1]
        else:
            final_output = outputs
        
        dice = dice_coefficient(final_output, masks)
        
        # Update metrics (use unscaled loss for display)
        metrics.update(loss.item() * Config.ACCUMULATION_STEPS, dice)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * Config.ACCUMULATION_STEPS:.4f}',
            'dice': f'{dice:.4f}'
        })
    
    return metrics.get_average()


def validate_epoch(model, val_loader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    metrics = MetricsTracker()
    all_metrics = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{Config.EPOCHS} [VAL]  ')
    
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Use final output for metrics
            if isinstance(outputs, tuple):
                final_output = outputs[-1]
            else:
                final_output = outputs
            
            dice = dice_coefficient(final_output, masks)
            batch_metrics = calculate_metrics(final_output, masks)
            
            metrics.update(loss.item(), dice)
            all_metrics.append(batch_metrics)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}'
            })
    
    # Average all metrics
    avg_metrics = {
        'dice': np.mean([m['dice'] for m in all_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
        'sensitivity': np.mean([m['sensitivity'] for m in all_metrics]),
        'specificity': np.mean([m['specificity'] for m in all_metrics]),
        'auc': np.mean([m['auc'] for m in all_metrics])
    }
    
    result = metrics.get_average()
    result.update(avg_metrics)
    
    return result


def save_checkpoint(model, optimizer, epoch, metrics, is_best, checkpoint_dir):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'latest.pth')
    torch.save(checkpoint, latest_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best.pth')
        torch.save(checkpoint, best_path)
        print(f"[SAVE] Best model saved with Dice: {metrics['dice']:.4f}")


def plot_training_history(history, checkpoint_dir):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Dice
    axes[1].plot(history['train_dice'], label='Train Dice', marker='o')
    axes[1].plot(history['val_dice'], label='Val Dice', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Coefficient')
    axes[1].set_title('Training and Validation Dice')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'training_history.png'), dpi=150)
    print(f"[SAVE] Training curves saved to {checkpoint_dir}/training_history.png")


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    """Main training function"""
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Print configuration
    print_config()
    
    # Create checkpoint directory
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # Create data loaders
    print("[LOAD] Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        Config.TRAIN_IMG_DIR,
        Config.TRAIN_MASK_DIR,
        Config.TEST_IMG_DIR,  # Use test set as validation
        Config.TEST_MASK_DIR,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        patch_size=Config.PATCH_SIZE,
        stride=Config.STRIDE
    )
    
    print(f"[OK] Train batches: {len(train_loader)}")
    print(f"[OK] Val batches: {len(val_loader)}")
    
    # Create model
    print("\n[LOAD] Creating U-Net++ model...")
    model = UNetPlusPlus(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        deep_supervision=Config.DEEP_SUPERVISION
    ).to(Config.DEVICE)
    
    params = count_parameters(model)
    print(f"[OK] Model created with {params:,} parameters ({params/1e6:.1f}M)")
    
    # Create loss function
    if Config.DEEP_SUPERVISION:
        criterion = DeepSupervisionLoss(weights=[0.25, 0.25, 0.25, 1.0])
    else:
        criterion = BCEDiceLoss()
    
    # Create optimizer
    optimizer = Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=Config.LR_FACTOR,
        patience=Config.LR_PATIENCE,
        min_lr=Config.MIN_LR,
        verbose=True
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if Config.MIXED_PRECISION else None
    
    # Training history
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': [],
        'val_accuracy': [],
        'val_sensitivity': [],
        'val_specificity': [],
        'val_auc': []
    }
    
    # Early stopping
    best_val_dice = 0
    patience_counter = 0
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, Config.DEVICE, epoch
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, Config.DEVICE, epoch
        )
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_dice'].append(train_metrics['dice'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_sensitivity'].append(val_metrics['sensitivity'])
        history['val_specificity'].append(val_metrics['specificity'])
        history['val_auc'].append(val_metrics['auc'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{Config.EPOCHS} Summary:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}")
        print(f"  Val   - Acc: {val_metrics['accuracy']:.4f}, Sens: {val_metrics['sensitivity']:.4f}, "
              f"Spec: {val_metrics['specificity']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Check for best model
        is_best = val_metrics['dice'] > best_val_dice
        if is_best:
            best_val_dice = val_metrics['dice']
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, val_metrics, is_best, Config.CHECKPOINT_DIR)
        
        # Early stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\n[WARNING] Early stopping triggered after {epoch} epochs")
            print(f"[WARNING] Best Val Dice: {best_val_dice:.4f}")
            break
        
        print("-" * 80)
    
    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nTotal Time: {elapsed_time/60:.1f} minutes")
    print(f"Best Validation Dice: {best_val_dice:.4f}")
    print(f"Final Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Plot training history
    plot_training_history(history, Config.CHECKPOINT_DIR)
    
    # Save final metrics
    final_metrics = {
        'best_val_dice': best_val_dice,
        'total_epochs': epoch,
        'training_time': elapsed_time,
        'history': history
    }
    
    import json
    with open(os.path.join(Config.CHECKPOINT_DIR, 'metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    print(f"\n[OK] Training complete! Checkpoints saved to: {Config.CHECKPOINT_DIR}")
    print(f"[OK] Best model: {os.path.join(Config.CHECKPOINT_DIR, 'best.pth')}")


if __name__ == "__main__":
    main()
