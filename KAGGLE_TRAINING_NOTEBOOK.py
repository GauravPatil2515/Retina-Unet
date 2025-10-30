# Retina Blood Vessel Segmentation - Kaggle Notebook
# Complete training pipeline optimized for Kaggle GPU (T4/P100)
# Expected training time: ~30-40 minutes for 200 epochs

# ============================================================================
# CELL 1: Setup and Installation
# ============================================================================

!pip install -q segmentation-models-pytorch albumentations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# Check GPU
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")


# ============================================================================
# CELL 2: Upload Your Dataset
# ============================================================================

# METHOD 1: Upload from local computer
# Click "Add data" → "Upload" → Upload your Retina.zip file
# Then unzip it

!unzip -q /kaggle/input/your-dataset/Retina.zip -d /kaggle/working/

# OR METHOD 2: If dataset is already on Kaggle
# Just add it as a data source and it will be in /kaggle/input/

# Verify data structure
print("Dataset structure:")
!ls -R /kaggle/working/Retina | head -20


# ============================================================================
# CELL 3: Configuration - OPTIMIZED FOR BEST RESULTS
# ============================================================================

class Config:
    # Paths (adjust based on your data location)
    TRAIN_IMG_DIR = "/kaggle/working/Retina/train/image"
    TRAIN_MASK_DIR = "/kaggle/working/Retina/train/mask"
    TEST_IMG_DIR = "/kaggle/working/Retina/test/image"
    TEST_MASK_DIR = "/kaggle/working/Retina/test/mask"
    
    # Model
    IMAGE_SIZE = 512
    IN_CHANNELS = 3
    OUT_CHANNELS = 2  # Background + Vessel
    
    # Training - OPTIMIZED
    EPOCHS = 200              # ⭐ More epochs = better results
    BATCH_SIZE = 8            # ⭐ Kaggle GPUs have more memory
    LEARNING_RATE = 0.0001    # ⭐ Lower LR for stable training
    WEIGHT_DECAY = 1e-4
    
    # Loss weights
    DICE_WEIGHT = 0.7
    CE_WEIGHT = 0.3
    VESSEL_WEIGHT = 3.0       # ⭐ Handle class imbalance
    
    # Augmentation
    USE_AUGMENTATION = True   # ⭐ Critical for better performance
    
    # Validation
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 30
    
    # Optimization
    USE_AMP = True            # ⭐ Mixed precision for faster training
    GRADIENT_CLIPPING = True
    MAX_GRAD_NORM = 1.0
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Seed
    RANDOM_SEED = 42

cfg = Config()

# Set seed
torch.manual_seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(cfg.RANDOM_SEED)

print("✅ Configuration loaded")
print(f"Device: {cfg.DEVICE}")
print(f"Epochs: {cfg.EPOCHS} | Batch Size: {cfg.BATCH_SIZE} | LR: {cfg.LEARNING_RATE}")


# ============================================================================
# CELL 4: Data Augmentation - POWERFUL AUGMENTATIONS
# ============================================================================

def get_train_transforms():
    """Aggressive augmentation for training"""
    return A.Compose([
        A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        
        # Color augmentations (only for image)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.CLAHE(clip_limit=4.0, p=1),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1),
        ], p=0.5),
        
        # Elastic transforms
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1),
            A.GridDistortion(p=1),
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=1),
        ], p=0.3),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms():
    """No augmentation for validation"""
    return A.Compose([
        A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

print("✅ Augmentation pipelines created")


# ============================================================================
# CELL 5: Dataset Class
# ============================================================================

class RetinaDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 128).astype(np.uint8)  # Binary mask
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.long()

# Create datasets
full_dataset = RetinaDataset(cfg.TRAIN_IMG_DIR, cfg.TRAIN_MASK_DIR)

# Split into train/val
val_size = int(cfg.VALIDATION_SPLIT * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(cfg.RANDOM_SEED)
)

# Apply transforms
train_dataset.dataset.transform = get_train_transforms()
val_dataset.dataset.transform = get_val_transforms()

# Create dataloaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=cfg.BATCH_SIZE, 
    shuffle=True, 
    num_workers=2,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=cfg.BATCH_SIZE, 
    shuffle=False, 
    num_workers=2,
    pin_memory=True
)

print(f"✅ Training samples: {len(train_dataset)}")
print(f"✅ Validation samples: {len(val_dataset)}")


# ============================================================================
# CELL 6: U-Net Model Architecture
# ============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder (Down)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Decoder (Up)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)

model = UNet(in_channels=cfg.IN_CHANNELS, out_channels=cfg.OUT_CHANNELS).to(cfg.DEVICE)
print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")


# ============================================================================
# CELL 7: Loss Functions - COMBINED LOSS FOR BEST RESULTS
# ============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)[:, 1, :]
        target = target.float()
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.3, dice_weight=0.7):
        super(CombinedLoss, self).__init__()
        # Weighted CrossEntropy for class imbalance
        class_weights = torch.tensor([1.0, cfg.VESSEL_WEIGHT]).to(cfg.DEVICE)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice

criterion = CombinedLoss(cfg.CE_WEIGHT, cfg.DICE_WEIGHT)
print("✅ Combined loss function created (Dice + CrossEntropy)")


# ============================================================================
# CELL 8: Optimizer and Scheduler
# ============================================================================

optimizer = AdamW(
    model.parameters(), 
    lr=cfg.LEARNING_RATE, 
    weight_decay=cfg.WEIGHT_DECAY
)

scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)

# Mixed precision scaler
scaler = GradScaler() if cfg.USE_AMP else None

print("✅ Optimizer: AdamW")
print("✅ Scheduler: CosineAnnealingLR")
print("✅ Mixed Precision:", cfg.USE_AMP)


# ============================================================================
# CELL 9: Metrics Calculation
# ============================================================================

def calculate_metrics(pred, target):
    """Calculate Dice, IoU, Accuracy, Precision, Recall"""
    pred = torch.softmax(pred, dim=1)[:, 1, :] > 0.5
    target = target.bool()
    
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    tp = (pred_flat & target_flat).sum().float()
    fp = (pred_flat & ~target_flat).sum().float()
    tn = (~pred_flat & ~target_flat).sum().float()
    fn = (~pred_flat & target_flat).sum().float()
    
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }

print("✅ Metrics functions ready")


# ============================================================================
# CELL 10: Training and Validation Functions
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    epoch_loss = 0
    epoch_metrics = {'dice': 0, 'iou': 0, 'accuracy': 0}
    
    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        if cfg.USE_AMP:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            
            if cfg.GRADIENT_CLIPPING:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            
            if cfg.GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
            
            optimizer.step()
        
        with torch.no_grad():
            metrics = calculate_metrics(outputs, masks)
        
        epoch_loss += loss.item()
        for key in epoch_metrics:
            epoch_metrics[key] += metrics[key]
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{metrics["dice"]:.4f}'})
    
    num_batches = len(loader)
    return epoch_loss / num_batches, {k: v / num_batches for k, v in epoch_metrics.items()}

def validate_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_metrics = {'dice': 0, 'iou': 0, 'accuracy': 0}
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            if cfg.USE_AMP:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            metrics = calculate_metrics(outputs, masks)
            
            epoch_loss += loss.item()
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{metrics["dice"]:.4f}'})
    
    num_batches = len(loader)
    return epoch_loss / num_batches, {k: v / num_batches for k, v in epoch_metrics.items()}

print("✅ Training functions ready")


# ============================================================================
# CELL 11: MAIN TRAINING LOOP - RUN THIS!
# ============================================================================

import time

history = {
    'train_loss': [], 'val_loss': [],
    'train_dice': [], 'val_dice': [],
    'train_accuracy': [], 'val_accuracy': []
}

best_val_dice = 0
epochs_no_improve = 0

print("\n" + "="*70)
print("🚀 STARTING TRAINING - OPTIMIZED FOR MAXIMUM PERFORMANCE")
print("="*70)
print(f"Expected time: ~30-40 minutes on Kaggle GPU")
print(f"Epochs: {cfg.EPOCHS} | Batch Size: {cfg.BATCH_SIZE}")
print("="*70 + "\n")

start_time = time.time()

for epoch in range(1, cfg.EPOCHS + 1):
    epoch_start = time.time()
    
    # Train
    train_loss, train_metrics = train_epoch(
        model, train_loader, criterion, optimizer, scaler, cfg.DEVICE
    )
    
    # Validate
    val_loss, val_metrics = validate_epoch(
        model, val_loader, criterion, cfg.DEVICE
    )
    
    # Update scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Record history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_dice'].append(train_metrics['dice'])
    history['val_dice'].append(val_metrics['dice'])
    history['train_accuracy'].append(train_metrics['accuracy'])
    history['val_accuracy'].append(val_metrics['accuracy'])
    
    # Print summary
    epoch_time = time.time() - epoch_start
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"Epoch {epoch}/{cfg.EPOCHS} | Time: {epoch_time:.1f}s | Total: {elapsed_time/60:.1f}min")
    print(f"LR: {current_lr:.6f}")
    print(f"Train - Loss: {train_loss:.4f} | Dice: {train_metrics['dice']:.4f} | Acc: {train_metrics['accuracy']:.4f}")
    print(f"Val   - Loss: {val_loss:.4f} | Dice: {val_metrics['dice']:.4f} | Acc: {val_metrics['accuracy']:.4f}")
    
    # Save best model
    if val_metrics['dice'] > best_val_dice:
        best_val_dice = val_metrics['dice']
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"⭐ New best! Dice: {best_val_dice:.4f} → Model saved")
    else:
        epochs_no_improve += 1
    
    # Early stopping
    if epochs_no_improve >= cfg.EARLY_STOPPING_PATIENCE:
        print(f"\n⚠️  Early stopping! No improvement for {cfg.EARLY_STOPPING_PATIENCE} epochs")
        break
    
    print(f"{'='*70}")

total_time = time.time() - start_time
print(f"\n{'='*70}")
print("✅ TRAINING COMPLETE!")
print(f"{'='*70}")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Best validation Dice: {best_val_dice:.4f}")
print(f"{'='*70}\n")


# ============================================================================
# CELL 12: Plot Training History
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Dice plot
axes[1].plot(history['train_dice'], label='Train Dice', linewidth=2)
axes[1].plot(history['val_dice'], label='Val Dice', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Dice Score', fontsize=12)
axes[1].set_title('Training and Validation Dice Score', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Best validation Dice: {best_val_dice:.4f}")
print(f"✅ Final train Dice: {history['train_dice'][-1]:.4f}")


# ============================================================================
# CELL 13: Evaluate on Test Set
# ============================================================================

# Load test dataset
test_dataset = RetinaDataset(cfg.TEST_IMG_DIR, cfg.TEST_MASK_DIR, get_val_transforms())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

test_metrics = {'dice': 0, 'iou': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}
predictions = []

print("\n🔬 Evaluating on test set...")

with torch.no_grad():
    for images, masks in tqdm(test_loader):
        images = images.to(cfg.DEVICE)
        masks = masks.to(cfg.DEVICE)
        
        outputs = model(images)
        metrics = calculate_metrics(outputs, masks)
        
        for key in test_metrics:
            test_metrics[key] += metrics[key]
        
        # Save prediction
        pred = torch.softmax(outputs, dim=1)[:, 1, :] > 0.5
        predictions.append(pred.cpu().numpy()[0])

# Average metrics
num_test = len(test_loader)
test_metrics = {k: v / num_test for k, v in test_metrics.items()}

print("\n" + "="*70)
print("📊 TEST SET RESULTS")
print("="*70)
print(f"Dice Score:    {test_metrics['dice']:.4f} ⭐")
print(f"IoU:           {test_metrics['iou']:.4f}")
print(f"Accuracy:      {test_metrics['accuracy']:.4f}")
print(f"Precision:     {test_metrics['precision']:.4f}")
print(f"Recall:        {test_metrics['recall']:.4f}")
print("="*70)


# ============================================================================
# CELL 14: Visualize Predictions
# ============================================================================

def visualize_predictions(num_samples=5):
    test_dataset_vis = RetinaDataset(cfg.TEST_IMG_DIR, cfg.TEST_MASK_DIR, get_val_transforms())
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    
    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            image, mask = test_dataset_vis[i]
            image_input = image.unsqueeze(0).to(cfg.DEVICE)
            
            output = model(image_input)
            pred = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
            pred_binary = (pred > 0.5).astype(np.uint8)
            
            # Denormalize image for display
            img_display = image.permute(1, 2, 0).numpy()
            img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_display = np.clip(img_display, 0, 1)
            
            # Plot
            axes[i, 0].imshow(img_display)
            axes[i, 0].set_title('Original Image', fontsize=12)
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask.numpy(), cmap='gray')
            axes[i, 1].set_title('Ground Truth', fontsize=12)
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_binary, cmap='gray')
            axes[i, 2].set_title('Prediction', fontsize=12)
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_predictions(num_samples=6)


# ============================================================================
# CELL 15: Download Model
# ============================================================================

# Download the trained model
from IPython.display import FileLink

print("📥 Download your trained model:")
FileLink('best_model.pth')

print("\n✅ Training complete! Your model achieved:")
print(f"   Validation Dice: {best_val_dice:.4f}")
print(f"   Test Dice: {test_metrics['dice']:.4f}")
print("\n💡 This is your trained model that you can use for inference!")
