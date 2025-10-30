# Retina Blood Vessel Segmentation - Kaggle Ready
# Optimized for Kaggle GPU with DRIVE dataset from Kaggle
# Just clone repo and run!

# ===========================================================================
# CELL 1: Clone Repository and Install Dependencies
# ===========================================================================

# Clone your GitHub repository
!git clone https://github.com/Harshtherocking/retina-unet-segmentation.git
%cd retina-unet-segmentation

# Install required packages
!pip install -q albumentations segmentation-models-pytorch

print("✅ Repository cloned and dependencies installed!")


# ===========================================================================
# CELL 2: Download DRIVE Dataset from Kaggle
# ===========================================================================

import kagglehub
import os
import shutil

# Download DRIVE dataset
print("📥 Downloading DRIVE dataset from Kaggle...")
path = kagglehub.dataset_download("andrewmvd/drive-digital-retinal-images-for-vessel-extraction")
print(f"✅ Dataset downloaded to: {path}")

# The dataset structure will be:
# path/training/images/  - training images
# path/training/1st_manual/  - training masks
# path/test/images/  - test images
# path/test/1st_manual/  - test masks


# ===========================================================================
# CELL 3: Prepare Dataset Structure
# ===========================================================================

import os
import shutil
from pathlib import Path

# Create output directories
os.makedirs("data/train/image", exist_ok=True)
os.makedirs("data/train/mask", exist_ok=True)
os.makedirs("data/test/image", exist_ok=True)
os.makedirs("data/test/mask", exist_ok=True)

# Copy training data
train_img_src = os.path.join(path, "training/images")
train_mask_src = os.path.join(path, "training/1st_manual")

if os.path.exists(train_img_src):
    for file in os.listdir(train_img_src):
        if file.endswith(('.tif', '.png', '.jpg')):
            src = os.path.join(train_img_src, file)
            # Convert to PNG and rename
            dst = os.path.join("data/train/image", file.replace('.tif', '.png'))
            shutil.copy2(src, dst)
    print(f"✅ Copied {len(os.listdir('data/train/image'))} training images")

if os.path.exists(train_mask_src):
    for file in os.listdir(train_mask_src):
        if file.endswith(('.gif', '.png', '.tif')):
            src = os.path.join(train_mask_src, file)
            dst = os.path.join("data/train/mask", file.replace('.gif', '.png').replace('_manual1', ''))
            shutil.copy2(src, dst)
    print(f"✅ Copied {len(os.listdir('data/train/mask'))} training masks")

# Copy test data
test_img_src = os.path.join(path, "test/images")
test_mask_src = os.path.join(path, "test/1st_manual")

if os.path.exists(test_img_src):
    for file in os.listdir(test_img_src):
        if file.endswith(('.tif', '.png', '.jpg')):
            src = os.path.join(test_img_src, file)
            dst = os.path.join("data/test/image", file.replace('.tif', '.png'))
            shutil.copy2(src, dst)
    print(f"✅ Copied {len(os.listdir('data/test/image'))} test images")

if os.path.exists(test_mask_src):
    for file in os.listdir(test_mask_src):
        if file.endswith(('.gif', '.png', '.tif')):
            src = os.path.join(test_mask_src, file)
            dst = os.path.join("data/test/mask", file.replace('.gif', '.png').replace('_manual1', ''))
            shutil.copy2(src, dst)
    print(f"✅ Copied {len(os.listdir('data/test/mask'))} test masks")

print("\n✅ Dataset prepared successfully!")


# ===========================================================================
# CELL 4: Update Config for Kaggle Paths
# ===========================================================================

# Update paths in config
import sys
sys.path.insert(0, '/kaggle/working/retina-unet-segmentation')

# Create Kaggle-specific config
with open('config_kaggle.py', 'w') as f:
    f.write('''"""
Kaggle-specific configuration
"""

import torch

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths - Updated for Kaggle
TRAIN_IMG_DIR = "/kaggle/working/retina-unet-segmentation/data/train/image"
TRAIN_MASK_DIR = "/kaggle/working/retina-unet-segmentation/data/train/mask"
TEST_IMG_DIR = "/kaggle/working/retina-unet-segmentation/data/test/image"
TEST_MASK_DIR = "/kaggle/working/retina-unet-segmentation/data/test/mask"

# Image settings
IMAGE_SIZE = 512
IN_CHANNELS = 3
OUT_CHANNELS = 2

# Training - OPTIMIZED FOR KAGGLE
EPOCHS = 200
BATCH_SIZE = 8  # Kaggle GPU can handle larger batches
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4

# Loss
LOSS_TYPE = "combined"
DICE_WEIGHT = 0.7
CE_WEIGHT = 0.3
USE_CLASS_WEIGHTS = True
VESSEL_WEIGHT = 3.0
BACKGROUND_WEIGHT = 1.0

# Scheduler
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = "cosine"
LR_MIN = 1e-6
T_MAX = EPOCHS

# Augmentation
USE_AUGMENTATION = True
AUG_ROTATION = 45
AUG_HORIZONTAL_FLIP = 0.5
AUG_VERTICAL_FLIP = 0.5
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.2

# Validation
VALIDATION_SPLIT = 0.2
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 30

# Optimization
USE_AMP = True
GRADIENT_CLIPPING = True
MAX_GRAD_NORM = 1.0

# Checkpointing
CHECKPOINT_DIR = "checkpoints"
MODEL_DIR = "models"
SAVE_CHECKPOINT_EVERY = 10
KEEP_BEST_MODEL = True
KEEP_LAST_MODEL = True

# Logging
USE_TENSORBOARD = True
LOG_DIR = "runs"

# Seed
RANDOM_SEED = 42
''')

print("✅ Kaggle config created!")


# ===========================================================================
# CELL 5: Check GPU and Setup
# ===========================================================================

import torch
from config_kaggle import *

print("="*70)
print("🚀 KAGGLE TRAINING SETUP")
print("="*70)
print(f"\n🖥️  Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️  No GPU detected! Training will be very slow.")

print(f"\n📊 Configuration:")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   Loss Type: {LOSS_TYPE.upper()}")
print(f"   Augmentation: {USE_AUGMENTATION}")
print(f"   Mixed Precision: {USE_AMP}")

print("\n" + "="*70)


# ===========================================================================
# CELL 6: Train Model - MAIN TRAINING
# ===========================================================================

# Import training script
from train_optimized import train

# Start training
print("\n🚀 Starting training...")
print("⏱️  Expected time: 30-40 minutes on Kaggle T4 GPU")
print("🎯 Expected result: 75-82% Dice score\n")

# Run training
train()

print("\n✅ Training complete!")


# ===========================================================================
# CELL 7: Evaluate on Test Set
# ===========================================================================

import torch
import os
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import numpy as np

from unet import Unet
from dataloader import ImageDataset
from config_kaggle import *

# Load test dataset
print("📊 Evaluating on test set...")
test_dataset = ImageDataset(
    folder_path=TEST_IMG_DIR.replace('/image', ''),
    is_train=False
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load best model
model = Unet(IN_CHANNELS, OUT_CHANNELS).to(DEVICE)
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()

# Calculate metrics
def calculate_metrics(pred, target):
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

test_metrics = {'dice': 0, 'iou': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}

with torch.no_grad():
    for images, masks in tqdm(test_loader, desc="Testing"):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        
        outputs = model(images)
        metrics = calculate_metrics(outputs, masks)
        
        for key in test_metrics:
            test_metrics[key] += metrics[key]

# Average metrics
num_test = len(test_loader)
test_metrics = {k: v / num_test for k, v in test_metrics.items()}

print("\n" + "="*70)
print("📊 TEST SET RESULTS")
print("="*70)
print(f"🎯 Dice Score:    {test_metrics['dice']:.4f}")
print(f"🎯 IoU:           {test_metrics['iou']:.4f}")
print(f"🎯 Accuracy:      {test_metrics['accuracy']:.4f}")
print(f"🎯 Precision:     {test_metrics['precision']:.4f}")
print(f"🎯 Recall:        {test_metrics['recall']:.4f}")
print("="*70)


# ===========================================================================
# CELL 8: Visualize Results
# ===========================================================================

import matplotlib.pyplot as plt
import cv2

def visualize_predictions(num_samples=6):
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            image, mask = test_dataset[i]
            image_input = image.unsqueeze(0).to(DEVICE)
            
            output = model(image_input)
            pred = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
            pred_binary = (pred > 0.5).astype(np.uint8)
            
            # Denormalize image
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

print("🎨 Generating visualizations...")
visualize_predictions(num_samples=6)
print("✅ Visualizations complete!")


# ===========================================================================
# CELL 9: Download Trained Model
# ===========================================================================

from IPython.display import FileLink

print("\n" + "="*70)
print("📥 DOWNLOAD YOUR TRAINED MODEL")
print("="*70)

# Create download link
print("\n✅ Click below to download your trained model:")
display(FileLink('models/best_model.pth'))

print(f"\n🎉 Congratulations! Your model achieved:")
print(f"   • Dice Score: {test_metrics['dice']:.4f}")
print(f"   • Accuracy: {test_metrics['accuracy']:.4f}")
print("\n💡 Use this model for inference on new retina images!")
print("="*70)
