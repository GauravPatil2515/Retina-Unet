"""
OPTIMIZED CONFIGURATION FOR BETTER PERFORMANCE
This config uses best practices for retina vessel segmentation
"""

import torch

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
# Paths
TRAIN_IMG_DIR = "Retina/train/image"
TRAIN_MASK_DIR = "Retina/train/mask"
TEST_IMG_DIR = "Retina/test/image"
TEST_MASK_DIR = "Retina/test/mask"

# Image settings
IMAGE_SIZE = 512  # Keep at 512x512
IN_CHANNELS = 3   # RGB images
OUT_CHANNELS = 1  # Binary mask (vessel vs background)

# =============================================================================
# TRAINING HYPERPARAMETERS - OPTIMIZED FOR BETTER PERFORMANCE
# =============================================================================

# Training duration
EPOCHS = 200  # ⬆️ Increased from 100 (IMPROVEMENT #1)
              # More epochs = better convergence
              # Expected: +5-10% Dice improvement

# Batch size (adjust based on GPU memory)
BATCH_SIZE = 4  # ⬆️ Increased from 2 (IMPROVEMENT #2)
                # Larger batch = more stable gradients
                # RTX 3050 6GB can handle batch_size=4 with 512x512 images

# Learning rate
LEARNING_RATE = 0.0001  # ⬇️ Reduced from 0.001 (IMPROVEMENT #3)
                        # Lower LR = more stable, fine-grained learning
                        # Prevents overshooting optimal weights

# Optimizer
OPTIMIZER = "adamw"  # AdamW with weight decay for regularization
WEIGHT_DECAY = 1e-4  # L2 regularization to prevent overfitting

# Learning rate scheduler
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = "cosine"  # 🔄 Changed from ReduceLROnPlateau (IMPROVEMENT #4)
                              # Options: "cosine", "plateau", "step"
                              # Cosine annealing smoothly reduces LR over training

# Cosine scheduler settings (if LR_SCHEDULER_TYPE = "cosine")
LR_MIN = 1e-6  # Minimum learning rate
T_MAX = EPOCHS  # Period of cosine schedule

# Plateau scheduler settings (if LR_SCHEDULER_TYPE = "plateau")
LR_PATIENCE = 10
LR_FACTOR = 0.5

# =============================================================================
# LOSS FUNCTION - OPTIMIZED
# =============================================================================
LOSS_TYPE = "combined"  # 🎯 Changed from "crossentropy" (IMPROVEMENT #5)
                        # Options: "crossentropy", "dice", "combined"
                        # Combined loss handles class imbalance better
                        # Dice loss focuses on overlap, CE on pixel classification

# Combined loss weights
DICE_WEIGHT = 0.7      # Weight for Dice loss
CE_WEIGHT = 0.3        # Weight for CrossEntropy loss

# Class weights for CrossEntropy (handles imbalance)
# Retina: ~10-20% pixels are vessels, 80-90% are background
USE_CLASS_WEIGHTS = True
VESSEL_WEIGHT = 3.0    # Give more importance to vessel pixels
BACKGROUND_WEIGHT = 1.0

# =============================================================================
# REGULARIZATION & STABILITY
# =============================================================================
GRADIENT_CLIPPING = True      # Prevent exploding gradients
MAX_GRAD_NORM = 1.0           # Clip gradients to this norm

DROPOUT_RATE = 0.1            # Dropout in U-Net decoder (prevents overfitting)

# =============================================================================
# DATA AUGMENTATION - CRITICAL FOR BETTER PERFORMANCE
# =============================================================================
USE_AUGMENTATION = True  # 🔥 NEW (IMPROVEMENT #6)

# Augmentation parameters
AUG_ROTATION = 45        # Random rotation ±45 degrees
AUG_HORIZONTAL_FLIP = 0.5  # 50% chance of horizontal flip
AUG_VERTICAL_FLIP = 0.5    # 50% chance of vertical flip
AUG_BRIGHTNESS = 0.2       # Brightness adjustment ±20%
AUG_CONTRAST = 0.2         # Contrast adjustment ±20%
AUG_ELASTIC_TRANSFORM = True  # Elastic deformation
AUG_GRID_DISTORTION = True    # Grid distortion

# =============================================================================
# VALIDATION & EARLY STOPPING
# =============================================================================
VALIDATION_SPLIT = 0.2  # 20% of training data for validation
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 30  # ⬆️ Increased from 20 (IMPROVEMENT #7)
                              # More patience with lower learning rate

# =============================================================================
# CHECKPOINTING
# =============================================================================
CHECKPOINT_DIR = "checkpoints"
MODEL_DIR = "models"
SAVE_CHECKPOINT_EVERY = 5  # Save every 5 epochs
KEEP_BEST_MODEL = True
KEEP_LAST_MODEL = True

# =============================================================================
# LOGGING & MONITORING
# =============================================================================
USE_TENSORBOARD = True
LOG_DIR = "runs"
LOG_INTERVAL = 10  # Log every 10 batches

# =============================================================================
# ADVANCED: MODEL ARCHITECTURE TWEAKS
# =============================================================================
# U-Net architecture parameters
UNET_FEATURES = [64, 128, 256, 512]  # Feature maps at each level
USE_BATCH_NORM = True                 # Batch normalization (stabilizes training)
USE_ATTENTION = False                 # Attention gates (experimental, slower)

# =============================================================================
# POST-PROCESSING
# =============================================================================
PREDICTION_THRESHOLD = 0.5  # Threshold for binary prediction
USE_MORPHOLOGICAL_CLEANUP = False  # Apply morphological operations (erosion/dilation)

# =============================================================================
# MIXED PRECISION TRAINING - FOR FASTER TRAINING
# =============================================================================
USE_AMP = True  # 🚀 Automatic Mixed Precision (IMPROVEMENT #8)
                # Trains faster with less memory
                # RTX 3050 supports this (Tensor Cores)

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
RANDOM_SEED = 42
DETERMINISTIC = False  # Set True for reproducibility (slower)

# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================
def print_config():
    """Print current configuration"""
    print("\n" + "="*70)
    print("⚙️  OPTIMIZED TRAINING CONFIGURATION")
    print("="*70)
    
    if torch.cuda.is_available():
        print(f"\n🖥️  Device: {DEVICE}")
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"\n⚠️  Device: {DEVICE} (GPU not available)")
    
    print(f"\n📊 TRAINING SETTINGS:")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Optimizer: {OPTIMIZER.upper()}")
    print(f"   Weight Decay: {WEIGHT_DECAY}")
    
    print(f"\n🎯 LOSS FUNCTION:")
    print(f"   Type: {LOSS_TYPE.upper()}")
    if LOSS_TYPE == "combined":
        print(f"   Dice Weight: {DICE_WEIGHT}")
        print(f"   CE Weight: {CE_WEIGHT}")
    if USE_CLASS_WEIGHTS:
        print(f"   Vessel Weight: {VESSEL_WEIGHT}x")
    
    print(f"\n📈 LEARNING RATE SCHEDULER:")
    print(f"   Type: {LR_SCHEDULER_TYPE.upper()}")
    if LR_SCHEDULER_TYPE == "cosine":
        print(f"   Min LR: {LR_MIN}")
    
    print(f"\n🎨 DATA AUGMENTATION:")
    print(f"   Enabled: {USE_AUGMENTATION}")
    if USE_AUGMENTATION:
        print(f"   Rotation: ±{AUG_ROTATION}°")
        print(f"   Flips: H={AUG_HORIZONTAL_FLIP} V={AUG_VERTICAL_FLIP}")
        print(f"   Brightness: ±{AUG_BRIGHTNESS*100}%")
        print(f"   Contrast: ±{AUG_CONTRAST*100}%")
    
    print(f"\n⚡ OPTIMIZATIONS:")
    print(f"   Mixed Precision (AMP): {USE_AMP}")
    print(f"   Gradient Clipping: {GRADIENT_CLIPPING}")
    print(f"   Dropout Rate: {DROPOUT_RATE}")
    
    print(f"\n✋ EARLY STOPPING:")
    print(f"   Enabled: {EARLY_STOPPING}")
    print(f"   Patience: {EARLY_STOPPING_PATIENCE} epochs")
    
    print("\n" + "="*70)
    print("💡 EXPECTED IMPROVEMENTS:")
    print("="*70)
    print("  📈 Current Dice: 68.22%")
    print("  🎯 Target Dice: 75-80%")
    print("  🚀 Improvements from:")
    print("     • More epochs (200): +5-8%")
    print("     • Combined loss: +3-5%")
    print("     • Data augmentation: +2-4%")
    print("     • Lower learning rate: +1-2%")
    print("     • Larger batch size: +1-2%")
    print("  ⭐ Expected Final: 75-82% Dice")
    print("="*70 + "\n")

if __name__ == "__main__":
    print_config()
