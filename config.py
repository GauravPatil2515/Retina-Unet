"""
Configuration file for Retina U-Net Segmentation Project
All hyperparameters and paths are defined here for easy modification
"""

import os
import torch

# ============ PATHS ============
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "Retina")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Create directories if they don't exist
for directory in [MODEL_DIR, CHECKPOINT_DIR, RESULT_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============ MODEL PARAMETERS ============
INPUT_CHANNELS = 3      # RGB images
OUTPUT_CHANNELS = 2     # Background + Vessel (binary segmentation)
IMAGE_SIZE = 512        # All images will be resized to this

# ============ TRAINING HYPERPARAMETERS ============
BATCH_SIZE = 2          # Reduce to 1 if you run out of GPU memory
LEARNING_RATE = 0.001   # Initial learning rate
EPOCHS = 100            # Total number of training epochs
NUM_WORKERS = 4         # Number of parallel data loading workers (reduce to 0 if issues)

# ============ OPTIMIZER SETTINGS ============
WEIGHT_DECAY = 1e-5     # L2 regularization
AMSGRAD = True          # Variant of Adam optimizer

# ============ DEVICE CONFIGURATION ============
# Automatically detect if CUDA (GPU) is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Using device: {DEVICE}")

if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  No GPU found. Training will be slower on CPU.")

# ============ TRAINING SCHEDULE ============
SAVE_CHECKPOINT_EVERY = 10      # Save checkpoint every N epochs
VALIDATE_EVERY = 5              # Run validation every N epochs
LOG_IMAGES_EVERY = 10           # Log sample predictions every N epochs

# ============ DATA AUGMENTATION ============
USE_AUGMENTATION = True         # Enable/disable data augmentation
AUGMENTATION_PROB = 0.5         # Probability of applying each augmentation

# ============ EARLY STOPPING ============
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 20    # Stop if no improvement for N epochs

# ============ LEARNING RATE SCHEDULER ============
USE_LR_SCHEDULER = True
LR_SCHEDULER_PATIENCE = 10      # Reduce LR if no improvement for N epochs
LR_SCHEDULER_FACTOR = 0.5       # Multiply LR by this factor when reducing

# ============ LOSS FUNCTION ============
LOSS_TYPE = "cross_entropy"     # Options: "cross_entropy", "dice", "combined"
DICE_WEIGHT = 0.5               # Weight for dice loss in combined mode

# ============ VALIDATION SPLIT ============
VALIDATION_SPLIT = 0.2          # 20% of training data for validation

# ============ RANDOM SEED ============
RANDOM_SEED = 42                # For reproducibility
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# ============ DISPLAY SETTINGS ============
def print_config():
    """Print current configuration"""
    print("\n" + "="*50)
    print("CONFIGURATION SETTINGS")
    print("="*50)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Validation Split: {VALIDATION_SPLIT*100}%")
    print(f"Data Augmentation: {USE_AUGMENTATION}")
    print(f"Loss Function: {LOSS_TYPE}")
    print("="*50 + "\n")

if __name__ == "__main__":
    print_config()
