# U-Net++ Retinal Vessel Segmentation

Complete PyTorch implementation of U-Net++ (Nested U-Net) for retinal vessel segmentation on the DRIVE dataset.

## 🎯 Key Features

### Architecture
- **U-Net++ (Nested U-Net)** with dense skip connections
- **Deep Supervision**: 4 output heads with weighted losses (0.25, 0.25, 0.25, 1.0)
- **9.0M parameters** (optimized from original 35-40M)
- **Nested skip paths** to fill semantic gap between encoder/decoder

### Data Processing
- **Patch-based training**: 128×128 patches with 50% overlap
- **Patch filtering**: Only trains on patches containing vessels
- **Data augmentation**: Random flips, brightness, contrast
- **FOV-aware evaluation**: Only evaluates retinal region

### Training Features
- **Combined BCE + Dice Loss** for better segmentation
- **Mixed Precision Training** for faster computation
- **Learning Rate Scheduling**: ReduceLROnPlateau
- **Early Stopping**: Patience-based stopping
- **Model Checkpointing**: Saves best model automatically

### Metrics
- Dice Coefficient
- Accuracy
- Sensitivity (True Positive Rate)
- Specificity (True Negative Rate)
- AUC (Area Under ROC Curve)

## 📁 Project Structure

```
retina-unet-segmentation/
├── unet_plus_plus.py          # U-Net++ model architecture
├── dataloader_unetpp.py       # Advanced dataloader with patches
├── losses_unetpp.py           # Loss functions and metrics
├── train_unetpp.py            # Main training script
├── requirements_unetpp.txt    # Dependencies
├── README_UNETPP.md           # This file
├── Retina/                    # Dataset
│   ├── train/
│   │   ├── image/
│   │   └── mask/
│   └── test/
│       ├── image/
│       └── mask/
└── checkpoints_unetpp/        # Saved models (created during training)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_unetpp.txt
```

### 2. Verify Setup
```bash
# Test model architecture
python unet_plus_plus.py

# Test dataloader
python dataloader_unetpp.py

# Test loss functions
python losses_unetpp.py
```

### 3. Train Model
```bash
python train_unetpp.py
```

## ⚙️ Configuration

Edit `train_unetpp.py` to modify training parameters:

```python
class Config:
    # Training
    EPOCHS = 60
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    
    # Data
    PATCH_SIZE = 128
    STRIDE = 64  # 50% overlap
    
    # Model
    DEEP_SUPERVISION = True
    
    # Training features
    MIXED_PRECISION = True
    EARLY_STOPPING_PATIENCE = 10
    LR_PATIENCE = 5
```

## 📊 Expected Performance

Based on DRIVE dataset benchmarks:

| Metric | Expected Range |
|--------|---------------|
| Dice Coefficient | 78-85% |
| Accuracy | 94-96% |
| Sensitivity | 75-82% |
| Specificity | 96-98% |
| AUC | 0.96-0.98 |

**Training Time**: ~60 minutes on NVIDIA RTX 3050 (60 epochs)

## 🏗️ Architecture Details

### U-Net++ Structure

```
Encoder Levels:
  Level 0: 32 filters
  Level 1: 64 filters
  Level 2: 128 filters
  Level 3: 256 filters
  Level 4: 512 filters (Bottleneck)

Nested Skip Connections:
  X_0_0 (input) → X_0_1 → X_0_2 → X_0_3 → X_0_4 (output)
                  X_1_0 → X_1_1 → X_1_2 → X_1_3
                          X_2_0 → X_2_1 → X_2_2
                                  X_3_0 → X_3_1
                                          X_4_0 (bottleneck)

Deep Supervision Outputs:
  - Output 1 from X_0_1 (weight: 0.25)
  - Output 2 from X_0_2 (weight: 0.25)
  - Output 3 from X_0_3 (weight: 0.25)
  - Output 4 from X_0_4 (weight: 1.0) ← Main output
```

### Convolutional Block

```
Conv2D(3×3) → BatchNorm → ReLU → Conv2D(3×3) → BatchNorm → ReLU
```

## 📈 Training Process

1. **Epoch Training**:
   - Extract 128×128 patches with 50% overlap
   - Filter patches to only include those with vessels
   - Apply random augmentation
   - Train with mixed precision
   - Gradient clipping

2. **Validation**:
   - Evaluate on test set patches
   - Calculate comprehensive metrics
   - Update learning rate based on validation loss

3. **Callbacks**:
   - Save best model based on Dice coefficient
   - Reduce learning rate on plateau
   - Early stopping if no improvement

4. **Outputs**:
   - `checkpoints_unetpp/best.pth` - Best model
   - `checkpoints_unetpp/latest.pth` - Latest checkpoint
   - `checkpoints_unetpp/training_history.png` - Training curves
   - `checkpoints_unetpp/metrics.json` - Final metrics

## 🔬 Key Innovations vs Standard U-Net

1. **Nested Skip Connections**:
   - Intermediate convolutions along skip paths
   - Gradual feature transformation
   - Better gradient flow

2. **Dense Connections**:
   - Each decoder node receives multiple inputs
   - Reduces semantic gap between encoder/decoder
   - Improves feature reuse

3. **Deep Supervision**:
   - 4 output heads at different scales
   - Forces intermediate layers to learn useful features
   - Can prune model at inference for speed/accuracy tradeoff

4. **Patch Filtering**:
   - Only trains on patches containing vessels
   - Reduces training time
   - Increases vessel examples per batch

5. **FOV-Aware Evaluation**:
   - Ignores black background pixels
   - More accurate metrics for clinical use
   - Focuses on actual retinal region

## 💡 Usage Tips

1. **GPU Memory**:
   - If out of memory, reduce `BATCH_SIZE` in config
   - Or reduce `PATCH_SIZE` to 96 or 64

2. **Training Time**:
   - Use `MIXED_PRECISION = True` for faster training
   - Adjust `NUM_WORKERS` based on CPU cores

3. **Overfitting**:
   - Model has early stopping and regularization
   - Increase augmentation if needed
   - Reduce model size by decreasing filter counts

4. **Fine-tuning**:
   - Load checkpoint and resume training
   - Adjust learning rate for fine-tuning
   - Can disable deep supervision for faster inference

## 📚 References

- Zhou et al. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation" (2018)
- DRIVE Dataset: Digital Retinal Images for Vessel Extraction
- Original Kaggle Notebook: u-net-segmentation (TensorFlow implementation)

## 📄 License

This implementation is for research and educational purposes.

## 🙏 Acknowledgments

- Based on the TensorFlow U-Net++ Kaggle notebook
- DRIVE dataset providers
- PyTorch team for the excellent framework
