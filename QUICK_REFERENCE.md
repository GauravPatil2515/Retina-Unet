# 🚀 Quick Reference Guide

## File Organization

### 📁 Core Directories

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| **models/** | Model architectures & losses | `unet_plus_plus.py`, `losses_unetpp.py` |
| **scripts/** | Training & inference scripts | `train_unetpp.py`, `evaluate_unetpp.py`, `test_model.py` |
| **results/** | Training outputs & checkpoints | `checkpoints_unetpp/`, `evaluation_results_unetpp/` |
| **docs/** | Documentation | `README_UNETPP.md`, `FINAL_RESULTS.md` |
| **Retina/** | DRIVE dataset | `train/`, `test/` |
| **legacy/** | Old U-Net implementation | Legacy files for reference |

---

## 🎯 Common Tasks

### Test the Model (Quick)
```bash
python scripts/test_model.py
```
**Output:** `results/test_result.png` (4-panel comparison)

---

### Train from Scratch
```bash
python scripts/train_unetpp.py
```
**Output:** 
- `results/checkpoints_unetpp/best.pth` (Best model)
- `results/checkpoints_unetpp/latest.pth` (Latest checkpoint)
- `results/checkpoints_unetpp/metrics.json` (Training history)
- `results/checkpoints_unetpp/training_history.png` (Curves)

**Time:** ~30 minutes on RTX 3050  
**GPU Memory:** ~4GB (batch_size=8)

---

### Evaluate on Test Set
```bash
python scripts/evaluate_unetpp.py
```
**Output:**
- `results/evaluation_results_unetpp/test_metrics.json`
- `results/evaluation_results_unetpp/prediction_*.png` (5 samples)

---

### Inference on Custom Image
```bash
python scripts/inference.py --image path/to/image.png --output result.png
```

---

## 📊 Model Files

### Checkpoints Location
```
results/checkpoints_unetpp/
├── best.pth (103.75 MB) - Epoch 10, Val Dice: 83.67%
├── latest.pth (103.78 MB) - Epoch 20
├── metrics.json - Training history
└── training_history.png - Loss/Dice curves
```

### Load Trained Model
```python
import torch
from models.unet_plus_plus import UNetPlusPlus

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNetPlusPlus(in_channels=3, out_channels=1, deep_supervision=True).to(device)

checkpoint = torch.load('results/checkpoints_unetpp/best.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## 🔧 Configuration

### Training Settings
Edit `scripts/train_unetpp.py`:
```python
class Config:
    batch_size = 8           # Reduce to 4 if GPU memory issues
    num_epochs = 60          # Max epochs (early stopping at 10)
    learning_rate = 1e-4     # Adam optimizer
    accumulation_steps = 2   # Effective batch_size = 16
    patience_early_stopping = 10
    patience_reduce_lr = 5
```

### Data Paths
```python
train_image_dir = "Retina/train/image"
train_mask_dir = "Retina/train/mask"
test_image_dir = "Retina/test/image"
test_mask_dir = "Retina/test/mask"
```

---

## 📈 Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Dice Coefficient | 83.82% | 78-85% | ✅ |
| Accuracy | 96.08% | 94-96% | ✅ |
| Sensitivity | 82.91% | 75-82% | ✅ |
| Specificity | 97.97% | 96-98% | ✅ |
| AUC-ROC | 97.82% | 96-98% | ✅ |

---

## 🐛 Troubleshooting

### GPU Memory Error
```python
# In scripts/train_unetpp.py
batch_size = 4  # Reduce from 8
accumulation_steps = 4  # Increase to maintain effective batch_size=16
```

### Import Errors
```bash
pip install -r requirements_unetpp.txt
```

### Model Not Found
```bash
# Train a new model
python scripts/train_unetpp.py
```

---

## 📦 Dependencies

**Main Requirements:**
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
Pillow>=9.5.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

**Install:**
```bash
pip install -r requirements_unetpp.txt
```

---

## 🔍 Code Structure

### models/unet_plus_plus.py
- **UNetPlusPlus** - Main model class (9.0M params)
- **ConvBlock** - Basic conv-bn-relu block
- Nested skip connections with dense connections
- 4 output heads for deep supervision

### models/losses_unetpp.py
- **DiceLoss** - Dice coefficient loss
- **BCEDiceLoss** - Combined BCE + Dice
- **DeepSupervisionLoss** - Multi-output loss
- **calculate_metrics** - Dice, Acc, Sens, Spec, AUC

### scripts/dataloader_unetpp.py
- **PatchDataset** - Extract 128×128 patches
- **FullImageDataset** - Load full images for evaluation
- Augmentation: flips, brightness, contrast
- Patch filtering: vessel-containing only

### scripts/train_unetpp.py
- **Config** - Training configuration
- **EarlyStopping** - Stop when no improvement
- **ReduceLROnPlateau** - Reduce LR on plateau
- Mixed precision training (FP16)
- Gradient accumulation

### scripts/evaluate_unetpp.py
- **reconstruct_from_patches** - Merge overlapping patches
- **evaluate_model** - Full evaluation pipeline
- **visualize_predictions** - Generate comparison images
- Save metrics to JSON

### scripts/test_model.py
- Quick sanity check on one test image
- 4-panel visualization (original, ground truth, probability, binary)
- Fast inference test

---

## 📚 Documentation

| File | Description |
|------|-------------|
| **README.md** | Main project overview (this file) |
| **docs/README_UNETPP.md** | Detailed U-Net++ documentation |
| **docs/FINAL_RESULTS.md** | Complete results & analysis |
| **QUICK_REFERENCE.md** | This quick reference guide |

---

## 🎓 References

1. **U-Net++:** Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation" (2018)
2. **DRIVE Dataset:** Staal et al., "Ridge-based vessel segmentation in color images of the retina" (2004)
3. **Original U-Net:** Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)

---

## ⚡ Quick Commands Cheatsheet

```bash
# Test model (fastest)
python scripts/test_model.py

# Train new model
python scripts/train_unetpp.py

# Evaluate on test set
python scripts/evaluate_unetpp.py

# Inference on single image
python scripts/inference.py --image path/to/image.png --output result.png

# Install dependencies
pip install -r requirements_unetpp.txt

# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# View training history
python -c "import json; print(json.load(open('results/checkpoints_unetpp/metrics.json')))"
```

---

**Updated:** October 31, 2025  
**Version:** 1.0  
**Author:** Gaurav Patil
