# 🩺 Retina Blood Vessel Segmentation with U-Net++# 🩺 Retina Blood Vessel Segmentation



[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)Deep learning model for automated retinal blood vessel segmentation using U-Net architecture with PyTorch.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)[![Made with PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**State-of-the-art retinal blood vessel segmentation using U-Net++ (Nested U-Net) architecture.**

## � Performance

🎯 **Performance:** 83.82% Dice Score | 96.08% Accuracy | 97.97% Specificity

| Metric | Score |

---|--------|-------|

| **Dice Coefficient** | **68-82%** |

## 📁 Project Structure| IoU | 60-68% |

| Accuracy | 95-96% |

```

retina-unet-segmentation/## 📁 Quick Start

├── models/                      # Model architectures

│   ├── unet_plus_plus.py       # U-Net++ implementation (9.0M params)```bash

│   └── losses_unetpp.py        # Loss functions & metrics# Install dependencies

│pip install -r requirements.txt

├── scripts/                     # Training & inference scripts

│   ├── train_unetpp.py         # Training script# Train the model

│   ├── evaluate_unetpp.py      # Evaluation on test setpython train_optimized.py

│   ├── test_model.py           # Quick model test

│   ├── dataloader_unetpp.py    # Data loading & augmentation# Or use basic training

│   └── inference.py            # Single image inferencepython train_improved.py

│```

├── results/                     # Training outputs

│   ├── checkpoints_unetpp/     # Model checkpoints## 📁 Key Files

│   │   ├── best.pth            # Best model (83.67% val Dice)

│   │   ├── latest.pth          # Latest checkpoint- **[train_optimized.py](train_optimized.py)** ⭐ - Advanced training (recommended)

│   │   ├── metrics.json        # Training history- **[train_improved.py](train_improved.py)** - Basic training script

│   │   └── training_history.png- **[config_optimized.py](config_optimized.py)** - Optimized hyperparameters

│   ├── evaluation_results_unetpp/  # Test set results- **[unet.py](unet.py)** - U-Net model architecture

│   │   ├── test_metrics.json- **[dataloader.py](dataloader.py)** - Dataset loading with augmentation

│   │   └── prediction_*.png- **[inference.py](inference.py)** - Make predictions on new images

│   └── test_result.png         # Quick test output- **[evaluate_results.py](evaluate_results.py)** - Calculate metrics

│- **[visualize.py](visualize.py)** - Visualization tools

├── docs/                        # Documentation

│   ├── README_UNETPP.md        # Detailed U-Net++ docs## 🎯 Features

│   └── FINAL_RESULTS.md        # Complete results report

│✅ **Optimized U-Net** - 31M parameters for precise segmentation  

├── Retina/                      # DRIVE dataset✅ **Combined Loss** - Dice + CrossEntropy for better results  

│   ├── train/                  # 20 training images✅ **Data Augmentation** - Rotation, flips, elastic transforms  

│   └── test/                   # 20 test images✅ **Mixed Precision** - Faster training with AMP  

│✅ **GPU Accelerated** - CUDA support for RTX/T4 GPUs  

├── requirements.txt             # Old dependencies (legacy)✅ **Complete Pipeline** - Training to inference

├── requirements_unetpp.txt      # U-Net++ dependencies

├── unet.py                      # Legacy U-Net## 📖 Project Structure

├── dataloader.py                # Legacy dataloader

├── utils.py                     # Utility functions```text

└── download_datasets.py         # Dataset download scriptRetina-Unet/

```├── 🎯 Core Files

│   ├── unet.py                        # U-Net model architecture (31M params)

---│   ├── dataloader.py                  # Dataset loading with augmentation

│   ├── utils.py                       # Helper functions (Dice, IoU)

## 🚀 Quick Start│   └── inference.py                   # Predict on new images

│

### 1️⃣ Installation├── 🚀 Training

│   ├── train_optimized.py             # Advanced training (recommended) ⭐

```bash│   ├── train_improved.py              # Basic training script

# Clone repository│   ├── config_optimized.py            # Optimized hyperparameters

git clone https://github.com/GauravPatil2515/Retina-Unet.git│   └── config.py                      # Basic configuration

cd retina-unet-segmentation│

├── 📊 Evaluation

# Install dependencies│   ├── evaluate_results.py            # Calculate metrics (Dice, IoU, Acc)

pip install -r requirements_unetpp.txt│   └── visualize.py                   # Visualization tools

```│

├──  Other

**Requirements:**│   ├── README.md                      # This file

- Python 3.8+│   ├── requirements.txt               # Python dependencies

- PyTorch 2.0+ with CUDA support│   └── download_datasets.py           # Download DRIVE/CHASE datasets

- 6GB+ GPU memory (RTX 3050 or better)│

└── 📂 Data (create these folders)

### 2️⃣ Quick Test    └── Retina/

        ├── train/

Test the trained model on a sample image:        │   ├── image/                 # Training images

        │   └── mask/                  # Training masks

```bash        └── test/

python scripts/test_model.py            ├── image/                 # Test images

```            └── mask/                  # Test masks

```

Output: `results/test_result.png` with 4-panel visualization

## 🛠️ Training Options

### 3️⃣ Train Your Own Model

### Option 1: Optimized Training (Recommended)

```bashBest performance with all optimizations enabled:

# Train U-Net++ from scratch```bash

python scripts/train_unetpp.pypython train_optimized.py

``````

- 200 epochs, batch size 8

**Training Configuration:**- Combined Dice + CrossEntropy loss

- **Model:** U-Net++ (9.0M parameters)- Cosine annealing scheduler

- **Batch Size:** 8 (effective 16 with gradient accumulation)- Mixed precision training

- **Epochs:** 60 (early stopping enabled)- Advanced data augmentation

- **Learning Rate:** 1e-4 with ReduceLROnPlateau

- **Loss:** BCE + Dice with deep supervision### Option 2: Basic Training

- **Training Time:** ~30 minutes on RTX 3050Simple training for quick testing:

```bash

### 4️⃣ Evaluate Modelpython train_improved.py

```

```bash- 100 epochs, batch size 4

# Evaluate on test set- Binary CrossEntropy loss

python scripts/evaluate_unetpp.py- Step learning rate decay

```

## 📊 Evaluation

Outputs:

- `results/evaluation_results_unetpp/test_metrics.json`After training, evaluate your model:

- `results/evaluation_results_unetpp/prediction_*.png` (5 samples)```bash

python evaluate_results.py

### 5️⃣ Inference on Single Image```



```bashThis will show:

python scripts/inference.py --image path/to/your/image.png --output output.png- Dice Coefficient

```- IoU (Intersection over Union)

- Accuracy, Sensitivity, Specificity

---- Visualizations of predictions



## 📊 Performance Metrics## 🔮 Inference on New Images



| Metric | U-Net++ (Ours) | Target Range | Status |```bash

|--------|----------------|--------------|--------|python inference.py

| **Dice Coefficient** | **83.82%** | 78-85% | ✅ |```

| **Accuracy** | **96.08%** | 94-96% | ✅ |

| **Sensitivity** | **82.91%** | 75-82% | ✅ |## 💡 Dataset

| **Specificity** | **97.97%** | 96-98% | ✅ |

| **AUC-ROC** | **97.82%** | 96-98% | ✅ |This project uses the DRIVE dataset (Digital Retinal Images for Vessel Extraction):

- 40 training images (20 with manual annotations)

**Test Set:** DRIVE dataset (20 images)  - 20 test images

**GPU:** NVIDIA RTX 3050 6GB  - Resolution: 584×565 pixels

**Training Time:** 28.2 minutes (20 epochs with early stopping)

Use `download_datasets.py` to get the dataset automatically.

---

## 🤝 Contributing

## 🏗️ Model Architecture

Contributions welcome! Feel free to:

**U-Net++ (Nested U-Net)** with key features:- Report bugs

- Suggest features

✨ **Nested Skip Connections** - Dense connections for better feature propagation  - Submit pull requests

✨ **Deep Supervision** - 4 output heads with weighted loss  - Improve documentation

✨ **9.0M Parameters** - Optimized for retinal vessel segmentation  

✨ **Mixed Precision Training** - FP16 for faster training## 📜 License



**Architecture Highlights:**MIT License - Free for research and commercial use

- **Encoder:** 5 levels (32→64→128→256→512 filters)

- **Decoder:** Nested structure with intermediate supervision---

- **Skip Connections:** Dense connections at each level

- **Output:** 4 segmentation heads (deep supervision weights: [0.25, 0.25, 0.25, 1.0])**Made with ❤️ for medical imaging research**


---

## 📖 Usage Examples

### Load Trained Model

```python
import torch
from models.unet_plus_plus import UNetPlusPlus

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNetPlusPlus(in_channels=3, out_channels=1, deep_supervision=True).to(device)

# Load checkpoint
checkpoint = torch.load('results/checkpoints_unetpp/best.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model from epoch {checkpoint['epoch']}")
print(f"Validation Dice: {checkpoint['metrics']['dice']:.4f}")
```

### Predict on Image

```python
import torch
from PIL import Image
import numpy as np

# Load and preprocess image
img = Image.open('path/to/image.png').convert('RGB')
img_array = np.array(img) / 255.0
img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(img_tensor)
    pred_logits = outputs[-1]  # Use final output
    pred = torch.sigmoid(pred_logits).squeeze().cpu().numpy()

# Threshold
pred_binary = (pred > 0.5).astype(np.uint8)
```

---

## 🔧 Configuration

Edit `scripts/train_unetpp.py` to customize:

```python
class Config:
    # Data
    batch_size = 8
    num_workers = 2
    
    # Training
    num_epochs = 60
    learning_rate = 1e-4
    accumulation_steps = 2  # Effective batch size = 16
    
    # Model
    deep_supervision = True
    deep_supervision_weights = [0.25, 0.25, 0.25, 1.0]
    
    # Optimization
    patience_early_stopping = 10
    patience_reduce_lr = 5
    lr_factor = 0.5
```

---

## 📚 Dataset

**DRIVE Dataset** (Digital Retinal Images for Vessel Extraction)

- **Training:** 20 images → 3,896 patches (128×128)
- **Test:** 20 images (full resolution)
- **Patch Extraction:** 50% overlap, vessel filtering
- **Augmentation:** Flips, brightness, contrast

Download automatically with:
```bash
python download_datasets.py
```

---

## 🎯 Key Features

✅ **State-of-the-art Architecture** - U-Net++ with nested skip connections  
✅ **Production Ready** - Clean, modular, well-documented code  
✅ **Efficient Training** - Mixed precision, gradient accumulation  
✅ **Robust Evaluation** - Patch-based reconstruction for large images  
✅ **Comprehensive Metrics** - Dice, Accuracy, Sensitivity, Specificity, AUC  
✅ **Easy to Use** - Simple scripts for training, evaluation, inference  

---

## 📈 Training Tips

1. **GPU Memory Issues?**
   - Reduce `batch_size` from 8 to 4
   - Increase `accumulation_steps` to maintain effective batch size

2. **Want Better Performance?**
   - Train longer (remove early stopping)
   - Try different learning rates (5e-5, 1e-3)
   - Experiment with deep supervision weights

3. **Custom Dataset?**
   - Modify `scripts/dataloader_unetpp.py`
   - Update data paths in config
   - Adjust patch size if needed

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **DRIVE Dataset:** Staal et al., "Ridge-based vessel segmentation in color images of the retina"
- **U-Net++:** Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
- **Original U-Net:** Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"

---

## 📞 Contact

**Author:** Gaurav Patil  
**Repository:** [github.com/GauravPatil2515/Retina-Unet](https://github.com/GauravPatil2515/Retina-Unet)

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{retina_unet_plus_plus,
  author = {Patil, Gaurav},
  title = {Retina Blood Vessel Segmentation with U-Net++},
  year = {2025},
  url = {https://github.com/GauravPatil2515/Retina-Unet}
}
```

---

<div align="center">
Made with ❤️ for medical image analysis
</div>
