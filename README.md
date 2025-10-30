# 🩺 Retina Blood Vessel Segmentation

Deep learning model for automated retinal blood vessel segmentation using U-Net architecture with PyTorch.

[![Made with PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## � Performance

| Metric | Score |
|--------|-------|
| **Dice Coefficient** | **68-82%** |
| IoU | 60-68% |
| Accuracy | 95-96% |

## 📁 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train_optimized.py

# Or use basic training
python train_improved.py
```

## 📁 Key Files

- **[train_optimized.py](train_optimized.py)** ⭐ - Advanced training (recommended)
- **[train_improved.py](train_improved.py)** - Basic training script
- **[config_optimized.py](config_optimized.py)** - Optimized hyperparameters
- **[unet.py](unet.py)** - U-Net model architecture
- **[dataloader.py](dataloader.py)** - Dataset loading with augmentation
- **[inference.py](inference.py)** - Make predictions on new images
- **[evaluate_results.py](evaluate_results.py)** - Calculate metrics
- **[visualize.py](visualize.py)** - Visualization tools

## 🎯 Features

✅ **Optimized U-Net** - 31M parameters for precise segmentation  
✅ **Combined Loss** - Dice + CrossEntropy for better results  
✅ **Data Augmentation** - Rotation, flips, elastic transforms  
✅ **Mixed Precision** - Faster training with AMP  
✅ **GPU Accelerated** - CUDA support for RTX/T4 GPUs  
✅ **Complete Pipeline** - Training to inference

## 📖 Project Structure

```text
Retina-Unet/
├── 🎯 Core Files
│   ├── unet.py                        # U-Net model architecture (31M params)
│   ├── dataloader.py                  # Dataset loading with augmentation
│   ├── utils.py                       # Helper functions (Dice, IoU)
│   └── inference.py                   # Predict on new images
│
├── 🚀 Training
│   ├── train_optimized.py             # Advanced training (recommended) ⭐
│   ├── train_improved.py              # Basic training script
│   ├── config_optimized.py            # Optimized hyperparameters
│   └── config.py                      # Basic configuration
│
├── 📊 Evaluation
│   ├── evaluate_results.py            # Calculate metrics (Dice, IoU, Acc)
│   └── visualize.py                   # Visualization tools
│
├──  Other
│   ├── README.md                      # This file
│   ├── requirements.txt               # Python dependencies
│   └── download_datasets.py           # Download DRIVE/CHASE datasets
│
└── 📂 Data (create these folders)
    └── Retina/
        ├── train/
        │   ├── image/                 # Training images
        │   └── mask/                  # Training masks
        └── test/
            ├── image/                 # Test images
            └── mask/                  # Test masks
```

## 🛠️ Training Options

### Option 1: Optimized Training (Recommended)
Best performance with all optimizations enabled:
```bash
python train_optimized.py
```
- 200 epochs, batch size 8
- Combined Dice + CrossEntropy loss
- Cosine annealing scheduler
- Mixed precision training
- Advanced data augmentation

### Option 2: Basic Training
Simple training for quick testing:
```bash
python train_improved.py
```
- 100 epochs, batch size 4
- Binary CrossEntropy loss
- Step learning rate decay

## 📊 Evaluation

After training, evaluate your model:
```bash
python evaluate_results.py
```

This will show:
- Dice Coefficient
- IoU (Intersection over Union)
- Accuracy, Sensitivity, Specificity
- Visualizations of predictions

## 🔮 Inference on New Images

```bash
python inference.py
```

## 💡 Dataset

This project uses the DRIVE dataset (Digital Retinal Images for Vessel Extraction):
- 40 training images (20 with manual annotations)
- 20 test images
- Resolution: 584×565 pixels

Use `download_datasets.py` to get the dataset automatically.

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 📜 License

MIT License - Free for research and commercial use

---

**Made with ❤️ for medical imaging research**
