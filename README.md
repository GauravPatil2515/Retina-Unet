# 🩺 Retina Blood Vessel Segmentation with U-Net++# 🩺 Retina Blood Vessel Segmentation with U-Net++# 🩺 Retina Blood Vessel Segmentation



[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)Deep learning model for automated retinal blood vessel segmentation using U-Net architecture with PyTorch.



**State-of-the-art retinal blood vessel segmentation using U-Net++ (Nested U-Net) architecture with PyTorch.**[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)



## 🎯 Performance[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)[![Made with PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org)



| Metric | Score | Status |[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

|--------|-------|--------|

| **Dice Coefficient** | **83.82%** | ✅ |**State-of-the-art retinal blood vessel segmentation using U-Net++ (Nested U-Net) architecture.**

| **Accuracy** | **96.08%** | ✅ |

| **Sensitivity** | **82.91%** | ✅ |## � Performance

| **Specificity** | **97.97%** | ✅ |

| **AUC-ROC** | **97.82%** | ✅ |🎯 **Performance:** 83.82% Dice Score | 96.08% Accuracy | 97.97% Specificity



**Test Set:** DRIVE dataset (20 images)  | Metric | Score |

**GPU:** NVIDIA RTX 3050 6GB  

**Training Time:** ~30 minutes---|--------|-------|



---| **Dice Coefficient** | **68-82%** |



## 🚀 Quick Start## 📁 Project Structure| IoU | 60-68% |



### 1️⃣ Installation| Accuracy | 95-96% |



```bash```

# Clone repository

git clone https://github.com/GauravPatil2515/Retina-Unet.gitretina-unet-segmentation/## 📁 Quick Start

cd Retina-Unet

├── models/                      # Model architectures

# Install dependencies

pip install -r requirements_unetpp.txt│   ├── unet_plus_plus.py       # U-Net++ implementation (9.0M params)```bash

```

│   └── losses_unetpp.py        # Loss functions & metrics# Install dependencies

**Requirements:**

- Python 3.8+│pip install -r requirements.txt

- PyTorch 2.0+ with CUDA

- 6GB+ GPU memory (RTX 3050 or better)├── scripts/                     # Training & inference scripts



### 2️⃣ Quick Test│   ├── train_unetpp.py         # Training script# Train the model



Test the pre-trained model:│   ├── evaluate_unetpp.py      # Evaluation on test setpython train_optimized.py



```bash│   ├── test_model.py           # Quick model test

python scripts/test_model.py

```│   ├── dataloader_unetpp.py    # Data loading & augmentation# Or use basic training



Output: `results/test_result.png` with 4-panel visualization│   └── inference.py            # Single image inferencepython train_improved.py



### 3️⃣ Web Dashboard│```



Launch the interactive dashboard:├── results/                     # Training outputs



```bash│   ├── checkpoints_unetpp/     # Model checkpoints## 📁 Key Files

cd dashboard

uvicorn app:app --host 127.0.0.1 --port 8000│   │   ├── best.pth            # Best model (83.67% val Dice)

```

│   │   ├── latest.pth          # Latest checkpoint- **[train_optimized.py](train_optimized.py)** ⭐ - Advanced training (recommended)

Then open: **http://localhost:8000**

│   │   ├── metrics.json        # Training history- **[train_improved.py](train_improved.py)** - Basic training script

**Features:**

- 📤 Drag & drop image upload│   │   └── training_history.png- **[config_optimized.py](config_optimized.py)** - Optimized hyperparameters

- 🔍 Real-time vessel segmentation

- 📊 Probability heatmaps│   ├── evaluation_results_unetpp/  # Test set results- **[unet.py](unet.py)** - U-Net model architecture

- 🎨 Overlay visualizations

- 📈 Statistical metrics│   │   ├── test_metrics.json- **[dataloader.py](dataloader.py)** - Dataset loading with augmentation



### 4️⃣ Inference on Custom Image│   │   └── prediction_*.png- **[inference.py](inference.py)** - Make predictions on new images



```bash│   └── test_result.png         # Quick test output- **[evaluate_results.py](evaluate_results.py)** - Calculate metrics

python run_on_custom_image.py "path/to/image.png" "output.png"

```│- **[visualize.py](visualize.py)** - Visualization tools



---├── docs/                        # Documentation



## 📁 Project Structure│   ├── README_UNETPP.md        # Detailed U-Net++ docs## 🎯 Features



```│   └── FINAL_RESULTS.md        # Complete results report

retina-unet-segmentation/

├── models/                      # Model architectures│✅ **Optimized U-Net** - 31M parameters for precise segmentation  

│   ├── unet_plus_plus.py       # U-Net++ implementation (9.0M params)

│   └── losses_unetpp.py        # Loss functions & metrics├── Retina/                      # DRIVE dataset✅ **Combined Loss** - Dice + CrossEntropy for better results  

│

├── scripts/                     # Training & inference scripts│   ├── train/                  # 20 training images✅ **Data Augmentation** - Rotation, flips, elastic transforms  

│   ├── train_unetpp.py         # Training script

│   ├── evaluate_unetpp.py      # Evaluation on test set│   └── test/                   # 20 test images✅ **Mixed Precision** - Faster training with AMP  

│   ├── test_model.py           # Quick model test

│   ├── dataloader_unetpp.py    # Data loading & augmentation│✅ **GPU Accelerated** - CUDA support for RTX/T4 GPUs  

│   └── inference.py            # Single image inference

│├── requirements.txt             # Old dependencies (legacy)✅ **Complete Pipeline** - Training to inference

├── dashboard/                   # Web interface

│   ├── app.py                  # FastAPI backend├── requirements_unetpp.txt      # U-Net++ dependencies

│   ├── templates/              # HTML templates

│   └── static/                 # CSS & JavaScript├── unet.py                      # Legacy U-Net## 📖 Project Structure

│

├── results/                     # Training outputs├── dataloader.py                # Legacy dataloader

│   ├── checkpoints_unetpp/     # Model checkpoints

│   │   ├── best.pth            # Best model (83.82% Dice)├── utils.py                     # Utility functions```text

│   │   └── metrics.json        # Training history

│   └── evaluation_results_unetpp/  # Test set results└── download_datasets.py         # Dataset download scriptRetina-Unet/

│

├── Retina/                      # DRIVE dataset```├── 🎯 Core Files

│   ├── train/                  # 80 training images

│   └── test/                   # 20 test images│   ├── unet.py                        # U-Net model architecture (31M params)

│

├── requirements_unetpp.txt      # Python dependencies---│   ├── dataloader.py                  # Dataset loading with augmentation

└── run_on_custom_image.py      # Simple inference script

```│   ├── utils.py                       # Helper functions (Dice, IoU)



---## 🚀 Quick Start│   └── inference.py                   # Predict on new images



## 🏗️ Model Architecture│



**U-Net++ (Nested U-Net)** with key features:### 1️⃣ Installation├── 🚀 Training



✨ **Nested Skip Connections** - Dense connections for better feature propagation  │   ├── train_optimized.py             # Advanced training (recommended) ⭐

✨ **Deep Supervision** - 4 output heads with weighted loss  

✨ **9.0M Parameters** - Optimized for retinal vessel segmentation  ```bash│   ├── train_improved.py              # Basic training script

✨ **Mixed Precision Training** - FP16 for faster training

# Clone repository│   ├── config_optimized.py            # Optimized hyperparameters

**Architecture Highlights:**

- **Encoder:** 5 levels (32→64→128→256→512 filters)git clone https://github.com/GauravPatil2515/Retina-Unet.git│   └── config.py                      # Basic configuration

- **Decoder:** Nested structure with intermediate supervision

- **Skip Connections:** Dense connections at each levelcd retina-unet-segmentation│

- **Output:** Sigmoid activation for vessel probability

├── 📊 Evaluation

---

# Install dependencies│   ├── evaluate_results.py            # Calculate metrics (Dice, IoU, Acc)

## 🎓 Training

pip install -r requirements_unetpp.txt│   └── visualize.py                   # Visualization tools

### Train from Scratch

```│

```bash

python scripts/train_unetpp.py├──  Other

```

**Requirements:**│   ├── README.md                      # This file

**Training Configuration:**

- **Model:** U-Net++ (9.0M parameters)- Python 3.8+│   ├── requirements.txt               # Python dependencies

- **Batch Size:** 8 (effective 16 with gradient accumulation)

- **Epochs:** 60 (early stopping enabled)- PyTorch 2.0+ with CUDA support│   └── download_datasets.py           # Download DRIVE/CHASE datasets

- **Learning Rate:** 1e-4 with ReduceLROnPlateau

- **Loss:** BCE + Dice with deep supervision- 6GB+ GPU memory (RTX 3050 or better)│

- **Training Time:** ~30 minutes on RTX 3050

└── 📂 Data (create these folders)

### Evaluate Model

### 2️⃣ Quick Test    └── Retina/

```bash

python scripts/evaluate_unetpp.py        ├── train/

```

Test the trained model on a sample image:        │   ├── image/                 # Training images

Outputs:

- `results/evaluation_results_unetpp/test_metrics.json`        │   └── mask/                  # Training masks

- `results/evaluation_results_unetpp/prediction_*.png` (sample visualizations)

```bash        └── test/

---

python scripts/test_model.py            ├── image/                 # Test images

## 📊 Dataset

```            └── mask/                  # Test masks

**DRIVE Dataset** (Digital Retinal Images for Vessel Extraction)

```

- **Training:** 80 images (20 originals → 3,896 patches)

- **Test:** 20 images (full resolution evaluation)Output: `results/test_result.png` with 4-panel visualization

- **Patch Size:** 128×128 with 50% overlap

- **Augmentation:** Flips, brightness, contrast variations## 🛠️ Training Options



Download automatically with:### 3️⃣ Train Your Own Model

```bash

python download_datasets.py### Option 1: Optimized Training (Recommended)

```

```bashBest performance with all optimizations enabled:

---

# Train U-Net++ from scratch```bash

## 💻 API Usage

python scripts/train_unetpp.pypython train_optimized.py

### Load Pre-trained Model

``````

```python

import torch- 200 epochs, batch size 8

from models.unet_plus_plus import UNetPlusPlus

**Training Configuration:**- Combined Dice + CrossEntropy loss

# Load model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')- **Model:** U-Net++ (9.0M parameters)- Cosine annealing scheduler

model = UNetPlusPlus(in_channels=3, out_channels=1, deep_supervision=True).to(device)

- **Batch Size:** 8 (effective 16 with gradient accumulation)- Mixed precision training

# Load checkpoint

checkpoint = torch.load('results/checkpoints_unetpp/best.pth', weights_only=False)- **Epochs:** 60 (early stopping enabled)- Advanced data augmentation

model.load_state_dict(checkpoint['model_state_dict'])

model.eval()- **Learning Rate:** 1e-4 with ReduceLROnPlateau



print(f"Model from epoch {checkpoint['epoch']}")- **Loss:** BCE + Dice with deep supervision### Option 2: Basic Training

print(f"Validation Dice: {checkpoint['metrics']['dice']:.4f}")

```- **Training Time:** ~30 minutes on RTX 3050Simple training for quick testing:



### Predict on Image```bash



```python### 4️⃣ Evaluate Modelpython train_improved.py

import numpy as np

from PIL import Image```



# Load and preprocess image```bash- 100 epochs, batch size 4

img = Image.open('path/to/image.png').convert('RGB')

img = img.resize((512, 512))  # Resize for model compatibility# Evaluate on test set- Binary CrossEntropy loss

img_array = np.array(img) / 255.0

img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().unsqueeze(0).to(device)python scripts/evaluate_unetpp.py- Step learning rate decay



# Predict```

with torch.no_grad():

    outputs = model(img_tensor)## 📊 Evaluation

    pred_logits = outputs[-1]  # Use final output

    pred_prob = torch.sigmoid(pred_logits).squeeze().cpu().numpy()Outputs:



# Binary segmentation- `results/evaluation_results_unetpp/test_metrics.json`After training, evaluate your model:

pred_binary = (pred_prob > 0.5).astype(np.uint8)

```- `results/evaluation_results_unetpp/prediction_*.png` (5 samples)```bash



---python evaluate_results.py



## 🎯 Key Features### 5️⃣ Inference on Single Image```



✅ **State-of-the-art Performance** - 83.82% Dice Score on DRIVE dataset  

✅ **Production Ready** - Clean, modular, well-documented code  

✅ **Web Dashboard** - Interactive interface for real-time segmentation  ```bashThis will show:

✅ **Efficient Training** - Mixed precision, gradient accumulation  

✅ **Robust Evaluation** - Patch-based reconstruction for large images  python scripts/inference.py --image path/to/your/image.png --output output.png- Dice Coefficient

✅ **Easy to Use** - Simple scripts for training, evaluation, inference  

```- IoU (Intersection over Union)

---

- Accuracy, Sensitivity, Specificity

## 🔧 Configuration

---- Visualizations of predictions

Customize training in `scripts/train_unetpp.py`:



```python

class Config:## 📊 Performance Metrics## 🔮 Inference on New Images

    # Data

    BATCH_SIZE = 8

    NUM_WORKERS = 2

    | Metric | U-Net++ (Ours) | Target Range | Status |```bash

    # Training

    NUM_EPOCHS = 60|--------|----------------|--------------|--------|python inference.py

    LEARNING_RATE = 1e-4

    ACCUMULATION_STEPS = 2  # Effective batch size = 16| **Dice Coefficient** | **83.82%** | 78-85% | ✅ |```

    

    # Model| **Accuracy** | **96.08%** | 94-96% | ✅ |

    DEEP_SUPERVISION = True

    DEEP_SUPERVISION_WEIGHTS = [0.25, 0.25, 0.25, 1.0]| **Sensitivity** | **82.91%** | 75-82% | ✅ |## 💡 Dataset

    

    # Optimization| **Specificity** | **97.97%** | 96-98% | ✅ |

    PATIENCE_EARLY_STOPPING = 10

    PATIENCE_REDUCE_LR = 5| **AUC-ROC** | **97.82%** | 96-98% | ✅ |This project uses the DRIVE dataset (Digital Retinal Images for Vessel Extraction):

```

- 40 training images (20 with manual annotations)

---

**Test Set:** DRIVE dataset (20 images)  - 20 test images

## 📈 Training Tips

**GPU:** NVIDIA RTX 3050 6GB  - Resolution: 584×565 pixels

1. **GPU Memory Issues?**

   - Reduce `BATCH_SIZE` from 8 to 4**Training Time:** 28.2 minutes (20 epochs with early stopping)

   - Increase `ACCUMULATION_STEPS` to maintain effective batch size

Use `download_datasets.py` to get the dataset automatically.

2. **Want Better Performance?**

   - Train longer (increase `NUM_EPOCHS`)---

   - Experiment with learning rates (5e-5, 2e-4)

   - Adjust deep supervision weights## 🤝 Contributing



3. **Custom Dataset?**## 🏗️ Model Architecture

   - Modify `scripts/dataloader_unetpp.py`

   - Update data paths in `Config` classContributions welcome! Feel free to:

   - Adjust patch size if needed

**U-Net++ (Nested U-Net)** with key features:- Report bugs

---

- Suggest features

## 🤝 Contributing

✨ **Nested Skip Connections** - Dense connections for better feature propagation  - Submit pull requests

Contributions welcome! Please:

✨ **Deep Supervision** - 4 output heads with weighted loss  - Improve documentation

1. Fork the repository

2. Create a feature branch (`git checkout -b feature/amazing-feature`)✨ **9.0M Parameters** - Optimized for retinal vessel segmentation  

3. Commit your changes (`git commit -m 'Add amazing feature'`)

4. Push to branch (`git push origin feature/amazing-feature`)✨ **Mixed Precision Training** - FP16 for faster training## 📜 License

5. Open a Pull Request



---

**Architecture Highlights:**MIT License - Free for research and commercial use

## 📄 License

- **Encoder:** 5 levels (32→64→128→256→512 filters)

MIT License - see [LICENSE](LICENSE) file for details.

- **Decoder:** Nested structure with intermediate supervision---

---

- **Skip Connections:** Dense connections at each level

## 🙏 Acknowledgments

- **Output:** 4 segmentation heads (deep supervision weights: [0.25, 0.25, 0.25, 1.0])**Made with ❤️ for medical imaging research**

- **DRIVE Dataset:** Staal et al., "Ridge-based vessel segmentation in color images of the retina"

- **U-Net++:** Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"

- **Original U-Net:** Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"---



---## 📖 Usage Examples



## 📞 Contact### Load Trained Model



**Author:** Gaurav Patil  ```python

**Repository:** [github.com/GauravPatil2515/Retina-Unet](https://github.com/GauravPatil2515/Retina-Unet)  import torch

**Issues:** [Report bugs or request features](https://github.com/GauravPatil2515/Retina-Unet/issues)from models.unet_plus_plus import UNetPlusPlus



---# Load model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 📝 Citationmodel = UNetPlusPlus(in_channels=3, out_channels=1, deep_supervision=True).to(device)



If you use this code in your research, please cite:# Load checkpoint

checkpoint = torch.load('results/checkpoints_unetpp/best.pth', weights_only=False)

```bibtexmodel.load_state_dict(checkpoint['model_state_dict'])

@software{retina_unet_plus_plus,model.eval()

  author = {Patil, Gaurav},

  title = {Retina Blood Vessel Segmentation with U-Net++},print(f"Model from epoch {checkpoint['epoch']}")

  year = {2025},print(f"Validation Dice: {checkpoint['metrics']['dice']:.4f}")

  url = {https://github.com/GauravPatil2515/Retina-Unet}```

}

```### Predict on Image



---```python

import torch

<div align="center">from PIL import Image

import numpy as np

**⭐ Star this repository if you find it helpful!**

# Load and preprocess image

Made with ❤️ for medical image analysisimg = Image.open('path/to/image.png').convert('RGB')

img_array = np.array(img) / 255.0

</div>img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().unsqueeze(0).to(device)


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
