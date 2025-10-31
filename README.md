# 🩺 Retina Blood Vessel Segmentation with U-Net++# 🩺 Retina Blood Vessel Segmentation with U-Net++# 🩺 Retina Blood Vessel Segmentation with U-Net++# 🩺 Retina Blood Vessel Segmentation



[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)



**State-of-the-art retinal blood vessel segmentation using U-Net++ (Nested U-Net) architecture with PyTorch.**[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)



## 🎯 Performance[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)Deep learning model for automated retinal blood vessel segmentation using U-Net architecture with PyTorch.



| Metric | Score | Status |

|--------|-------|--------|

| **Dice Coefficient** | **83.82%** | ✅ |**State-of-the-art retinal blood vessel segmentation using U-Net++ (Nested U-Net) architecture with PyTorch.**[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

| **Accuracy** | **96.08%** | ✅ |

| **Sensitivity** | **82.91%** | ✅ |

| **Specificity** | **97.97%** | ✅ |

| **AUC-ROC** | **97.82%** | ✅ |## 🎯 Performance[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)[![Made with PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org)



- **Test Set:** DRIVE dataset (20 images)

- **GPU:** NVIDIA RTX 3050 6GB

- **Training Time:** ~30 minutes| Metric | Score | Status |[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

- **Model Size:** 9.0M parameters

|--------|-------|--------|

## 📁 Project Structure

| **Dice Coefficient** | **83.82%** | ✅ |**State-of-the-art retinal blood vessel segmentation using U-Net++ (Nested U-Net) architecture.**

```text

retina-unet-segmentation/| **Accuracy** | **96.08%** | ✅ |

├── models/                      # Model architectures

│   ├── unet_plus_plus.py       # U-Net++ implementation (9.0M params)| **Sensitivity** | **82.91%** | ✅ |## � Performance

│   └── losses_unetpp.py        # Loss functions & metrics

├── scripts/                     # Training & inference scripts| **Specificity** | **97.97%** | ✅ |

│   ├── train_unetpp.py         # Training script

│   ├── evaluate_unetpp.py      # Evaluation on test set| **AUC-ROC** | **97.82%** | ✅ |🎯 **Performance:** 83.82% Dice Score | 96.08% Accuracy | 97.97% Specificity

│   ├── test_model.py           # Quick model test

│   ├── dataloader_unetpp.py    # Data loading & augmentation

│   └── inference.py            # Single image inference

├── dashboard/                   # Web interface (FastAPI)**Test Set:** DRIVE dataset (20 images)  | Metric | Score |

│   ├── app.py                  # Backend API

│   ├── templates/              # HTML templates**GPU:** NVIDIA RTX 3050 6GB  

│   │   └── index_platform.html # Multi-page platform UI

│   └── static/                 # CSS & JavaScript**Training Time:** ~30 minutes---|--------|-------|

│       ├── style_platform.css  # Professional styling

│       └── script_platform.js  # Upload & visualization logic

├── results/                     # Model outputs & checkpoints

│   ├── checkpoints_unetpp/     # Saved models---| **Dice Coefficient** | **68-82%** |

│   │   ├── best.pth           # Best model (Dice 0.8382)

│   │   └── latest.pth         # Latest checkpoint

│   └── evaluation_results_unetpp/

│       └── test_metrics.json   # Test set metrics## 🚀 Quick Start## 📁 Project Structure| IoU | 60-68% |

├── Retina/                      # DRIVE dataset

│   ├── train/                  # Training images & masks

│   └── test/                   # Test images & masks

└── docs/                        # Documentation### 1️⃣ Installation| Accuracy | 95-96% |

    └── FINAL_RESULTS.md        # Detailed results & analysis

```



## 🚀 Quick Start```bash```



### 1️⃣ Installation# Clone repository



```bashgit clone https://github.com/GauravPatil2515/Retina-Unet.gitretina-unet-segmentation/## 📁 Quick Start

# Clone repository

git clone https://github.com/GauravPatil2515/Retina-Unet.gitcd Retina-Unet

cd Retina-Unet

├── models/                      # Model architectures

# Install dependencies

pip install -r requirements_unetpp.txt# Install dependencies

```

pip install -r requirements_unetpp.txt│   ├── unet_plus_plus.py       # U-Net++ implementation (9.0M params)```bash

**Requirements:**

```

- Python 3.8+

- PyTorch 2.0+ with CUDA│   └── losses_unetpp.py        # Loss functions & metrics# Install dependencies

- 6GB+ GPU memory (RTX 3050 or better)

**Requirements:**

### 2️⃣ Quick Test

- Python 3.8+│pip install -r requirements.txt

Test the pre-trained model:

- PyTorch 2.0+ with CUDA

```bash

python scripts/test_model.py- 6GB+ GPU memory (RTX 3050 or better)├── scripts/                     # Training & inference scripts

```



Output: `results/test_result.png` with 4-panel visualization

### 2️⃣ Quick Test│   ├── train_unetpp.py         # Training script# Train the model

### 3️⃣ Download Dataset



```bash

python download_datasets.pyTest the pre-trained model:│   ├── evaluate_unetpp.py      # Evaluation on test setpython train_optimized.py

```



This downloads and extracts the DRIVE dataset automatically to the `Retina/` folder.

```bash│   ├── test_model.py           # Quick model test

### 4️⃣ Train Model

python scripts/test_model.py

```bash

python scripts/train_unetpp.py```│   ├── dataloader_unetpp.py    # Data loading & augmentation# Or use basic training

```



**Training Configuration:**

Output: `results/test_result.png` with 4-panel visualization│   └── inference.py            # Single image inferencepython train_improved.py

- **Epochs:** 10

- **Batch Size:** 8

- **Learning Rate:** 0.001 (Adam optimizer)

- **Image Size:** 512×512### 3️⃣ Web Dashboard│```

- **Loss Function:** BCEDiceLoss (BCE + Dice)

- **Data Augmentation:** Random flips, rotations, elastic transforms



Checkpoints saved to `results/checkpoints_unetpp/`Launch the interactive dashboard:├── results/                     # Training outputs



### 5️⃣ Evaluate Model



```bash```bash│   ├── checkpoints_unetpp/     # Model checkpoints## 📁 Key Files

python scripts/evaluate_unetpp.py

```cd dashboard



Generates metrics on test set and saves to `results/evaluation_results_unetpp/test_metrics.json`uvicorn app:app --host 127.0.0.1 --port 8000│   │   ├── best.pth            # Best model (83.67% val Dice)



### 6️⃣ Run Web Dashboard```



```bash│   │   ├── latest.pth          # Latest checkpoint- **[train_optimized.py](train_optimized.py)** ⭐ - Advanced training (recommended)

cd dashboard

uvicorn app:app --reload --host localhost --port 8000Then open: **http://localhost:8000**

```

│   │   ├── metrics.json        # Training history- **[train_improved.py](train_improved.py)** - Basic training script

Open browser: [http://localhost:8000](http://localhost:8000)

**Features:**

**Features:**

- 📤 Drag & drop image upload│   │   └── training_history.png- **[config_optimized.py](config_optimized.py)** - Optimized hyperparameters

- 🖼️ Drag-and-drop image upload

- ⚡ Real-time segmentation (<1s)- 🔍 Real-time vessel segmentation

- 📊 4 visualization modes: Original, Mask, Overlay, Heatmap

- 📈 Interactive dashboard with metrics- 📊 Probability heatmaps│   ├── evaluation_results_unetpp/  # Test set results- **[unet.py](unet.py)** - U-Net model architecture

- 💾 Recent uploads history

- 🎨 Professional medical UI design- 🎨 Overlay visualizations



## 🧠 Model Architecture- 📈 Statistical metrics│   │   ├── test_metrics.json- **[dataloader.py](dataloader.py)** - Dataset loading with augmentation



### U-Net++ (Nested U-Net)



Advanced encoder-decoder architecture with nested skip connections:### 4️⃣ Inference on Custom Image│   │   └── prediction_*.png- **[inference.py](inference.py)** - Make predictions on new images



```text

Encoder: 5 levels (16→32→64→128→256 channels)

Skip Connections: Dense nested pathways (X^0,1 to X^0,4)```bash│   └── test_result.png         # Quick test output- **[evaluate_results.py](evaluate_results.py)** - Calculate metrics

Decoder: 4 levels with concatenated features

Output: Sigmoid activation → Binary maskpython run_on_custom_image.py "path/to/image.png" "output.png"

```

```│- **[visualize.py](visualize.py)** - Visualization tools

**Key Features:**



- **9.0M parameters** (efficient yet powerful)

- **Deep supervision** during training---├── docs/                        # Documentation

- **Dense skip connections** for better gradient flow

- **Batch normalization** for stable training



## 🔬 Dataset## 📁 Project Structure│   ├── README_UNETPP.md        # Detailed U-Net++ docs## 🎯 Features



### DRIVE (Digital Retinal Images for Vessel Extraction)



- **Training Set:** 20 fundus images + manual segmentations```│   └── FINAL_RESULTS.md        # Complete results report

- **Test Set:** 20 fundus images + ground truth

- **Resolution:** 565×584 pixels (resized to 512×512)retina-unet-segmentation/

- **Format:** RGB images + binary masks

├── models/                      # Model architectures│✅ **Optimized U-Net** - 31M parameters for precise segmentation  

**Preprocessing:**

│   ├── unet_plus_plus.py       # U-Net++ implementation (9.0M params)

1. Resize to 512×512

2. Normalize to [0, 1]│   └── losses_unetpp.py        # Loss functions & metrics├── Retina/                      # DRIVE dataset✅ **Combined Loss** - Dice + CrossEntropy for better results  

3. Data augmentation (training only)

4. Automatic vessel mask extraction│



## 📊 Results├── scripts/                     # Training & inference scripts│   ├── train/                  # 20 training images✅ **Data Augmentation** - Rotation, flips, elastic transforms  



### Quantitative Metrics│   ├── train_unetpp.py         # Training script



| Metric | Formula | Score |│   ├── evaluate_unetpp.py      # Evaluation on test set│   └── test/                   # 20 test images✅ **Mixed Precision** - Faster training with AMP  

|--------|---------|-------|

| Dice Coefficient | 2TP / (2TP + FP + FN) | **83.82%** |│   ├── test_model.py           # Quick model test

| IoU (Jaccard) | TP / (TP + FP + FN) | **72.13%** |

| Accuracy | (TP + TN) / Total | **96.08%** |│   ├── dataloader_unetpp.py    # Data loading & augmentation│✅ **GPU Accelerated** - CUDA support for RTX/T4 GPUs  

| Sensitivity | TP / (TP + FN) | **82.91%** |

| Specificity | TN / (TN + FP) | **97.97%** |│   └── inference.py            # Single image inference

| AUC-ROC | Area Under ROC Curve | **97.82%** |

│├── requirements.txt             # Old dependencies (legacy)✅ **Complete Pipeline** - Training to inference

### Qualitative Results

├── dashboard/                   # Web interface

**Visual Inspection:**

│   ├── app.py                  # FastAPI backend├── requirements_unetpp.txt      # U-Net++ dependencies

- ✅ Accurate vessel detection (major & minor vessels)

- ✅ Clean boundaries with minimal noise│   ├── templates/              # HTML templates

- ✅ Correct vessel width preservation

- ✅ Low false positive rate│   └── static/                 # CSS & JavaScript├── unet.py                      # Legacy U-Net## 📖 Project Structure



**Example Results:**│



![Sample Segmentation](results/test_result.png)├── results/                     # Training outputs├── dataloader.py                # Legacy dataloader



*4-panel view: Original Image | Ground Truth | Prediction | Overlay*│   ├── checkpoints_unetpp/     # Model checkpoints



## 🛠️ Usage Examples│   │   ├── best.pth            # Best model (83.82% Dice)├── utils.py                     # Utility functions```text



### Inference on Single Image│   │   └── metrics.json        # Training history



```python│   └── evaluation_results_unetpp/  # Test set results└── download_datasets.py         # Dataset download scriptRetina-Unet/

from scripts.inference import segment_image

│

# Load and segment image

result = segment_image('path/to/retina_image.png')├── Retina/                      # DRIVE dataset```├── 🎯 Core Files



# result contains:│   ├── train/                  # 80 training images

# - 'original': Original image

# - 'mask': Binary segmentation mask│   └── test/                   # 20 test images│   ├── unet.py                        # U-Net model architecture (31M params)

# - 'overlay': Overlay visualization

# - 'heatmap': Probability heatmap│

```

├── requirements_unetpp.txt      # Python dependencies---│   ├── dataloader.py                  # Dataset loading with augmentation

### Custom Training

└── run_on_custom_image.py      # Simple inference script

```python

from scripts.train_unetpp import train_model```│   ├── utils.py                       # Helper functions (Dice, IoU)



# Train with custom parameters

train_model(

    epochs=15,---## 🚀 Quick Start│   └── inference.py                   # Predict on new images

    batch_size=8,

    learning_rate=0.0005,

    checkpoint_dir='results/my_checkpoints/'

)## 🏗️ Model Architecture│

```



### Batch Processing

**U-Net++ (Nested U-Net)** with key features:### 1️⃣ Installation├── 🚀 Training

```bash

# Process multiple images

python scripts/inference.py --input_dir images/ --output_dir results/

```✨ **Nested Skip Connections** - Dense connections for better feature propagation  │   ├── train_optimized.py             # Advanced training (recommended) ⭐



## 📈 Training Details✨ **Deep Supervision** - 4 output heads with weighted loss  



### Loss Function✨ **9.0M Parameters** - Optimized for retinal vessel segmentation  ```bash│   ├── train_improved.py              # Basic training script



**BCEDiceLoss** (combined loss):✨ **Mixed Precision Training** - FP16 for faster training



```python# Clone repository│   ├── config_optimized.py            # Optimized hyperparameters

Loss = BCE_Loss + Dice_Loss

BCE_Loss = -[y*log(p) + (1-y)*log(1-p)]**Architecture Highlights:**

Dice_Loss = 1 - (2*|X∩Y| / |X|+|Y|)

```- **Encoder:** 5 levels (32→64→128→256→512 filters)git clone https://github.com/GauravPatil2515/Retina-Unet.git│   └── config.py                      # Basic configuration



### Optimization- **Decoder:** Nested structure with intermediate supervision



- **Optimizer:** Adam- **Skip Connections:** Dense connections at each levelcd retina-unet-segmentation│

- **Learning Rate:** 0.001 (constant)

- **Weight Decay:** 1e-5- **Output:** Sigmoid activation for vessel probability

- **Gradient Clipping:** None

- **Early Stopping:** Best Dice score├── 📊 Evaluation



### Data Augmentation---



- Random horizontal flip (p=0.5)# Install dependencies│   ├── evaluate_results.py            # Calculate metrics (Dice, IoU, Acc)

- Random vertical flip (p=0.5)

- Random rotation (±15°)## 🎓 Training

- Elastic deformation

- Grid distortionpip install -r requirements_unetpp.txt│   └── visualize.py                   # Visualization tools



## 🚧 Troubleshooting### Train from Scratch



### Common Issues```│



**1. CUDA Out of Memory**```bash



```bashpython scripts/train_unetpp.py├──  Other

# Reduce batch size

python scripts/train_unetpp.py --batch_size 4```

```

**Requirements:**│   ├── README.md                      # This file

**2. Model Loading Error**

**Training Configuration:**

```bash

# Check checkpoint path- **Model:** U-Net++ (9.0M parameters)- Python 3.8+│   ├── requirements.txt               # Python dependencies

ls results/checkpoints_unetpp/best.pth

```- **Batch Size:** 8 (effective 16 with gradient accumulation)



**3. Dataset Not Found**- **Epochs:** 60 (early stopping enabled)- PyTorch 2.0+ with CUDA support│   └── download_datasets.py           # Download DRIVE/CHASE datasets



```bash- **Learning Rate:** 1e-4 with ReduceLROnPlateau

# Re-download dataset

python download_datasets.py- **Loss:** BCE + Dice with deep supervision- 6GB+ GPU memory (RTX 3050 or better)│

```

- **Training Time:** ~30 minutes on RTX 3050

**4. Web Dashboard Not Starting**

└── 📂 Data (create these folders)

```bash

# Check port availability### Evaluate Model

cd dashboard

uvicorn app:app --reload --host 127.0.0.1 --port 8001### 2️⃣ Quick Test    └── Retina/

```

```bash

## 🔍 File Descriptions

python scripts/evaluate_unetpp.py        ├── train/

### Core Files

```

- **`models/unet_plus_plus.py`**: U-Net++ architecture implementation

- **`models/losses_unetpp.py`**: Custom loss functions (BCEDiceLoss)Test the trained model on a sample image:        │   ├── image/                 # Training images

- **`scripts/train_unetpp.py`**: Training pipeline with checkpointing

- **`scripts/evaluate_unetpp.py`**: Test set evaluationOutputs:

- **`scripts/dataloader_unetpp.py`**: Data loading with augmentation

- **`dashboard/app.py`**: FastAPI backend with model inference- `results/evaluation_results_unetpp/test_metrics.json`        │   └── mask/                  # Training masks



### Utility Files- `results/evaluation_results_unetpp/prediction_*.png` (sample visualizations)



- **`download_datasets.py`**: Automatic dataset downloader```bash        └── test/

- **`run_dashboard.ps1`**: PowerShell script to launch dashboard

- **`requirements_unetpp.txt`**: Python dependencies---



## 📚 Referencespython scripts/test_model.py            ├── image/                 # Test images



### Papers## 📊 Dataset



1. **U-Net++**: Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation" (2018)```            └── mask/                  # Test masks

2. **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)

3. **DRIVE Dataset**: Staal et al., "Ridge-based vessel segmentation in color images of the retina" (2004)**DRIVE Dataset** (Digital Retinal Images for Vessel Extraction)



### Resources```



- PyTorch Documentation: <https://pytorch.org/docs/>- **Training:** 80 images (20 originals → 3,896 patches)

- FastAPI Documentation: <https://fastapi.tiangolo.com/>

- DRIVE Dataset: <https://drive.grand-challenge.org/>- **Test:** 20 images (full resolution evaluation)Output: `results/test_result.png` with 4-panel visualization



## 🤝 Contributing- **Patch Size:** 128×128 with 50% overlap



Contributions are welcome! Please follow these steps:- **Augmentation:** Flips, brightness, contrast variations## 🛠️ Training Options



1. Fork the repository

2. Create a feature branch (`git checkout -b feature/YourFeature`)

3. Commit your changes (`git commit -m 'Add YourFeature'`)Download automatically with:### 3️⃣ Train Your Own Model

4. Push to the branch (`git push origin feature/YourFeature`)

5. Open a Pull Request```bash



## 📝 Licensepython download_datasets.py### Option 1: Optimized Training (Recommended)



This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.```



## 👨‍💻 Author```bashBest performance with all optimizations enabled:



**Gaurav Patil**---



- GitHub: [@GauravPatil2515](https://github.com/GauravPatil2515)# Train U-Net++ from scratch```bash

- Repository: [Retina-Unet](https://github.com/GauravPatil2515/Retina-Unet)

## 💻 API Usage

## 🙏 Acknowledgments

python scripts/train_unetpp.pypython train_optimized.py

- DRIVE dataset creators for providing high-quality annotated retinal images

- PyTorch team for the excellent deep learning framework### Load Pre-trained Model

- U-Net++ authors for the innovative nested architecture design

- Medical imaging community for continued support and feedback``````



---```python



**⭐ Star this repo if you find it useful!**import torch- 200 epochs, batch size 8



**🔗 Repository**: <https://github.com/GauravPatil2515/Retina-Unet>from models.unet_plus_plus import UNetPlusPlus


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
