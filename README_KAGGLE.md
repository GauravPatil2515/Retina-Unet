# 🚀 Retina Blood Vessel Segmentation - Quick Start

## Train on Kaggle in 3 Steps!

### Step 1: Create Kaggle Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Settings → Accelerator → **"GPU T4 x2"**
4. Settings → Internet → **ON**

### Step 2: Add Dataset
1. In Kaggle, click **"Add Data"** (right panel)
2. Search: **"andrewmvd/drive-digital-retinal-images-for-vessel-extraction"**
3. Click **"Add"**

### Step 3: Run Training Code

Copy-paste these cells into your Kaggle notebook:

---

#### **CELL 1: Setup**
```python
# Clone repository
!git clone https://github.com/Harshtherocking/retina-unet-segmentation.git
%cd retina-unet-segmentation

# Install dependencies
!pip install -q albumentations
```

#### **CELL 2: Prepare Data**
```python
import kagglehub
import os
import shutil
from pathlib import Path

# Download DRIVE dataset
path = kagglehub.dataset_download("andrewmvd/drive-digital-retinal-images-for-vessel-extraction")
print(f"Dataset path: {path}")

# Create directories
os.makedirs("data/train/image", exist_ok=True)
os.makedirs("data/train/mask", exist_ok=True)
os.makedirs("data/test/image", exist_ok=True)
os.makedirs("data/test/mask", exist_ok=True)

# Copy training data
for file in os.listdir(f"{path}/training/images"):
    shutil.copy2(f"{path}/training/images/{file}", "data/train/image/")
for file in os.listdir(f"{path}/training/1st_manual"):
    shutil.copy2(f"{path}/training/1st_manual/{file}", "data/train/mask/")

# Copy test data  
for file in os.listdir(f"{path}/test/images"):
    shutil.copy2(f"{path}/test/images/{file}", "data/test/image/")
for file in os.listdir(f"{path}/test/1st_manual"):
    shutil.copy2(f"{path}/test/1st_manual/{file}", "data/test/mask/")

print(f"✅ Data prepared!")
```

#### **CELL 3: Update Config**
```python
# Update config paths for Kaggle
with open('config.py', 'r') as f:
    config = f.read()

config = config.replace('Retina/train/image', 'data/train/image')
config = config.replace('Retina/train/mask', 'data/train/mask')
config = config.replace('Retina/test/image', 'data/test/image')
config = config.replace('Retina/test/mask', 'data/test/mask')
config = config.replace('EPOCHS = 100', 'EPOCHS = 200')
config = config.replace('BATCH_SIZE = 2', 'BATCH_SIZE = 8')

with open('config.py', 'w') as f:
    f.write(config)

print("✅ Config updated for Kaggle!")
```

#### **CELL 4: Train Model** ⚡
```python
# Run training (30-40 minutes)
!python train_optimized.py
```

#### **CELL 5: Evaluate**
```python
!python evaluate_results.py
```

#### **CELL 6: Download Model**
```python
from IPython.display import FileLink
display(FileLink('models/best_model.pth'))
```

---

## Expected Results

- **Time**: 30-40 minutes on Kaggle T4 GPU
- **Dice Score**: 75-82%
- **Accuracy**: 95-96%

## Files in This Repo

```
retina-unet-segmentation/
├── train_optimized.py          # Optimized training script
├── config_optimized.py         # Best parameters
├── unet.py                     # U-Net architecture
├── dataloader.py               # Dataset loader
├── utils.py                    # Utilities
├── inference.py                # Make predictions
├── evaluate_results.py         # Calculate metrics
└── README.md                   # This file
```

## Local Training (Optional)

If you want to train locally:

```bash
# Install dependencies
pip install torch torchvision opencv-python albumentations

# Train model
python train_optimized.py

# Evaluate
python evaluate_results.py

# Make predictions
python inference.py --model models/best_model.pth --input image.png
```

## Citation

If you use this code, please cite:

```
@misc{retina-unet-segmentation,
  author = {Harsh Patil},
  title = {Retina Blood Vessel Segmentation using U-Net},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Harshtherocking/retina-unet-segmentation}
}
```

## License

MIT License - feel free to use for research and commercial purposes!

---

**Ready? Start training on Kaggle now! 🚀**
