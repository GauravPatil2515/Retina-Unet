# 🎯 Command Reference Card - Retina U-Net Segmentation

## ⚡ Quick Commands (Copy & Paste)

### Initial Setup (One Time Only)
```powershell
# Navigate to project
cd "C:\Users\GAURAV PATIL\Desktop\gaurav's code\Project\unet\retina-unet-segmentation"

# Create virtual environment
python -m venv venv

# Activate (run this every time you open a new terminal)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Daily Commands

```powershell
# Activate environment (ALWAYS run this first)
.\venv\Scripts\Activate.ps1

# View configuration
python config.py

# Start training (improved version)
python train_improved.py

# Start training (original version)
python train.py

# View training progress (in new terminal)
tensorboard --logdir=logs

# Make prediction on single image
python inference.py --model models/best_model.pth --input image.png --output predictions

# Make predictions on folder
python inference.py --model models/best_model.pth --input test_folder/ --output predictions

# Visualize dataset
python visualize.py --action dataset

# View model architecture
python visualize.py --action architecture
```

### Testing

```powershell
# Test data loading
python dataloader.py

# Test model
python unet.py

# Check if GPU is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📁 Important File Locations

```
Your Models: models/
  └── best_model.pth      ← Use this for predictions
  └── last_model.pth      ← Latest training state

Your Data: Retina/
  └── train/image/        ← 80 training images
  └── train/mask/         ← 80 training masks

Training Logs: logs/
  └── View with: tensorboard --logdir=logs

Results: results/
  └── Sample predictions during training
```

---

## 🎛️ Quick Configuration Changes

### Make Training Faster (Lower Quality)
Edit `config.py`:
```python
BATCH_SIZE = 4
EPOCHS = 50
IMAGE_SIZE = 256
```

### Make Training Better (Slower)
Edit `config.py`:
```python
BATCH_SIZE = 1
EPOCHS = 200
IMAGE_SIZE = 512
LEARNING_RATE = 0.0001
```

### Fix "Out of Memory" Error
Edit `config.py`:
```python
BATCH_SIZE = 1
IMAGE_SIZE = 256
NUM_WORKERS = 0
```

---

## 📊 Expected Values

### Good Loss Values
- Epoch 1: 0.6-0.7
- Epoch 50: 0.15-0.25
- Epoch 100: 0.10-0.15

### Good Dice Scores
- Epoch 50: 0.70-0.75
- Epoch 100: 0.75-0.80
- Excellent: > 0.80

---

## 🆘 Emergency Fixes

### "No module named torch"
```powershell
.\venv\Scripts\Activate.ps1
pip install torch torchvision
```

### "CUDA out of memory"
Edit config.py:
```python
BATCH_SIZE = 1
```

### Training stopped accidentally
Training continues from last checkpoint automatically!

### Can't activate venv
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📖 Documentation Quick Links

- **Quick Setup:** Read `QUICKSTART.md`
- **Full Guide:** Read `COMPLETE_PROJECT_GUIDE.md`
- **Overview:** Read `PROJECT_SUMMARY.md`
- **This Card:** For daily reference

---

## ✅ Pre-Training Checklist

- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] `python config.py` works
- [ ] `python dataloader.py` works
- [ ] GPU detected (or using CPU)
- [ ] Ready to train!

---

## 🎯 Workflow

```
1. Activate venv
   ↓
2. python train_improved.py
   ↓
3. tensorboard --logdir=logs (optional)
   ↓
4. Wait for training...
   ↓
5. python inference.py --model models/best_model.pth --input test.png
```

---

**Print this page for quick reference! 📄**
