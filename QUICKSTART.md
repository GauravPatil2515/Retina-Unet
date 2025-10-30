# 🚀 Quick Start Guide - Retina Blood Vessel Segmentation

## ⚡ For Absolute Beginners

Follow these steps in order. Don't skip any!

### Step 1: Install Python (if not already installed)

1. Download Python 3.10 from: https://www.python.org/downloads/
2. **IMPORTANT:** Check "Add Python to PATH" during installation
3. Verify installation:
   ```powershell
   python --version
   ```
   You should see: `Python 3.10.x` or similar

### Step 2: Setup Virtual Environment

Open PowerShell in your project folder and run:

```powershell
# Navigate to project folder
cd "C:\Users\GAURAV PATIL\Desktop\gaurav's code\Project\unet\retina-unet-segmentation"

# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1
```

**If you get an error**, run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 3: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**For GPU users** (much faster training):
```powershell
# Instead of the above, use:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm matplotlib numpy pillow scikit-learn tensorboard
```

### Step 4: Verify Installation

```powershell
python config.py
```

You should see your configuration settings printed.

### Step 5: Check Your Data

Your data is already in place! You have:
- 80 training images in `Retina/train/image/`
- 80 training masks in `Retina/train/mask/`

Verify:
```powershell
python dataloader.py
```

### Step 6: Start Training (Improved Version)

```powershell
python train_improved.py
```

**What to expect:**
- Progress bars showing training
- Loss values (should decrease)
- Validation metrics every 5 epochs
- Checkpoints saved every 10 epochs

**Training time:**
- With GPU: ~30 seconds per epoch → ~50 minutes for 100 epochs
- With CPU: ~5 minutes per epoch → ~8 hours for 100 epochs

**Stop training:** Press `Ctrl+C`

### Step 7: Monitor Training (Optional)

Open a new PowerShell window and run:
```powershell
tensorboard --logdir=logs
```

Then open in browser: http://localhost:6006

### Step 8: Make Predictions

After training, predict on new images:

```powershell
# Single image
python inference.py --model models/best_model.pth --input path/to/image.png --output predictions

# Folder of images
python inference.py --model models/best_model.pth --input Retina/test/image --output predictions

# With overlay visualization
python inference.py --model models/best_model.pth --input Retina/test/image --output predictions --overlay
```

---

## 📁 Understanding Your Files

### Main Files:
- `train_improved.py` - **Use this for training** (better than original train.py)
- `inference.py` - Make predictions on new images
- `config.py` - All settings in one place
- `unet.py` - The neural network architecture
- `dataloader.py` - Loads your images
- `utils.py` - Helper functions

### Configuration Files:
- `requirements.txt` - List of packages to install
- `COMPLETE_PROJECT_GUIDE.md` - Detailed guide (you're reading the quick version!)

### Folders:
- `Retina/` - Your dataset
- `models/` - Saved models (created during training)
- `checkpoints/` - Training checkpoints
- `logs/` - Tensorboard logs
- `results/` - Sample predictions during training

---

## 🎯 Common Commands Cheat Sheet

```powershell
# Activate environment (do this every time you open a new terminal)
.\venv\Scripts\Activate.ps1

# Start training
python train_improved.py

# Check configuration
python config.py

# Test data loading
python dataloader.py

# Make predictions
python inference.py --model models/best_model.pth --input test_image.png --output predictions

# View training progress
tensorboard --logdir=logs

# Install new package
pip install package_name

# Deactivate environment
deactivate
```

---

## ⚙️ Changing Settings

Edit `config.py` to change:

```python
# Make training faster (lower quality)
BATCH_SIZE = 4
EPOCHS = 50

# Make training slower (better quality)
BATCH_SIZE = 1
EPOCHS = 200

# Use smaller images (if running out of memory)
IMAGE_SIZE = 256

# Change learning rate
LEARNING_RATE = 0.0001  # Slower learning
LEARNING_RATE = 0.01    # Faster learning
```

---

## 🐛 Quick Troubleshooting

### "No module named torch"
→ You forgot to activate the virtual environment
```powershell
.\venv\Scripts\Activate.ps1
```

### "CUDA out of memory"
→ Reduce batch size in `config.py`:
```python
BATCH_SIZE = 1
```

### "Loss not decreasing"
→ Try different learning rate in `config.py`:
```python
LEARNING_RATE = 0.0001
```

### "Permission denied"
→ Run PowerShell as Administrator

### Training too slow
→ You're using CPU. Consider:
1. Using Google Colab (free GPU): https://colab.research.google.com
2. Reducing epochs
3. Using smaller images

---

## 📊 What Good Results Look Like

After training, check:

1. **Loss values:**
   - Start: 0.6-0.7
   - After 50 epochs: 0.15-0.25
   - After 100 epochs: 0.10-0.15

2. **Dice score:**
   - After 50 epochs: 0.70-0.75
   - After 100 epochs: 0.75-0.80
   - Excellent: > 0.80

3. **Visual predictions:**
   - Check `results/` folder
   - Vessels should be clearly visible
   - Minimal noise/artifacts

---

## 🎓 Next Steps

1. **Complete basic training** (100 epochs)
2. **Read the full guide** (`COMPLETE_PROJECT_GUIDE.md`)
3. **Experiment with settings** (learning rate, batch size)
4. **Try data augmentation** (see full guide)
5. **Implement improvements** (different loss functions, etc.)

---

## 🆘 Getting Help

**Before asking for help, always provide:**
1. Exact error message
2. What command you ran
3. Your Python version (`python --version`)
4. GPU or CPU

**Where to ask:**
- ChatGPT (me!)
- PyTorch Forums: https://discuss.pytorch.org/
- Stack Overflow: https://stackoverflow.com/
- GitHub Issues (for package-specific problems)

---

## ✅ Success Checklist

- [ ] Python installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed
- [ ] Configuration verified
- [ ] Data loading works
- [ ] Training started successfully
- [ ] Can view Tensorboard
- [ ] Model saved after training
- [ ] Can make predictions

---

**🎉 Congratulations! You're ready to train your retina segmentation model!**

For detailed explanations of concepts, see `COMPLETE_PROJECT_GUIDE.md`

*Good luck! 🚀*
