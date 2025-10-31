readme_content = """# Retina Blood Vessel Segmentation with U-Net++

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

State-of-the-art retinal blood vessel segmentation using U-Net++ architecture.

## Performance

| Metric | Score |
|--------|-------|
| Dice Coefficient | 83.82% |
| Accuracy | 96.08% |
| Sensitivity | 82.91% |
| Specificity | 97.97% |
| AUC-ROC | 97.82% |

Test Set: DRIVE dataset (20 images)  
GPU: NVIDIA RTX 3050 6GB  
Training Time: 30 minutes  
Model Size: 9.0M parameters

## Quick Start

### Installation

```bash
git clone https://github.com/GauravPatil2515/Retina-Unet.git
cd Retina-Unet
pip install -r requirements_unetpp.txt
```

### Test Model

```bash
python scripts/test_model.py
```

### Train Model

```bash
python scripts/train_unetpp.py
```

### Run Web Dashboard

```bash
cd dashboard
uvicorn app:app --reload --host localhost --port 8000
```

Visit: http://localhost:8000

## Model Architecture

U-Net++ (Nested U-Net) with dense skip connections:
- 9.0M parameters
- Deep supervision
- 5 encoder levels (16→32→64→128→256 channels)
- 4 decoder levels with nested pathways

## Dataset

DRIVE (Digital Retinal Images for Vessel Extraction):
- Training: 20 images with manual annotations
- Test: 20 images with ground truth
- Resolution: 512×512 (resized from 565×584)

## Results

Quantitative metrics on DRIVE test set:
- Dice: 83.82%
- IoU: 72.13%
- Accuracy: 96.08%
- Sensitivity: 82.91%
- Specificity: 97.97%

## Usage

```python
from scripts.inference import segment_image
result = segment_image("path/to/image.png")
```

## File Structure

- `models/unet_plus_plus.py` - U-Net++ implementation
- `scripts/train_unetpp.py` - Training pipeline
- `scripts/evaluate_unetpp.py` - Evaluation
- `dashboard/app.py` - FastAPI web interface

## References

1. Zhou et al., "UNet++: A Nested U-Net Architecture" (2018)
2. Ronneberger et al., "U-Net" (2015)
3. DRIVE Dataset: https://drive.grand-challenge.org/

## License

MIT License

## Author

Gaurav Patil  
GitHub: @GauravPatil2515  
Repository: https://github.com/GauravPatil2515/Retina-Unet
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)
print('✅ README.md created successfully!')
