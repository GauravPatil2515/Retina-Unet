# Retina Blood Vessel Segmentation with U-Net++

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

State-of-the-art retinal blood vessel segmentation using U-Net++ (Nested U-Net) architecture with PyTorch.

## Performance

| Metric | Score | Status |
|--------|-------|--------|
| Dice Coefficient | 83.82% | ✅ |
| Accuracy | 96.08% | ✅ |
| Sensitivity | 82.91% | ✅ |
| Specificity | 97.97% | ✅ |
| AUC-ROC | 97.82% | ✅ |

- Test Set: DRIVE dataset (20 images)
- GPU: NVIDIA RTX 3050 6GB
- Training Time: 30 minutes
- Model Size: 9.0M parameters

## Project Structure

``text
retina-unet-segmentation/
├── models/              # U-Net++ architecture
├── scripts/             # Training and inference
├── dashboard/           # Web interface
├── results/             # Checkpoints and metrics
├── Retina/              # DRIVE dataset
└── docs/                # Documentation
``

## Quick Start

### Installation

``bash
git clone https://github.com/GauravPatil2515/Retina-Unet.git
cd Retina-Unet
pip install -r requirements_unetpp.txt
``

Requirements: Python 3.8+, PyTorch 2.0+, CUDA, 6GB+ GPU

### Test Model

``bash
python scripts/test_model.py
``

### Train Model

``bash
python scripts/train_unetpp.py
``

### Run Dashboard

```bash
cd dashboard
uvicorn app:app --reload --host localhost --port 8000
```

Visit: <http://localhost:8000>

## Model Architecture

U-Net++ with nested skip connections, 9.0M parameters, deep supervision.

## Dataset

DRIVE dataset: 20 training + 20 test retinal images with vessel annotations.

## Results

Achieves 83.82% Dice score, 96.08% accuracy on DRIVE test set.

## Usage

``python
from scripts.inference import segment_image
result = segment_image('path/to/image.png')
``

## References

- Zhou et al., UNet++: A Nested U-Net Architecture (2018)
- Ronneberger et al., U-Net (2015)
- DRIVE Dataset: <https://drive.grand-challenge.org/>

## License

MIT License

## Author

Gaurav Patil - GitHub: @GauravPatil2515

Repository: <https://github.com/GauravPatil2515/Retina-Unet>
