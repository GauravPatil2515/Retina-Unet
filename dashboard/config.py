"""
Configuration and settings for RetinaAI Dashboard
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional
import os

# Base directories
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent

# Model configuration
MODEL_INPUT_SIZE = (512, 512)
MODEL_NAME = "U-Net++"
MODEL_PARAMETERS = "9.0M"
DEEP_SUPERVISION = True

# File upload configuration
UPLOAD_MAX_SIZE_MB = 10
ALLOWED_IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/jpg', 'image/tiff', 'image/tif']
ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tiff', '.tif']
MIN_IMAGE_SIZE = (64, 64)
MAX_IMAGE_SIZE = (4096, 4096)

# Inference configuration
CONFIDENCE_THRESHOLD = 0.5
COMPRESSION_LEVEL = 1  # Fast compression

# History configuration
MAX_HISTORY_ENTRIES = 50
HISTORY_FILE = BASE_DIR / 'history.json'

# Checkpoint paths (in order of preference)
CHECKPOINT_PATHS = [
    PROJECT_DIR / 'results' / 'checkpoints_unetpp' / 'best.pth',
    Path('/opt/render/project/src/results/checkpoints_unetpp/best.pth'),  # Render.com path
    Path('./results/checkpoints_unetpp/best.pth'),  # Relative path
]

# Metrics file
METRICS_FILE = PROJECT_DIR / 'results' / 'evaluation_results_unetpp' / 'test_metrics.json'

# Default metrics (if file not found)
DEFAULT_METRICS = {
    'dice': 83.82,
    'accuracy': 96.08,
    'sensitivity': 82.91,
    'specificity': 97.97,
    'auc': 97.82
}

# Normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# API Configuration
class APISettings(BaseSettings):
    """API Configuration from environment variables"""
    
    app_name: str = "RetinaAI - Medical Vessel Segmentation Platform"
    app_version: str = "1.1.0"
    app_description: str = "Advanced deep learning system for retinal blood vessel segmentation using U-Net++ architecture"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Documentation settings
    docs_url: str = "/docs"
    redoc_url: Optional[str] = None  # Disabled for faster startup
    
    # CORS settings
    cors_enabled: bool = True
    cors_origins: list = ["*"]
    cors_credentials: bool = True
    cors_methods: list = ["*"]
    cors_headers: list = ["*"]
    
    # Device settings
    force_cpu: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Initialize settings
api_settings = APISettings()

# Model checkpoint info (will be populated on startup)
checkpoint_info = {
    'loaded': False,
    'epoch': None,
    'dice': 0.0,
    'error': None
}

# Global variables
model = None
device = None
_matplotlib_imported = False
