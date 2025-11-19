"""
Model service - handles model loading and inference
"""

import torch
import numpy as np
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.unet_plus_plus import UNetPlusPlus
from config import (
    MODEL_INPUT_SIZE, MODEL_NAME, CHECKPOINT_PATHS, 
    checkpoint_info, api_settings
)


class ModelService:
    """Service for model loading and inference operations"""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.checkpoint_info = {
            'loaded': False,
            'epoch': None,
            'dice': 0.0,
            'error': None,
            'warning': None
        }
    
    def setup_device(self) -> torch.device:
        """Setup compute device with optimizations"""
        if api_settings.force_cpu:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {device}")
        
        # Set torch optimizations
        torch.set_grad_enabled(False)  # Disable gradients for inference
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
            torch.cuda.empty_cache()  # Clear CUDA cache before loading
        
        self.device = device
        return device
    
    def create_model_architecture(self) -> UNetPlusPlus:
        """Create model architecture"""
        print("Creating model architecture...")
        
        with torch.device(self.device):
            model = UNetPlusPlus(in_channels=3, out_channels=1, deep_supervision=True)
        
        model = model.to(self.device)
        model.eval()  # Set to eval mode immediately
        
        print("✓ Model architecture created")
        return model
    
    def find_checkpoint(self) -> Optional[Path]:
        """Find checkpoint file from predefined paths"""
        for path in CHECKPOINT_PATHS:
            if path.exists():
                return path
        return None
    
    def _check_git_lfs_pointer(self, filepath: Path) -> bool:
        """Check if file is a Git LFS pointer instead of actual file"""
        try:
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            if file_size_mb < 1:  # LFS pointer files are tiny (<1 MB)
                with open(filepath, 'rb') as f:
                    first_bytes = f.read(100)
                    if b'version https://git-lfs.github.com' in first_bytes or b'oid sha256' in first_bytes:
                        return True
        except Exception as e:
            print(f"Warning: Could not check LFS status: {e}")
        
        return False
    
    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Load model checkpoint and weights"""
        try:
            print(f"Loading checkpoint from: {checkpoint_path}")
            print(f"Checkpoint size: {checkpoint_path.stat().st_size / (1024*1024):.2f} MB")
            
            # Check for Git LFS pointer
            if self._check_git_lfs_pointer(checkpoint_path):
                print("=" * 60)
                print("✗✗✗ CRITICAL: Git LFS POINTER DETECTED! ✗✗✗")
                print("=" * 60)
                print("The file is a Git LFS pointer, NOT the actual 104MB model!")
                print("Render.com free tier doesn't download Git LFS files properly.")
                print("")
                print("SOLUTION:")
                print("1. Upload best.pth to Google Drive (get direct link)")
                print("2. Update MODEL_URL in download_model.py")
                print("3. Redeploy on Render")
                print("=" * 60)
                
                self.checkpoint_info = {
                    'loaded': False,
                    'error': 'Git LFS pointer found - actual model not downloaded',
                    'warning': 'Use cloud storage (Google Drive) for model hosting'
                }
                return False
            
            # Load checkpoint
            checkpoint = torch.load(
                str(checkpoint_path), 
                map_location=self.device, 
                weights_only=False
            )
            
            # Verify checkpoint structure
            if 'model_state_dict' not in checkpoint:
                raise ValueError("Invalid checkpoint format - missing 'model_state_dict'")
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Checkpoint loaded successfully!")
            
            # Extract metrics
            dice_score = checkpoint.get('metrics', {}).get('dice', 0.0)
            epoch = checkpoint.get('epoch', 'N/A')
            
            self.checkpoint_info = {
                'epoch': epoch,
                'dice': dice_score,
                'loaded': True,
                'error': None,
                'warning': None
            }
            
            print(f"✓ Model ready: Epoch {epoch}, Dice Score: {dice_score:.4f}")
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            
            self.checkpoint_info = {
                'loaded': False,
                'error': str(e),
                'warning': 'Model will use random weights - predictions will be incorrect'
            }
            return False
    
    def load_model(self) -> bool:
        """Load complete model with architecture and weights"""
        try:
            print("=" * 60)
            print("🚀 Loading RetinaAI Model")
            print("=" * 60)
            
            # Setup device
            self.setup_device()
            
            # Create architecture
            self.model = self.create_model_architecture()
            
            # Find and load checkpoint
            checkpoint_path = self.find_checkpoint()
            
            if checkpoint_path:
                success = self.load_checkpoint(checkpoint_path)
            else:
                print("=" * 60)
                print("✗✗✗ CRITICAL ERROR: NO CHECKPOINT FOUND ✗✗✗")
                print("=" * 60)
                print("Searched paths:")
                for p in CHECKPOINT_PATHS:
                    print(f"  - {p} (exists: {p.exists()})")
                print("=" * 60)
                print("⚠ Model will use RANDOM WEIGHTS - predictions will be WRONG!")
                print("⚠ This is likely a Git LFS issue on Render.com")
                print("=" * 60)
                
                self.checkpoint_info = {
                    'loaded': False,
                    'error': 'CHECKPOINT NOT FOUND - Using random weights',
                    'warning': 'Git LFS may not be working on Render. Check deployment logs.'
                }
                success = False
            
            return success
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            
            self.checkpoint_info = {
                'loaded': False,
                'error': str(e),
                'warning': None
            }
            return False
    
    async def load_model_async(self) -> bool:
        """Load model asynchronously in background"""
        return await asyncio.to_thread(self.load_model)
    
    def infer(self, img_tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Run inference on input tensor
        
        Args:
            img_tensor: Input tensor (1, 3, H, W)
        
        Returns:
            Tuple of (probability_map, inference_time)
        """
        import time
        
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        with torch.no_grad():
            # Use autocast for faster inference if CUDA available
            autocast_enabled = self.device.type == 'cuda'
            with torch.amp.autocast(self.device.type, enabled=autocast_enabled):
                outputs = self.model(img_tensor)
                
                # Handle deep supervision (outputs may be tuple)
                if isinstance(outputs, (tuple, list)):
                    pred_logits = outputs[-1]  # Use final output
                else:
                    pred_logits = outputs
                
                pred_prob = torch.sigmoid(pred_logits)
        
        inference_time = time.time() - start_time
        return pred_prob, inference_time
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': MODEL_NAME,
            'parameters': '9.0M',
            'loaded': self.checkpoint_info.get('loaded', False),
            'epoch': self.checkpoint_info.get('epoch', 'N/A'),
            'dice': self.checkpoint_info.get('dice', 0.0),
            'device': str(self.device),
            'error': self.checkpoint_info.get('error'),
            'warning': self.checkpoint_info.get('warning')
        }


# Global instance
model_service = ModelService()
