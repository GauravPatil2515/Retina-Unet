"""Image service - handles image processing and preprocessing"""

import numpy as np
from PIL import Image
import io
import base64
from typing import Tuple, Dict, Any
import torch

from config import MODEL_INPUT_SIZE, CONFIDENCE_THRESHOLD, COMPRESSION_LEVEL


class ImageService:
    """Service for image processing and validation"""
    
    @staticmethod
    def validate_image_file(content: bytes, filename: str) -> Tuple[bool, str]:
        """
        Validate image file
        
        Args:
            content: File bytes
            filename: Filename
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not filename:
            return False, "No file provided"
        
        try:
            image = Image.open(io.BytesIO(content))
            image.verify()
            image = Image.open(io.BytesIO(content))  # Reopen after verify
            
            # Check dimensions
            if image.size[0] < 64 or image.size[1] < 64:
                return False, "Image too small. Minimum size is 64x64 pixels."
            
            if image.size[0] > 4096 or image.size[1] > 4096:
                return False, "Image too large. Maximum size is 4096x4096 pixels."
            
            return True, ""
            
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
    
    @staticmethod
    def load_image(content: bytes) -> Image.Image:
        """Load image from bytes and convert to RGB"""
        return Image.open(io.BytesIO(content)).convert('RGB')
    
    @staticmethod
    def preprocess_image(image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for model inference
        
        Args:
            image: PIL Image
        
        Returns:
            Tuple of (preprocessed_tensor, original_size)
        """
        original_size = image.size
        
        # Resize to model input size
        image_resized = image.resize(MODEL_INPUT_SIZE, Image.LANCZOS)
        
        # Convert to numpy and normalize
        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        
        # Convert to tensor (C, H, W) and add batch dimension
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor, original_size
    
    @staticmethod
    def postprocess_predictions(
        pred_prob: torch.Tensor,
        original_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Postprocess model predictions
        
        Args:
            pred_prob: Probability map from model (1, 1, H, W)
            original_size: Original image size
        
        Returns:
            Tuple of (prob_original_size, binary_mask, vessel_mask)
        """
        # Convert to numpy
        pred_prob_np = pred_prob.squeeze().cpu().numpy()
        
        # Create binary mask
        pred_binary = (pred_prob_np > CONFIDENCE_THRESHOLD).astype(np.uint8) * 255
        
        # Resize to original size
        pred_prob_resized = Image.fromarray(
            (pred_prob_np * 255).astype(np.uint8)
        ).resize(original_size, Image.BICUBIC)
        pred_prob_np_original = np.array(pred_prob_resized, dtype=np.float32) / 255.0
        
        pred_binary_resized = Image.fromarray(pred_binary).resize(
            original_size, Image.NEAREST
        )
        pred_binary_original = np.array(pred_binary_resized)
        
        # Create vessel mask
        vessel_mask = pred_prob_np_original > CONFIDENCE_THRESHOLD
        
        return pred_prob_np_original, pred_binary_original, vessel_mask
    
    @staticmethod
    def calculate_statistics(
        pred_prob: np.ndarray,
        vessel_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate segmentation statistics
        
        Args:
            pred_prob: Probability map
            vessel_mask: Binary vessel mask
        
        Returns:
            Dictionary of statistics
        """
        vessel_pixels = int(vessel_mask.sum())
        total_pixels = int(pred_prob.size)
        vessel_coverage = float((vessel_pixels / total_pixels) * 100)
        mean_confidence = float(pred_prob.mean())
        
        return {
            'vessel_pixels': vessel_pixels,
            'total_pixels': total_pixels,
            'vessel_coverage': vessel_coverage,
            'mean_confidence': mean_confidence
        }
    
    @staticmethod
    def create_overlay(
        original_image: Image.Image,
        vessel_mask: np.ndarray
    ) -> np.ndarray:
        """
        Create overlay visualization
        
        Args:
            original_image: Original PIL Image
            vessel_mask: Binary vessel mask
        
        Returns:
            Overlay image as numpy array
        """
        overlay = np.array(original_image, dtype=np.float32)
        overlay[vessel_mask] = overlay[vessel_mask] * 0.5 + np.array(
            [255, 0, 100], dtype=np.float32
        ) * 0.5
        return overlay.astype(np.uint8)
    
    @staticmethod
    def create_heatmap(pred_prob: np.ndarray) -> np.ndarray:
        """
        Create heatmap visualization with custom colormap
        
        Args:
            pred_prob: Probability map
        
        Returns:
            Heatmap image as numpy array
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
            
            # Create custom colormap
            colors = ['#2196F3', '#4CAF50', '#FFEB3B', '#FF9800', '#F44336']
            cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)
            
            # Apply colormap
            heatmap_colored = (cmap(pred_prob)[:, :, :3] * 255).astype(np.uint8)
            return heatmap_colored
            
        except ImportError:
            # Fallback: grayscale heatmap if matplotlib not available
            return (pred_prob * 255).astype(np.uint8)
    
    @staticmethod
    def image_to_base64(img_array: np.ndarray) -> str:
        """
        Convert image array to base64 string
        
        Args:
            img_array: Image as numpy array
        
        Returns:
            Base64 encoded image string
        """
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG', optimize=False, compress_level=COMPRESSION_LEVEL)
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


# Global instance
image_service = ImageService()
