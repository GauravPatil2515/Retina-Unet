"""
Advanced DataLoader for U-Net++ Training
Features:
- Patch extraction (128x128) with overlap
- Patch filtering (only patches with vessels)
- FOV (Field of View) mask support
- Data augmentation (flips, brightness, contrast)
- Efficient caching and prefetching
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
import torchvision.transforms.functional as TF
import random
from PIL import Image


class PatchDataset(Dataset):
    """
    Dataset that extracts patches from retinal images
    Implements patch filtering to only include patches with vessels
    """
    def __init__(self, image_dir, mask_dir, patch_size=128, stride=64, 
                 augment=True, filter_patches=True, min_vessel_ratio=0.01):
        """
        Args:
            image_dir: Directory containing training images
            mask_dir: Directory containing mask images
            patch_size: Size of patches to extract (default: 128x128)
            stride: Stride for patch extraction (default: 64, 50% overlap)
            augment: Apply data augmentation
            filter_patches: Only keep patches containing vessels
            min_vessel_ratio: Minimum ratio of vessel pixels to keep patch
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.filter_patches = filter_patches
        self.min_vessel_ratio = min_vessel_ratio
        
        # Get image list
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.tif', '.png', '.jpg'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.gif', '.png', '.jpg'))])
        
        # Extract all patches
        self.patches = []
        self._extract_patches()
        
        print(f"[LOAD] Extracted {len(self.patches)} patches from {len(self.image_files)} images")
    
    def _extract_patches(self):
        """Extract patches from all images"""
        for img_file, mask_file in zip(self.image_files, self.mask_files):
            img_path = os.path.join(self.image_dir, img_file)
            mask_path = os.path.join(self.mask_dir, mask_file)
            
            # Load image and mask
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            img = np.array(img)
            mask = np.array(mask)
            
            # Binarize mask
            mask = (mask > 0).astype(np.float32)
            
            H, W = img.shape[:2]
            
            # Extract patches with stride
            for y in range(0, H - self.patch_size + 1, self.stride):
                for x in range(0, W - self.patch_size + 1, self.stride):
                    img_patch = img[y:y+self.patch_size, x:x+self.patch_size]
                    mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                    
                    # Filter patches (only keep patches with vessels)
                    if self.filter_patches:
                        vessel_ratio = np.sum(mask_patch) / (self.patch_size * self.patch_size)
                        if vessel_ratio < self.min_vessel_ratio:
                            continue
                    
                    self.patches.append({
                        'image': img_patch,
                        'mask': mask_patch
                    })
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        image = patch['image'].copy()
        mask = patch['mask'].copy()
        
        # Apply augmentation if enabled
        if self.augment:
            image, mask = self._augment(image, mask)
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # [H, W, C] -> [C, H, W], normalize
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # [H, W] -> [1, H, W]
        
        return image, mask
    
    def _augment(self, image, mask):
        """
        Apply synchronized augmentation to image and mask
        - Random horizontal flip
        - Random vertical flip
        - Random brightness
        - Random contrast
        """
        # Random horizontal flip
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        
        # Random vertical flip
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        
        # Random brightness (±10%)
        if random.random() > 0.5:
            delta = random.uniform(-0.1, 0.1)
            image = np.clip(image.astype(np.float32) + delta * 255, 0, 255).astype(np.uint8)
        
        # Random contrast (0.9-1.1)
        if random.random() > 0.5:
            factor = random.uniform(0.9, 1.1)
            mean = image.mean()
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        return image, mask


class FullImageDataset(Dataset):
    """
    Dataset for full-size images (used for validation/testing)
    No patch extraction, returns full images
    """
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.tif', '.png', '.jpg'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.gif', '.png', '.jpg'))])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        # Load image and mask
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        img = np.array(img)
        mask = np.array(mask)
        
        # Binarize mask
        mask = (mask > 0).astype(np.float32)
        
        # Convert to tensors
        image = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask


def create_data_loaders(train_img_dir, train_mask_dir, 
                        val_img_dir=None, val_mask_dir=None,
                        batch_size=16, num_workers=4, 
                        patch_size=128, stride=64):
    """
    Create train and validation data loaders
    
    Args:
        train_img_dir: Training images directory
        train_mask_dir: Training masks directory
        val_img_dir: Validation images directory (optional)
        val_mask_dir: Validation masks directory (optional)
        batch_size: Batch size for training
        num_workers: Number of worker processes
        patch_size: Size of patches
        stride: Stride for patch extraction
    
    Returns:
        train_loader, val_loader (or train_loader, None if no validation data)
    """
    # Training dataset with augmentation
    train_dataset = PatchDataset(
        train_img_dir, train_mask_dir,
        patch_size=patch_size,
        stride=stride,
        augment=True,
        filter_patches=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation dataset (if provided)
    val_loader = None
    if val_img_dir and val_mask_dir:
        val_dataset = PatchDataset(
            val_img_dir, val_mask_dir,
            patch_size=patch_size,
            stride=stride,
            augment=False,  # No augmentation for validation
            filter_patches=False  # Use all patches for validation
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataloader
    data_root = os.path.join(os.getcwd(), "Retina")
    train_img_dir = os.path.join(data_root, "train", "image")
    train_mask_dir = os.path.join(data_root, "train", "mask")
    
    print(f"Loading data from: {train_img_dir}")
    
    # Create dataset
    dataset = PatchDataset(
        train_img_dir, train_mask_dir,
        patch_size=128, stride=64,
        augment=True, filter_patches=True
    )
    
    print(f"\nDataset size: {len(dataset)} patches")
    
    # Test a sample
    img, mask = dataset[0]
    print(f"\nSample patch:")
    print(f"  Image shape: {img.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Image range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"  Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
    print(f"  Vessel ratio: {mask.mean():.3%}")
    
    # Create dataloader
    train_loader, _ = create_data_loaders(
        train_img_dir, train_mask_dir,
        batch_size=16, num_workers=0
    )
    
    print(f"\nDataLoader:")
    print(f"  Batches per epoch: {len(train_loader)}")
    print(f"  Batch size: {train_loader.batch_size}")
    
    # Test one batch
    images, masks = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Masks: {masks.shape}")
    
    print("\n[OK] DataLoader test complete!")
