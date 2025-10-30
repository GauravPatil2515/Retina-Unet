"""
Inference Script for Retina Blood Vessel Segmentation
Use this to make predictions on new retina images after training
"""

import torch
import os
from torchvision.io import read_image, write_png
from torchvision.transforms import Resize
from torch.nn.functional import softmax
import argparse

from unet import Unet
from utils import uint8_to_float32, float32_to_uint8
from config import *

def load_model(model_path, device):
    """Load trained model from checkpoint"""
    model = Unet(INPUT_CHANNELS, OUTPUT_CHANNELS).to(device)
    
    if model_path.endswith('.pth'):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    
    model.eval()
    print(f"✅ Model loaded from: {model_path}")
    return model

def predict_image(model, image_path, device, output_path=None):
    """
    Make prediction on a single image
    
    Args:
        model: Trained U-Net model
        image_path: Path to input retina image
        device: cuda or cpu
        output_path: Where to save the prediction (optional)
    
    Returns:
        Prediction mask as tensor
    """
    # Load and preprocess image
    image = read_image(image_path)
    original_shape = image.shape
    
    # Resize to model input size
    resize = Resize((IMAGE_SIZE, IMAGE_SIZE))
    image_resized = resize(image)
    
    # Convert to float and normalize
    image_normalized = uint8_to_float32(image_resized)
    
    # Add batch dimension
    image_batch = image_normalized.unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_batch)
        
        # Convert to probabilities
        probabilities = softmax(output, dim=1)
        
        # Get vessel probability (channel 1)
        vessel_prob = probabilities[0, 1, :, :]
        
        # Threshold to binary mask
        binary_mask = (vessel_prob > 0.5).float()
        
        # Convert to uint8 for saving (move to CPU first)
        mask_uint8 = (binary_mask * 255).cpu().to(torch.uint8).unsqueeze(0)
    
    # Save if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        write_png(mask_uint8, output_path)
        print(f"💾 Prediction saved to: {output_path}")
    
    return binary_mask

def predict_folder(model, input_folder, output_folder, device):
    """
    Make predictions on all images in a folder
    
    Args:
        model: Trained U-Net model
        input_folder: Folder containing retina images
        output_folder: Folder to save predictions
        device: cuda or cpu
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"📂 Found {len(image_files)} images in {input_folder}")
    print(f"🚀 Starting predictions...\n")
    
    for idx, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"pred_{filename}")
        
        print(f"[{idx}/{len(image_files)}] Processing: {filename}")
        predict_image(model, input_path, device, output_path)
    
    print(f"\n✅ All predictions completed!")
    print(f"📁 Results saved in: {output_folder}")

def create_overlay(original_image_path, prediction_path, output_path, alpha=0.5):
    """
    Create an overlay of the prediction on the original image
    
    Args:
        original_image_path: Path to original retina image
        prediction_path: Path to prediction mask
        output_path: Path to save overlay
        alpha: Transparency of overlay (0-1)
    """
    # Load images
    original = read_image(original_image_path).float() / 255.0
    prediction = read_image(prediction_path).float() / 255.0
    
    # Ensure same size
    if original.shape != prediction.shape:
        resize = Resize((original.shape[1], original.shape[2]))
        prediction = resize(prediction)
    
    # Create colored overlay (red for vessels)
    overlay = original.clone()
    if prediction.shape[0] == 1:  # Grayscale mask
        overlay[0] = torch.maximum(overlay[0], prediction[0] * alpha)  # Add red channel
    
    # Save overlay
    overlay_uint8 = (overlay * 255).to(torch.uint8)
    write_png(overlay_uint8, output_path)
    print(f"🎨 Overlay saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Retina Blood Vessel Segmentation Inference')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or folder')
    parser.add_argument('--output', type=str, default='predictions',
                        help='Path to output folder')
    parser.add_argument('--overlay', action='store_true',
                        help='Create overlay visualization')
    
    args = parser.parse_args()
    
    # Load model
    print(f"\n{'='*50}")
    print("RETINA VESSEL SEGMENTATION - INFERENCE")
    print(f"{'='*50}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}\n")
    
    model = load_model(args.model, device)
    
    # Check if input is file or folder
    if os.path.isfile(args.input):
        # Single image prediction
        output_path = os.path.join(args.output, f"pred_{os.path.basename(args.input)}")
        predict_image(model, args.input, device, output_path)
        
        if args.overlay:
            overlay_path = os.path.join(args.output, f"overlay_{os.path.basename(args.input)}")
            create_overlay(args.input, output_path, overlay_path)
    
    elif os.path.isdir(args.input):
        # Folder prediction
        predict_folder(model, args.input, args.output, device)
        
        if args.overlay:
            print("\n🎨 Creating overlays...")
            overlay_folder = os.path.join(args.output, "overlays")
            os.makedirs(overlay_folder, exist_ok=True)
            
            for pred_file in os.listdir(args.output):
                if pred_file.startswith('pred_'):
                    original_file = pred_file.replace('pred_', '')
                    original_path = os.path.join(args.input, original_file)
                    pred_path = os.path.join(args.output, pred_file)
                    overlay_path = os.path.join(overlay_folder, f"overlay_{original_file}")
                    
                    if os.path.exists(original_path):
                        create_overlay(original_path, pred_path, overlay_path)
    
    else:
        print(f"❌ Error: {args.input} is neither a file nor a directory!")
        return
    
    print(f"\n{'='*50}")
    print("✅ INFERENCE COMPLETED SUCCESSFULLY")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
