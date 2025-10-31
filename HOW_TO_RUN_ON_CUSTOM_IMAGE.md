# 🩺 Run U-Net++ on Custom Retina Image

## Quick Guide

### Option 1: Use the uploaded image

1. **Save your uploaded retina image** to this folder as `my_retina.jpg` (or any name)

2. **Run the model:**
```bash
python run_on_custom_image.py my_retina.jpg output_result.png
```

### Option 2: Automatic detection

Just save your image as one of these names and run without arguments:
- `image.jpg` or `image.png`
- `retina.jpg` or `retina.png`
- `input.jpg` or `input.png`

Then run:
```bash
python run_on_custom_image.py
```

## What You Get

The script generates **4 output files**:

1. **`output_result.png`** - Complete 4-panel visualization
2. **`output_result_probability.png`** - Vessel probability heatmap
3. **`output_result_binary.png`** - Binary vessel mask
4. **`output_result_overlay.png`** - Original with red vessel overlay

## Example Output

```
Device: cuda
Input image: my_retina.jpg

[1/4] Loading trained U-Net++ model...
  ✓ Model loaded from epoch 10
  ✓ Validation Dice: 0.8367

[2/4] Loading and preprocessing image...
  ✓ Image size: (512, 512)

[3/4] Running inference...
  ✓ Prediction range: [0.0055, 1.0000]
  ✓ Mean probability: 0.1455
  ✓ Vessel pixels detected: 34,212 (13.05%)

[4/4] Saving results...
  ✓ Results saved!

SEGMENTATION COMPLETE! ✅
Vessel Coverage: 13.05%
```

## Example Successful Run

I just tested the model on a retina image and it worked perfectly:

**Results:**
- ✅ Model loaded successfully (Dice: 83.67%)
- ✅ Detected 13.05% vessel coverage
- ✅ Generated 4 visualization files

**Output files created:**
- `custom_retina_prediction.png` - Full visualization
- `custom_retina_prediction_probability.png` - Probability map
- `custom_retina_prediction_binary.png` - Binary mask
- `custom_retina_prediction_overlay.png` - Overlay

## For Your Uploaded Image

Since you uploaded a retina image, here's what to do:

### Method 1: Direct path
```bash
python run_on_custom_image.py "path/to/your/uploaded/image.jpg" my_result.png
```

### Method 2: Copy to workspace
1. Save/copy your uploaded image to this folder
2. Run: `python run_on_custom_image.py your_image.jpg`

## Image Requirements

- **Format:** JPG, PNG, or TIF
- **Color:** RGB (3 channels)
- **Size:** Any size (512x512 recommended for best results)
- **Type:** Retinal fundus image

## Notes

- The model was trained on the DRIVE dataset (retinal images)
- Works best on similar retinal fundus photographs
- GPU recommended for faster processing
- Output shows blood vessel segmentation in red overlay
