"""
U-Net++ (Nested U-Net) Implementation in PyTorch
Based on: Zhou et al. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"

Key Features:
- Nested skip connections with intermediate convolutions
- Dense connections - each decoder node receives multiple inputs
- Deep supervision with 4 output heads
- ~35-40M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Convolutional block: Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU
    Equivalent to the TensorFlow implementation in the Kaggle notebook
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class UNetPlusPlus(nn.Module):
    """
    U-Net++ Architecture
    
    Nested skip connections create intermediate feature transformations:
    X_0_0 (input) -> X_0_1 -> X_0_2 -> X_0_3 -> X_0_4 (output)
                     X_1_0 -> X_1_1 -> X_1_2 -> X_1_3
                              X_2_0 -> X_2_1 -> X_2_2
                                       X_3_0 -> X_3_1
                                                X_4_0 (bottleneck)
    
    Deep Supervision: 4 outputs from X_0_1, X_0_2, X_0_3, X_0_4
    """
    def __init__(self, in_channels=3, out_channels=1, deep_supervision=True):
        super(UNetPlusPlus, self).__init__()
        
        self.deep_supervision = deep_supervision
        
        # Filter sizes at each level (same as Kaggle notebook)
        filters = [32, 64, 128, 256, 512]
        
        # Encoder (Downsampling path) - X_0_0, X_1_0, X_2_0, X_3_0, X_4_0
        self.conv0_0 = ConvBlock(in_channels, filters[0])
        self.conv1_0 = ConvBlock(filters[0], filters[1])
        self.conv2_0 = ConvBlock(filters[1], filters[2])
        self.conv3_0 = ConvBlock(filters[2], filters[3])
        self.conv4_0 = ConvBlock(filters[3], filters[4])  # Bottleneck
        
        # MaxPooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Upsampling (learned via ConvTranspose2d, like TensorFlow implementation)
        self.up1_0 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.up3_0 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.up3_1 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.up3_2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.up4_0 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.up4_1 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.up4_2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.up4_3 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        
        # Nested skip connections (decoder nodes)
        # X_0_1: receives [X_0_0, upsample(X_1_0)]
        self.conv0_1 = ConvBlock(filters[0] + filters[0], filters[0])
        
        # X_1_1: receives [X_1_0, upsample(X_2_0)]
        self.conv1_1 = ConvBlock(filters[1] + filters[1], filters[1])
        
        # X_0_2: receives [X_0_0, X_0_1, upsample(X_1_1)]
        self.conv0_2 = ConvBlock(filters[0] * 2 + filters[0], filters[0])
        
        # X_2_1: receives [X_2_0, upsample(X_3_0)]
        self.conv2_1 = ConvBlock(filters[2] + filters[2], filters[2])
        
        # X_1_2: receives [X_1_0, X_1_1, upsample(X_2_1)]
        self.conv1_2 = ConvBlock(filters[1] * 2 + filters[1], filters[1])
        
        # X_0_3: receives [X_0_0, X_0_1, X_0_2, upsample(X_1_2)]
        self.conv0_3 = ConvBlock(filters[0] * 3 + filters[0], filters[0])
        
        # X_3_1: receives [X_3_0, upsample(X_4_0)]
        self.conv3_1 = ConvBlock(filters[3] + filters[3], filters[3])
        
        # X_2_2: receives [X_2_0, X_2_1, upsample(X_3_1)]
        self.conv2_2 = ConvBlock(filters[2] * 2 + filters[2], filters[2])
        
        # X_1_3: receives [X_1_0, X_1_1, X_1_2, upsample(X_2_2)]
        self.conv1_3 = ConvBlock(filters[1] * 3 + filters[1], filters[1])
        
        # X_0_4: receives [X_0_0, X_0_1, X_0_2, X_0_3, upsample(X_1_3)]
        self.conv0_4 = ConvBlock(filters[0] * 4 + filters[0], filters[0])
        
        # Deep supervision: 4 output heads
        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder path
        x0_0 = self.conv0_0(x)  # Level 0
        x1_0 = self.conv1_0(self.pool(x0_0))  # Level 1
        x2_0 = self.conv2_0(self.pool(x1_0))  # Level 2
        x3_0 = self.conv3_0(self.pool(x2_0))  # Level 3
        x4_0 = self.conv4_0(self.pool(x3_0))  # Level 4 (Bottleneck)
        
        # Nested skip connections - Column 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], dim=1))
        
        # Nested skip connections - Column 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up2_1(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up3_1(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up4_1(x3_1)], dim=1))
        
        # Nested skip connections - Column 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up3_2(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up4_2(x2_2)], dim=1))
        
        # Nested skip connections - Column 4 (Final)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up4_3(x1_3)], dim=1))
        
        # Deep supervision: Return 4 outputs (logits, not sigmoid)
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return output1, output2, output3, output4
        else:
            return self.final(x0_4)


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = UNetPlusPlus(in_channels=3, out_channels=1, deep_supervision=True).to(device)
    
    # Count parameters
    params = count_parameters(model)
    print(f"\nTotal Parameters: {params:,} ({params/1e6:.1f}M)")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 128, 128).to(device)
    
    with torch.no_grad():
        outputs = model(x)
    
    print(f"\nInput shape: {x.shape}")
    if isinstance(outputs, tuple):
        print(f"Deep Supervision Outputs (logits):")
        for i, out in enumerate(outputs, 1):
            print(f"  Output {i}: {out.shape}")
        
        # Test with sigmoid
        print(f"\nWith Sigmoid Applied:")
        for i, out in enumerate(outputs, 1):
            prob = torch.sigmoid(out)
            print(f"  Output {i} range: [{prob.min():.3f}, {prob.max():.3f}]")
    else:
        print(f"Output shape: {outputs.shape}")
        prob = torch.sigmoid(outputs)
        print(f"Output range (with sigmoid): [{prob.min():.3f}, {prob.max():.3f}]")
    
    print("\n[OK] U-Net++ model created successfully!")
