# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ConvBlock(nn.Module):
#     """Basic convolutional block with batch normalization and ReLU activation."""
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x

# class Encoder(nn.Module):
#     """Encoder network for a single modality (CT or MRI)."""
#     def __init__(self, in_channels=1, base_filters=64):
#         super(Encoder, self).__init__()
        
#         # Downsampling path
#         self.enc1 = ConvBlock(in_channels, base_filters)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         self.enc2 = ConvBlock(base_filters, base_filters*2)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         self.enc3 = ConvBlock(base_filters*2, base_filters*4)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         self.enc4 = ConvBlock(base_filters*4, base_filters*8)
        
#         # Store intermediate features for skip connections
#         self.features = []
    
#     def forward(self, x):
#         self.features = []
        
#         # Encoder path with feature storage
#         x1 = self.enc1(x)
#         self.features.append(x1)
#         x = self.pool1(x1)
        
#         x2 = self.enc2(x)
#         self.features.append(x2)
#         x = self.pool2(x2)
        
#         x3 = self.enc3(x)
#         self.features.append(x3)
#         x = self.pool3(x3)
        
#         x4 = self.enc4(x)
#         self.features.append(x4)
        
#         return x4, self.features

# class FusionModule(nn.Module):
#     """Fusion module to combine features from CT and MRI encoders."""
#     def __init__(self, in_channels, out_channels):
#         super(FusionModule, self).__init__()
        
#         # Fusion convolution layer
#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
    
#     def forward(self, ct_features, mri_features):
#         # Concatenate features along channel dimension
#         fused_features = torch.cat([ct_features, mri_features], dim=1)
#         # Apply fusion convolution
#         fused_features = self.fusion_conv(fused_features)
#         return fused_features

# class Decoder(nn.Module):
#     """Decoder network (pseudo-sensing module) to reconstruct the fused image."""
#     def __init__(self, in_channels, base_filters=64):
#         super(Decoder, self).__init__()
        
#         # Upsampling path
#         self.upconv3 = nn.ConvTranspose2d(in_channels, base_filters*4, kernel_size=2, stride=2)
#         self.dec3 = ConvBlock(base_filters*4, base_filters*4)
        
#         self.upconv2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
#         self.dec2 = ConvBlock(base_filters*2, base_filters*2)
        
#         self.upconv1 = nn.ConvTranspose2d(base_filters*2, base_filters, kernel_size=2, stride=2)
#         self.dec1 = ConvBlock(base_filters, base_filters)
        
#         # Final output layer
#         self.final_conv = nn.Conv2d(base_filters, 1, kernel_size=1)
    
#     def forward(self, x):
#         # Decoder path
#         x = self.upconv3(x)
#         x = self.dec3(x)
        
#         x = self.upconv2(x)
#         x = self.dec2(x)
        
#         x = self.upconv1(x)
#         x = self.dec1(x)
        
#         # Final convolution to get output image
#         x = self.final_conv(x)
        
#         return x

# class MultiModalFusionModel(nn.Module):
#     """Complete multi-modal fusion model with dual-stream encoders, fusion module, and decoder."""
#     def __init__(self, in_channels=1, base_filters=64):
#         super(MultiModalFusionModel, self).__init__()
        
#         # Dual-stream encoders
#         self.ct_encoder = Encoder(in_channels, base_filters)
#         self.mri_encoder = Encoder(in_channels, base_filters)
        
#         # Fusion module
#         self.fusion = FusionModule(base_filters*8, base_filters*8)
        
#         # Decoder (pseudo-sensing module)
#         self.decoder = Decoder(base_filters*8, base_filters)
    
#     def forward(self, ct_image, mri_image):
#         # Encode CT and MRI images
#         ct_features, ct_skip_features = self.ct_encoder(ct_image)
#         mri_features, mri_skip_features = self.mri_encoder(mri_image)
        
#         # Fuse features
#         fused_features = self.fusion(ct_features, mri_features)
        
#         # Decode fused features to generate output image
#         output = self.decoder(fused_features)
        
#         return output

# def get_loss_function(loss_type='l1'):
#     """Return the specified loss function."""
#     if loss_type.lower() == 'l1':
#         return nn.L1Loss()
#     elif loss_type.lower() == 'l2' or loss_type.lower() == 'mse':
#         return nn.MSELoss()
#     else:
#         raise ValueError(f"Unsupported loss type: {loss_type}") 


import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and ReLU activation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    """Encoder network for a single modality (CT or MRI)."""
    def __init__(self, in_channels=1, base_filters=64):
        super(Encoder, self).__init__()
        
        # Downsampling path
        self.enc1 = ConvBlock(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = ConvBlock(base_filters, base_filters*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = ConvBlock(base_filters*2, base_filters*4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = ConvBlock(base_filters*4, base_filters*8)
        
        # Store intermediate features for skip connections
        self.features = []
    
    def forward(self, x):
        self.features = []
        
        # Encoder path with feature storage
        x1 = self.enc1(x)
        self.features.append(x1)
        x = self.pool1(x1)
        
        x2 = self.enc2(x)
        self.features.append(x2)
        x = self.pool2(x2)
        
        x3 = self.enc3(x)
        self.features.append(x3)
        x = self.pool3(x3)
        
        x4 = self.enc4(x)
        self.features.append(x4)
        
        return x4, self.features

class FusionModule(nn.Module):
    """Fusion module to combine features from CT and MRI encoders."""
    def __init__(self, in_channels, out_channels):
        super(FusionModule, self).__init__()
        
        # Fusion convolution layer
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, ct_features, mri_features):
        # Concatenate features along channel dimension
        fused_features = torch.cat([ct_features, mri_features], dim=1)
        # Apply fusion convolution
        fused_features = self.fusion_conv(fused_features)
        return fused_features

class KolmogorovArnoldModule(nn.Module):
    """
    Kolmogorov–Arnold-inspired module.
    This module approximates the idea that a multivariate function can be represented
    as a sum of univariate functions (applied via simple 1x1 convolutions here).
    """
    def __init__(self, in_channels, num_terms=3):
        super(KolmogorovArnoldModule, self).__init__()
        self.num_terms = num_terms
        # 'psi' transforms: project each channel to a scalar (via 1x1 conv)
        self.psi = nn.ModuleList([nn.Conv2d(in_channels, 1, kernel_size=1) for _ in range(num_terms)])
        # 'phi' transforms: project back from scalar to original channel dimension
        self.phi = nn.ModuleList([nn.Conv2d(1, in_channels, kernel_size=1) for _ in range(num_terms)])
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = 0
        # Apply each pair of psi and phi transforms and sum the results.
        for i in range(self.num_terms):
            term = self.psi[i](x)
            term = self.relu(term)
            term = self.phi[i](term)
            out = out + term
        return out

class Decoder(nn.Module):
    """Decoder network (pseudo-sensing module) to reconstruct the fused image."""
    def __init__(self, in_channels, base_filters=64):
        super(Decoder, self).__init__()
        
        # Upsampling path
        self.upconv3 = nn.ConvTranspose2d(in_channels, base_filters*4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_filters*4, base_filters*4)
        
        self.upconv2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_filters*2, base_filters*2)
        
        self.upconv1 = nn.ConvTranspose2d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_filters, base_filters)
        
        # Final output layer
        self.final_conv = nn.Conv2d(base_filters, 1, kernel_size=1)
    
    def forward(self, x):
        # Decoder path
        x = self.upconv3(x)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = self.dec1(x)
        
        # Final convolution to get output image
        x = self.final_conv(x)
        
        return x

class MultiModalFusionModel(nn.Module):
    """Complete multi-modal fusion model with dual-stream encoders, fusion module, a Kolmogorov–Arnold-inspired module, and decoder."""
    def __init__(self, in_channels=1, base_filters=64):
        super(MultiModalFusionModel, self).__init__()
        
        # Dual-stream encoders for CT and MRI
        self.ct_encoder = Encoder(in_channels, base_filters)
        self.mri_encoder = Encoder(in_channels, base_filters)
        
        # Fusion module
        self.fusion = FusionModule(base_filters*8, base_filters*8)
        
        # Kolmogorov–Arnold-inspired module to further process the fused features
        self.kam_module = KolmogorovArnoldModule(base_filters*8, num_terms=3)
        
        # Decoder (pseudo-sensing module)
        self.decoder = Decoder(base_filters*8, base_filters)
    
    def forward(self, ct_image, mri_image):
        # Encode CT and MRI images
        ct_features, ct_skip_features = self.ct_encoder(ct_image)
        mri_features, mri_skip_features = self.mri_encoder(mri_image)
        
        # Fuse features from both modalities
        fused_features = self.fusion(ct_features, mri_features)
        # Apply the Kolmogorov–Arnold-inspired module
        fused_features = self.kam_module(fused_features)
        
        # Decode fused features to generate the output image
        output = self.decoder(fused_features)
        
        return output

def get_loss_function(loss_type='l1'):
    """Return the specified loss function."""
    if loss_type.lower() == 'l1':
        return nn.L1Loss()
    elif loss_type.lower() in ['l2', 'mse']:
        return nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
