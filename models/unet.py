import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnet18

class UNet34(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(UNet34, self).__init__()
        # Load a pretrained ResNet encoder
        self.encoder = resnet34(pretrained=pretrained)
        self.base_layers = list(self.encoder.children())

        # Encoder layers
        self.enc1 = nn.Sequential(*self.base_layers[:3])  # Conv1 + BN + ReLU
        self.enc2 = nn.Sequential(*self.base_layers[3:5])  # MaxPool + Layer1
        self.enc3 = self.base_layers[5]  # Layer2
        self.enc4 = self.base_layers[6]  # Layer3
        self.enc5 = self.base_layers[7]  # Layer4

        # Decoder with extra upsampling to restore original size
        self.up4 = self._upsample_block(512, 256)
        self.up3 = self._upsample_block(256, 128)
        self.up2 = self._upsample_block(128, 64)
        self.up1 = self._upsample_block(64, 64)
        self.up_final = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Final output layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # Decoder with skip connections
        d4 = self.up4(e5) + e4
        d3 = self.up3(d4) + e3
        d2 = self.up2(d3) + e2
        d1 = self.up1(d2) + e1
        
        # Final upsampling to restore original size
        d_final = self.up_final(d1)

        # Final output
        out = self.final_conv(d_final)
        return out


class UNet18(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(UNet18, self).__init__()
        self.encoder = resnet18(pretrained=pretrained)
        self.base_layers = list(self.encoder.children())

        # Encoder layers from ResNet18
        self.enc1 = nn.Sequential(*self.base_layers[:3])  # Conv1 + BN + ReLU
        self.enc2 = nn.Sequential(*self.base_layers[3:5])  # MaxPool + Layer1
        self.enc3 = self.base_layers[5]  # Layer2
        self.enc4 = self.base_layers[6]  # Layer3
        self.enc5 = self.base_layers[7]  # Layer4

        # Decoder upsampling blocks
        self.up4 = self._upsample_block(512, 256)
        self.up3 = self._upsample_block(256, 128)
        self.up2 = self._upsample_block(128, 64)
        self.up1 = self._upsample_block(64, 64)
        self.up_final = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Final output layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)   # 64 channels
        e2 = self.enc2(e1)  # 64
        e3 = self.enc3(e2)  # 128
        e4 = self.enc4(e3)  # 256
        e5 = self.enc5(e4)  # 512

        # Decoder with skip connections
        d4 = self.up4(e5) + e4
        d3 = self.up3(d4) + e3
        d2 = self.up2(d3) + e2
        d1 = self.up1(d2) + e1
        d_final = self.up_final(d1)

        out = self.final_conv(d_final)
        return out