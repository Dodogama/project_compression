import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class TeacherUNet(nn.Module):
    """
    Teacher model: Standard UNet architecture with more parameters
    ~31 million parameters
    """
    def __init__(self, in_channels=3, num_classes=2):
        super(TeacherUNet, self).__init__()
        
        # Encoder
        self.enc1_1 = ConvBlock(in_channels, 64)
        self.enc1_2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2_1 = ConvBlock(64, 128)
        self.enc2_2 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3_1 = ConvBlock(128, 256)
        self.enc3_2 = ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4_1 = ConvBlock(256, 512)
        self.enc4_2 = ConvBlock(512, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck_1 = ConvBlock(512, 1024)
        self.bottleneck_2 = ConvBlock(1024, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4_1 = ConvBlock(1024, 512)  # 1024 = 512 + 512 (skip connection)
        self.dec4_2 = ConvBlock(512, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3_1 = ConvBlock(512, 256)  # 512 = 256 + 256 (skip connection)
        self.dec3_2 = ConvBlock(256, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2_1 = ConvBlock(256, 128)  # 256 = 128 + 128 (skip connection)
        self.dec2_2 = ConvBlock(128, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1_1 = ConvBlock(128, 64)  # 128 = 64 + 64 (skip connection)
        self.dec1_2 = ConvBlock(64, 64)
        
        # Output layer
        self.output = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1_2(self.enc1_1(x))
        enc2 = self.enc2_2(self.enc2_1(self.pool1(enc1)))
        enc3 = self.enc3_2(self.enc3_1(self.pool2(enc2)))
        enc4 = self.enc4_2(self.enc4_1(self.pool3(enc3)))
        
        # Bottleneck
        bottleneck = self.bottleneck_2(self.bottleneck_1(self.pool4(enc4)))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4_2(self.dec4_1(dec4))
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3_2(self.dec3_1(dec3))
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2_2(self.dec2_1(dec2))
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1_2(self.dec1_1(dec1))
        
        # Output layer
        out = self.output(dec1)
        
        return out


class StudentUNet(nn.Module):
    """
    Student model: Simplified UNet architecture with fewer parameters
    ~485,000 parameters (~64x smaller than teacher)
    """
    def __init__(self, in_channels=3, num_classes=2):
        super(StudentUNet, self).__init__()
        
        # Encoder - fewer filters and simpler architecture
        self.enc1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck - smaller than teacher
        self.bottleneck = ConvBlock(128, 256)
        
        # Decoder - fewer filters and simpler architecture
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)  # 256 = 128 + 128 (skip connection)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)  # 128 = 64 + 64 (skip connection)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)  # 64 = 32 + 32 (skip connection)
        
        # Output layer
        self.output = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output layer
        out = self.output(dec1)
        
        return out


def knowledge_distillation_loss(student_outputs, teacher_outputs, targets, alpha=0.5, temperature=2.0):
    """
    Combined loss function for knowledge distillation:
    - soft targets from teacher (KL divergence loss)
    - hard targets from ground truth (cross entropy loss)
    """
    # Soft targets loss (KL divergence)
    soft_targets = F.softmax(teacher_outputs / temperature, dim=1)
    log_probs = F.log_softmax(student_outputs / temperature, dim=1)
    distillation_loss = F.kl_div(log_probs, soft_targets, reduction='batchmean') * temperature * temperature
    
    # Hard targets loss (cross entropy)
    ce_loss = F.cross_entropy(student_outputs, targets)
    
    # Combined loss
    return alpha * ce_loss + (1 - alpha) * distillation_loss


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage
if __name__ == "__main__":
    # Create models
    teacher = TeacherUNet(in_channels=3, num_classes=2)
    student = StudentUNet(in_channels=3, num_classes=2)
    
    # Print model sizes
    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)
    
    print(f"Teacher model parameters: {teacher_params:,}")
    print(f"Student model parameters: {student_params:,}")
    print(f"Compression ratio: {teacher_params/student_params:.2f}x")
    
    # Test with a sample input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher.to(device)
    student.to(device)
    
    # Create a random sample (batch_size=2, channels=3, height=64, width=64)
    x = torch.randn(2, 3, 64, 64).to(device)
    
    # Generate random target masks (class indices for each pixel)
    target = torch.randint(0, 2, (2, 64, 64)).to(device)
    
    # Forward pass
    with torch.no_grad():
        teacher_output = teacher(x)
        student_output = student(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Teacher output shape: {teacher_output.shape}")
        print(f"Student output shape: {student_output.shape}")
    
    # Calculate distillation loss
    loss = knowledge_distillation_loss(student_output, teacher_output, target)
    print(f"Distillation loss: {loss.item()}")