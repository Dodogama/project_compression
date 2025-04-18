import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.resblock import ResBlock
from .layers.bottleneck import BottleneckBlock
  
class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=10):
    super().__init__()

    #Initial channel number
    self.in_channels = 64

    #1st Conv Layer for CIFAR-10
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)

    #Residuals
    self.layer1 = self.make_layer(block, 64,  layers[0], stride=1)
    self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

    #Pooling
    self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    #Classification
    self.fc = nn.Linear(512 * block.expansion, num_classes)

  def make_layer(self, block, out_channels, num_blocks, stride):
    layers = []

    #First block with stride
    layers.append(block(self.in_channels, out_channels, stride))
    self.in_channels = out_channels * block.expansion

    #Remaining blocks have stride=1
    for _ in range(1, num_blocks):
      layers.append(block(self.in_channels, out_channels, stride=1))

    return nn.Sequential(*layers)

  def forward(self, x):
    #Initial Conv Layer -> Batch Norm -> ReLU
    out = F.relu(self.bn1(self.conv1(x)))

    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)

    #Reduce dim to 1x1
    out = self.global_avg_pool(out)
    out = torch.flatten(out, 1)

    #FC Layer then classify output
    out = self.fc(out)
    return F.log_softmax(out, dim=1)
  
def resnet34(num_classes=10):
    return ResNet(ResBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes=10):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes=num_classes)