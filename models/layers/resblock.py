import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
  #Output channel multiplier
  expansion = 1

  def __init__(self, in_channels, out_channels, stride):
    super().__init__()

    #1st Layer: 3x3 Conv
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)

    #2nd Layer: 3x3 Conv
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)

    #Shortcut for dim matching
    self.shortcut = nn.Sequential()
    if stride != 1 or in_channels != out_channels:
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

  def forward(self, x):
    #1st Conv -> Batch Norm -> ReLU
    out = F.relu(self.bn1(self.conv1(x)))

    #2nd Conv -> Batch Norm
    out = self.bn2(self.conv2(out))

    #Shortcut (if needed)
    out += self.shortcut(x)

    #ReLU
    return F.relu(out)