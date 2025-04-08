import torch
import torch.nn as nn
from typing import Callable, Optional
from .conv import conv1x1, conv3x3

class BasicBlock(nn.Module):
    """Pair of convolutional layers forming basic residual block."""
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # skip connection
        identity = x
        # conv layer 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # conv layer 2
        out = self.conv2(out)
        out = self.bn2(out)
        # downsample
        if self.downsample is not None:
            identity = self.downsample(x)
        # skip connection
        out += identity
        # output
        out = self.relu(out)
        return out
    

class Bottleneck(nn.Module):
    """Stack of convolutional layers forming bottleneck residual block."""
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None and groups > 1:
            norm_layer = nn.GroupNorm
        elif norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # skip connection
        identity = x
        # conv layer 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # conv layer 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # conv layer 3
        out = self.conv3(out)
        out = self.bn3(out)
        # downsample
        if self.downsample is not None:
            identity = self.downsample(x)
        # skip connection
        out += identity
        # output
        out = self.relu(out)
        return out