import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from collections import OrderedDict
from .operations import OPS, conv2d, ConvBnAct, SqueezeExcite


'''ResNet'''
OPS['maxp_3x3'] = lambda inp, oup, t, stride, kwargs: nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
OPS['conv7x7'] = lambda inp, oup, t, stride, kwargs: ConvBnAct(inp, oup, kernel_size=7, stride=stride, **kwargs)
OPS['res_3x3'] = lambda inp, oup, t, stride, kwargs: Bottleneck(inplanes=inp, outplanes=oup, kernel_size=3, stride=stride, **kwargs)
OPS['res_5x5'] = lambda inp, oup, t, stride, kwargs: Bottleneck(inplanes=inp, outplanes=oup, kernel_size=5, stride=stride, **kwargs)
OPS['res_7x7'] = lambda inp, oup, t, stride, kwargs: Bottleneck(inplanes=inp, outplanes=oup, kernel_size=7, stride=stride, **kwargs)
OPS['res_3x3_se'] = lambda inp, oup, t, stride, kwargs: ResNeXtBottleneck(inplanes=inp, outplanes=oup, kernel_size=3, stride=stride, use_se=True, expansion=4, **kwargs)
OPS['res_5x5_se'] = lambda inp, oup, t, stride, kwargs: ResNeXtBottleneck(inplanes=inp, outplanes=oup, kernel_size=5, stride=stride, use_se=True, expansion=4, **kwargs)
OPS['res_7x7_se'] = lambda inp, oup, t, stride, kwargs: ResNeXtBottleneck(inplanes=inp, outplanes=oup, kernel_size=7, stride=stride, use_se=True, expansion=4, **kwargs)
OPS['res_3x3_se_e'] = lambda inp, oup, t, stride, kwargs: ResNeXtBottleneck(inplanes=inp, outplanes=oup, kernel_size=3, stride=stride, use_se=True, expansion=t, **kwargs)
OPS['res_5x5_se_e'] = lambda inp, oup, t, stride, kwargs: ResNeXtBottleneck(inplanes=inp, outplanes=oup, kernel_size=5, stride=stride, use_se=True, expansion=t, **kwargs)
OPS['res_7x7_se_e'] = lambda inp, oup, t, stride, kwargs: ResNeXtBottleneck(inplanes=inp, outplanes=oup, kernel_size=7, stride=stride, use_se=True, expansion=t, **kwargs)
OPS['resnext_3x3'] = lambda inp, oup, t, stride, kwargs: ResNeXtBottleneck(inplanes=inp, outplanes=oup, kernel_size=3, stride=stride, **kwargs)
OPS['resnext_5x5'] = lambda inp, oup, t, stride, kwargs: ResNeXtBottleneck(inplanes=inp, outplanes=oup, kernel_size=5, stride=stride, **kwargs)
OPS['resnext_7x7'] = lambda inp, oup, t, stride, kwargs: ResNeXtBottleneck(inplanes=inp, outplanes=oup, kernel_size=7, stride=stride, **kwargs)
OPS['resnext_3x3_se'] = lambda inp, oup, t, stride, kwargs: ResNeXtBottleneck(inplanes=inp, outplanes=oup, kernel_size=3, stride=stride, use_se=True, expansion=4, **kwargs)
OPS['resnext_5x5_se'] = lambda inp, oup, t, stride, kwargs: ResNeXtBottleneck(inplanes=inp, outplanes=oup, kernel_size=5, stride=stride, use_se=True, expansion=4, **kwargs)
OPS['resnext_7x7_se'] = lambda inp, oup, t, stride, kwargs: ResNeXtBottleneck(inplanes=inp, outplanes=oup, kernel_size=7, stride=stride, use_se=True, expansion=4, **kwargs)
OPS['resnext_3x3_se_e'] = lambda inp, oup, t, stride, kwargs: ResNeXtBottleneck(inplanes=inp, outplanes=oup, kernel_size=3, stride=stride, use_se=True, expansion=t, **kwargs)
OPS['resnext_5x5_se_e'] = lambda inp, oup, t, stride, kwargs: ResNeXtBottleneck(inplanes=inp, outplanes=oup, kernel_size=5, stride=stride, use_se=True, expansion=t, **kwargs)
OPS['resnext_7x7_se_e'] = lambda inp, oup, t, stride, kwargs: ResNeXtBottleneck(inplanes=inp, outplanes=oup, kernel_size=7, stride=stride, use_se=True, expansion=t, **kwargs)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        outplanes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        kernel_size: int = 3,
        use_se: bool = False,
        planes: int = None,
        expansion = 4
    ) -> None:
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if stride != 1 or inplanes != outplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, stride=stride, kernel_size=1, bias=False),
                norm_layer(outplanes),
            )
            if planes is None:
                planes = int(inplanes // self.expansion * 2)
        else:
            self.downsample = None
            planes = int(inplanes // self.expansion)

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = conv2d(width, width, kernel_size, stride, bias=False, groups=groups)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)
        if use_se:
            self.se = SqueezeExcite(outplanes, reduce_channels=max(1, outplanes // 16)) 
        else:
            self.se = None
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNeXtBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    def __init__(
        self,
        inplanes: int,
        outplanes: int,
        stride: int = 1,
        groups: int = 32,
        base_width: int = 4,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        kernel_size: int = 3,
        use_se: bool = False,
        planes: int = None,
        expansion = 4,
    ) -> None:
        super(ResNeXtBottleneck, self).__init__()
        self.expansion = expansion

        if stride != 1 or inplanes != outplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(outplanes),
            )
            if planes is None:
                planes = int(inplanes // self.expansion * 2 )
        else:
            self.downsample = None
            planes = int(inplanes // self.expansion)

        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv2d(width, width, kernel_size=kernel_size, stride=stride,
                            groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        if use_se:
            self.se = SqueezeExcite(outplanes, reduce_channels=max(1, outplanes // 16)) 
        else:
            self.se = None
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




