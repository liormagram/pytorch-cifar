'''EfficientNet in PyTorch.

Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".

Reference: https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from lp_norm import MyLpNorm2d
from my_layers import Identity, norm


def swish(x):
    return x * x.sigmoid()


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class SE(nn.Module):
    '''Squeeze-and-Excitation block with Swish.'''

    def __init__(self, in_planes, se_planes):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_planes, se_planes, kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_planes, in_planes, kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = swish(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block(nn.Module):
    '''expansion + depthwise + pointwise + squeeze-excitation'''

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride,
                 expand_ratio=1,
                 se_ratio=0.,
                 drop_rate=0.,
                 norm_type='BN', lp_norm=2, device='cpu'):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        self.norm_type = norm_type
        self.lp_norm = lp_norm
        self.device = device

        # Expansion
        planes = expand_ratio * in_planes
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = norm(norm_type=self.norm_type, num_features=planes, lp_norm=self.lp_norm, device=self.device)

        # Depthwise conv
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=(1 if kernel_size == 3 else 2),
                               groups=planes,
                               bias=False)
        self.bn2 = norm(norm_type=self.norm_type, num_features=planes, lp_norm=self.lp_norm, device=self.device)
        # self.bn2 = nn.BatchNorm2d(planes)

        # SE layers
        se_planes = int(in_planes * se_ratio)
        self.se = SE(planes, se_planes)

        # Output
        self.conv3 = nn.Conv2d(planes,
                               out_planes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = norm(norm_type=self.norm_type, num_features=out_planes, lp_norm=self.lp_norm, device=self.device)
        # self.bn3 = nn.BatchNorm2d(out_planes)

        # Skip connection if in and out shapes are the same (MV-V2 style)
        self.has_skip = (stride == 1) and (in_planes == out_planes)

    def forward(self, x):
        if self.norm_type == 'ST':
            out = x if self.expand_ratio == 1 else swish(self.conv1(x))
            out = swish(self.conv2(out))
            out = self.se(out)
            out = self.conv3(out)
        else:
            out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
            out = swish(self.bn2(self.conv2(out)))
            out = self.se(out)
            out = self.bn3(self.conv3(out))

        if self.has_skip:
            if self.training and self.drop_rate > 0:
                out = drop_connect(out, self.drop_rate)
            out = out + x
        return out


class EfficientNet(nn.Module):
    def __init__(self, cfg, num_classes=10, norm_type='BN', lp_norm=2, device='cpu'):
        super(EfficientNet, self).__init__()

        self.norm_type = norm_type
        self.lp_norm = lp_norm
        self.device = device

        self.cfg = cfg
        self.conv1 = nn.Conv2d(3,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = norm(norm_type=self.norm_type, num_features=32, lp_norm=self.lp_norm, device=self.device)
        # self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(cfg['out_planes'][-1], num_classes)

    def _make_layers(self, in_planes):
        layers = []
        cfg = [self.cfg[k] for k in ['expansion', 'out_planes', 'num_blocks', 'kernel_size',
                                     'stride']]
        for expansion, out_planes, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(
                    Block(in_planes,
                          out_planes,
                          kernel_size,
                          stride,
                          expansion,
                          se_ratio=0.25,
                          drop_rate=0,
                          norm_type=self.norm_type, lp_norm=self.lp_norm, device=self.device))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.norm_type == 'ST':
            out = swish(self.conv1(x))
        else:
            out = swish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def EfficientNetB0(norm_type='BN', lp_norm=2, device='cpu'):
    cfg = {
        'num_blocks': [1, 2, 2, 3, 3, 4, 1],
        'expansion': [1, 6, 6, 6, 6, 6, 6],
        'out_planes': [16, 24, 40, 80, 112, 192, 320],
        'kernel_size': [3, 3, 5, 3, 5, 5, 3],
        'stride': [1, 2, 2, 2, 1, 2, 1],
    }
    return EfficientNet(cfg, norm_type=norm_type, lp_norm=lp_norm, device=device)


# def test():
#     net = EfficientNetB0()
#     x = torch.randn(2, 3, 32, 32)
#     y = net(x)
#     print(y.shape)
#
#
# if __name__ == '__main__':
#     test()
