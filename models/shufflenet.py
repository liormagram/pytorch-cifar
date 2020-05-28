'''ShuffleNet in PyTorch.

See the paper "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from my_layers import norm


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups, norm_type='BN', lp_norm=2, device='cpu'):
        super(Bottleneck, self).__init__()
        self.stride = stride

        self.norm_type = norm_type
        self.lp_norm = lp_norm
        self.device = device

        mid_planes = out_planes/4
        g = 1 if in_planes==24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        # self.bn1 = nn.BatchNorm2d(mid_planes)
        self.bn1 = norm(norm_type=self.norm_type, num_features=mid_planes, lp_norm=self.lp_norm, device=self.device)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        # self.bn2 = nn.BatchNorm2d(mid_planes)
        self.bn2 = norm(norm_type=self.norm_type, num_features=mid_planes, lp_norm=self.lp_norm, device=self.device)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        # self.bn3 = nn.BatchNorm2d(out_planes)
        self.bn3 = norm(norm_type=self.norm_type, num_features=out_planes, lp_norm=self.lp_norm, device=self.device)


        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = F.relu(torch.cat([out,res], 1)) if self.stride==2 else F.relu(out+res)
        return out


class ShuffleNet(nn.Module):
    def __init__(self, cfg, norm_type='BN', lp_norm=2, device='cpu'):
        super(ShuffleNet, self).__init__()
        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']

        self.norm_type = norm_type
        self.lp_norm = lp_norm
        self.device = device

        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(24)
        self.bn1 = norm(norm_type=self.norm_type, num_features=24, lp_norm=self.lp_norm, device=self.device)

        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.linear = nn.Linear(out_planes[2], 10)

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck(self.in_planes, out_planes-cat_planes, stride=stride, groups=groups,
                                     norm_type=self.norm_type, lp_norm=self.lp_norm, device=self.device))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ShuffleNetG2(norm_type='BN', lp_norm=2, device='cpu'):
    cfg = {
        'out_planes': [200,400,800],
        'num_blocks': [4,8,4],
        'groups': 2
    }
    return ShuffleNet(cfg, norm_type, lp_norm, device)

def ShuffleNetG3(norm_type='BN', lp_norm=2, device='cpu'):
    cfg = {
        'out_planes': [240,480,960],
        'num_blocks': [4,8,4],
        'groups': 3
    }
    return ShuffleNet(cfg, norm_type, lp_norm, device)


def test():
    net = ShuffleNetG2()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()
