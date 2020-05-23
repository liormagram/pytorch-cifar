'''Dual Path Networks in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from my_layers import norm


class Bottleneck(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer,
                 norm_type, lp_norm, device):
        super(Bottleneck, self).__init__()

        self.norm_type = norm_type
        self.lp_norm = lp_norm
        self.device = device

        self.out_planes = out_planes
        self.dense_depth = dense_depth

        self.conv1 = nn.Conv2d(last_planes, in_planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1 = norm(norm_type=self.norm_type, num_features=in_planes, lp_norm=self.lp_norm, device=self.device)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
        # self.bn2 = nn.BatchNorm2d(in_planes)
        self.bn2 = norm(norm_type=self.norm_type, num_features=in_planes, lp_norm=self.lp_norm, device=self.device)
        self.conv3 = nn.Conv2d(in_planes, out_planes+dense_depth, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(out_planes+dense_depth)
        self.bn3 = norm(norm_type=self.norm_type, num_features=out_planes+dense_depth, lp_norm=self.lp_norm, device=self.device)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv2d(last_planes, out_planes+dense_depth, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(out_planes+dense_depth)
                norm(norm_type=self.norm_type, num_features=out_planes + dense_depth, lp_norm=self.lp_norm, device=self.device)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        x = self.shortcut(x)
        d = self.out_planes
        out = torch.cat([x[:,:d,:,:]+out[:,:d,:,:], x[:,d:,:,:], out[:,d:,:,:]], 1)
        out = F.relu(out)
        return out


class DPN(nn.Module):
    def __init__(self, cfg, norm_type='BN', lp_norm=2, device='cpu'):
        super(DPN, self).__init__()

        self.norm_type = norm_type
        self.lp_norm = lp_norm
        self.device = device

        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = norm(norm_type=self.norm_type, num_features=64, lp_norm=self.lp_norm, device=self.device)
        self.last_planes = 64
        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.linear = nn.Linear(out_planes[3]+(num_blocks[3]+1)*dense_depth[3], 10)

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i,stride in enumerate(strides):
            layers.append(Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i==0,
                                     norm_type=self.norm_type, lp_norm=self.lp_norm, device=self.device))
            self.last_planes = out_planes + (i+2) * dense_depth
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def DPN26(norm_type='BN', lp_norm=2, device='cpu'):
    cfg = {
        'in_planes': (96,192,384,768),
        'out_planes': (256,512,1024,2048),
        'num_blocks': (2,2,2,2),
        'dense_depth': (16,32,24,128)
    }
    return DPN(cfg, norm_type=norm_type, lp_norm=lp_norm, device=device)

def DPN92(norm_type='BN', lp_norm=2, device='cpu'):
    cfg = {
        'in_planes': (96,192,384,768),
        'out_planes': (256,512,1024,2048),
        'num_blocks': (3,4,20,3),
        'dense_depth': (16,32,24,128)
    }
    return DPN(cfg, norm_type=norm_type, lp_norm=lp_norm, device=device)


# def test():
#     net = DPN92()
#     x = torch.randn(1,3,32,32)
#     y = net(x)
#     print(y)

# test()
