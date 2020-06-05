import torch
import torch.nn as nn
import numpy as np
import math

class MyHarNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, device='cpu'):
        super(MyHarNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.device = device
        self.running_har = torch.zeros(num_features)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            n = input.numel() / input.size(1)
            # mean = input.mean([0, 2, 3])
            # use biased var in train
            har = (torch.tensor([n-1])) / \
                 (torch.sum(input.pow(-1)))

            with torch.no_grad():
                # self.running_mean = exponential_average_factor * mean\
                #     + (1 - exponential_average_factor) * self.running_mean

                self.running_har = (exponential_average_factor * har.to(self.device) * n / (n - 1)\
                                    + (1 - exponential_average_factor) * self.running_har.to(self.device)).to(self.device)
        else:
            # mean = self.running_mean
            har = self.running_har
        input = har.to(self.device)

        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input

class MyGeomNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, device='cpu'):
        super(MyHarNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.device = device
        self.running_geom = torch.zeros(num_features)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            n = input.numel() / input.size(1)
            # mean = input.mean([0, 2, 3])
            # use biased var in train
            math.e
            geom = torch.pow(torch.tensor([math.e]),
                             (torch.sum(torch.log(input))) / (torch.tensor([n-1])) )

            with torch.no_grad():
                # self.running_mean = exponential_average_factor * mean\
                #     + (1 - exponential_average_factor) * self.running_mean

                self.running_geom = (exponential_average_factor * geom.to(self.device) * n / (n - 1)\
                                    + (1 - exponential_average_factor) * self.running_geom.to(self.device)).to(self.device)
        else:
            # mean = self.running_mean
            geom = self.running_har
        input = geom.to(self.device)

        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input