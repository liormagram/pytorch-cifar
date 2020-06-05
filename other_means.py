import torch
import torch.nn as nn

class MyNewLpNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, norm=2, device='cpu'):
        super(MyNewLpNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.norm = norm
        self.device = device
        self.running_lp = torch.zeros(num_features)

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
            mean = input.mean([0, 2, 3])
            # use biased var in train
            lp = (input-mean[None, :, None, None]) / \
                 (input.norm(self.norm, [0, 2, 3]) * torch.pow(torch.tensor([n-1]).to(self.device), 1/float(self.norm)))

            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean

                self.running_lp = (exponential_average_factor * lp.to(self.device) * n / (n - 1)\
                                    + (1 - exponential_average_factor) * self.running_lp.to(self.device)).to(self.device)
        else:
            mean = self.running_mean
            lp = self.running_lp
        input = ((input - mean[None, :, None, None]) / (lp[None, :, None, None].to(self.device) + self.eps)).to(self.device)

        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input