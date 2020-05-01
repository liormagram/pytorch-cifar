import torch
import torch.nn as nn

class MyBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MyBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        # self.running_l2 = torch.zeros(num_features).to('cuda:0')
        self.running_l3 = torch.zeros(num_features)
        self.running_l2 = torch.zeros(num_features)
        self.running_l1 = torch.zeros(num_features)
        a=1

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
            var = input.var([0, 2, 3], unbiased=False)
            l3 = (input-mean[None, :, None, None]).norm(2, [0, 2, 3])/torch.pow(torch.tensor([n-1]).to('cuda'), 1/3.0)
            l2 = (input-mean[None, :, None, None]).norm(2, [0, 2, 3])/torch.sqrt(torch.tensor([n-1]).to('cuda'))
            l1 = (input-mean[None, :, None, None]).norm(1, [0, 2, 3])/(torch.tensor([n-1]).to('cuda'))

            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                # self.running_var = (exponential_average_factor * var * n / (n - 1)\
                #     + (1 - exponential_average_factor) * self.running_var).to('cuda')

                self.running_l3 = (exponential_average_factor * l3.to('cuda') * n / (n - 1)\
                                    + (1 - exponential_average_factor) * self.running_l3.to('cuda')).to('cuda')

                self.running_l2 = (exponential_average_factor * l2.to('cuda') * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_l2.to('cuda')).to('cuda')

                self.running_l1 = (exponential_average_factor * l1.to('cuda') * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_l1.to('cuda')).to('cuda')
        else:
            mean = self.running_mean
            # var = self.running_var
            l3 = self.running_l3
            l2 = self.running_l2
            l1 = self.running_l1

        # input_var = ((input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))).to('cuda')
        # input = ((input - mean[None, :, None, None]) / (l2[None, :, None, None] + self.eps)).to('cuda')
        input = ((input - mean[None, :, None, None]) / (l3[None, :, None, None].to('cuda') + self.eps)).to('cuda')

        # print('input: ')
        # print(input)
        # print('input_var: ')
        # print(input_var)
        # input = (input - mean[None, :, None, None]) / (torch.norm(input[None, :, None, None], 2) + self.eps)
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input