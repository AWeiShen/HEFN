import torch
from torch import nn

class WeightedBatchNorm(nn.BatchNorm1d):
    def __init__(self,
                 n_channels,
                 n_dim,
                 eps=1e-5,
                 momentum=0.1,
                 affine=False,
                 track_running_stats=True):
        super(WeightedBatchNorm, self).__init__(
            n_channels,
            eps,
            momentum,
            affine,
            track_running_stats
        )
        self.n_channels = n_channels
        self.n_dim = n_dim

    def forward(self, inp):
        features, part_weight = inp
        part_weight = part_weight.unsqueeze(+1)
        features_contract = features
        for i in range(self.n_dim):
            features_contract = torch.sum(features_contract * part_weight.reshape(
                part_weight.shape + (1,) * (self.n_dim - i - 1)),
                                          dim=2)
        self._check_input_dim(features_contract)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            mean = features_contract.mean(0)
            var = features_contract.var(0, unbiased=True)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
        mean = mean.reshape((self.num_features,) + (1,) * self.n_dim)
        var = var.reshape((self.num_features,) + (1,) * self.n_dim)
        features_bn = (features - mean) / torch.sqrt(var + self.eps)
        return features_bn
