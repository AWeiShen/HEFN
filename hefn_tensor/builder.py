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

# Tree Examples
class Quadrangle(nn.Module):
    """
    i       j
      O---O
      |   |
      O---O
    k       l
    """

    def __init__(self, n_terms, n_channels, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.n_terms = n_terms
        self.n_channels = n_channels

        self.weight_1 = nn.Parameter(torch.rand((n_terms, n_channels), **factory_kwargs))
        torch.nn.init.kaiming_uniform_(self.weight_1, mode='fan_in', nonlinearity='relu')
        self.weight_2 = nn.Parameter(torch.rand((n_terms, n_channels), **factory_kwargs))
        torch.nn.init.kaiming_uniform_(self.weight_2, mode='fan_in', nonlinearity='relu')
        self.weight_3 = nn.Parameter(torch.rand((n_terms, n_channels), **factory_kwargs))
        torch.nn.init.kaiming_uniform_(self.weight_3, mode='fan_in', nonlinearity='relu')
        self.weight_4 = nn.Parameter(torch.rand((n_terms, n_channels), **factory_kwargs))
        torch.nn.init.kaiming_uniform_(self.weight_4, mode='fan_in', nonlinearity='relu')

        self.bias_1 = nn.Parameter(torch.zeros((1, n_channels, 1, 1), **factory_kwargs))
        self.bias_2 = nn.Parameter(torch.zeros((1, n_channels, 1, 1), **factory_kwargs))
        self.bias_3 = nn.Parameter(torch.zeros((1, n_channels, 1), **factory_kwargs))

        self.bn_1 = WeightedBatchNorm(n_channels=n_channels, n_dim=2)
        self.bn_2 = WeightedBatchNorm(n_channels=n_channels, n_dim=2)

    def forward(self, inp):
        part_weight, pair_weight = inp
        features = pair_weight
        features = features * part_weight.unsqueeze(+1).unsqueeze(+1)
        features = torch.einsum(
            'paik, pblk, at, bt -> ptil',
            features,
            pair_weight,
            self.weight_1,
            self.weight_2
        )
        features = features + self.bias_1
        features = torch.nn.functional.relu(features)
        features = self.bn_1([features, part_weight])

        short_cut = features
        features = features * part_weight.unsqueeze(+1).unsqueeze(+1)
        features = torch.einsum(
            'ptil, pcjl, ct -> ptij',
            features,
            pair_weight,
            self.weight_3,
        )
        features = features + self.bias_2
        features = torch.nn.functional.relu(features)
        features = features + short_cut
        features = self.bn_2([features, part_weight])

        features = features * part_weight.unsqueeze(+1).unsqueeze(+1)
        short_cut = torch.sum(
            features,
            dim=-1
        )
        features = torch.einsum(
            'ptij, pdij, dt -> pti',
            features,
            pair_weight,
            self.weight_4,
        )
        features = features + self.bias_3
        features = torch.nn.functional.relu(features)
        features = features + short_cut

        jet_features = torch.sum(
            features * part_weight.unsqueeze(+1),
            dim=-1
        )
        return jet_features


class Triangle(nn.Module):
    """
        i
        O
       / \
    j O---O k

    \sum_{i} z_{i} ReLU(\sum_{j} z_{j} P_{ij} ReLU(\sum_{k} z_{k} P_{jk} P_{ik}) )

    """

    def __init__(self, n_terms, n_channels, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.n_terms = n_terms
        self.n_channels = n_channels

        self.weight_1 = nn.Parameter(torch.rand((n_terms, n_channels), **factory_kwargs))
        torch.nn.init.kaiming_uniform_(self.weight_1, mode='fan_in', nonlinearity='relu')
        self.weight_2 = nn.Parameter(torch.rand((n_terms, n_channels), **factory_kwargs))
        torch.nn.init.kaiming_uniform_(self.weight_2, mode='fan_in', nonlinearity='relu')
        self.weight_3 = nn.Parameter(torch.rand((n_terms, n_channels), **factory_kwargs))
        torch.nn.init.kaiming_uniform_(self.weight_3, mode='fan_in', nonlinearity='relu')

        self.bias_1 = nn.Parameter(torch.zeros((1, n_channels, 1, 1), **factory_kwargs))
        self.bias_2 = nn.Parameter(torch.zeros((1, n_channels, 1), **factory_kwargs))

        self.bn_1 = WeightedBatchNorm(n_channels=n_channels, n_dim=2)

    def forward(self, inp):
        part_weight, pair_weight = inp  # z_{i}, p_{ij}

        features = pair_weight
        features = features * part_weight.unsqueeze(+1).unsqueeze(+1)
        features = torch.einsum(
            'pajk, pbik, at, bt -> ptij',
            features,
            pair_weight,
            self.weight_1,
            self.weight_2
        )
        features = features + self.bias_1
        features = torch.nn.functional.relu(features)

        features = self.bn_1([features, part_weight])
        features = features * part_weight.unsqueeze(+1).unsqueeze(+1)
        short_cut = torch.sum(
            features,
            dim=-1
        )
        features = torch.einsum(
            'ptij, paij, at -> pti',
            features,
            pair_weight,
            self.weight_3
        )
        features = features + self.bias_2
        features = torch.nn.functional.relu(features)
        features = features + short_cut

        jet_features = torch.sum(
            features * part_weight.unsqueeze(+1),
            dim=-1
        )
        return jet_features


class Path(nn.Module):
    """
    O--O--O--O-...
    i--j--k--l-...
    \sum_{i}z_{i} ReLU(\sum_{j}z_{j} P_{ij} ReLU(\sum_{k}z_{k} P_{kl}(...) ) )
    """

    def __init__(self, n_terms, n_channels, n_point, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        assert n_point >= 2
        self.n_terms = n_terms
        self.n_channels = n_channels
        self.n_point = n_point

        self.weight_list = nn.ParameterList([
            nn.Parameter(torch.rand((n_terms, n_channels), **factory_kwargs))
            for _ in range(n_point - 1)
        ])
        self.bias_list = nn.ParameterList([
            nn.Parameter(torch.zeros((n_channels, 1), **factory_kwargs))
            for _ in range(n_point - 1)
        ])
        for param in self.weight_list:
            torch.nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')

        self.bn_list = nn.ModuleList([
            WeightedBatchNorm(n_channels=n_channels, n_dim=1)
            for _ in range(n_point - 2)
        ])

    def forward(self, inp):
        part_weight, pair_weight = inp
        features = torch.einsum(
            'pj, paij, at -> pti',
            part_weight,
            pair_weight,
            self.weight_list[0]
        )
        features = features + self.bias_list[0]
        features = torch.nn.functional.relu(features)

        for i in range(1, self.n_point - 1):
            short_cut = features
            features = self.bn_list[i - 1]([features, part_weight])
            features = features * part_weight.unsqueeze(+1)
            features = torch.einsum(
                'ptj, paij, at -> pti',
                features,
                pair_weight,
                self.weight_list[i]
            )
            features = features + self.bias_list[i]
            features = torch.nn.functional.relu(features)
            features = features + short_cut

        jet_features = torch.sum(
            features * part_weight.unsqueeze(+1),
            dim=-1
        )
        return jet_features


