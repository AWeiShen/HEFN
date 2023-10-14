import torch
from torch import nn


# part_mask may not be needed

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
        features, part_weight, part_mask = inp
        part_weight = part_weight.unsqueeze(+1)
        part_mask = part_mask.unsqueeze(+1)
        features_contract = features
        for i in range(self.n_dim):
            features_contract = torch.sum(features_contract * part_weight.reshape(
                part_weight.shape + (1,) * (self.n_dim - i - 1)
            ) * part_mask.reshape(
                part_mask.shape + (1,) * (self.n_dim - i - 1)
            ),
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

        self.weight_1 = nn.Parameter(torch.rand((n_terms, n_channels), **factory_kwargs), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.weight_1, mode='fan_in', nonlinearity='relu')
        self.weight_2 = nn.Parameter(torch.rand((n_terms, n_channels), **factory_kwargs), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.weight_2, mode='fan_in', nonlinearity='relu')
        self.weight_3 = nn.Parameter(torch.rand((n_terms, n_channels), **factory_kwargs), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.weight_3, mode='fan_in', nonlinearity='relu')
        self.weight_4 = nn.Parameter(torch.rand((n_terms, n_channels), **factory_kwargs), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.weight_4, mode='fan_in', nonlinearity='relu')

        self.bias_1 = nn.Parameter(torch.zeros((1, n_channels, 1, 1), **factory_kwargs), requires_grad=True)
        self.bias_2 = nn.Parameter(torch.zeros((1, n_channels, 1, 1), **factory_kwargs), requires_grad=True)
        self.bias_3 = nn.Parameter(torch.zeros((1, n_channels, 1), **factory_kwargs), requires_grad=True)

        self.bn_1 = WeightedBatchNorm(n_channels=n_channels, n_dim=2)
        self.bn_2 = WeightedBatchNorm(n_channels=n_channels, n_dim=2)

    def forward(self, inp):
        part_weight, part_mask, pair_weight = inp
        features = pair_weight
        features = features * part_weight.unsqueeze(+1).unsqueeze(+1) * part_mask.unsqueeze(+1).unsqueeze(+1)
        features = torch.einsum(
            'paik, pblk, at, bt -> ptil',
            features,
            pair_weight,
            self.weight_1,
            self.weight_2
        )
        features = features + self.bias_1
        features = torch.nn.functional.relu(features)

        features = self.bn_1([features, part_weight, part_mask])

        short_cut = features
        features = features * part_weight.unsqueeze(+1).unsqueeze(+1) * part_mask.unsqueeze(+1).unsqueeze(+1)
        features = torch.einsum(
            'ptil, pcjl, ct -> ptij',
            features,
            pair_weight,
            self.weight_3,
        )
        features = features + self.bias_2
        features = torch.nn.functional.relu(features)
        features = features + short_cut
        features = self.bn_2([features, part_weight, part_mask])

        features = features * part_weight.unsqueeze(+1).unsqueeze(+1) * part_mask.unsqueeze(+1).unsqueeze(+1)
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
            features * part_weight.unsqueeze(+1) * part_mask.unsqueeze(+1),
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
        part_weight, part_mask, pair_weight = inp

        features = pair_weight
        features = features * part_weight.unsqueeze(+1).unsqueeze(+1) * part_mask.unsqueeze(+1).unsqueeze(+1)
        features = torch.einsum(
            'pajk, pbik, at, bt -> ptij',
            features,
            pair_weight,
            self.weight_1,
            self.weight_2
        )
        features = features + self.bias_1
        features = torch.nn.functional.relu(features)

        features = self.bn_1([features, part_weight, part_mask])
        features = features * part_weight.unsqueeze(+1).unsqueeze(+1) * part_mask.unsqueeze(+1).unsqueeze(+1)
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
            features * part_weight.unsqueeze(+1) * part_mask.unsqueeze(+1),
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

        self.weight = nn.Parameter(torch.rand((n_terms, n_channels), **factory_kwargs), requires_grad=True)
        self.higher_point = nn.ModuleList([
            HigherPointV2(
                n_dim=1,
                n_index=1,
                n_terms=n_terms,
                n_channels=n_channels
            )
            for _ in range(n_point - 2)
        ])
        self.bn_list = nn.ModuleList([
            WeightedBatchNorm(n_channels=n_channels, n_dim=1)
            for _ in range(n_point - 2)
        ])

    def forward(self, inp):
        part_weight, part_mask, pair_weight = inp

        features = pair_weight * part_weight.unsqueeze(+1).unsqueeze(+1) * part_mask.unsqueeze(+1).unsqueeze(+1)
        features = torch.sum(features, dim=-1)
        features = torch.einsum('pzi, zv -> pvi', features, self.weight)
        for i in range(self.n_point - 2):
            short_cut = features
            features = self.bn_list[i]([features, part_weight, part_mask])
            features = features * part_weight.unsqueeze(+1) * part_mask.unsqueeze(+1)
            features = self.higher_point[i]([features, pair_weight])
            features = features + short_cut
        jet_features = torch.sum(
            features * part_weight.unsqueeze(+1) * part_mask.unsqueeze(+1),
            dim=-1
        )

        return jet_features


class HigherPointV1(nn.Module):
    """
    input weighted p^{z}_{ijkl..,x}
    output p^{z}_{ijkl...}
    """

    def __init__(self, n_dim, n_index, n_channels, n_terms, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        assert n_dim > 0
        assert n_index > 0
        super().__init__()
        subscript_characters = 'ijklrst'
        superscript_characters = 'abcdefg'
        self.n_dim = n_dim
        self.n_index = n_index
        self.n_terms = n_terms
        self.n_channels = n_channels
        self.weight_list = nn.ParameterList([
            nn.Parameter(torch.rand((n_terms, n_channels), **factory_kwargs), requires_grad=True)
            for _ in range(n_index)
        ])
        for para in self.weight_list:
            torch.nn.init.kaiming_uniform_(para, mode='fan_in', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros((n_channels,) + (1,) * (n_dim - 1)))
        self.einsum_string = ('pz' + subscript_characters[:n_dim - 1] + 'x'
                              + ', '
                              + ', '.join(['p' + superscript_characters[i] + subscript_characters[i] + 'x'
                                           for i in range(n_index)
                                           ])
                              + ', '
                              + ', '.join([superscript_characters[i] + 'z'
                                           for i in range(n_index)
                                           ])
                              + ' -> '
                              + 'pz' + subscript_characters[:n_dim - 1]
                              )

    def forward(self, inp):
        features_weighted, pair_features = inp
        # features_weighted.shape = (B, C, M, ..., M),
        # the numbers of M = n_dim,
        # the last index are weighted,
        # C = n_channels,
        # pair_features.shape = (B, T, M, M), T = n_terms
        # \sum_{i_{n_dim}} z_{i_{n_dim}} p_{i_{1}i_{2}...i_{n_dim}}
        # * P_{i_{1}i_{n_dim}}...P_{i_{n_index}i_{n_dim}}
        features = torch.einsum(
            self.einsum_string,
            (features_weighted,)
            + tuple([pair_features for _ in range(self.n_index)])
            + tuple(self.weight_list)
        )
        features = features + self.bias
        features = torch.nn.functional.relu(features)
        return features


class HigherPointV3(nn.Module):
    """
    input weighted p^{z}_{ijkl..,x}
    output p^{v}_{ijkl...}
    """

    def __init__(self, n_dim, n_index, in_channels, out_channels, n_terms, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        assert n_dim > 0
        assert n_index > 0
        subscript_characters = 'ijklrst'
        superscript_characters = 'abcdefg'
        self.n_dim = n_dim
        self.n_index = n_index
        self.n_terms = n_terms
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_list = nn.ParameterList([
            nn.Parameter(torch.rand((n_terms, in_channels, out_channels), **factory_kwargs), requires_grad=True)
            for _ in range(n_index)
        ])
        for para in self.weight_list:
            torch.nn.init.kaiming_uniform_(para, mode='fan_in', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros((out_channels,) + (1,) * (n_dim - 1)))
        self.einsum_string = ('pz' + subscript_characters[:n_dim - 1] + 'x'
                              + ', '
                              + ', '.join(['p' + superscript_characters[i] + subscript_characters[i] + 'x'
                                           for i in range(n_index)
                                           ])
                              + ', '
                              + ', '.join([superscript_characters[i] + 'zv'
                                           for i in range(n_index)
                                           ])
                              + ' -> '
                              + 'pv' + subscript_characters[:n_dim - 1]
                              )

    def forward(self, inp):
        features_weighted, pair_features = inp
        # features_weighted.shape = (B, C, M, ..., M),
        # the numbers of M = n_dim,
        # the last index are weighted,
        # C = n_channels,
        # pair_features.shape = (B, T, M, M), T = n_terms
        # \sum_{i_{n_dim}} z_{i_{n_dim}} p_{i_{1}i_{2}...i_{n_dim}}
        # * P_{i_{1}i_{n_dim}}...P_{i_{n_index}i_{n_dim}}
        features = torch.einsum(
            self.einsum_string,
            (features_weighted,)
            + tuple([pair_features for _ in range(self.n_index)])
            + tuple(self.weight_list)
        )
        features = features + self.bias
        features = torch.nn.functional.relu(features)
        return features


class HigherPointV2(nn.Module):
    """
        input weighted p^{z}_{ijkl..,x}
        output p^{z}_{ijkl...y}
        """

    def __init__(self, n_dim, n_index, n_channels, n_terms, device=None, dtype=None):
        assert n_dim > 0
        assert n_index > 0
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        subscript_characters = 'ijklrst'
        superscript_characters = 'abcdefg'
        self.n_dim = n_dim
        self.n_index = n_index
        self.n_terms = n_terms
        self.n_channels = n_channels
        self.weight_list = nn.ParameterList([
            nn.Parameter(torch.rand((n_terms, n_channels), **factory_kwargs), requires_grad=True)
            for _ in range(n_index + 1)
        ])
        for para in self.weight_list:
            torch.nn.init.kaiming_uniform_(para, mode='fan_in', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros((n_channels,) + (1,) * n_dim))
        self.einsum_string = ('pz' + subscript_characters[:n_dim - 1] + 'x'
                              + ', '
                              + ', '.join(['p' + superscript_characters[i] + subscript_characters[i] + 'x'
                                           for i in range(n_index)
                                           ])
                              + ', '
                              + 'phyx'
                              + ', '
                              + ', '.join([superscript_characters[i] + 'z'
                                           for i in range(n_index)
                                           ])
                              + ', '
                              + 'hz'
                              + ' -> '
                              + 'pz' + subscript_characters[:n_dim - 1] + 'y'
                              )

    def forward(self, inp):
        features_weighted, pair_features = inp
        # features_weighted.shape = (B, C, M, ..., M),
        # the numbers of M = n_dim,
        # the last index are weighted,
        # C = n_channels,
        # pair_features.shape = (B, T, M, M), T = n_terms
        # \sum_{i_{n_dim}} z_{i_{n_dim}} p_{i_{1}i_{2}...i_{n_dim}}
        #  * P_{i_{1}i_{n_dim}}...P_{i_{n_index}i_{n_dim}} P_{i_{n_index}i_{new}}
        features = torch.einsum(
            self.einsum_string,
            (features_weighted,)
            + tuple([pair_features for _ in range(self.n_index + 1)])
            + tuple(self.weight_list)
        )
        features = features + self.bias
        features = torch.nn.functional.relu(features)
        return features


class HigherPointV4(nn.Module):
    """
        input weighted p^{z}_{ijkl..,x}
        output p^{v}_{ijkl...y}
        """

    def __init__(self, n_dim, n_index, in_channels, out_channels, n_terms, device=None, dtype=None):
        assert n_dim > 0
        assert n_index > 0
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        subscript_characters = 'ijklrst'
        superscript_characters = 'abcdefg'
        self.n_dim = n_dim
        self.n_index = n_index
        self.n_terms = n_terms
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_list = nn.ParameterList([
            nn.Parameter(torch.rand((n_terms, in_channels, out_channels), **factory_kwargs), requires_grad=True)
            for _ in range(n_index + 1)
        ])
        for para in self.weight_list:
            torch.nn.init.kaiming_uniform_(para, mode='fan_in', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros((out_channels,) + (1,) * n_dim))
        self.einsum_string = ('pz' + subscript_characters[:n_dim - 1] + 'x'
                              + ', '
                              + ', '.join(['p' + superscript_characters[i] + subscript_characters[i] + 'x'
                                           for i in range(n_index)
                                           ])
                              + ', '
                              + 'phyx'
                              + ', '
                              + ', '.join([superscript_characters[i] + 'zv'
                                           for i in range(n_index)
                                           ])
                              + ', '
                              + 'hzv'
                              + ' -> '
                              + 'pv' + subscript_characters[:n_dim - 1] + 'y'
                              )

    def forward(self, inp):
        features_weighted, pair_features = inp
        # features_weighted.shape = (B, C, M, ..., M),
        # the numbers of M = n_dim,
        # the last index are weighted,
        # C = n_channels,
        # pair_features.shape = (B, T, M, M), T = n_terms
        # \sum_{i_{n_dim}} z_{i_{n_dim}} p_{i_{1}i_{2}...i_{n_dim}}
        #  * P_{i_{1}i_{n_dim}}...P_{i_{n_index}i_{n_dim}} P_{i_{n_index}i_{new}}
        features = torch.einsum(
            self.einsum_string,
            (features_weighted,)
            + tuple([pair_features for _ in range(self.n_index + 1)])
            + tuple(self.weight_list)
        )
        features = features + self.bias
        features = torch.nn.functional.relu(features)
        return features


class JetCLS(nn.Module):
    def __init__(
            self,
            n_terms=8,
            n_channels=32
    ):
        super(JetCLS, self).__init__()
        self.n_terms = n_terms
        self.n_channels = n_channels
        # self.triangle = Triangle(n_terms=n_terms, n_channels=n_channels)
        # self.quadrangle = Quadrangle(n_terms=n_terms, n_channels=n_channels)
        self.path = Path(n_terms=n_terms, n_channels=n_channels, n_point=4)
        self.linear_classifier = nn.Linear(n_channels, 1)

    def forward(self, inp):
        part_weight, part_mask, pair_weight = inp
        jet_features = self.path([part_weight, part_mask, pair_weight])
        jet_features = torch.nn.functional.layer_norm(
            jet_features,
            normalized_shape=(self.n_channels,)
        )
        jet_features = self.linear_classifier(jet_features)
        jet = torch.nn.functional.sigmoid(jet_features)
        return jet
