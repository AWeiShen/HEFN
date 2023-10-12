import torch
from torch import nn
from loader import legendre, chebyshev
import torch.nn.functional as F


class WeightedBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        # Custom implementation of IRC-safe BN based on
        # https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py
        super(WeightedBatchNorm1d, self).__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats
        )
        self.num_features = num_features

    def forward(self, inp):
        part_features, part_weight, part_indicator = inp
        batch_size = torch.max(part_indicator) + 1
        jet_features = torch.scatter_add(
            torch.zeros(size=(batch_size, self.num_features), dtype=torch.float32, device="cuda"),
            dim=0,
            index=torch.broadcast_to(part_indicator, (-1, self.num_features)),
            src=torch.mul(part_features, part_weight)
        )
        # we use jet_features.mean and jet_features.var here to keep IRC-safe

        self._check_input_dim(part_features)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            mean = jet_features.mean(0)
            var = jet_features.var(0, unbiased=True)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        part_features = (part_features - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            part_features = part_features * self.weight + self.bias
        return part_features


class PartInitMinimal(nn.Module):
    def __init__(self, n_terms, out_channels):
        super().__init__()
        self.n_terms = n_terms
        self.out_channels = out_channels
        self.bn = WeightedBatchNorm1d(num_features=self.n_terms, affine=False)
        self.fc = nn.Linear(in_features=self.n_terms, out_features=self.out_channels)
        # trainable parameters are self.fc.weight and self.fc.bias

    def forward(self, inp):
        part_weight, pair_head, pair_tail, pair_func, part_indicator = inp
        part_features = self.bn([torch.scatter_add(
            input=torch.zeros_like(part_weight).repeat(1, self.n_terms),
            dim=0,
            index=torch.broadcast_to(pair_head, (-1, self.n_terms)),
            src=part_weight.gather(dim=0, index=pair_tail) * pair_func
        ), part_weight, part_indicator])
        part_features = self.fc(part_features)
        return part_features


class HigherPointPartMinimal(nn.Module):
    def __init__(self, n_terms, in_channels, out_channels):
        super().__init__()
        self.n_terms = n_terms
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activate = nn.ReLU()
        self.bn = WeightedBatchNorm1d(num_features=self.n_terms * self.in_channels, affine=False)
        self.weight = nn.Parameter(
            torch.rand(size=(self.n_terms, self.in_channels),
                       dtype=torch.float32,
                       device='cuda',
                       requires_grad=True)
        )
        torch.nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
        self.bias = nn.Parameter(
            torch.zeros(size=(self.in_channels,),
                        dtype=torch.float32,
                        device='cuda',
                        requires_grad=True
                        )
        )
        self.shortcut = nn.Identity() if self.in_channels == self.out_channels else nn.Linear(
            in_features=self.in_channels,
            out_features=self.out_channels
        )
        self.output = nn.Identity() if self.in_channels == self.out_channels else nn.Linear(
            in_features=self.in_channels,
            out_features=self.out_channels
        )
        # trainable parameters are self.weight and self.bias,
        # self.shortcut and self.output actually are nn.Identity() normally.

    def forward(self, inp):
        part_features, part_weight, pair_head, pair_tail, pair_func, part_indicator = inp
        short_cut = self.shortcut(part_features)
        part_features = torch.scatter_add(
            input=torch.zeros_like(part_weight).repeat(1, self.n_terms * self.in_channels),
            dim=0,
            index=torch.broadcast_to(pair_head, (-1, self.n_terms * self.in_channels)),
            src=(((self.activate(part_features) * part_weight).gather(dim=0,
                                                                               index=torch.broadcast_to(pair_tail,
                                                                                                        (-1,
                                                                                                         self.in_channels))
                                                                               )).unsqueeze(+1) * pair_func.unsqueeze(
                -1)
                 ).reshape(-1, self.n_terms * self.in_channels)
        )
        part_features = self.bn([part_features, part_weight, part_indicator])
        part_features = part_features.reshape((-1, self.n_terms, self.in_channels))
        part_features = torch.sum(part_features * self.weight, dim=+1)
        part_features = part_features + self.bias

        return self.output(part_features) + short_cut


class JetCLSMinimal(nn.Module):
    def __init__(
            self,
            n_terms=8,
            n_channels=16
    ):
        super(JetCLSMinimal, self).__init__()
        self.n_terms = n_terms
        self.n_channels = n_channels
        self.higher_point_init = PartInitMinimal(n_terms=n_terms,
                                                 out_channels=self.n_channels)
        self.higher_point_a = HigherPointPartMinimal(n_terms=n_terms,
                                                     in_channels=self.n_channels,
                                                     out_channels=self.n_channels)
        self.higher_point_b = HigherPointPartMinimal(n_terms=n_terms,
                                                     in_channels=self.n_channels,
                                                     out_channels=self.n_channels)
        self.higher_point_c = HigherPointPartMinimal(n_terms=n_terms,
                                                     in_channels=self.n_channels,
                                                     out_channels=self.n_channels)

        self.linear_classifier = nn.Linear(in_features=self.n_channels, out_features=1)

    def forward(self, inp):
        part_weight, pair_head, pair_tail, pair_rho, part_indicator = inp
        pair_func = legendre(pair_rho, n_terms=self.n_terms)
        batch_size = part_indicator.max() + 1
        part_features = self.higher_point_init(
            [part_weight, pair_head, pair_tail, pair_func, part_indicator]
        )
        part_features = self.higher_point_a(
            [part_features, part_weight, pair_head, pair_tail, pair_func, part_indicator]
        )
        part_features = self.higher_point_b(
            [part_features, part_weight, pair_head, pair_tail, pair_func, part_indicator]
        )
        part_features = self.higher_point_c(
            [part_features, part_weight, pair_head, pair_tail, pair_func, part_indicator]
        )

        jet_features = torch.scatter_add(
            torch.zeros(size=(batch_size, self.n_channels), dtype=torch.float32, device="cuda"),
            dim=0,
            index=torch.broadcast_to(part_indicator, (-1, self.n_channels)),
            src=part_features * part_weight
        )
        jet_features = F.layer_norm(jet_features, (self.n_channels,))
        jet_features = self.linear_classifier(jet_features)
        jet = F.sigmoid(jet_features)
        return jet


class PartInit(nn.Module):
    def __init__(self, n_terms, out_channels):
        super(PartInit, self).__init__()
        self.n_terms = n_terms
        self.out_channels = out_channels

        self.mlp_bn1 = WeightedBatchNorm1d(num_features=self.n_terms)
        self.mlp_relu1 = nn.ReLU(inplace=True)
        self.mlp_fc1 = nn.Linear(in_features=self.n_terms, out_features=self.out_channels, bias=False)

        self.mlp_bn2 = WeightedBatchNorm1d(num_features=self.out_channels)
        self.mlp_relu2 = nn.ReLU(inplace=True)
        self.mlp_fc2 = nn.Linear(in_features=self.out_channels, out_features=self.out_channels, bias=False)

        self.mlp_bn3 = WeightedBatchNorm1d(num_features=self.out_channels)
        self.mlp_relu3 = nn.ReLU(inplace=True)
        self.mlp_fc3 = nn.Linear(in_features=self.out_channels, out_features=self.out_channels, bias=False)

    def forward(self, inp):
        part_features, part_weight, pair_head, pair_tail, pair_func, part_indicator = inp

        part_features = self.mlp_bn1([part_features, part_weight, part_indicator])
        part_features = self.mlp_relu1(part_features)
        part_features = self.mlp_fc1(part_features)

        part_features = self.mlp_bn2([part_features, part_weight, part_indicator])
        part_features = self.mlp_relu2(part_features)
        part_features = self.mlp_fc2(part_features)

        part_features = self.mlp_bn3([part_features, part_weight, part_indicator])
        part_features = self.mlp_relu3(part_features)
        part_features = self.mlp_fc3(part_features)
        return part_features


class HigherPointPart(nn.Module):
    def __init__(self, n_terms, in_channels, mid_channels, out_channels, hidden_channels):
        super(HigherPointPart, self).__init__()
        self.n_terms = n_terms
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.shortcut = nn.Linear(in_features=self.in_channels,
                                  out_features=self.out_channels,
                                  bias=False) if self.in_channels != self.out_channels else nn.Identity()

        self.mlp_bn1 = WeightedBatchNorm1d(num_features=self.in_channels)
        self.mlp_relu1 = nn.ReLU(inplace=True)
        self.mlp_fc1 = nn.Linear(in_features=self.in_channels, out_features=self.hidden_channels, bias=False)

        self.mlp_bn2 = WeightedBatchNorm1d(num_features=self.n_terms * self.hidden_channels)
        self.mlp_relu2 = nn.ReLU(inplace=True)
        self.mlp_fc2 = nn.Linear(in_features=self.n_terms * self.hidden_channels, out_features=self.mid_channels,
                                 bias=False)

        self.mlp_bn3 = WeightedBatchNorm1d(num_features=self.mid_channels)
        self.mlp_relu3 = nn.ReLU(inplace=True)
        self.mlp_fc3 = nn.Linear(in_features=self.mid_channels, out_features=self.out_channels, bias=False)

    def forward(self, inp):
        part_features, part_weight, pair_func, pair_head, pair_tail, part_indicator = inp
        short_cut = self.shortcut(part_features)

        part_features = self.mlp_bn1([part_features, part_weight, part_indicator])
        part_features = self.mlp_relu1(part_features)
        part_features = self.mlp_fc1(part_features)

        part_features = torch.mul(part_features, part_weight)

        part_features = torch.scatter_add(
            input=torch.zeros_like(part_weight).repeat(1, self.hidden_channels * self.n_terms),
            dim=0,
            index=torch.broadcast_to(pair_head, (-1, self.hidden_channels * self.n_terms)),
            src=(part_features.gather(dim=0,
                                      index=torch.broadcast_to(pair_tail, (-1, self.hidden_channels))).unsqueeze(
                -1) * pair_func[:, :self.n_terms].unsqueeze(+1)
                 ).reshape((-1, self.n_terms * self.hidden_channels))
        )

        part_features = self.mlp_bn2([part_features, part_weight, part_indicator])
        part_features = self.mlp_relu2(part_features)
        part_features = self.mlp_fc2(part_features)
        part_features = self.mlp_bn3([part_features, part_weight, part_indicator])
        part_features = self.mlp_relu3(part_features)
        part_features = self.mlp_fc3(part_features)

        return part_features + short_cut


class JetCLS(nn.Module):
    def __init__(
            self,
            n_terms=16,
            n_point=2,
            part_dim=256,
            classes=2,
    ):
        super(JetCLS, self).__init__()
        self.n_terms = n_terms
        self.part_dim = part_dim
        self.n_point = n_point
        self.part_init = PartInit(
            n_terms=n_terms,
            out_channels=part_dim
        )

        self.blocks = nn.ModuleList([
            HigherPointPart(in_channels=part_dim,
                            hidden_channels=int(1024 / n_terms),
                            n_terms=n_terms,
                            mid_channels=256,
                            out_channels=part_dim)
            for _ in range(n_point - 2)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(in_features=part_dim, out_features=part_dim, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=part_dim, out_features=part_dim, bias=False),
            nn.ReLU(),
            nn.Linear(part_dim, classes)
        )

    def forward(self, inp):
        part_weight, pair_head, pair_tail, pair_rho, part_indicator = inp
        batch_size = torch.max(part_indicator) + 1
        pair_func = legendre(pair_rho, self.n_terms)
        part_features = torch.scatter_add(
            input=torch.zeros_like(part_weight).repeat(1, self.n_terms),
            dim=0,
            index=torch.broadcast_to(pair_head, (-1, self.n_terms)),
            src=part_weight.gather(dim=0, index=pair_tail) * pair_func
        )
        part_features = self.part_init(part_features)
        for m in self.blocks:
            part_features = m([part_features, part_weight, pair_func, pair_head, pair_tail, part_indicator])
        part_features = torch.nn.functional.relu(part_features)
        jet = torch.scatter_add(
            torch.zeros(size=(batch_size, self.part_dim), dtype=torch.float32, device="cuda"),
            dim=0,
            index=torch.broadcast_to(part_indicator, (-1, self.part_dim)),
            src=torch.mul(part_features, part_weight)
        )
        jet = torch.nn.functional.layer_norm(jet, normalized_shape=(self.part_dim,))
        return self.classifier(jet)
