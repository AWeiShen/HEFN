import torch
from torch.utils import data


def legendre(rho, n_terms):
    p0 = torch.ones_like(rho)
    p1 = rho
    pa = p0
    pb = p1
    result = [p0, p1]
    for l in range(1, n_terms - 1):
        pc = (2 * l + 1) / (l + 1) * rho * pb - l / (l + 1) * pa
        # (n+1)P_{n+1}(x) = (2n+1)xP_{n}(x) - nP_{n-1}(x)
        result.append(pc)
        pa = pb
        pb = pc
    result = torch.concat(result, dim=-1)
    return result


def chebyshev(rho, n_terms):
    p0 = torch.ones_like(rho)
    p1 = rho
    pa = p0
    pb = p1
    result = [p0, p1]
    for l in range(1, n_terms - 1):
        pc = 2 * rho * pb - pa
        # T_{n+1}(x) = 2xT_{n}(x) - T_{n-1}(x)
        result.append(pc)
        pa = pb
        pb = pc
    result = torch.concat(result, dim=-1)
    return result


class JetDataset(data.Dataset):
    def __init__(self, dataset_file):
        super(JetDataset, self).__init__()
        self.data = torch.load(dataset_file)
        self.label = self.data["label"]
        self.len = self.label.size()[0]

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.data['jet'][item], \
            self.data['size'][item], \
            self.label[item]


class JetCollater(object):
    def __init__(self, r_max=1.6):
        # r_max is the maximum distance between two particles,
        # r_max could be set as 2R0, where R0 is the radius of jet
        self.r_max = r_max

    def __call__(self, batch_data):
        batch_size = len(batch_data)
        batch_energy = []
        batch_head = []
        batch_tail = []
        batch_value = []
        batch_indicator = []
        batch_label = []
        begin_i = 0
        x_index = 0
        for batch_idx in range(batch_size):
            x_i = batch_data[batch_idx][0]  # x_i.shape = (M_padded, 3), [energy, eta, phi]
            s_i = batch_data[batch_idx][1]  # s_i is the length M of the jet, M < M_padded
            y_i = batch_data[batch_idx][2]  # label
            energy_i, head_i, tail_i, value_i = self.single_jet_preprocessing(x_i, s_i)
            head_i = begin_i + head_i
            tail_i = begin_i + tail_i
            indicator_i = torch.full_like(energy_i, fill_value=x_index, dtype=torch.int64)
            batch_energy.append(energy_i)
            batch_head.append(head_i)
            batch_tail.append(tail_i)
            batch_value.append(value_i)
            batch_indicator.append(indicator_i)
            batch_label.append(y_i)
            begin_i += s_i
            x_index += 1
        batch_energy = torch.concat(batch_energy, dim=0)
        batch_head = torch.concat(batch_head, dim=0)
        batch_tail = torch.concat(batch_tail, dim=0)
        batch_value = torch.concat(batch_value, dim=0)
        batch_indicator = torch.concat(batch_indicator, dim=0)
        batch_label = torch.stack(batch_label, dim=0).long()
        return batch_energy, batch_head.unsqueeze(-1), batch_tail.unsqueeze(-1), batch_value.unsqueeze(
            -1), batch_indicator, batch_label

    def single_jet_preprocessing(self, _kinematic, _size):
        _energy = _kinematic[:_size, 0:1]
        _energy = _energy / _energy.sum()  # energy fraction

        _pair_head, _pair_tail = torch.triu_indices(_size, _size, offset=1)
        _diag_indices = torch.arange(_size)

        _co = _kinematic[:_size, 1:3]
        _pair_ii = torch.sum(torch.square(_co), dim=-1)
        _pair_ij = torch.sum(_co[_pair_head] * _co[_pair_tail], dim=-1)
        _pair_values = torch.clamp(torch.sqrt(torch.clamp_min(_pair_ii[_pair_head]
                                                              + _pair_ii[_pair_tail]
                                                              - 2 * _pair_ij
                                                              , 1e-12)) * 2. / self.r_max - 1.,
                                   min=-0.999999,
                                   max=+0.999999
                                   )  # here we set theta_ij = R_ij * 2 / r_max - 1
        _head = torch.concat([_pair_head, _diag_indices, _pair_tail])
        _tail = torch.concat([_pair_tail, _diag_indices, _pair_head])
        _value = torch.concat([_pair_values,
                               torch.full(size=(_size,), fill_value=-0.999999),
                               _pair_values])  # theta_ij  = -1 if R_ij = 0.0

        return _energy, \
            _head, \
            _tail, \
            _value
