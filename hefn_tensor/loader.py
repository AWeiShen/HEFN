import torch
from torch.utils import data


class JetDataset(data.Dataset):
    def __init__(self, dataset_file, radius_max=1.6, n_terms=8, max_length=64):
        super(JetDataset, self).__init__()
        self.data = torch.load(dataset_file)
        self.radius_max = radius_max
        self.label = self.data["label"]
        self.len = self.label.size()[0]
        self.n_terms = n_terms
        self.max_length = max_length

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        part_kin = self.data['jet'][item]
        jet_size = self.data['size'][item]
        part_weight = part_kin[:, 0:1] / torch.sum(part_kin[:, 0])
        part_coord = part_kin[0:jet_size, 1:3]
        pair_indices = torch.triu_indices(jet_size, jet_size, offset=1)
        diag_indices = torch.arange(jet_size)
        pair_ii = torch.sum(torch.square(part_coord), dim=-1, keepdim=True)
        pair_ij = torch.matmul(part_coord, torch.transpose(part_coord, 1, 0))
        pair_radius_square = (pair_ii[pair_indices[0], 0]
                              + pair_ii[pair_indices[1], 0]
                              - 2 * pair_ij[pair_indices[0], pair_indices[1]]
                              )

        pair_radius = torch.sqrt(torch.clamp_min(pair_radius_square, 1e-9))
        pair_weight_sparse = pair_radius * 2. / self.radius_max - 1.
        pair_weight_dense = torch.zeros(size=(self.n_terms, self.max_length , self.max_length), dtype=torch.float32)
        p0 = torch.ones_like(pair_weight_sparse)
        p1 = pair_weight_sparse
        pa = p0
        pb = p1
        pair_weight_dense[0, pair_indices[0], pair_indices[1]] = p0
        pair_weight_dense[0, diag_indices, diag_indices] = 1
        pair_weight_dense[0, pair_indices[1], pair_indices[0]] = p0
        pair_weight_dense[1, pair_indices[0], pair_indices[1]] = p1
        pair_weight_dense[1, diag_indices, diag_indices] = 0
        pair_weight_dense[1, pair_indices[1], pair_indices[0]] = p1
        for term_idx in range(1, self.n_terms - 1):
            pc = (2 * term_idx + 1) / (term_idx + 1) * pair_weight_sparse * pb - term_idx / (term_idx + 1) * pa
            pair_weight_dense[term_idx + 1, pair_indices[0], pair_indices[1]] = pc
            pair_weight_dense[term_idx + 1, diag_indices, diag_indices] = (term_idx + 1) % 2
            pair_weight_dense[term_idx + 1, pair_indices[1], pair_indices[0]] = pc
            pa = pb
            pb = pc

        return part_weight.squeeze(-1), pair_weight_dense, self.label[item].float()

