import fastjet
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


def read_dataset(dataset_source):
    """
    read dataset and return jet_kin, jet_size, jet_label
    here we use some code from
    https://github.com/jet-universe/particle_transformer/blob/main/utils/convert_top_datasets.py
    to read
    https://zenodo.org/record/2603256 Top Quark Tagging Reference Dataset
    """
    df = pd.read_hdf(dataset_source, key='table')
    df = df.iloc[0:-1]

    def _col_list(prefix, max_particles=200):
        return ['%s_%d' % (prefix, i) for i in range(max_particles)]

    _px = df[_col_list('PX')].values.astype(np.float64)
    _py = df[_col_list('PY')].values.astype(np.float64)
    _pz = df[_col_list('PZ')].values.astype(np.float64)
    _e = df[_col_list('E')].values.astype(np.float64)

    _label = df['is_signal_new'].values
    mask = _e > 0
    _n_particles = np.sum(mask, axis=1)

    return np.stack([_px, _py, _pz, _e], axis=-1), _n_particles, _label


def convert_top_dataset(_jets, _jets_size, _jets_label):
    """
    input variable length jet.constituents.px, py, pz and e
    return max_length jet.constituents.pt, deta, dphi
    we use exclusive-kt algorithm re-cluster jet (size > max_length) to max_length subjets,
    """
    min_length = 5
    max_length = 64
    jet_radiu = 0.8

    num_jets = _jets_size.shape[0]
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, jet_radiu)
    pbar = tqdm(total=num_jets)
    jets_list = []
    jets_size = []
    jets_label = []
    for i in range(num_jets):
        if _jets_size[i] <= min_length:
            pass
        else:
            array = [fastjet.PseudoJet(_jets[i][j][0],
                                       _jets[i][j][1],
                                       _jets[i][j][2],
                                       _jets[i][j][3])
                     for j in range(_jets_size[i])]
            cluster = fastjet.ClusterSequence(array, jetdef=jetdef)
            inc_jets = cluster.inclusive_jets()
            if len(inc_jets) != 1:
                print(len(inc_jets[1].constituents()))
            inc_jet = inc_jets[0]
            inc_jet_pt = inc_jet.pt()
            inc_jet_eta = inc_jet.eta()
            inc_subjets = cluster.exclusive_subjets_up_to(inc_jet, max_length)
            jet_size = len(inc_subjets)
            jet_tensor = torch.tensor(data=np.pad(np.array([[s.pt() / inc_jet_pt,
                                                             s.eta() - inc_jet_eta,
                                                             s.delta_phi_to(inc_jet)]
                                                            for s in inc_subjets]),
                                                  pad_width=((0, max_length - jet_size), (0, 0)),
                                                  constant_values=0.0),
                                      dtype=torch.float32)
            jets_list.append(jet_tensor)
            jets_size.append(jet_size)
            jets_label.append(_jets_label[i])
            pbar.update(1)
    jets = torch.stack(jets_list, dim=0)
    jets_size = torch.tensor(jets_size, dtype=torch.int32)
    jets_label = torch.tensor(jets_label, dtype=torch.int8)
    return jets, jets_size, jets_label


def save_top_dataset(dataset_source, dataset_target):
    jets, jets_size, jets_label = read_dataset(dataset_source)
    jets, jets_size, jets_label = convert_top_dataset(jets, jets_size, jets_label)
    torch.save({
        "jet": jets,
        'size': jets_size,
        'label': jets_label
    }, dataset_target)


save_top_dataset("../dataset_download/Top/test.h5",
                 "../Top/test/test.pt")
