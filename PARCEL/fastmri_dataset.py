import random
import pathlib
import numpy as np
import h5py
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from utils import complex2real


class FastMRIData(Dataset):
    def __init__(self, data_path, u_mask_path, s_mask_up_path, s_mask_down_path, sample_rate):
        super(FastMRIData, self).__init__()
        self.data_path = data_path
        self.u_mask_path = u_mask_path
        self.s_mask_up_path = s_mask_up_path
        self.s_mask_down_path = s_mask_down_path
        self.sample_rate = sample_rate

        self.examples = []
        files = list(pathlib.Path(self.data_path).iterdir())
        if self.sample_rate < 1:
            num_examples = round(int(len(files) * self.sample_rate))
            files = files[:num_examples]
        for file in sorted(files):
            slices = int(file.name.split('_')[1])
            self.examples += [(file, slice_id) for slice_id in range(slices)]

        self.mask_under = np.array(sio.loadmat(self.u_mask_path)['mask'])
        self.s_mask_up = np.array(sio.loadmat(self.s_mask_up_path)['mask'])
        self.s_mask_down = np.array(sio.loadmat(self.s_mask_down_path)['mask'])

        self.mask_net_up = self.mask_under * self.s_mask_up
        self.mask_net_down = self.mask_under * self.s_mask_down

        self.mask_under = torch.from_numpy(self.mask_under).float()
        self.mask_net_up = torch.from_numpy(self.mask_net_up).float()
        self.mask_net_down = torch.from_numpy(self.mask_net_down).float()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        file, slice_id = self.examples[item]
        with h5py.File(file, 'r') as data:
            full_kspace = np.array(data['kspace'][slice_id, ...])
            csm = np.array(data['csm'][slice_id, ...])

        full_kspace = complex2real(full_kspace)
        csm = complex2real(csm)
        full_kspace = torch.from_numpy(full_kspace).float()
        csm = torch.from_numpy(csm).float()

        full_kspace = torch.view_as_complex(full_kspace)
        csm = torch.view_as_complex(csm)

        return full_kspace, csm, self.mask_under, self.mask_net_up, self.mask_net_down, file.name, slice_id
