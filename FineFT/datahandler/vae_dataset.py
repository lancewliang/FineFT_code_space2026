import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class One_Dim_Dataset(Dataset):
    def __init__(self, dataset_path):
        numpy_data = np.load(dataset_path)
        self.data = torch.from_numpy(numpy_data).float()
        self.input_dim = numpy_data[0].shape[
            0
        ]  # assume the data shape is (NSamples, Sample_Length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
