import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torch.distributions import MultivariateNormal
import numpy as np


def generate_toeplitz(size: int, param_c: float, param_alpha: float) -> np.array:
    n, c, a = size, param_c, param_alpha
    
    i = np.arange(n, dtype=float)
    j = np.arange(n, dtype=float)
    diff = np.abs(i[:, None] - j)
    toeplitz = np.identity(n) + c * diff ** a
    return toeplitz


def generate_decay_matrix(size: int, decay_rate: float):
    i = np.arange(1, size+1, dtype=float)
    return np.diag((size / i) ** (decay_rate))
    


class SyntheticDataset(Dataset):
    def __init__(self, decay_rate=1.5 ,size=(3, 32, 32), length=50000, covariance_matrix=None):
        """
        size: データのサイズ (例: CIFAR10の場合は (3, 32, 32))
        length: データセットの長さ
        covariance_matrix: 共分散行列
        """

        self.size = size
        self.length = length
        self.decay_rate = decay_rate


        C, H, W = size
        if covariance_matrix is None:
            single_channel_cov = generate_decay_matrix(H*W, self.decay_rate)
            covariance_matrix = torch.tensor(single_channel_cov, dtype=torch.float32)
            mean = torch.zeros(H * W)


        self.distribution = MultivariateNormal(mean, covariance_matrix)
        single_channel_data = self.distribution.sample((length,))
        self.data = single_channel_data.unsqueeze(1).repeat(1, 3, 1).reshape(length, *size)
        self.labels = torch.randint(0, 10, (length,))  # 10クラス分類の場合


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SyntheticDataModule(LightningDataModule):
    def __init__(self, num_workers=None, data_dir=None, batch_size=32, size=(3, 32, 32), train_length=50000, val_length=10000, test_length=10000, decay_rate=1.5):
        super().__init__()
        self.batch_size = batch_size
        self.size = size
        self.dims = size
        self.train_length = train_length
        self.val_length = val_length
        self.test_length = test_length
        self.decay_rate = decay_rate

    def setup(self, stage=None):
        self.train_dataset = SyntheticDataset(size=self.size, length=self.train_length, decay_rate=self.decay_rate)
        self.val_dataset = SyntheticDataset(size=self.size, length=self.val_length, decay_rate=self.decay_rate)
        self.test_dataset = SyntheticDataset(size=self.size, length=self.test_length, decay_rate=self.decay_rate)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
