import pytorch_lightning as pl
import torch
from torchvision import datasets, transforms


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="/home/shinzato/GitHub/rbm/data/FashionMNIST", batch_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_datasets = datasets.FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)]),
        )
        self.test_datasets = datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)]),
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_datasets, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_datasets,
            batch_size=self.batch_size,
            num_wokers=4,
        )


class WhitenedFashionMNISTDataModule(pl.LightningDataModule):
    # WIP
    # What is the bast way to white the dataset?
    # transforms.LinearTransformation(P, mu) will smart, but how compute the matrix?
    def __init__(self, data_dir="/home/shinzato/GitHub/rbm/data/FashionMNIST", batch_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_datasets = datasets.FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)]),
        )
        self.test_datasets = datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)]),
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_datasets, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_datasets,
            batch_size=self.batch_size,
            num_wokers=4,
        )
