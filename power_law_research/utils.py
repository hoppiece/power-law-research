from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from models import EncoderDecoderModel


def sample_neuron_firings(
    model: EncoderDecoderModel, dataloader: torch.utils.data.DataLoader
) -> np.array:
    """
    入力データに対する中間ニューロンの発火状況を numpy.array で取得. 中間ニューロンは, model.encode メソッドで
    得られることを仮定しています.
    Basic usage:
        firings = sample_neuron_firings(model, FashionMNISTDataModule().train_dataloader())

    Args:
        model (nn.Module): Model to embedding vectors. model must have `encode` method to
            embedd input into hidden layers.
        dataloader (torch.utils.data.DataLoader): Input dataloader.

    Returns:
        np.array: Neuron firing matrix, shape of (#data, vec_size).
    """
    firings = []
    for data, target in dataloader:
        vv = model.encode(data.view(data.size(0), -1)).detach()
        firings.append(vv)
    firings = torch.cat(firings).numpy()
    return firings


def pca_variance(firings: np.array) -> Tuple[np.array, np.array]:
    """
    Shape が (#data, vec_size) の ニューロン発火行列のPC次元解析をします.

    Args:
        firings (np.array): Shape is (#data, vec_size)

    Returns:
        Tuple[np.array, np.array]: Shape of pc_dim and var: (vec_size, )
    """
    pca = PCA().fit(firings)
    n_hid = firings.shape[1]
    pc_dim = np.arange(n_hid) + 1
    pc_var = pca.explained_variance_
    return pc_dim, pc_var
