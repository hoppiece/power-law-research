import abc
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class EncoderDecoderModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.training_step_outputs = []
        self.epoch_loss_dict = {
            "train_loss_epoch_ave": [],
            "train_recon_loss_epoch_ave": [],
        }

    def encode(self, tensor):
        raise NotImplementedError

    def decode(self, tensor):
        raise NotImplementedError

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def configure_optimizers(self):
        if self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError
        return optimizer

    def sample_neuron_firings(self, dataloader: torch.utils.data.DataLoader):
        """
        入力データに対する中間ニューロンの発火状況を numpy.array で取得. 中間ニューロンは, model.encode メソッドで
        得られることを仮定しています.

        Args:
            dataloader (torch.utils.data.DataLoader): Input dataloader.

        Returns:
            np.array: Neuron firing matrix, shape of (#data, vec_size).
        """
        firings = []
        for data, target in dataloader:
            vv = self.encode(data.view(data.size(0), -1)).detach()
            firings.append(vv)
        firings = torch.cat(firings).numpy()
        return firings

    def on_train_epoch_end(self):
        epoch_average_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        epoch_average_recon_loss = torch.stack(
            [x["recon_loss"] for x in self.training_step_outputs]
        ).mean()

        self.epoch_loss_dict["train_loss_epoch_ave"].append(epoch_average_loss)
        self.epoch_loss_dict["train_recon_loss_epoch_ave"].append(epoch_average_recon_loss)

        self.logger.log_hyperparams(
            self.hparams,
            {
                f"metrics/ave_loss_epoch/{self.current_epoch}": epoch_average_loss,
                f"metrics/ave_recon_loss_epoch/{self.current_epoch}": epoch_average_recon_loss,
            },
        )
        self.log(f"metrics/ave_loss_epoch/{self.current_epoch}", epoch_average_loss)
        self.log(f"metrics/ave_recon_loss_epoch/{self.current_epoch}", epoch_average_recon_loss)
        self.training_step_outputs.clear()


class LitVanillaVAE(EncoderDecoderModel):
    """
    Original Paper: "Auto-Encoding Variational Bayes" Diederik et. al., 2014 https://arxiv.org/abs/1312.6114
    Implemetnation reference: https://github.com/pytorch/examples/tree/main/vae
    """

    def __init__(
        self,
        n_vis,
        n_hid=100,
        optimizer_name="sgd",
        lr=0.01,
        normalize_loss=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.is_normalize_loss = normalize_loss

        self.set_encoder()
        self.set_decoder()

    def set_encoder(self):
        self.encoder_mean = nn.Sequential(nn.Linear(self.n_vis, self.n_hid), nn.ReLU())

        # In order to avoid numerical instability, the parameters of the encoder
        # for a variance should be initialized to zero, not Gaussian. See
        # https://stackoverflow.com/questions/49634488/keras-variational-autoencoder-nan-loss
        self.encoder_var = nn.Sequential(
            OrderedDict(
                [
                    ("enc_var_fc1", nn.Linear(self.n_vis, self.n_hid)),
                    ("enc_var_relu1", nn.ReLU()),
                ]
            )
        )
        torch.nn.init.zeros_(self.encoder_var.enc_var_fc1.weight)

    def set_decoder(self):
        self.decoder = nn.Sequential(nn.Linear(self.n_hid, self.n_vis), nn.Sigmoid())

    def _sample_hidden(self, mean, log_var):
        epsilon = torch.randn(mean.shape).to(mean.device)
        h = mean + epsilon * torch.exp(0.5 * log_var)
        return h

    def encode(self, v):
        mean, log_var = self.encoder_mean(v), self.encoder_var(v)
        h = self._sample_hidden(mean, log_var)
        return h

    def decode(self, h):
        return self.decoder(h)

    def train_loss(self, tensor):
        batch_size = tensor.size(0)
        x = tensor.view(batch_size, -1)
        mean, log_var = self.encoder_mean(x), self.encoder_var(x)
        delta = 1e-8
        KL = 0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

        z = self._sample_hidden(mean, log_var)
        x_hat = self.decode(z)

        # Reconstruction loss is binary cross entorpy
        reconstruction = torch.sum(
            x * torch.log(x_hat + delta) + (1 - x) * torch.log(1 - x_hat + delta)
        )
        # This is same as
        # -F.binary_cross_entropy(x_hat, x, reduction='sum')

        # VAE loss does not simply divide by batch: https://github.com/pytorch/examples/issues/652
        if self.is_normalize_loss:
            # https://github.com/pytorch/examples/issues/234
            # This occurs something worng behavior.
            lower_bound = KL / (batch_size * self.n_vis) + reconstruction / batch_size
        else:
            lower_bound = KL + reconstruction
        return -lower_bound

    def training_step(self, batch, batch_idx):
        tensor, label = batch
        batch_size = tensor.size(0)
        loss = self.train_loss(tensor)
        recon_loss = self.recon_loss(tensor)
        self.log("loss/train", loss)
        self.log("loss/train_batch", loss / batch_size)
        self.log("loss/recon", recon_loss)
        self.log("loss/recon_batch", recon_loss / batch_size)

        outputs = {"loss": loss, "recon_loss": loss}
        self.training_step_outputs.append(outputs)
        return outputs

    def recon_loss(self, tensor):
        batch_size = tensor.size(0)
        x = tensor.view(batch_size, -1)
        recon = self.forward(x)
        return F.mse_loss(x, recon) / batch_size


class LitScaleFreeVAE(LitVanillaVAE):
    def __init__(self, n_vis, n_hid=100, optimizer_name="sgd", lr=0.01, power_law_gamma=1):
        super().__init__(n_vis, n_hid, optimizer_name, lr)
        self.var_decay_1d = torch.tensor([n ** (-power_law_gamma) for n in range(1, n_hid + 1)])

    def _sample_hidden(self, mean, log_var):
        batch_size = mean.size(0)
        var_decay = self.var_decay_1d.repeat(batch_size, 1)
        epsilon = torch.mul(torch.randn(mean.shape), var_decay).to(mean.device)
        h = mean + epsilon * torch.exp(0.5 * log_var)
        return h


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, n_vis, n_hid=100, optimizer_name="sgd", lr=0.01):
        super().__init__()
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.encoder = nn.Sequential(nn.Linear(n_vis, n_hid), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(n_hid, n_vis), nn.Sigmoid())

    def encode(self, v):
        return self.encoder(v)

    def decode(self, h):
        return self.decoder(h)

    def train_loss(self, tensor):
        batch = tensor.size(0)
        x = tensor.view(batch, -1)
        recon = self.forward(x)
        return F.mse_loss(x, recon)

    def training_step(self, batch, batch_idx):
        tensor, label = batch
        loss = self.train_loss(tensor)
        self.log("train_loss", loss)
        return loss
