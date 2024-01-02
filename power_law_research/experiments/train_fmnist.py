from datetime import datetime

import pytorch_lightning as pl
import torch
from logzero import logger
from pl_bolts.datamodules.fashion_mnist_datamodule import FashionMNISTDataModule
from pl_bolts.models.autoencoders import VAE
from torch import nn

# VAE の Implementation: https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py


class Encoder(nn.Module):
    def __init__(self, dims: tuple, enc_out_dim: int):
        super().__init__()
        self.channels, self.width, self.height = dims
        self.input_dim = self.channels * self.width * self.height
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, enc_out_dim),
            nn.ReLU(),
        )

    def forward(self, input):
        batch_size, channels, width, height = input.size()
        x = input.view(batch_size, -1)
        x = self.net(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dims, latent_dim: int, enc_out_dim: int):
        super().__init__()
        self.channels, self.width, self.height = dims
        self.input_dim = self.channels * self.width * self.height
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, enc_out_dim),
            nn.ReLU(),
            nn.Linear(enc_out_dim, self.input_dim),
            nn.Sigmoid(),
        )

    def forward(self, input):
        batch_size, latent_dim = input.size()
        x = input.view(batch_size, -1)
        x = self.net(x)
        x = x.reshape(batch_size, self.channels, self.width, self.height)
        return x


class ScaleFreeVAE(VAE):
    def __init__(
        self,
        input_height: int,
        power_law_gamma=1.0,
        clamp_dim=None,
        enc_type: str = "resnet18",  # dummy
        first_conv: bool = False,  # dummy
        maxpool1: bool = False,  # dummy
        enc_out_dim: int = 786,
        kl_coeff: float = 0.1,
        latent_dim: int = 786,
        lr: float = 1e-4,
        **kwargs,
    ) -> None:
        super().__init__(
            input_height,
            enc_type,
            first_conv,
            maxpool1,
            enc_out_dim,
            kl_coeff,
            latent_dim,
            lr,
            **kwargs,
        )

        self.save_hyperparameters()
        self.power_law_gamma = power_law_gamma

        self.clamp_dim = clamp_dim
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.input_dim = self.input_height**2

        self.encoder = Encoder((1, input_height, input_height), enc_out_dim)
        self.decoder = Decoder((1, input_height, input_height), latent_dim, enc_out_dim)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        device = mu.device
        dtype = mu.dtype

        if not self.clamp_dim:
            var_decay_1d = torch.pow(
                torch.arange(1, self.latent_dim + 1, dtype=dtype, device=device),
                -self.power_law_gamma,
            )
        else:
            var_decay_1d = torch.pow(
                torch.clamp(
                    torch.arange(1, self.latent_dim + 1, dtype=dtype, device=device),
                    max=self.clip_dim,
                )
                / self.latent_dim,
                -self.power_law_gamma,
            )
        var_decay = var_decay_1d.unsqueeze(0).expand(mu.shape[0], -1)
        p = torch.distributions.Normal(
            torch.zeros_like(mu), torch.ones_like(std) * var_decay
        )
        q = torch.distributions.Normal(mu, std * var_decay)
        z = q.rsample()  # z = mu + std * epsilon

        return p, q, z


if __name__ == "__main__":
    trials = [_ for _ in range(20)]
    gammas = [0.1 * i for i in range(20)]

    total = len(trials) * len(gammas)

    for trial in trials:
        for i, gamma in enumerate(gammas):
            experiment_idx = trial * len(gammas) + i
            logger.info(
                f"[{experiment_idx+1} / {total}]: gamma={gamma} on trial {trial}"
            )

            log_dir_base = (
                f"../log/fmnist/gamma_search_sgd_all_786/trial={trial}/gamma={gamma}"
            )
            tensorboard_dir = f"{log_dir_base}/tb"
            csv_dir = f"{log_dir_base}/csv"
            tl_logger = pl.loggers.TensorBoardLogger(
                tensorboard_dir, default_hp_metric=False
            )
            csv_logger = pl.loggers.CSVLogger(csv_dir)
            data_module = FashionMNISTDataModule(
                num_workers=16,
                data_dir="../data",
            )

            model = ScaleFreeVAE(
                data_module.dims[-1],
            )
            trainer = pl.Trainer(
                max_epochs=1,
                accelerator="gpu",
                devices=[2],
                logger=[tl_logger, csv_logger],
                enable_progress_bar=True,
            )
            trainer.fit(model, data_module)
