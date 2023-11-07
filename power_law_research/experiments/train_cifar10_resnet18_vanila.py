from datetime import datetime

import pytorch_lightning as pl
import torch
from logzero import logger
from pl_bolts.datamodules.cifar10_datamodule import CIFAR10DataModule
from pl_bolts.models.autoencoders import VAE

# VAE の Implementation: https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py


class BoltScalefreeVAE(VAE):
    def __init__(
        self,
        input_height: int,
        power_law_gamma=1.0,
        enc_type: str = "resnet18",
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
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
        self.power_law_gamma = power_law_gamma

    def sample(self, mu, log_var):
        """
        Original:

        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.sample() # z = mu + std * eps (eps ~ N(0, 1))
        return p, q, z

        つぎの関数を書き換える
        z = q.rsample()
        はつぎの関数と等価:
        z = mu + torch.randn(mu.shape, dtype=mu.dtype, device=mu.device) * std

        Normal.rsample() の実装:
        rsample は誤差逆伝播法が可能なサンプリングを行う.内部では Reparametrization trick が行われている

        class Normal(ExponentialFamily):
            ...
            def __init__(self, loc, scale, validate_args=None):
                pass
            ...

            def rsample(self, sample_shape=torch.Size()):
                shape = self._extended_shape(sample_shape)
                eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
                return self.loc + eps * self.scale
        """
        std = torch.exp(log_var / 2)
        device = mu.device
        dtype = mu.dtype

        var_decay_1d = torch.pow(
            torch.arange(1, self.latent_dim + 1, dtype=dtype, device=device) / self.latent_dim,
            -self.power_law_gamma,
        )
        var_decay = var_decay_1d.unsqueeze(0).expand(mu.shape[0], -1)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std) * var_decay)
        q = torch.distributions.Normal(mu, std * var_decay)
        z = q.rsample()  # z = mu + std * epsilon

        return p, q, z


class BoltScalefreeClampVAE(VAE):
    def __init__(
        self,
        input_height: int,
        power_law_gamma=1.0,
        clip_dim=100,
        enc_type: str = "resnet18",
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
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
        self.power_law_gamma = power_law_gamma
        self.clip_dim = 100

    def sample(self, mu, log_var):
        # p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        # q = torch.distributions.Normal(mu, std)
        """
        つぎの関数を書き換える
        z = q.rsample()
        はつぎの関数と等価:
        z = mu + torch.randn(mu.shape, dtype=mu.dtype, device=mu.device) * std

        Normal.rsample() の実装:
        rsample は誤差逆伝播法が可能なサンプリングを行う.内部では Reparametrization trick が行われている

        class Normal(ExponentialFamily):
            ...
            def __init__(self, loc, scale, validate_args=None):
                pass
            ...

            def rsample(self, sample_shape=torch.Size()):
                shape = self._extended_shape(sample_shape)
                eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
                return self.loc + eps * self.scale
        """
        std = torch.exp(log_var / 2)
        device = mu.device
        dtype = mu.dtype

        var_decay_1d = torch.pow(
            torch.clamp(
                torch.arange(1, self.latent_dim + 1, dtype=dtype, device=device), max=self.clip_dim
            ),
            -self.power_law_gamma,
        )
        var_decay = var_decay_1d.unsqueeze(0).expand(mu.shape[0], -1)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std) * var_decay)
        q = torch.distributions.Normal(mu, std * var_decay)
        # z = mu + std * epsilon
        z = q.rsample()

        return p, q, z


if __name__ == "__main__":
    trials = [_ for _ in range(10)]
    gammas = [0.1 * i for i in range(20)]

    total = len(trials) * len(gammas)
    for trial in trials:
        for i, gamma in enumerate(gammas):
            experiment_idx = trial * len(gammas) + i
            logger.info(f"[{experiment_idx+1} / {total}]: gamma={gamma} on trial {trial}")
            tensorboard_dir = (
                f"../log/scale_free_vae/gamma_search_fix1/trial={trial}/gamma={gamma}/tb"
            )
            csv_dir = f"../log/scale_free_vae/gamma_search_fix1/trial={trial}/gamma={gamma}/csv"
            tl_logger = pl.loggers.TensorBoardLogger(tensorboard_dir, default_hp_metric=False)
            csv_logger = pl.loggers.CSVLogger(csv_dir)
            data_module = CIFAR10DataModule(num_workers=16, data_dir="../data")

            model = BoltScalefreeVAE(data_module.dims[-1], power_law_gamma=gamma)
            trainer = pl.Trainer(
                max_epochs=2,
                accelerator="gpu",
                devices=[3],
                logger=[tl_logger, csv_logger],
                enable_progress_bar=True,
            )
            trainer.fit(model, data_module)
