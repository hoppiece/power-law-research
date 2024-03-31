from datetime import datetime

import pytorch_lightning as pl
import torch
from draw_and_save_log import main as draw_and_save_log
from logzero import logger
from pl_bolts.datamodules.cifar10_datamodule import CIFAR10DataModule
from pl_bolts.models.autoencoders import VAE
from synthetic_cgd_datamodule import SyntheticDataModule
from torch.nn import functional as F

# VAE の Implementation: https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py


class VAELogBeforeTraining(VAE):
    # Logging validation loss before the first epoch
    # Base class for experiments.
    def on_train_start(self):
        # 初期バリデーションチェックを実行
        self.run_initial_validation()

    def run_initial_validation(self):
        # バリデーションデータローダーを取得
        val_dataloader = self.trainer.datamodule.val_dataloader()

        # モデルが使用するデバイスを取得
        device = self.device

        # バリデーションLossの初期値を計算
        self.eval()  # モデルを評価モードに設定
        with torch.no_grad():
            for batch in val_dataloader:
                # データをモデルと同じデバイスに移動
                batch = [b.to(device) for b in batch]
                
                # self.step メソッドを使用してバリデーションステップを実行
                _, logs = self.step(batch, 0)

                # 通常のログメソッドを使用して記録
                self.log_dict({f"initial_val_{k}": v for k, v in logs.items()})

        # モデルをトレーニングモードに戻す
        self.train()


class BoltScalefreeVAE(VAELogBeforeTraining):
    # Vanilla のScale-free VAE
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
            torch.arange(1, self.latent_dim + 1, dtype=dtype, device=device),
            -self.power_law_gamma,
        )
        var_decay = var_decay_1d.unsqueeze(0).expand(mu.shape[0], -1)
        p = torch.distributions.Normal(
            torch.zeros_like(mu), torch.ones_like(std) * var_decay
        )
        q = torch.distributions.Normal(mu, std * var_decay)
        z = q.rsample()  # z = mu + std * epsilon

        return p, q, z


class BoltScalefreeClampVAE(VAELogBeforeTraining):
    def __init__(
        self,
        input_height: int,
        power_law_gamma=1.0,
        clip_dim=None,
        min_decay=1e-3,
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
        self.clip_dim = clip_dim
        self.min_decay = min_decay

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

        if self.clip_dim is not None:
            var_decay_1d = torch.pow(
                torch.clamp(
                    torch.arange(1, self.latent_dim + 1, dtype=dtype, device=device),
                    max=self.clip_dim,
                ),
                -self.power_law_gamma,
            )
            var_decay = var_decay_1d.unsqueeze(0).expand(mu.shape[0], -1)

        else:
            var_decay_1d = torch.pow(
                torch.arange(1, self.latent_dim + 1, dtype=dtype, device=device),
                -self.power_law_gamma,
            )
            var_decay_1d_clamped = torch.clamp(var_decay_1d, min=self.min_decay)
            var_decay = var_decay_1d_clamped.unsqueeze(0).expand(mu.shape[0], -1)

        p = torch.distributions.Normal(
            torch.zeros_like(mu), torch.ones_like(std) * var_decay
        )
        q = torch.distributions.Normal(mu, std * var_decay)
        # z = mu + std * epsilon
        z = q.rsample()

        return p, q, z



class BoltScalefreeClampVAENotDecayFc(VAELogBeforeTraining):
    def __init__(
        self,
        input_height: int,
        power_law_gamma=1.0,
        clip_dim=None,
        min_decay=1e-3,
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
        self.clip_dim = clip_dim
        self.min_decay = min_decay

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

        if self.clip_dim is not None:
            var_decay_1d = torch.pow(
                torch.clamp(
                    torch.arange(1, self.latent_dim + 1, dtype=dtype, device=device),
                    max=self.clip_dim,
                ),
                -self.power_law_gamma,
            )
            var_decay = var_decay_1d.unsqueeze(0).expand(mu.shape[0], -1)

        else:
            var_decay_1d = torch.pow(
                torch.arange(1, self.latent_dim + 1, dtype=dtype, device=device),
                -self.power_law_gamma,
            )
            var_decay_1d_clamped = torch.clamp(var_decay_1d, min=self.min_decay)
            var_decay = var_decay_1d_clamped.unsqueeze(0).expand(mu.shape[0], -1)

        p = torch.distributions.Normal(
            torch.zeros_like(mu), torch.ones_like(std) * var_decay
        )
        q = torch.distributions.Normal(mu, std) # Decay させない
        # z = mu + std * epsilon
        z = q.rsample()

        return p, q, z


class BoltScalefreeClampWAE(VAELogBeforeTraining):
    def __init__(
        self,
        input_height: int,
        power_law_gamma=1.0,
        clip_dim=None,
        min_decay=1e-3,
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
        self.clip_dim = clip_dim
        self.min_decay = min_decay

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
        std = torch.exp(log_var / 2).clamp(min=1e-6) # WS Loss の安定性のために最小値を設定
        device = mu.device
        dtype = mu.dtype

        if self.clip_dim is not None:
            var_decay_1d = torch.pow(
                torch.clamp(
                    torch.arange(1, self.latent_dim + 1, dtype=dtype, device=device),
                    max=self.clip_dim,
                ),
                -self.power_law_gamma,
            )
            var_decay = var_decay_1d.unsqueeze(0).expand(mu.shape[0], -1)

        else:
            var_decay_1d = torch.pow(
                torch.arange(1, self.latent_dim + 1, dtype=dtype, device=device),
                -self.power_law_gamma,
            )
            var_decay_1d_clamped = torch.clamp(var_decay_1d, min=self.min_decay)
            var_decay = var_decay_1d_clamped.unsqueeze(0).expand(mu.shape[0], -1)

        p = torch.distributions.Normal(
            torch.zeros_like(mu), torch.ones_like(std) * var_decay
        )
        q = torch.distributions.Normal(mu, std * var_decay)
        # z = mu + std * epsilon
        z = q.rsample()

        return p, q, z
    

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        # Wasserstein距離の計算
        # https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/
        wd = F.mse_loss(p.mean, q.mean, reduction="mean") + F.mse_loss(p.scale, q.scale, reduction="mean")
        wd *= self.kl_coeff  # kl_coeffはWasserstein距離の係数に再利用

        loss = wd + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "wasserstein": wd,
            "loss": loss,
        }
        return loss, logs



class BoltScalefreeOtherDecayVAE(VAELogBeforeTraining):
    def __init__(
        self,
        input_height: int,
        linear_decay_rate=None,
        exponential_decay_rate=None,
        min_decay=1e-3,
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
        self.linear_decay_rate = linear_decay_rate
        self.exponential_decay_rate = exponential_decay_rate
        self.min_decay = min_decay

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

        if self.linear_decay_rate is not None:
            var_decay_1d = torch.linspace(1, 1 - self.linear_decay_rate, self.latent_dim, dtype=dtype, device=device)
            var_decay_1d_clamped = torch.clamp(var_decay_1d, min=self.min_decay)
            var_decay = var_decay_1d_clamped.unsqueeze(0).expand(mu.shape[0], -1)

        elif self.exponential_decay_rate is not None:
            decay_rate = self.exponential_decay_rate
            exp_decay_1d = torch.exp(-decay_rate * torch.arange(self.latent_dim, dtype=dtype, device=device))
            exp_decay_1d_clamped = torch.clamp(exp_decay_1d, min=self.min_decay)
            var_decay = exp_decay_1d_clamped.unsqueeze(0).expand(mu.shape[0], -1)

        else:
            raise ValueError("Either linear_decay or exponential_decay_rate must be True")

        p = torch.distributions.Normal(
            torch.zeros_like(mu), torch.ones_like(std) * var_decay
        )
        q = torch.distributions.Normal(mu, std * var_decay)
        # z = mu + std * epsilon
        z = q.rsample()

        return p, q, z



if __name__ == "__main__":
    trials = [_ for _ in range(15, 20)]
    gammas = [0.1 * i for i in range(20)]
    total = len(trials) * len(gammas)

    exp_name = "clamp_min_decay-0.001-ep5"
    for trial in trials:
        for i, gamma in enumerate(gammas):
            experiment_idx = trial * len(gammas) + i
            logger.info(
                f"[{experiment_idx+1} / {total}]: gamma={gamma} on trial {trial}"
            )
            log_dir_base = f"../log/scale_free_vae/{exp_name}/trial={trial}/gamma={gamma}"
            tensorboard_dir = f"{log_dir_base}/tb"
            csv_dir = f"{log_dir_base}/csv"
            tl_logger = pl.loggers.TensorBoardLogger(
                tensorboard_dir, default_hp_metric=False
            )
            csv_logger = pl.loggers.CSVLogger(csv_dir)
            data_module = CIFAR10DataModule(num_workers=16, data_dir="../data")

            model = BoltScalefreeClampVAE(
                data_module.dims[-1],
                power_law_gamma=gamma,
                min_decay=0.001,
            )
            trainer = pl.Trainer(
                max_epochs=5,
                accelerator="gpu",
                devices=[3],
                logger=[tl_logger, csv_logger],
                enable_progress_bar=True,
            )
            trainer.fit(model, data_module)

    draw_and_save_log(exp_name)