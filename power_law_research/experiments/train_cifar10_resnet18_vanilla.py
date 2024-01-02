from datetime import datetime

import pytorch_lightning as pl
import torch
from logzero import logger
from pl_bolts.datamodules.cifar10_datamodule import CIFAR10DataModule
from pl_bolts.models.autoencoders import VAE

# VAE の Implementation: https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py


if __name__ == "__main__":
    trials = [_ for _ in range(80)]

    total = len(trials)
    for trial in trials:
        experiment_idx = trial
        logger.info(f"[{experiment_idx+1} / {total}]: on trial {trial}")
        log_dir_base = (
            f"../log/scale_free_vae/gamma_zero_vanilla_vae/trial={trial}/gamma=0"
        )
        tensorboard_dir = f"{log_dir_base}/tb"
        csv_dir = f"{log_dir_base}/csv"
        tl_logger = pl.loggers.TensorBoardLogger(
            tensorboard_dir, default_hp_metric=False
        )
        csv_logger = pl.loggers.CSVLogger(csv_dir)
        data_module = CIFAR10DataModule(num_workers=16, data_dir="../data")

        model = VAE(data_module.dims[-1])
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=[2],
            logger=[tl_logger, csv_logger],
            enable_progress_bar=True,
        )
        trainer.fit(model, data_module)
