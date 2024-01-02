import sys

sys.path.append("../")
import itertools
import logging
from datetime import datetime
from string import Template

import click
import pytorch_lightning as pl
from data_modules import FashionMNISTDataModule
from logzero import logger
from models import LitScaleFreeVAE, LitVanillaVAE
from tqdm import tqdm
from utils import pca_variance, sample_neuron_firings


@click.group()
def cli():
    pass


def gammas():
    n_sample = 20
    N = 100
    delta_gamma = 0.5
    gammas = list(
        itertools.chain.from_iterable(
            [[(delta_gamma * i, j) for j in range(n_sample)] for i in range(N)]
        )
    )
    # [(0.1, 1), (0.1, 2), ...]
    return gammas


@cli.command()
def compile():
    # cat search_gamma.sh | parallel --bar -j 8 {}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    bash_str = "#!/bin/bash\n"

    for item, rank in zip(gammas(), itertools.cycle([0, 1, 2, 3])):
        gamma, j = item
        cmd = f"python search_gamma.py try-gamma \
--gamma {gamma} -j {j} --name {timestamp} --rank {rank}\n"
        bash_str += cmd

    with open("search_gamma.sh", "w") as fp:
        fp.write(bash_str)


@cli.command()
@click.option("--gamma", type=float)
@click.option("-j", type=int)
@click.option("--name", "experiment_name")
@click.option("--rank", type=int)
def try_gamma(gamma: float, j: int, experiment_name: str, rank: int):
    log_frequencey = 5
    log_base = "../log/exp01/gamma_sample_prod"
    tensorboard_dir = f"{log_base}/tensorboard/{experiment_name}"
    csv_dir = f"{log_base}/csv/{experiment_name}"

    model = LitScaleFreeVAE(n_vis=784, power_law_gamma=gamma)
    tensorboard_logger = pl.loggers.TensorBoardLogger(
        save_dir=tensorboard_dir,
        name=f"gamma={gamma}_sample={j}",
        default_hp_metric=False,
    )
    csv_logger = pl.loggers.CSVLogger(save_dir=csv_dir, name=f"gamma={gamma}_sample={j}")
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    data_module = FashionMNISTDataModule(batch_size=512)
    trainer = pl.Trainer(
        max_epochs=8,
        devices=[rank],
        accelerator="gpu",
        logger=[tensorboard_logger, csv_logger],
        enable_progress_bar=False,
        log_every_n_steps=log_frequencey,
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    cli()
