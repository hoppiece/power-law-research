import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import logging

logger = logging.getLogger(__name__)

def metrics_parser(metrics_path):
    trial_str, gamma_str = metrics_path.split("/")[-6], metrics_path.split("/")[-5]
    trial = int(trial_str.replace("trial=", ""))
    gamma = float(gamma_str.replace("gamma=", ""))
    gamma = round(gamma, 3)
    return trial, gamma

def main(exp_name, epoch=0):
    logger.info("Exporting the graph...")
    logdirs = f"../log/scale_free_vae/{exp_name}/trial=*/gamma=*/csv"
    metrics_files = sorted(glob.glob(f"{logdirs}/**/metrics.csv", recursive=True))

    save_dir = f"../result_figures/{exp_name}/"
    # Create directory if not exists
    os.makedirs(save_dir, exist_ok=True)


    draw_graph(metrics_files, "val_recon_loss", epoch, save_dir)
    draw_graph(metrics_files, "val_loss", epoch, save_dir)
    draw_graph(metrics_files, "val_kl", epoch, save_dir)
    draw_graph(metrics_files, "val_wasserstein", epoch, save_dir)
    if epoch == 0:
        draw_graph(metrics_files, "initial_val_recon_loss", epoch, save_dir)
        draw_graph(metrics_files, "initial_val_kl", epoch, save_dir)


def get_value_from_csv(csv_path, column_name, epoch):
    # Get the not a nan value of the column_name at the last epoch
    try:
        df = pd.read_csv(csv_path)
        df = df[df["epoch"] == epoch]
        df = df[df[column_name].notna()]
        df =  df[column_name].values[0]
    except Exception as e:
        df = np.nan
    return df

def draw_graph(metrics_files, column_name, epoch, save_dir):
    results = dict()
    for path in metrics_files:
        trial, gamma = metrics_parser(path)
        value = get_value_from_csv(path, column_name, epoch)
        if gamma not in results.keys():
            results[gamma] = []

        results[gamma].append(value)
    
    gammas = []
    means = []
    stds = []

    for gamma, values in results.items():
        gammas.append(gamma)
        means.append(np.mean(values))
        stds.append(np.std(values))

    # Draw the graph with error bar and save it
    plt.errorbar(gammas, means, yerr=stds, fmt="o", ecolor='r')
    plt.ylabel(column_name)
    plt.grid()
    fname = f"{save_dir.rstrip('/')}/{column_name}_ep{epoch}.png"
    plt.savefig(fname)
    # clear the figure
    plt.clf()
    logger.info(f"Saved {fname}")


if __name__ == "__main__":
    import argparse

    paser = argparse.ArgumentParser()
    paser.add_argument("--exp_name", type=str, required=True)
    args = paser.parse_args()
    main(exp_name=args.exp_name)