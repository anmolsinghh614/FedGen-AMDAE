import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import h5py
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.ticker import StrMethodFormatter
import os
from utils.model_utils import get_log_path, METRICS
import seaborn as sns
import string
import matplotlib.colors as mcolors
import os
COLORS=list(mcolors.TABLEAU_COLORS)
MARKERS=["o", "v", "s", "*", "x", "P"]

plt.rcParams.update({'font.size': 14})
n_seeds=3

def load_results(args, algorithm, seed):
    alg = get_log_path(args, algorithm, seed, args.gen_batch_size)
    hf = h5py.File("./{}/{}.h5".format(args.result_path, alg), 'r')
    metrics = {}
    for key in METRICS:
        metrics[key] = np.array(hf.get(key)[:])
    return metrics


def get_label_name(name):
    name = name.split("_")[0]
    if 'Distill' in name:
        if '-FL' in name:
            name = 'FedDistill' + r'$^+$'
        else:
            name = 'FedDistill'
    elif 'FedDF' in name:
        name = 'FedFusion'
    elif 'FedEnsemble' in name:
        name = 'Ensemble'
    elif 'FedAvg' in name:
        name = 'FedAvg'
    elif 'FedProx' in name:
        name = 'FedProx'
    return name

def plot_results(args, algorithms):
    n_seeds = args.times

    # Split once and use strings (not the list) to build paths/labels
    parts = args.dataset.split('-')
    dataset_name = parts
    subset = parts[1] if len(parts) > 1 else "default"
    sub_dir = f"{dataset_name}/{subset}"  # e.g., Mnist/ratio0.5
    os.makedirs(f"figs/{sub_dir}", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    TOP_N = 5
    global_min = np.inf
    global_max = -np.inf
    global_len = None

    for i, algorithm in enumerate(algorithms):
        algo_name = get_label_name(algorithm)
        metrics = [load_results(args, algorithm, seed) for seed in range(n_seeds)]
        curves = [np.asarray(m['glob_acc'], dtype=float) for m in metrics]

        min_len = min(len(c) for c in curves)
        curves = [c[:min_len] for c in curves]

        all_curves = np.concatenate(curves)
        global_min = min(global_min, float(np.min(all_curves)))
        global_max = max(global_max, float(np.max(all_curves)))
        global_len = min_len if global_len is None else min(global_len, min_len)

        top_accs = np.concatenate([np.sort(c)[-TOP_N:] for c in curves])
        acc_avg = np.mean(top_accs)
        acc_std = np.std(top_accs)
        info = 'Algorithm: {:<10s}, Accuracy = {:.2f} %, deviation = {:.2f}'.format(
            algo_name, acc_avg * 100, acc_std * 100
        )
        print(info)

        x = np.tile(np.arange(min_len), n_seeds)

        try:
            sns.lineplot(
                x=x,
                y=all_curves,
                ax=ax,
                color=list(mcolors.TABLEAU_COLORS.values())[i % len(mcolors.TABLEAU_COLORS)],
                label=algo_name,
                errorbar="sd",
            )
        except TypeError:
            sns.lineplot(
                x=x,
                y=all_curves,
                ax=ax,
                color=list(mcolors.TABLEAU_COLORS.values())[i % len(mcolors.TABLEAU_COLORS)],
                label=algo_name,
                ci="sd",
            )

    ax.grid(True)
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Accuracy')
    yticks = np.arange(0, 1.01, 0.05) 
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    y_top = min(1.0, global_max + 0.02)
    ax.set_ylim(0.0, y_top)
    if global_len is not None:
        ax.set_xlim(0, max(0, global_len - 1))

    fig.tight_layout()
    fig_save_path = os.path.join('figs', sub_dir, f"{dataset_name}.png")
    fig.savefig(fig_save_path, bbox_inches='tight', pad_inches=0.05, dpi=400)
    print('file saved to {}'.format(fig_save_path))
