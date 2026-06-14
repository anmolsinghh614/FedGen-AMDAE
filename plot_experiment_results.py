#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import seaborn as sns
import matplotlib.ticker as mticker
from utils.model_utils import get_log_path

def get_label_name(name):
    if 'Distill' in name:
        return 'FedDistill'
    elif 'FedEnsemble' in name:
        return 'Ensemble'
    return name

def load_results(args, algorithm, seed=0):
    try:
        alg = get_log_path(args, algorithm, seed, args.gen_batch_size)
        hf = h5py.File(os.path.join(args.result_path, f"{alg}.h5"), 'r')
        glob_acc = np.array(hf.get('glob_acc')[:])
        return glob_acc
    except Exception as e:
        print(f"Warning: Could not load results for {algorithm}: {e}")
        return None


def load_curves(args, algorithm, seed=0):
    """Return (glob_acc, glob_loss) for one algorithm/seed. Either may be None."""
    try:
        alg = get_log_path(args, algorithm, seed, args.gen_batch_size)
        with h5py.File(os.path.join(args.result_path, f"{alg}.h5"), 'r') as hf:
            acc = np.array(hf['glob_acc'][:]) if 'glob_acc' in hf else None
            loss = np.array(hf['glob_loss'][:]) if 'glob_loss' in hf else None
        return acc, loss
    except Exception as e:
        print(f"Warning: Could not load curves for {algorithm}: {e}")
        return None, None

def main(args):
    algorithms = [a.strip() for a in args.algorithms.split(',')]
    assert len(algorithms) > 0, "No algorithms provided"
    
    n_seeds = args.times
    results_dict = {}
    
    # Setup directory for saving figures and tables
    parts = args.dataset.split('-')
    dataset_name = parts[0]
    alpha_str = parts[1].replace('alpha', '') if len(parts) > 1 else 'default'
    
    out_dir = 'results/experiment_summary'
    os.makedirs(out_dir, exist_ok=True)
    
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {
        'FedGen': '#1f77b4',     # blue
        'FedAvg': '#ff7f0e',     # orange
        'FedProx': '#2ca02c',    # green
        'Ensemble': '#9467bd',   # purple
        'FedDistill': '#d62728'  # red
    }
    
    TOP_N = 5
    summary_text = f"Performance Comparison on {dataset_name} (Alpha={alpha_str}, Missing={args.missing_rate*100}%)\n"
    summary_text += "-" * 70 + "\n"
    summary_text += f"{'Setting':<25} | {'Acc (Mean ± Std) %'}\n"
    summary_text += "-" * 70 + "\n"
    
    print("\n" + summary_text, end="")
    
    max_len = 0
    all_accuracies_for_plot = []
    
    for algorithm in algorithms:
        label = get_label_name(algorithm)
        
        curves = []
        for seed in range(n_seeds):
            acc = load_results(args, algorithm, seed)
            if acc is not None:
                curves.append(acc)
                
        if not curves:
            continue
            
        min_len = min(len(c) for c in curves)
        curves = [c[:min_len] for c in curves]
        if min_len > max_len:
            max_len = min_len
            
        # Get top-N accuracy for table
        top_accs = np.concatenate([np.sort(c)[-TOP_N:] for c in curves])
        acc_avg = np.mean(top_accs) * 100
        acc_std = np.std(top_accs) * 100
        
        algo_summary = f"{label:<25} | {acc_avg:.2f} ± {acc_std:.2f}\n"
        summary_text += algo_summary
        print(algo_summary, end="")

        all_curves = np.concatenate(curves)
        x = np.tile(np.arange(min_len), n_seeds)
        
        sns.lineplot(
            x=x,
            y=all_curves,
            ax=ax,
            color=colors.get(label, 'black'),
            label=label,
            errorbar="sd" if hasattr(sns, "lineplot") and "errorbar" in sns.lineplot.__code__.co_varnames else None,
            ci="sd" if not (hasattr(sns, "lineplot") and "errorbar" in sns.lineplot.__code__.co_varnames) else None
        )

    # Save summary table
    table_path = os.path.join(out_dir, f"table_{dataset_name}_alpha{alpha_str}_miss{args.missing_rate}.txt")
    with open(table_path, 'w') as f:
        f.write(summary_text)

    # Customize plot to match image
    ax.grid(True)
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Accuracy')
    
    # Replicate Y-axis percentage formatting (0% to 70%+ in steps of 5%)
    yticks = np.arange(0, 0.75, 0.05)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0.0, max((yticks[-1], ax.get_ylim()[1])))
    
    if max_len > 0:
        ax.set_xlim(0, max_len - 1)
        
    ax.legend(loc='lower right')
    
    # Title matching format in paper
    alpha_print = int(float(alpha_str)) if float(alpha_str).is_integer() else float(alpha_str)
    miss_pct = int(args.missing_rate * 100)
    plt.title(f"$\\alpha = {alpha_print}$, {miss_pct}% Missing", y=-0.2)
    
    fig.tight_layout()
    fig_path = os.path.join(out_dir, f"plot_{dataset_name}_alpha{alpha_str}_miss{args.missing_rate}.png")
    fig.savefig(fig_path, bbox_inches='tight', dpi=300)
    print(f"\nSaved summary table to: {table_path}")
    print(f"Saved performance graph to: {fig_path}")

def plot_acc_loss_side_by_side(args):
    """Reproduces paper Fig. 13: test accuracy + training loss side-by-side."""
    algorithms = [a.strip() for a in args.algorithms.split(',')]
    parts = args.dataset.split('-')
    dataset_name = parts[0]
    alpha_str = parts[1].replace('alpha', '') if len(parts) > 1 else 'default'
    out_dir = 'results/experiment_summary'
    os.makedirs(out_dir, exist_ok=True)

    colors = {
        'FedGen': '#1f77b4', 'FedAvg': '#ff7f0e', 'FedProx': '#2ca02c',
        'Ensemble': '#9467bd', 'FedDistill': '#d62728',
    }

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))
    plotted = 0
    for algorithm in algorithms:
        label = get_label_name(algorithm)
        accs, losses = [], []
        for seed in range(args.times):
            a, l = load_curves(args, algorithm, seed)
            if a is not None:
                accs.append(a)
            if l is not None:
                losses.append(l)
        if not accs and not losses:
            continue

        if accs:
            min_len_a = min(len(c) for c in accs)
            stacked = np.stack([c[:min_len_a] for c in accs], axis=0)
            mean_a = stacked.mean(axis=0)
            std_a = stacked.std(axis=0)
            x = np.arange(min_len_a)
            ax_acc.plot(x, mean_a, color=colors.get(label, 'black'),
                        label=label, linewidth=2)
            ax_acc.fill_between(x, mean_a - std_a, mean_a + std_a,
                                color=colors.get(label, 'black'), alpha=0.15)

        if losses:
            min_len_l = min(len(c) for c in losses)
            stacked = np.stack([c[:min_len_l] for c in losses], axis=0)
            mean_l = stacked.mean(axis=0)
            std_l = stacked.std(axis=0)
            x = np.arange(min_len_l)
            ax_loss.plot(x, mean_l, color=colors.get(label, 'black'),
                         label=label, linewidth=2)
            ax_loss.fill_between(x, mean_l - std_l, mean_l + std_l,
                                 color=colors.get(label, 'black'), alpha=0.15)
        plotted += 1

    ax_acc.set_title("Test Accuracy")
    ax_acc.set_xlabel("Communication Rounds")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax_acc.grid(True)
    ax_acc.legend(loc="lower right")

    ax_loss.set_title("Training Loss")
    ax_loss.set_xlabel("Communication Rounds")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True)
    ax_loss.legend(loc="upper right")

    miss_pct = int(args.missing_rate * 100)
    fig.suptitle(f"{dataset_name} (alpha={alpha_str}, {miss_pct}% Missing)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(
        out_dir,
        f"acc_loss_{dataset_name}_alpha{alpha_str}_miss{args.missing_rate}.png")
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved acc/loss panel to: {out_path}  (algorithms plotted: {plotted})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--algorithms", type=str, required=True, help='comma separated algorithms')
    parser.add_argument("--missing_rate", type=float, required=True)
    parser.add_argument("--result_path", type=str, default="results/models")
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--num_users", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--embedding", type=int, default=0)
    parser.add_argument("--gen_batch_size", type=int, default=64)
    parser.add_argument("--num_glob_iters", type=int, default=100)
    parser.add_argument("--plot_loss", action="store_true",
                        help="Also produce a paper Fig.13-style side-by-side accuracy + loss subplot.")
    args = parser.parse_args()
    main(args)
    if args.plot_loss:
        plot_acc_loss_side_by_side(args)
