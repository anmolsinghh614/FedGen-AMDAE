import argparse
import glob
import json
import os
from typing import List, Optional, Tuple, Dict
import re

import h5py
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Candidate keys to discover labels/predictions in your HDF5 files
CANDIDATE_TRUE_KEYS = ["y_true", "test_y", "labels", "targets", "test_targets"]
CANDIDATE_PRED_KEYS = ["y_pred", "preds", "test_pred", "test_predictions", "predictions"]
CANDIDATE_PROBA_KEYS = ["y_prob", "probs", "probabilities", "logits", "outputs"]

# HDF5 magic signature (first 8 bytes)
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"

# ---------------------------
# Utility helpers
# ---------------------------
def is_hdf5(path: str) -> bool:
    """Quick signature check to avoid h5py OSError on non-HDF5 files."""
    try:
        with open(path, "rb") as f:
            return f.read(8) == HDF5_MAGIC
    except OSError:
        return False

def _first_existing_key(hf: h5py.File, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in hf:
            return k
    return None

def _ensure_1d_labels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 0:
        x = x.reshape(1)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x.reshape(-1)
    return x

def _proba_to_pred(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert probabilities/logits to class predictions.
    Binary: shape (N,) or (N,1) -> thresholding.
    Multiclass: shape (N,C) -> argmax along C.
    """
    proba = np.asarray(proba)
    if proba.ndim == 1 or (proba.ndim == 2 and proba.shape[1] == 1):
        return (proba.reshape(-1) >= threshold).astype(int)
    elif proba.ndim == 2:
        return np.argmax(proba, axis=1)
    else:
        # Fallback for higher-rank arrays: take argmax on last axis and flatten
        pred = np.argmax(proba, axis=-1)
        return pred.reshape(-1)

def extract_round_number(filename: str) -> int:
    """Extract round number from filename."""
    # Look for patterns like "round_X" or just numbers
    match = re.search(r'round_(\d+)', filename)
    if match:
        return int(match.group(1))
    # Fallback: extract any number from filename
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[-1])  # Take the last number found
    return 0

def load_labels_and_preds(h5_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Open a valid HDF5 file, find y_true, y_pred, y_prob and return them
    along with discovered labels and the list of file keys for debugging.
    """
    if not is_hdf5(h5_path):
        raise OSError("Not an HDF5 file (magic signature mismatch)")

    with h5py.File(h5_path, "r") as hf:
        keys = list(hf.keys())

        true_key = _first_existing_key(hf, CANDIDATE_TRUE_KEYS)
        pred_key = _first_existing_key(hf, CANDIDATE_PRED_KEYS)
        proba_key = _first_existing_key(hf, CANDIDATE_PROBA_KEYS)

        if true_key is None:
            raise KeyError(f"Missing ground-truth; keys present: {keys}")

        y_true = _ensure_1d_labels(hf[true_key][()])

        # Get probabilities if available
        y_prob = None
        if proba_key is not None:
            y_prob = np.array(hf[proba_key][()])

        if pred_key is not None:
            y_pred = _ensure_1d_labels(hf[pred_key][()])
        elif proba_key is not None:
            y_pred = _proba_to_pred(hf[proba_key][()])
            y_pred = _ensure_1d_labels(y_pred)
        else:
            raise KeyError(f"Missing predictions/probabilities; keys present: {keys}")

        if y_true.shape != y_pred.shape:
            raise ValueError(f"Length mismatch y_true={y_true.shape} vs y_pred={y_pred.shape}")

        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())

    return y_true, y_pred, y_prob, labels, keys

def plot_roc_auc_curve(y_true: np.ndarray, y_prob: np.ndarray, labels: List[int], out_path: str, show_classes: int = 8):
    """Plot ROC AUC curve showing only top and bottom performing classes."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    plt.figure(figsize=(12, 8))
    
    y_true_bin = label_binarize(y_true, classes=labels)
    n_classes = len(labels)
    roc_data = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        roc_data.append((i, fpr, tpr, roc_auc, labels[i]))
    
    roc_data.sort(key=lambda x: x[3], reverse=True)
    
    top_classes = roc_data[:show_classes//2]
    bottom_classes = roc_data[-(show_classes//2):]
    selected_classes = top_classes + bottom_classes
    
    # Plot selected classes
    colors = plt.cm.Set3(np.linspace(0, 1, len(selected_classes)))
    for idx, (class_idx, fpr, tpr, roc_auc, class_label) in enumerate(selected_classes):
        line_style = '-' if idx < len(top_classes) else '--'
        plt.plot(fpr, tpr, color=colors[idx], lw=2, linestyle=line_style,
                 label=f'Class {class_label} (AUC = {roc_auc:.3f})')
    
    # Add micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, 
             label=f'Micro-average (AUC = {auc_micro:.3f})',
             color='red', linestyle=':', linewidth=3)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - Top {show_classes//2} Best & {show_classes//2} Worst Performing Classes', fontsize=14)
    plt.legend(loc='lower right', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix_last_round(y_true: np.ndarray, y_pred: np.ndarray, 
                                   labels: List[int], out_path: str, normalize: str = "none"):
    """Plot confusion matrix for the last round with better formatting."""
    norm_arg = None if normalize == "none" else normalize
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=norm_arg)
    
    # Determine figure size based on number of classes
    n_classes = len(labels)
    fig_size = max(8, n_classes * 0.4)  # Scale figure size with classes
    
    plt.figure(figsize=(fig_size, fig_size))
    
    # Create the plot with smaller font sizes
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", colorbar=True, 
              values_format=".2f" if norm_arg else "d",
              xticks_rotation='vertical')
    
    # Adjust font sizes
    plt.setp(disp.ax_.get_xticklabels(), fontsize=max(6, 12 - n_classes//4))
    plt.setp(disp.ax_.get_yticklabels(), fontsize=max(6, 12 - n_classes//4))
    
    # Adjust text in cells
    for text in disp.text_.ravel():
        if text is not None:
            text.set_fontsize(max(4, 10 - n_classes//3))
    
    plt.title("Confusion Matrix - Last Round", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_f1_vs_round(rounds_f1_data: Dict[int, float], out_path: str):
    """Plot F1 score vs round number."""
    rounds = sorted(rounds_f1_data.keys())
    f1_scores = [rounds_f1_data[r] for r in rounds]
    
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, f1_scores, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Round')
    plt.ylabel('F1 Score (Macro)')
    plt.title('F1 Score vs Training Round')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate ROC/AUC for last round, confusion matrix for last round, and F1 vs round plot.")
    parser.add_argument("--results_dir", type=str, default="results/metrics", help="Directory containing .h5 files.")
    parser.add_argument("--glob", type=str, default="*.h5", help="Glob pattern, e.g., \"*.h5\".")
    parser.add_argument("--output_dir", type=str, default="eval", help="Subdirectory under results_dir for outputs.")
    parser.add_argument("--normalize", type=str, choices=["none", "true", "pred", "all"], default="none",
                        help="Normalization for confusion_matrix.")
    args = parser.parse_args()

    # Discover files
    pattern = os.path.join(args.results_dir, args.glob)
    h5_paths = sorted(glob.glob(pattern))
    if not h5_paths:
        print(f"No .h5 files found under {args.results_dir} with pattern {args.glob}")
        return

    out_root = os.path.join(args.results_dir, args.output_dir)
    os.makedirs(out_root, exist_ok=True)

    # Process each file to collect F1 scores and find last round
    rounds_f1_data = {}
    last_round_data = None
    max_round = -1

    for p in h5_paths:
        base, _ = os.path.splitext(os.path.basename(p))
        round_num = extract_round_number(base)
        
        try:
            y_true, y_pred, y_prob, labels, keys = load_labels_and_preds(p)
            
            # Calculate F1 score
            f1_macro = f1_score(y_true, y_pred, average="macro")
            rounds_f1_data[round_num] = f1_macro
            
            # Check if this is the last round
            if round_num > max_round:
                max_round = round_num
                last_round_data = {
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'y_prob': y_prob,
                    'labels': labels,
                    'filename': base
                }
            
            print(f"[OK] {base} (round {round_num}, F1={f1_macro:.4f}) | keys={keys}")
            
        except (OSError, KeyError, ValueError) as e:
            print(f"[SKIP] {base}: {e}")

    # Generate plots
    if rounds_f1_data:
        # Plot F1 vs Round
        f1_vs_round_path = os.path.join(out_root, "f1_score_vs_round.png")
        plot_f1_vs_round(rounds_f1_data, f1_vs_round_path)
        print(f"F1 vs Round plot saved to: {f1_vs_round_path}")
        
        # Save F1 data to CSV
        f1_csv_path = os.path.join(out_root, "f1_scores_by_round.csv")
        with open(f1_csv_path, 'w') as f:
            f.write("round,f1_macro\n")
            for round_num in sorted(rounds_f1_data.keys()):
                f.write(f"{round_num},{rounds_f1_data[round_num]:.6f}\n")
        print(f"F1 scores CSV saved to: {f1_csv_path}")

    if last_round_data:
        print(f"\nGenerating plots for last round (round {max_round}): {last_round_data['filename']}")
        
        # Plot ROC-AUC for last round
        if last_round_data['y_prob'] is not None:
            roc_path = os.path.join(out_root, "roc_auc_last_round.png")
            plot_roc_auc_curve(
                last_round_data['y_true'], 
                last_round_data['y_prob'], 
                last_round_data['labels'], 
                roc_path
            )
            print(f"ROC-AUC plot saved to: {roc_path}")
        else:
            print("No probability data available for ROC curve in last round")
        
        # Plot confusion matrix for last round
        cm_path = os.path.join(out_root, "confusion_matrix_last_round.png")
        plot_confusion_matrix_last_round(
            last_round_data['y_true'], 
            last_round_data['y_pred'], 
            last_round_data['labels'], 
            cm_path, 
            args.normalize
        )
        print(f"Confusion matrix saved to: {cm_path}")
        
        # Save classification report for last round
        report = classification_report(
            last_round_data['y_true'], 
            last_round_data['y_pred'], 
            labels=last_round_data['labels'], 
            zero_division=0, 
            output_dict=True
        )
        
        # Save detailed metrics for last round
        import pandas as pd
        report_path = os.path.join(out_root, "classification_report_last_round.csv")
        pd.DataFrame(report).transpose().to_csv(report_path, index=True)
        print(f"Classification report saved to: {report_path}")
        
        # Save summary JSON for last round
        summary = {
            "last_round": max_round,
            "f1_micro": float(f1_score(last_round_data['y_true'], last_round_data['y_pred'], average="micro")),
            "f1_macro": float(f1_score(last_round_data['y_true'], last_round_data['y_pred'], average="macro")),
            "f1_weighted": float(f1_score(last_round_data['y_true'], last_round_data['y_pred'], average="weighted")),
            "accuracy": float(report.get("accuracy", np.nan)),
        }
        
        summary_path = os.path.join(out_root, "summary_last_round.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary JSON saved to: {summary_path}")

    else:
        print("No valid data found for generating plots")

if __name__ == "__main__":
    main()
