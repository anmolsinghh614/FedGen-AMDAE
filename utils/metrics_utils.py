import torch
import torch.nn.functional as F
import numpy as np
import h5py
import os
from typing import List, Tuple, Union, Any

def collect_evaluation_metrics(model: torch.nn.Module, users: List[Any], device: Union[str, torch.device] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect evaluation metrics (true labels, predictions, probabilities) from all users.
    
    Args:
        model: The trained model to evaluate
        users: List of user objects with testloaderfull attribute
        device: Device to run evaluation on (auto-detected if None)
    
    Returns:
        Tuple of (y_true, y_pred, y_prob) as numpy arrays
    """
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        for user in users:
            user.model.eval()
            for batch_idx, (X, y) in enumerate(user.testloaderfull):
                X, y = X.to(device), y.to(device)
                output = user.model(X)
                
                # Handle different output formats
                if isinstance(output, dict):
                    # Try different possible keys for logits
                    if 'logit' in output:
                        logits = output['logit']
                    elif 'logits' in output:
                        logits = output['logits']
                    elif 'output' in output:
                        logits = output['output']
                    else:
                        logits = list(output.values())[0]
                else:
                    logits = output
                
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_y_true.extend(y.cpu().numpy())
                all_y_pred.extend(preds.cpu().numpy())
                all_y_prob.extend(probs.cpu().numpy())
    
    return np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob)


def save_evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                          algorithm: str, dataset: str, glob_iter: int,
                          results_dir: str = 'results/metrics') -> str:
    """
    Save per-round evaluation metrics to an HDF5 file.

    The dump stores three arrays under their canonical names:
        y_true : (N,)   int64  ground-truth labels
        y_pred : (N,)   int64  argmax(class) predictions
        y_prob : (N, K) float  per-class probabilities (soft outputs)

    Historical note (Jul 2026)
    --------------------------
    Earlier versions of this function wrote **5 duplicate copies** of each
    array under aliases like ``test_y`` / ``labels`` / ``targets`` /
    ``test_targets`` (for y_true), ``preds`` / ``test_pred`` / ``test_predictions``
    / ``predictions`` (for y_pred), and ``probs`` / ``probabilities`` /
    ``logits`` / ``outputs`` (for y_prob). That "compatibility layer"
    bloated every dump to 5x its useful size and was directly responsible
    for the ~800 GB PAMAP2 explosion observed during the Option A sweep.

    All downstream readers already probe multiple candidate keys and fall
    back to ``y_true`` / ``y_pred`` / ``y_prob``, so dropping the aliases
    is a lossless win. Arrays are additionally gzipped (level 4) which
    gives another ~2-3x on integer label vectors.
    """
    dataset_dir = os.path.join(results_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    filename = os.path.join(dataset_dir, f'{algorithm}_{dataset}_round_{glob_iter}.h5')

    # Cast to compact dtypes before writing:
    #   labels -> int32 (all datasets in this repo have < 2**31 classes)
    #   probs  -> float32 (float64 gives no meaningful precision here)
    y_true_arr = np.asarray(y_true).astype(np.int32, copy=False)
    y_pred_arr = np.asarray(y_pred).astype(np.int32, copy=False)
    y_prob_arr = np.asarray(y_prob).astype(np.float32, copy=False)

    def _kwargs_for(arr):
        # Only bother chunking + gzipping arrays large enough to benefit;
        # tiny arrays get no compression (avoids per-file overhead).
        if arr.size < 128:
            return {}
        return dict(
            chunks=True,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

    with h5py.File(filename, 'w') as f:
        f.create_dataset('y_true', data=y_true_arr, **_kwargs_for(y_true_arr))
        f.create_dataset('y_pred', data=y_pred_arr, **_kwargs_for(y_pred_arr))
        f.create_dataset('y_prob', data=y_prob_arr, **_kwargs_for(y_prob_arr))
        # A tiny attrs stamp helps forensic debugging when reviewing a
        # dump months later. Cheap; adds ~50 bytes.
        f.attrs['schema_version'] = 2
        f.attrs['algorithm'] = str(algorithm)
        f.attrs['dataset'] = str(dataset)
        f.attrs['glob_iter'] = int(glob_iter)

    return filename


def save_evaluation_metrics_from_model(model: torch.nn.Module, users: List[Any], 
                                     algorithm: str, dataset: str, glob_iter: int,
                                     results_dir: str = 'results/metrics', 
                                     device: Union[str, torch.device] = None) -> str:
    """
    Complete pipeline: collect metrics from model and users, then save to file.
    
    Args:
        model: The trained model to evaluate
        users: List of user objects with testloaderfull attribute
        algorithm: Algorithm name for filename
        dataset: Dataset name for filename
        glob_iter: Global iteration number for filename
        results_dir: Directory to save results (default: 'results/metrics')
        device: Device to run evaluation on (auto-detected if None)
    
    Returns:
        Path to saved file
    """
    y_true, y_pred, y_prob = collect_evaluation_metrics(model, users, device)
    return save_evaluation_metrics(y_true, y_pred, y_prob, algorithm, dataset, glob_iter, results_dir)
