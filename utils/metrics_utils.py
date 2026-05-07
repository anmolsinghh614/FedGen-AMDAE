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
    Save evaluation metrics to HDF5 file with multiple dataset names for compatibility.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        algorithm: Algorithm name for filename
        dataset: Dataset name for filename
        glob_iter: Global iteration number for filename
        results_dir: Directory to save results (default: 'results/metrics')
    
    Returns:
        Path to saved file
    """
    dataset_dir = os.path.join(results_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    filename = os.path.join(dataset_dir, f'{algorithm}_{dataset}_round_{glob_iter}.h5')
    
    with h5py.File(filename, 'w') as f:
        # Save true labels with multiple names for compatibility
        f.create_dataset('y_true', data=y_true)
        f.create_dataset('test_y', data=y_true)
        f.create_dataset('labels', data=y_true)
        f.create_dataset('targets', data=y_true)
        f.create_dataset('test_targets', data=y_true)
        
        # Save predictions with multiple names for compatibility
        f.create_dataset('y_pred', data=y_pred)
        f.create_dataset('preds', data=y_pred)
        f.create_dataset('test_pred', data=y_pred)
        f.create_dataset('test_predictions', data=y_pred)
        f.create_dataset('predictions', data=y_pred)
        
        # Save probabilities with multiple names for compatibility
        f.create_dataset('y_prob', data=y_prob)
        f.create_dataset('probs', data=y_prob)
        f.create_dataset('probabilities', data=y_prob)
        f.create_dataset('logits', data=y_prob)
        f.create_dataset('outputs', data=y_prob)
    
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
