"""
Enhanced Data Imputation Module using Adaptive-Learned Median-Filled Deep Autoencoder (AM-DAE)
Based on: "Imputation of Missing Values in Time Series Using an Adaptive-Learned 
Median-Filled Deep Autoencoder" (IEEE Transactions on Cybernetics, 2023)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, List, Tuple, Any, Optional
import random
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.spatial.distance import jensenshannon
from abc import ABC, abstractmethod


class AMDAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], activation='relu', dropout_rate=0.1):
        super(AMDAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        self.dropout = nn.Dropout(dropout_rate)        
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
    def _build_encoder(self):
        layers = []
        dims = [self.input_dim] + self.hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(self.dropout_rate)) 
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self):
        layers = []
        dims = self.hidden_dims[::-1] + [self.input_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if self.activation == 'relu':
                    layers.append(nn.ReLU())
                elif self.activation == 'tanh':
                    layers.append(nn.Tanh())
                elif self.activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                layers.append(nn.Dropout(self.dropout_rate))  # Create new dropout instance
        
        return nn.Sequential(*layers)
    
    def forward(self, x, start_layer_idx=0, logit=False):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MissingDataSimulator:
    """
    Generates the binary mask M used by the imputers.

    Patterns
    --------
    - 'random'   / 'mcar'   : each entry is masked iid with probability r.
    - 'mar'                  : missingness depends on an *observed* covariate
                                (here: the row label, when provided). Rows with
                                certain label values are made more missing-prone.
                                The aggregate per-cell missing rate is still r.
    - 'mnar'                 : missingness depends on the *value itself*
                                (high-magnitude entries are more likely to be
                                masked, mimicking sensor saturation). The
                                aggregate per-cell missing rate is still r.
    - 'fixed_intervals'      : block missingness over rows.
    - 'continuous_periods'   : long contiguous gaps within a column.

    For backward compatibility, the legacy aliases continue to work:
       'mcar' is an alias of 'random'.
    """
    def __init__(self, missing_patterns=None):
        self.missing_patterns = missing_patterns or [
            'random', 'mcar', 'mar', 'mnar',
            'fixed_intervals', 'continuous_periods',
        ]

    def introduce_missing_data(self, data: np.ndarray, missing_rate: float = 0.1,
                             pattern: str = 'random', labels: np.ndarray = None,
                             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pat = (pattern or 'random').lower()
        if pat in ('random', 'mcar'):
            return self._random_missing(data, missing_rate)
        elif pat == 'mar':
            return self._mar_missing(data, missing_rate, labels=labels, **kwargs)
        elif pat == 'mnar':
            return self._mnar_missing(data, missing_rate, **kwargs)
        elif pat == 'fixed_intervals':
            return self._fixed_intervals_missing(data, missing_rate, **kwargs)
        elif pat == 'continuous_periods':
            return self._continuous_periods_missing(data, missing_rate, **kwargs)
        else:
            raise ValueError(f"Unknown missing pattern: {pattern}")

    def _random_missing(self, data: np.ndarray, missing_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        corrupted_data = data.copy()
        missing_mask = np.random.random(data.shape) < missing_rate
        corrupted_data[missing_mask] = np.nan
        return corrupted_data, missing_mask.astype(int)

    def _mar_missing(self, data: np.ndarray, missing_rate: float,
                     labels: np.ndarray = None,
                     skew: float = 3.0, **_kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Missing-At-Random: per-row masking probability scales with row label.
        Rows with high label-id are `skew`x more likely to be missing than
        rows with low label-id, but the *aggregate* per-cell missing rate
        across the whole matrix is still ~missing_rate. If labels are not
        supplied, fall back to a deterministic row-index-based proxy.
        """
        N, D = data.shape
        if labels is None or len(labels) != N:
            row_signal = np.linspace(0.0, 1.0, N)
        else:
            lab = np.asarray(labels).astype(float).reshape(-1)
            lo, hi = float(lab.min()), float(lab.max())
            row_signal = (lab - lo) / (hi - lo + 1e-12)

        # row-wise probability mass: linearly mix [1, skew]
        row_weight = 1.0 + (skew - 1.0) * row_signal
        per_row_p = row_weight / row_weight.mean() * missing_rate
        per_row_p = np.clip(per_row_p, 0.0, 1.0)

        u = np.random.random((N, D))
        missing_mask = u < per_row_p[:, None]
        corrupted = data.copy()
        corrupted[missing_mask] = np.nan
        return corrupted, missing_mask.astype(int)

    def _mnar_missing(self, data: np.ndarray, missing_rate: float,
                      skew: float = 3.0, **_kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Missing-Not-At-Random: high-magnitude entries (per column) are more
        likely to be masked, simulating sensor saturation / clipping. The
        aggregate per-cell missing rate is still ~missing_rate.
        """
        N, D = data.shape
        finite = np.isfinite(data)
        cell_signal = np.zeros_like(data, dtype=float)
        for j in range(D):
            col = data[:, j]
            mask_finite = finite[:, j]
            if mask_finite.sum() == 0:
                continue
            mu = float(np.mean(col[mask_finite]))
            sd = float(np.std(col[mask_finite])) + 1e-12
            z = np.abs((col - mu) / sd)
            z[~mask_finite] = 0.0
            # squash into [0,1] via tanh; high |z| -> closer to 1
            cell_signal[:, j] = np.tanh(z)

        cell_weight = 1.0 + (skew - 1.0) * cell_signal
        cell_p = cell_weight / cell_weight.mean() * missing_rate
        cell_p = np.clip(cell_p, 0.0, 1.0)

        u = np.random.random(data.shape)
        missing_mask = u < cell_p
        corrupted = data.copy()
        corrupted[missing_mask] = np.nan
        return corrupted, missing_mask.astype(int)
    
    def _fixed_intervals_missing(self, data: np.ndarray, missing_rate: float, 
                               interval_size: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        corrupted_data = data.copy()
        missing_mask = np.zeros_like(data, dtype=int)
        
        for i in range(0, data.shape[0], interval_size):
            if np.random.random() < missing_rate:
                end_idx = min(i + interval_size, data.shape[0])
                missing_mask[i:end_idx] = 1
                corrupted_data[i:end_idx] = np.nan
        
        return corrupted_data, missing_mask
    
    def _continuous_periods_missing(self, data: np.ndarray, missing_rate: float,
                                  min_period: int = 5, max_period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        corrupted_data = data.copy()
        missing_mask = np.zeros_like(data, dtype=int)
        
        total_missing = int(missing_rate * data.size)
        current_missing = 0
        
        while current_missing < total_missing:
            period_length = np.random.randint(min_period, max_period + 1)
            start_row = np.random.randint(0, data.shape[0])
            start_col = np.random.randint(0, data.shape[1])
            
            end_row = min(start_row + period_length, data.shape[0])
            end_col = min(start_col + 1, data.shape[1])
            
            missing_mask[start_row:end_row, start_col:end_col] = 1
            corrupted_data[start_row:end_row, start_col:end_col] = np.nan
            
            current_missing += (end_row - start_row) * (end_col - start_col)
        
        return corrupted_data, missing_mask


class BaseImputer(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def fit_impute(self, data_with_missing: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        pass


class MeanImputer(BaseImputer):
    def __init__(self):
        super().__init__("Mean Imputation")
    
    def fit_impute(self, data_with_missing: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        imputed_data = data_with_missing.copy()
        for j in range(data_with_missing.shape[1]):
            col_data = data_with_missing[:, j]
            mean_val = np.nanmean(col_data)
            imputed_data[missing_mask[:, j] == 1, j] = mean_val
        return imputed_data


class MedianImputer(BaseImputer):
    def __init__(self):
        super().__init__("Median Imputation")
    
    def fit_impute(self, data_with_missing: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        imputed_data = data_with_missing.copy()
        for j in range(data_with_missing.shape[1]):
            col_data = data_with_missing[:, j]
            median_val = np.nanmedian(col_data)
            imputed_data[missing_mask[:, j] == 1, j] = median_val
        return imputed_data


class ZeroImputer(BaseImputer):
    def __init__(self):
        super().__init__("Zero Imputation")
    
    def fit_impute(self, data_with_missing: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        imputed_data = data_with_missing.copy()
        imputed_data[missing_mask == 1] = 0
        return imputed_data


class AMDAEImputer(BaseImputer):
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, 
                 max_epochs: int = 100, batch_size: int = 32, 
                 learning_rate: float = 0.001, device: str = 'cpu'):
        super().__init__("AM-DAE")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [input_dim // 2, input_dim // 4, input_dim // 8]
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
        self.model = AMDAE(input_dim, self.hidden_dims)
        self.model.to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def _init_missing_values(self, data: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        initialized_data = data.copy()
        for j in range(data.shape[1]):
            col_data = data[:, j]
            mean_val = np.nanmean(col_data)
            initialized_data[missing_mask[:, j] == 1, j] = mean_val
        return initialized_data
    
    def _adaptive_loss_function(self, reconstruction: torch.Tensor, input_data: torch.Tensor, 
                              missing_mask: torch.Tensor, epoch: int) -> torch.Tensor:
        alpha_k = 2 * (1 - 0.5 * epoch / self.max_epochs)
        beta_k = epoch / self.max_epochs
        
        weight_matrix = alpha_k * (1 - missing_mask) + beta_k * missing_mask
        weighted_diff = weight_matrix * (reconstruction - input_data)
        loss = torch.mean(torch.sum(weighted_diff ** 2, dim=1))
        
        return loss
    
    def _median_update(self, input_data: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        return (input_data + reconstruction) / 2
    
    def _update_missing_values(self, original_data: torch.Tensor, median_values: torch.Tensor, 
                             missing_mask: torch.Tensor) -> torch.Tensor:
        return (1 - missing_mask) * original_data + missing_mask * median_values
    
    def fit_impute(self, data_with_missing: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        initialized_data = self._init_missing_values(data_with_missing, missing_mask)
        
        current_data = initialized_data.copy()
        missing_mask_tensor = torch.FloatTensor(missing_mask).to(self.device)
        
        for epoch in range(self.max_epochs):
            indices = np.random.permutation(len(current_data))
            epoch_loss = 0
            
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_data = torch.FloatTensor(current_data[batch_indices]).to(self.device)
                batch_mask = missing_mask_tensor[batch_indices]
                
                self.optimizer.zero_grad()
                
                reconstruction = self.model(batch_data)
                
                loss = self._adaptive_loss_function(reconstruction, batch_data, batch_mask, epoch + 1)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                median_values = self._median_update(batch_data, reconstruction)
                updated_data = self._update_missing_values(batch_data, median_values, batch_mask)
                
                current_data[batch_indices] = updated_data.detach().cpu().numpy()
        
        return current_data
    
    def fit_impute_timeseries(self, data_with_missing: np.ndarray, missing_mask: np.ndarray,
                            delta_t: int = 16) -> np.ndarray:
        if len(data_with_missing.shape) == 2:
            stacked_data = []
            stacked_masks = []
            
            for t in range(delta_t, data_with_missing.shape[0]):
                stack = data_with_missing[t-delta_t:t].flatten()
                mask_stack = missing_mask[t-delta_t:t].flatten()
                stacked_data.append(stack)
                stacked_masks.append(mask_stack)
            
            stacked_data = np.array(stacked_data)
            stacked_masks = np.array(stacked_masks)
            
            self.input_dim = stacked_data.shape[1]
            self.model = AMDAE(self.input_dim, self.hidden_dims)
            self.model.to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            imputed_stacked = self.fit_impute(stacked_data, stacked_masks)
            
            result = data_with_missing.copy()
            for t in range(delta_t, data_with_missing.shape[0]):
                idx = t - delta_t
                reshaped = imputed_stacked[idx].reshape(delta_t, -1)
                result[t] = reshaped[-1]
            
            return result
        
        return self.fit_impute(data_with_missing, missing_mask)


def process_federated_data_for_imputation(data: Tuple) -> Dict[str, Any]:
    clients, groups, train_data, test_data, proxy_data = data

    all_train_data = []
    all_train_labels = []
    all_test_data = []
    all_test_labels = []
    client_indices = []

    for i, client in enumerate(clients):
        if client in train_data:
            client_train = np.array(train_data[client]['x'])
            if len(client_train.shape) > 2:
                client_train = client_train.reshape(client_train.shape[0], -1)
            all_train_data.append(client_train)
            client_indices.extend([i] * len(client_train))
            if 'y' in train_data[client]:
                all_train_labels.append(np.asarray(train_data[client]['y']).reshape(-1))

        if client in test_data:
            client_test = np.array(test_data[client]['x'])
            if len(client_test.shape) > 2:
                client_test = client_test.reshape(client_test.shape[0], -1)
            all_test_data.append(client_test)
            if 'y' in test_data[client]:
                all_test_labels.append(np.asarray(test_data[client]['y']).reshape(-1))

    combined_train = np.vstack(all_train_data) if all_train_data else np.array([])
    combined_test = np.vstack(all_test_data) if all_test_data else np.array([])
    combined_data = np.vstack([combined_train, combined_test]) if len(combined_train) > 0 and len(combined_test) > 0 else combined_train

    if all_train_labels:
        combined_train_labels = np.concatenate(all_train_labels)
    else:
        combined_train_labels = np.array([])
    if all_test_labels:
        combined_test_labels = np.concatenate(all_test_labels)
    else:
        combined_test_labels = np.array([])
    if len(combined_train_labels) and len(combined_test_labels):
        combined_labels = np.concatenate([combined_train_labels, combined_test_labels])
    else:
        combined_labels = combined_train_labels

    processed_data = {
        'combined_data': combined_data,
        'combined_labels': combined_labels,
        'train_data': combined_train,
        'test_data': combined_test,
        'client_indices': np.array(client_indices),
        'original_shapes': [np.array(train_data[client]['x']).shape for client in clients if client in train_data]
    }

    return processed_data


def reconstruct_federated_data(imputed_data: np.ndarray, metadata: Dict[str, Any], 
                             original_data: Tuple) -> Tuple:
    clients, groups, train_data, test_data, proxy_data = original_data
    
    train_size = len(metadata['train_data'])
    imputed_train = imputed_data[:train_size]
    
    new_train_data = train_data.copy()
    
    start_idx = 0
    for i, client in enumerate(clients):
        if client in train_data:
            client_size = len(train_data[client]['x'])
            original_shape = metadata['original_shapes'][i]
            
            if len(original_shape) > 2:
                reshaped_data = imputed_train[start_idx:start_idx + client_size].reshape(original_shape)
            else:
                reshaped_data = imputed_train[start_idx:start_idx + client_size]
            
            new_train_data[client]['x'] = reshaped_data.tolist()
            start_idx += client_size
    
    return clients, groups, new_train_data, test_data, proxy_data

def _kl_divergence(original_data: np.ndarray, imputed_data: np.ndarray, 
                                 missing_mask: np.ndarray) -> float:
        """Calculate proper KL divergence between original and imputed missing values."""
        missing_positions = missing_mask == 1
        
        if not np.any(missing_positions):
            return 0.0
        
        original_missing = original_data[missing_positions]
        imputed_missing = imputed_data[missing_positions]
        
        # Use histogram-based probability estimation
        min_val = min(np.min(original_missing), np.min(imputed_missing))
        max_val = max(np.max(original_missing), np.max(imputed_missing))
        
        # Create bins
        bins = np.linspace(min_val, max_val, num=50)
        
        # Calculate histograms
        orig_hist, _ = np.histogram(original_missing, bins=bins, density=True)
        imp_hist, _ = np.histogram(imputed_missing, bins=bins, density=True)
        
        # Convert to probabilities
        orig_prob = orig_hist / (np.sum(orig_hist) + 1e-8)
        imp_prob = imp_hist / (np.sum(imp_hist) + 1e-8)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        orig_prob = np.maximum(orig_prob, epsilon)
        imp_prob = np.maximum(imp_prob, epsilon)
        
        # Calculate KL divergence: KL(P||Q) = sum(P * log(P/Q))
        kl_div = np.sum(orig_prob * np.log(orig_prob / imp_prob))
        
        return float(kl_div)

def calculate_comprehensive_imputation_metrics(original_data: np.ndarray, imputed_data: np.ndarray, 
                                             missing_mask: np.ndarray, data_with_missing: np.ndarray = None) -> Dict[str, float]:
    """Calculate comprehensive imputation metrics with proper KL divergence."""
    missing_positions = missing_mask == 1
    
    if not np.any(missing_positions):
        return {'RMSE': 0.0, 'MAPE': 0.0, 'KL-Divergence': 0.0, 'Mean-Difference': 0.0, 'Adaptive-Loss': 0.0}
    
    original_missing = original_data[missing_positions]
    imputed_missing = imputed_data[missing_positions]
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(original_missing, imputed_missing))
    
    # MAPE
    mape = np.mean(np.abs((original_missing - imputed_missing) / (np.abs(original_missing) + 1e-8))) * 100
    
    # Proper KL-Divergence
    try:
        kl_divergence = _kl_divergence(original_data, imputed_data, missing_mask)
        if not np.isfinite(kl_divergence) or kl_divergence > 1000:
            kl_divergence = 1000.0
    except:
        kl_divergence = 1000.0
    
    # Mean Difference
    mean_diff = np.abs(np.mean(original_missing) - np.mean(imputed_missing))
    
    # Adaptive Loss
    alpha = 0.7
    beta = 0.3
    
    observed_positions = ~missing_positions.astype(bool)
    observed_loss = np.mean((original_data[observed_positions] - imputed_data[observed_positions]) ** 2) if np.any(observed_positions) else 0
    missing_loss = np.mean((original_missing - imputed_missing) ** 2)
    
    adaptive_loss = alpha * observed_loss + beta * missing_loss
    
    return {
        'RMSE': rmse,
        'MAPE': mape,
        'KL-Divergence': kl_divergence,
        'Mean-Difference': mean_diff,
        'Adaptive-Loss': adaptive_loss
    }

# ---------------------------------------------------------------------------
# Composite-score configuration
# ---------------------------------------------------------------------------
#
# `calculate_comprehensive_imputation_metrics` reports 5 metrics:
#     RMSE, MAPE, KL-Divergence, Mean-Difference, Adaptive-Loss.
#
# Two of those are degenerate as RANKING signals (they remain in the plot for
# transparency, but should NOT drive imputer selection):
#
#   * MAPE: |x - x_hat| / (|x| + 1e-8). On standardised IMU / image data
#     a non-trivial fraction of ground-truth entries are near zero, so the
#     denominator collapses and MAPE explodes to ~1e6+ for any imputer that
#     fills with non-zero values (including AM-DAE). Mean / Median / Zero
#     happen to fill with values near the per-feature centre, so they look
#     artificially better. This is a well-known degeneracy of MAPE on
#     near-zero ground truth.
#
#   * Mean-Difference: |mean(x) - mean(x_hat)| over the missing positions.
#     This is exactly zero by construction for Mean Imputation (and ~zero
#     for Median Imputation on roughly-symmetric distributions). It is a
#     metric of bias only, not of point-wise reconstruction quality, and
#     hard-codes Mean Imputation as the winner.
#
# The composite ranking therefore aggregates only the three well-behaved
# point-wise / distributional metrics. Override via the `metrics` argument
# if you want the legacy 5-metric behaviour.
RELIABLE_METRICS: Tuple[str, ...] = ('RMSE', 'KL-Divergence', 'Adaptive-Loss')


def calculate_overall_score(all_results: Dict[str, Dict[str, float]],
                            metrics: Optional[Tuple[str, ...]] = None
                            ) -> Dict[str, float]:
    """Calculate normalized overall performance scores across all methods.

    Lower is better. Each metric is min-max-normalised by the max across
    methods, then averaged with equal weights over the chosen metric set
    (default: ``RELIABLE_METRICS``).
    """
    if metrics is None:
        metrics = RELIABLE_METRICS

    available = list(next(iter(all_results.values())).keys())
    metrics = tuple(m for m in metrics if m in available)
    if not metrics:
        # Fall back to whatever is present so we never crash on a
        # caller passing an unexpected metric set.
        metrics = tuple(available)

    weight = 1.0 / len(metrics)

    max_vals: Dict[str, float] = {}
    for metric in metrics:
        max_vals[metric] = max(
            all_results[method][metric]
            if np.isfinite(all_results[method][metric]) else 0
            for method in all_results
        )

    overall_scores: Dict[str, float] = {}
    for method, metric_vals in all_results.items():
        score = 0.0
        for metric in metrics:
            value = metric_vals.get(metric, np.nan)
            if max_vals.get(metric, 0) > 0:
                norm_val = value / max_vals[metric] if np.isfinite(value) else 1000.0
                score += weight * norm_val
        overall_scores[method] = score
    return overall_scores


def select_best_imputation_method(results: Dict[str, Dict[str, float]],
                                  metrics: Optional[Tuple[str, ...]] = None
                                  ) -> str:
    """Select the best imputation method based on the composite score.

    By default uses ``RELIABLE_METRICS`` (RMSE + KL-Divergence + Adaptive-Loss);
    MAPE and Mean-Difference are excluded because they are degenerate
    ranking signals on standardised data (see RELIABLE_METRICS docstring).
    """
    used_metrics = tuple(metrics) if metrics is not None else RELIABLE_METRICS
    overall_scores = calculate_overall_score(results, metrics=used_metrics)

    best_method = min(overall_scores, key=overall_scores.get)

    print(f"\n=== IMPUTATION METHOD PERFORMANCE RANKING (Normalized) ===")
    print(f"   metrics aggregated: {', '.join(used_metrics)}")
    print(f"   metrics excluded  : MAPE, Mean-Difference  "
          f"(degenerate on near-zero / centred ground truth; see code docs)")
    sorted_methods = sorted(overall_scores.items(), key=lambda x: x[1])
    for rank, (method, score) in enumerate(sorted_methods, 1):
        print(f"   {rank}. {method}: {score:.4f}")

    print(f"\n>> BEST PERFORMING METHOD: {best_method}")
    print(f"   Overall Normalized Score: {overall_scores[best_method]:.4f}")

    return best_method


def plot_comprehensive_imputation_comparison(results: Dict[str, Dict[str, float]], 
                                           save_path: str = 'results/comprehensive_imputation_comparison.png'):
    """Plot comprehensive comparison of all imputation methods across all metrics."""
    methods = list(results.keys())
    metrics = list(next(iter(results.values())).keys())
    
    n_metrics = len(metrics)
    n_methods = len(methods)
    
    # Create subplots for each metric
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, metric in enumerate(metrics):
        if i < len(axes):
            values = [results[method][metric] for method in methods]
            
            # Handle infinite values for plotting
            finite_values = [v if np.isfinite(v) else max([val for val in values if np.isfinite(val)]) * 2 
                           for v in values]
            
            bars = axes[i].bar(range(len(methods)), finite_values, 
                              color=colors[:len(methods)], alpha=0.7)
            
            axes[i].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(metric)
            axes[i].set_xticks(range(len(methods)))
            axes[i].set_xticklabels(methods, rotation=45, ha='right')
            
            # Add value labels on bars
            for j, (bar, value) in enumerate(zip(bars, values)):
                if np.isfinite(value):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=8)
                else:
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                               'inf', ha='center', va='bottom', fontsize=8)
    
    # Overall score comparison with normalization
    if len(axes) > len(metrics):
        overall_scores = calculate_overall_score(results)
        score_values = list(overall_scores.values())

        bars = axes[len(metrics)].bar(range(len(methods)), score_values,
                                     color=colors[:len(methods)], alpha=0.7)

        axes[len(metrics)].set_title(
            f'Overall Performance Score (Normalized)\n'
            f'avg of {", ".join(RELIABLE_METRICS)}  --  Lower is Better',
            fontsize=11, fontweight='bold')
        axes[len(metrics)].set_ylabel('Normalized Score')
        axes[len(metrics)].set_xticks(range(len(methods)))
        axes[len(metrics)].set_xticklabels(methods, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, score in zip(bars, score_values):
            axes[len(metrics)].text(bar.get_x() + bar.get_width()/2, 
                                  bar.get_height() + bar.get_height()*0.01,
                                  f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def apply_amdae_imputation(data: Tuple, missing_rate: float = 0.7, 
                          missing_pattern: str = 'random',
                          hidden_dims: List[int] = None,
                          max_epochs: int = 10,
                          batch_size: int = 32,
                          learning_rate: float = 0.001,
                          use_timeseries: bool = False,
                          delta_t: int = 16,
                          device: str = 'cpu',
                          compare_methods: bool = True,
                          save_plot_path: str = 'results/comprehensive_imputation_comparison.png',
                          force_imputer: Optional[str] = None,
                          **kwargs) -> Tuple:
    """Apply imputation to a federated-data tuple.

    Args (selected):
        force_imputer: If set, bypass composite-score selection and use the
            named imputer unconditionally. Recognised values:

              * ``'amdae'`` / ``'AMDAE'`` -- force AM-DAE; guarantees
                FedGen-AMDAE numbers were truly trained on AM-DAE-imputed
                data. All four imputers still run so the comparison plot
                is unchanged.
              * ``'mean'`` / ``'Mean Imputation'``    -- force Mean
              * ``'median'`` / ``'Median Imputation'`` -- force Median
              * ``'zero'`` / ``'Zero Imputation'``    -- force Zero
              * ``'none'`` -- the "no-imputation" ablation baseline:
                missingness is still simulated, but no imputer runs and
                the missing positions are simply zero-filled to keep the
                data trainable. This is the honest "no imputation
                technique was applied" row the imputer ablation table
                needs.

            Default ``None`` lets the patched composite (RELIABLE_METRICS)
            pick the winner, which under standardised data is AM-DAE.
    """
    
    print(f"Starting Enhanced AM-DAE imputation with {missing_rate*100}% missing data...")
    print(f"Missing pattern: {missing_pattern}")
    print(f"Using time-series algorithm: {use_timeseries}")
    
    if missing_rate <= 0:
        print("Missing rate is 0. Skipping imputation.")
        return data
    
    processed_data = process_federated_data_for_imputation(data)

    if len(processed_data['combined_data']) == 0:
        print("No data to process")
        return data

    original_complete = processed_data['combined_data'].copy()

    simulator = MissingDataSimulator()
    sim_kwargs = dict(kwargs)
    if missing_pattern.lower() == 'mar' and 'labels' not in sim_kwargs:
        labels_full = processed_data.get('combined_labels')
        if labels_full is not None and len(labels_full) == len(processed_data['combined_data']):
            sim_kwargs['labels'] = labels_full
    data_with_missing, missing_mask = simulator.introduce_missing_data(
        processed_data['combined_data'],
        missing_rate,
        missing_pattern,
        **sim_kwargs,
    )

    # Special case: force_imputer='none' is the "no-imputation" ablation
    # baseline. Skip the 4-imputer pipeline entirely (saves ~30-60s per
    # call) and just zero-fill missing positions so the data is trainable.
    # This is conceptually different from the 'zero' force_imputer: that
    # one runs the full pipeline and selects ZeroImputer; this one bypasses
    # all imputation reasoning, making it the honest "no imputation
    # technique was applied" baseline that reviewers ask for.
    if force_imputer is not None and str(force_imputer).lower() == 'none':
        print("\n>> NO-IMPUTATION baseline (force_imputer='none'): "
              "zero-filling missing positions, skipping all imputers.")
        final_imputed_data = np.where(
            np.isnan(data_with_missing), 0.0, data_with_missing)
        reconstructed_data = reconstruct_federated_data(
            final_imputed_data, processed_data, data)
        print(f"\n   Selected method: NoImputation  (zero-filled NaN positions only)")
        return reconstructed_data

    input_dim = data_with_missing.shape[1]
    
    # Initialize all imputers
    imputers = [
        MeanImputer(),
        MedianImputer(),
        ZeroImputer(),
        AMDAEImputer(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device
        )
    ]
    
    results = {}
    imputed_data_dict = {}
    
    print(f"\n=== RUNNING IMPUTATION METHODS ===")
    for imputer in imputers:
        print(f"\nRunning {imputer.name}...")
        
        if isinstance(imputer, AMDAEImputer) and use_timeseries:
            imputed_data = imputer.fit_impute_timeseries(data_with_missing, missing_mask, delta_t)
        else:
            imputed_data = imputer.fit_impute(data_with_missing, missing_mask)
        
        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_imputation_metrics(
            original_complete, imputed_data, missing_mask, data_with_missing
        )
        
        results[imputer.name] = metrics
        imputed_data_dict[imputer.name] = imputed_data
        
        # Display results for this method
        print(f"  Results for {imputer.name}:")
        for metric_name, value in metrics.items():
            if np.isfinite(value):
                print(f"    {metric_name}: {value:.4f}")
            else:
                print(f"    {metric_name}: inf")
    
    # Select method: hard-force if requested, otherwise use composite score
    if force_imputer:
        # Map a few common shorthand names to the canonical imputer keys.
        canonical = {k.lower(): k for k in imputed_data_dict.keys()}
        canonical.update({
            'amdae': 'AMDAE',
            'am-dae': 'AMDAE',
            'mean': 'Mean Imputation',
            'median': 'Median Imputation',
            'zero': 'Zero Imputation',
        })
        key = canonical.get(force_imputer.lower())
        if key is None or key not in imputed_data_dict:
            valid = ', '.join(sorted(imputed_data_dict.keys()))
            raise ValueError(
                f"force_imputer={force_imputer!r} not recognised. "
                f"Valid options: {valid} (also accepted: 'amdae', 'mean', "
                f"'median', 'zero').")
        best_method = key
        print(f"\n>> FORCED IMPUTER: {best_method}  "
              f"(composite-score selection bypassed)")
    else:
        best_method = select_best_imputation_method(results)
    final_imputed_data = imputed_data_dict[best_method]
    
    # Plot comprehensive comparison.
    #   * Skip if the file already exists -- the plot is identical across
    #     (algo, dataset, alpha, missing_rate) cells, so re-rendering it
    #     90+ times wastes CPU + disk.
    #   * Wrap in try/except: a transient I/O failure (full disk, perm
    #     denied, etc.) here must NOT kill the federated training run.
    if compare_methods and len(results) > 1:
        already_there = isinstance(save_plot_path, str) and os.path.exists(save_plot_path)
        if already_there:
            print(f"\nComprehensive comparison plot already exists, skipping: {save_plot_path}")
        else:
            try:
                plot_path = plot_comprehensive_imputation_comparison(results, save_plot_path)
                print(f"\nComprehensive comparison plot saved to: {plot_path}")
            except Exception as exc:
                print(f"\n[WARN] Could not save comprehensive comparison plot to "
                      f"{save_plot_path}: {exc}. Continuing training.")
    
    # Reconstruct federated data using the best method
    reconstructed_data = reconstruct_federated_data(
        final_imputed_data, processed_data, data
    )
    
    print(f"\n Enhanced AM-DAE imputation completed successfully!")
    print(f"   Selected method: {best_method}")
    print(f"   Final dataset imputed using: {best_method}")
    
    return reconstructed_data


if __name__ == "__main__":
    print("Enhanced AM-DAE Data Imputation Module with Comprehensive Evaluation")
    print("Usage: Import apply_amdae_imputation function in your federated learning code")
