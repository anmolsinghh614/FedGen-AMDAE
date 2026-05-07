#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: ./run_experiments.sh <alpha> <dataset_name> <missing_ratio>
# Example: ./run_experiments.sh 0.1 EMnist 0.1

if [[ $# -ne 3 ]]; then
  echo "Usage: $(basename "$0") <alpha> <dataset_name> <missing_ratio>" >&2
  echo "  alpha: e.g. 0.1, 1.0, 10.0" >&2
  echo "  dataset_name: EMnist | Mnist | 'UCI HAR'" >&2
  echo "  missing_ratio: e.g. 0.1, 0.2" >&2
  exit 1
fi

ALPHA="$1"
DATA_NAME="$2"
MISSING_RATIO="$3"
GLOB_ITERS=100  # Default number of global rounds

# Determine the dataset sampling ratio based on the dataset name
if [[ "$DATA_NAME" == "EMnist" ]]; then
    DATA_RATIO=0.1
elif [[ "$DATA_NAME" == "Mnist" ]]; then
    DATA_RATIO=0.5
elif [[ "$DATA_NAME" == "UCI HAR" ]]; then
    DATA_RATIO=0.5
else
    echo "Unknown dataset: $DATA_NAME. Expected 'EMnist', 'Mnist', or 'UCI HAR'." >&2
    exit 1
fi

# Construct the full dataset string (e.g., EMnist-alpha0.1-ratio0.1)
DATASET="${DATA_NAME}-alpha${ALPHA}-ratio${DATA_RATIO}"
echo "====================================================================="
echo "Starting Experiments for $DATASET with Missing Ratio: $MISSING_RATIO"
echo "====================================================================="

# Algorithms as shown in the plot/table
ALGORITHMS=(FedGen FedAvg FedProx FedEnsemble FedDistill)

# Shared parameters
COMMON_ARGS=(
  --dataset "$DATASET"
  --batch_size 64
  --local_epochs 20
  --num_users 10
  --num_glob_iters "$GLOB_ITERS"
  --times 1
  --embedding 0
  --learning_rate 0.01
  --gen_batch_size 64
  --missing_rate "$MISSING_RATIO"
)

# Run algorithms sequentially
for algo in "${ALGORITHMS[@]}"; do
  echo "--------------------------------------------------------"
  echo "Running $algo ..."
  echo "--------------------------------------------------------"
  python main.py --algorithm "$algo" "${COMMON_ARGS[@]}"
done

echo "====================================================================="
echo "All algorithms finished training."
echo "====================================================================="

# ---------- Generate readable results summary ----------
RESULTS_FILE="results/experiment_results_${DATASET}_mr${MISSING_RATIO}.txt"
echo "Generating results summary → $RESULTS_FILE"

ALGO_CSV=$(IFS=,; echo "${ALGORITHMS[*]}")

python -c "
import h5py, numpy as np, os, sys, datetime

dataset      = '${DATASET}'
missing_ratio= '${MISSING_RATIO}'
algorithms   = '${ALGO_CSV}'.split(',')
result_path  = 'results/models'
lr           = 0.01
num_users    = 10
batch_size   = 64
local_epochs = 20
seed         = 0          # times=1 → only seed 0
gen_batch_size = 64
embedding    = 0

def build_h5_name(algo):
    name = dataset + '_' + algo
    name += '_' + str(lr) + '_' + str(num_users)
    name += 'u' + '_' + str(batch_size) + 'b' + '_' + str(local_epochs)
    name += '_' + str(seed)
    if 'FedGen' in algo:
        name += '_embed' + str(embedding)
        if int(gen_batch_size) != int(batch_size):
            name += '_gb' + str(gen_batch_size)
    return name

lines = []
lines.append('=' * 70)
lines.append('EXPERIMENT RESULTS SUMMARY')
lines.append('=' * 70)
lines.append(f'Date       : {datetime.datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
lines.append(f'Dataset    : {dataset}')
lines.append(f'Missing %  : {float(missing_ratio)*100:.0f}%')
lines.append(f'Glob Iters : ${GLOB_ITERS}')
lines.append(f'Local Epochs: {local_epochs}')
lines.append(f'Batch Size : {batch_size}')
lines.append(f'Num Users  : {num_users}')
lines.append('=' * 70)
lines.append('')
lines.append(f'{\"Algorithm\":<16} {\"Final Acc (%)\":>14} {\"Best Acc (%)\":>14} {\"Best Round\":>12}')
lines.append('-' * 60)

for algo in algorithms:
    h5_name = build_h5_name(algo)
    h5_path = os.path.join(result_path, h5_name + '.h5')
    if not os.path.exists(h5_path):
        lines.append(f'{algo:<16} {\"-- file not found --\":>42}')
        continue
    hf = h5py.File(h5_path, 'r')
    glob_acc = np.array(hf.get('glob_acc')[:])
    hf.close()
    final_acc = glob_acc[-1] * 100
    best_acc  = np.max(glob_acc) * 100
    best_round = int(np.argmax(glob_acc)) + 1
    lines.append(f'{algo:<16} {final_acc:>13.2f}% {best_acc:>13.2f}% {best_round:>10d}')

lines.append('-' * 60)
lines.append('')
lines.append('Note: Accuracy is on the global test set after each round.')

out_path = '${RESULTS_FILE}'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('Results written to', out_path)
"

echo ""
cat "$RESULTS_FILE"
echo ""

# ---------- Generate performance plots using main_plot.py ----------
echo "====================================================================="
echo "Generating performance graphs via main_plot.py ..."
echo "====================================================================="

python main_plot.py \
    --dataset "$DATASET" \
    --algorithms "$ALGO_CSV" \
    --num_glob_iters "$GLOB_ITERS" \
    --local_epochs 20 \
    --times 1 \
    --batch_size 64 \
    --gen_batch_size 64 \
    --num_users 10

echo "====================================================================="
echo "Done!  Results → $RESULTS_FILE"
echo "====================================================================="
