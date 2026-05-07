#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: ./run_all.sh <dataset_string> <num_glob_iters>
# Example: ./run_all.sh EMnist-alpha0.1-ratio0.1 60
# Example: ./run_all.sh "UCI HAR-alpha0.1-ratio0.1" 60

if [[ $# -ne 2 ]]; then
  echo "Usage: $(basename "$0") <dataset_string> <num_glob_iters>" >&2
  exit 1
fi

DATASET="$1"        # e.g., EMnist-alpha0.1-ratio0.1
GLOB_ITERS="$2"     # e.g., 60

ALGORITHMS=(FedGen FedAvg FedProx FedEnsemble FedDistill)

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
)

for algo in "${ALGORITHMS[@]}"; do
  echo "Running ${algo} with dataset=${DATASET}, num_glob_iters=${GLOB_ITERS}..."
  python main.py --algorithm "$algo" "${COMMON_ARGS[@]}"
done
