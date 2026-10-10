#!/usr/bin/env bash
set -euo pipefail

# The paper compares these two 1B MDLM runs at checkpoint index 124000.
# Set CUDA_VISIBLE_DEVICES to exactly two GPU IDs before calling this script.
: "${PRETRAIN_DATA:?Set PRETRAIN_DATA to the prepared data directory}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to two GPU IDs, for example 0,1}"
if [[ "$CUDA_VISIBLE_DEVICES" != *,* || "$CUDA_VISIBLE_DEVICES" == *,*,* ]]; then
  echo "Expected exactly two CUDA_VISIBLE_DEVICES entries" >&2
  exit 2
fi

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export MODEL_DIM=1536 MODEL_LAYERS=32 MODEL_HEADS=24 MODEL_KV_HEADS=6
export MODEL_INTERMEDIATE=4096 MHC_STREAMS=2 MHC_RESMIX=learned
export MHC_PAPER=1 MHC_PROPER_INIT=1
RUNS_DIR="${RUNS_DIR:-runs_pt}"

case "${1:-}" in
  baseline) name=diff1b_base_10b_s1337; arm=baseline ;;
  n2) name=diff1b_mhc_n2_10b_s1337; arm=mhc ;;
  *) echo "Usage: bash scripts/train_1b.sh {baseline|n2}" >&2; exit 2 ;;
esac

torchrun --standalone --nproc_per_node=2 model_and_train_ddp.py \
  --arm "$arm" --objective mdlm --target-tokens 10000000000 \
  --stop-after 124001 --micro 2 --accum 16 \
  --lr 3e-4 --router-lr 3e-4 --warmup-steps 2000 \
  --val-every 1000 --val-micro 1 --save-every 500 --keep-every 25000 \
  --out "$RUNS_DIR/$name"
