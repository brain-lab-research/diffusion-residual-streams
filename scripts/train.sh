#!/usr/bin/env bash
set -euo pipefail

: "${PRETRAIN_DATA:?Set PRETRAIN_DATA to the directory containing all_tokens.npy and meta.json}"
RUNS_DIR="${RUNS_DIR:-runs_pt}"
mkdir -p "$RUNS_DIR"
export MHC_PAPER=1 MHC_PROPER_INIT=1 MHC_RESMIX=learned

train() {
  local name="$1" arm="$2" streams="$3" objective="$4"
  export MHC_STREAMS="$streams"
  python model_and_train.py \
    --arm "$arm" --objective "$objective" --out "$RUNS_DIR/$name" \
    --steps 53400 --micro 16 --accum 4 --lr 3e-4 --router-lr 3e-4 \
    --warmup-steps 300 --val-every 500 --save-every 5000 --keep-every 5000
}

case "${1:-}" in
  baseline) train diff36_base baseline 1 mdlm ;;
  gate) train diff36_gate gate 1 mdlm ;;
  n1) train diff36_lit_s1 mhc 1 mdlm ;;
  n2) train diff36_lit_s2 mhc 2 mdlm ;;
  n3) train diff36_lit_s3 mhc 3 mdlm ;;
  n4) train diff36_lit mhc 4 mdlm ;;
  frr) train diff36_frr_s2 frr 2 mdlm ;;
  ar_baseline) train ar36_lit baseline 1 ar ;;
  ar_n1) train ar36_lit_s1 mhc 1 ar ;;
  ar_n2) train ar36_lit_s2 mhc 2 ar ;;
  ar_n3) train ar36_lit_s3 mhc 3 ar ;;
  ar_n4) train ar36_lit mhc 4 ar ;;
  *) echo "Usage: PRETRAIN_DATA=/path/to/data bash scripts/train.sh {baseline|gate|n1|n2|n3|n4|frr|ar_baseline|ar_n1|ar_n2|ar_n3|ar_n4}" >&2; exit 2 ;;
esac
