#!/usr/bin/env bash
set -euo pipefail

: "${PRETRAIN_DATA:?Set PRETRAIN_DATA to the prepared data directory}"
: "${GPU_ID:?Set GPU_ID to an idle physical GPU index}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

python -m scripts.configure_analysis --data-dir "$PRETRAIN_DATA" --runs-dir "${RUNS_DIR:-runs_pt}"
python -m scripts.capture_router --gpu "$GPU_ID"

python -m mechanism_audit_n2.side_drift --gpu "$GPU_ID"
python -m mechanism_audit_n2.side_retrieval --prepare
for role in masked visible; do
  for ratio in .1 .5 .9; do
    python -m mechanism_audit_n2.side_retrieval --ratio "$ratio" --role "$role" --gpu "$GPU_ID"
  done
done
python -m mechanism_audit_n2.side_retrieval --render
for role in masked visible; do
  python -m mechanism_audit_n2.side_identity --role "$role" --gpu "$GPU_ID"
done
python -m mechanism_audit_n2.side_identity --render
python -m mechanism_audit_n2.evidence_transfer --gpu "$GPU_ID"
python -m mechanism_audit_n2.evidence_transfer --render
python -m mechanism_audit_n2.stream_atlas

# The paper's write-volume claims use proportional physical-gain reconstruction.
# Its destination and joint rows use direct mean(g / sum(g)) calibration.
v="$root/volume_analysis"
CUDA_VISIBLE_DEVICES="$GPU_ID" python "$v/setup.py"
for group in smoke means shuffle_masked shuffle_visible depth_full noise_roles remove_exchange; do
  CUDA_VISIBLE_DEVICES="$GPU_ID" python "$v/run.py" --group "$group"
done
python "$v/analyze.py"
python "$v/verify.py"
python "$v/render.py"
CUDA_VISIBLE_DEVICES="$GPU_ID" python "$v/physical_destination_joint.py" --model 337m
