#!/usr/bin/env bash
set -euo pipefail

: "${PRETRAIN_DATA:?Set PRETRAIN_DATA to the prepared data directory}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one GPU ID}"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
d="$root/analysis_1b"

if [[ ! -f "$d/mhc_step124000_model_only.pt" ]]; then
  python "$d/freeze.py"
fi
python "$d/training_gain.py"
python "$d/audit.py" smoke
python "$d/audit.py" roles
python "$d/streams.py" routes
python "$d/streams.py" states
python "$d/scheme.py"
python "$d/early_state_boundary.py"
python "$d/middle_preservation.py"
python "$d/causal.py" states
python "$d/causal.py" first
python "$d/causal.py" rescue
python "$d/mixer_extension.py"
python "$d/stream_sufficiency_curve.py"

python "$d/volume_physical_control.py"
python "$d/volume_remove_exchange.py"
python "$d/volume_physical_followup.py"
python "$d/volume_pair_swap.py"
python "$d/verify_remove_exchange.py"

python "$d/transfer.py" --start 0 --end 64
python "$d/transfer.py" --start 64 --end 128 --prefix transfer_confirm
python "$d/combine_transfer.py"
python "$root/volume_analysis/physical_destination_joint.py" --model 1b

python "$d/plot_router_heatmap.py"
python "$d/plot_report.py"
python "$d/summarize_volume_physical.py"
python "$d/plot_volume_physical.py"
python "$d/plot_volume_physical_schedule.py"
