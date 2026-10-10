"""Create local paths for the frozen-checkpoint experiments."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "baseline": ("diff36_base", "baseline", 1, False),
    "full_n1": ("diff36_lit_s1", "mhc", 1, False),
    "full_n2": ("diff36_lit_s2", "mhc", 2, False),
    "full_n3": ("diff36_lit_s3", "mhc", 3, False),
    "full_n4": ("diff36_lit", "mhc", 4, False),
    "ar_baseline": ("ar36_lit", "baseline", 1, True),
    "ar_n1": ("ar36_lit_s1", "mhc", 1, True),
    "ar_n2": ("ar36_lit_s2", "mhc", 2, True),
    "ar_n3": ("ar36_lit_s3", "mhc", 3, True),
    "ar_n4": ("ar36_lit", "mhc", 4, True),
}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs_pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu-idle-seconds", type=float, default=5)
    args = parser.parse_args()
    data = args.data_dir.resolve()
    runs = args.runs_dir.resolve()
    meta = json.loads((data / "meta.json").read_text())
    if meta["seq_len"] != 1024 or meta["n_val_blocks"] <= 2047:
        parser.error("analysis requires 1024-token blocks and at least 2048 validation blocks")
    models = {key: {"arm": arm, "streams": streams, "causal": causal,
                    "checkpoint": str(runs / run / "ckpt_53399.pt")}
              for key, (run, arm, streams, causal) in RUNS.items()}
    sources = [ROOT / "model_and_train.py", ROOT / "textdiffusion/models/mhc_lite.py"]
    config = {
        "dataset": str(data / "all_tokens.npy"), "data_meta": str(data / "meta.json"),
        "output": str(ROOT / "mechanism_audit_n2/results"),
        "models": models,
        "splits": {"probe_train": list(range(64)), "calibration": list(range(128, 192)),
                   "discovery": list(range(256, 384)), "confirmation": list(range(1024, 2048))},
        "ratio": 0.5, "confirmation_ratios": [0.1, 0.3, 0.7, 0.9],
        "mask_seed": 782, "intervention_seed": 130913, "bootstrap_seed": 130914,
        "bootstrap_repeats": 5000, "batch_size": args.batch_size, "head_chunk": 128,
        "gpu_policy": {"max_idle_memory_mib": 1024, "max_idle_utilization_percent": 5,
                       "two_observations_seconds": args.gpu_idle_seconds,
                       "never_use_training_gpus": [], "stop_if_another_compute_process_appears": True},
        "source_files": {str(path): digest(path) for path in sources},
    }
    destination = ROOT / "mechanism_audit_n2/config.json"
    destination.write_text(json.dumps(config, indent=2) + "\n")
    circuit = ROOT / "residual_circuits_n2"
    (circuit / "captures").mkdir(parents=True, exist_ok=True)
    full_checkpoint = Path(models["full_n2"]["checkpoint"])
    (circuit / "config.json").write_text(json.dumps({
        "checkpoint_step": 53399, "training_input_tokens": 3_499_622_400,
        "checkpoint": str(full_checkpoint),
        "checkpoint_sha256": digest(full_checkpoint) if full_checkpoint.is_file() else "unavailable",
        "models": models, "dataset": config["dataset"]}, indent=2) + "\n")
    missing = [entry["checkpoint"] for entry in models.values() if not Path(entry["checkpoint"]).is_file()]
    print(f"Wrote {destination}")
    if missing:
        print(f"{len(missing)} optional or required checkpoint paths do not yet exist; see README for each experiment's inputs")


if __name__ == "__main__":
    main()
