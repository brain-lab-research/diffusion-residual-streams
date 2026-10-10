"""Freeze paired validation values through the audited checkpoint."""

import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS_DIR = Path(os.environ.get("RUNS_DIR", ROOT.parent / "runs_pt"))
RUNS = {
    "mhc": RUNS_DIR / "diff1b_mhc_n2_10b_s1337/metrics.jsonl",
    "baseline": RUNS_DIR / "diff1b_base_10b_s1337/metrics.jsonl",
}


def values(path):
    result = {}
    for line in path.open():
        row = json.loads(line)
        if "val_mean" in row and row["step"] <= 124000:
            result[row["step"]] = row
    return result


def main():
    logged = {name: values(path) for name, path in RUNS.items()}
    common = sorted(set(logged["mhc"]) & set(logged["baseline"]))
    assert common and common[-1] == 124000
    rows = []
    for step in common:
        mhc = logged["mhc"][step]["val_mean"]
        baseline = logged["baseline"][step]["val_mean"]
        rows.append({"step": step, "mhc": mhc, "baseline": baseline,
                     "gain_ce": baseline - mhc,
                     "perplexity_advantage_percent": (math.exp(baseline - mhc) - 1) * 100})
    latest_ratios = {}
    for ratio in (0.1, 0.3, 0.5, 0.7, 0.9):
        key = f"r{ratio}"
        mhc = logged["mhc"][124000][key]
        baseline = logged["baseline"][124000][key]
        latest_ratios[key] = {"mhc": mhc, "baseline": baseline,
                              "gain_ce": baseline - mhc}
    result = {"checkpoint_step": 124000,
              "validation_protocol": "mean of CE at mask ratios 0.1, 0.3, 0.5, 0.7, 0.9 on the fixed validation blocks",
              "matched_steps": len(rows), "latest": rows[-1],
              "latest_ratios": latest_ratios, "rows": rows}
    (ROOT / "training_gain.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["latest"], indent=2))


if __name__ == "__main__":
    main()
