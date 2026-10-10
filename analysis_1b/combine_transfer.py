"""Combine independent 64-sequence source-target searches at step 124000."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
NAMES = ("source_both", "target_both", "target_s0", "target_s1")


def main():
    packs = [json.loads((ROOT / name).read_text()) for name in
             ("transfer_raw.json", "transfer_confirm_raw.json")]
    assert [len(p["records"]) for p in packs] == [64, 64]
    assert all(p["checkpoint_step"] == 124000 for p in packs)
    groups = [[r for r in p["records"] if r["selected"]] for p in packs]
    rows = groups[0] + groups[1]
    result = {"checkpoint_step": 124000,
              "sequences_screened": 128,
              "pairs_pilot": len(groups[0]),
              "pairs_confirm": len(groups[1]),
              "pairs_total": len(rows),
              "adjacent_pairs": sum(abs(r["distance"]) == 1 for r in rows),
              "absolute_distance_counts": {str(distance): sum(abs(r["distance"]) == distance for r in rows)
                                           for distance in sorted({abs(r["distance"]) for r in rows})},
              "depths": list(packs[0]["depths"]),
              "bootstrap": {"unit": "sequence (all selected pairs per sampled sequence)",
                            "draws": 5000, "seed": 130914,
                            "sequences_selected": len({r["block"] for r in rows})},
              "curves": {}}
    rng = np.random.default_rng(130914)
    sequence_ids = np.array([r["block"] for r in rows])
    unique = np.unique(sequence_ids)
    members = [np.flatnonzero(sequence_ids == sequence_id) for sequence_id in unique]
    draws = [np.concatenate([members[i] for i in rng.integers(len(unique), size=len(unique))])
             for _ in range(5000)]
    for name in NAMES:
        vals = np.array([r["recovery"][name] for r in rows])
        boot = np.array([vals[index].mean(axis=0) for index in draws])
        result["curves"][name] = {
            "mean": vals.mean(axis=0).tolist(),
            "ci95_low": np.quantile(boot, 0.025, axis=0).tolist(),
            "ci95_high": np.quantile(boot, 0.975, axis=0).tolist(),
            "pilot_mean": np.array([r["recovery"][name] for r in groups[0]]).mean(axis=0).tolist(),
            "confirm_mean": np.array([r["recovery"][name] for r in groups[1]]).mean(axis=0).tolist(),
        }
    for label, positive, negative in (("target_minus_source", "target_both", "source_both"),
                                      ("target_s0_minus_s1", "target_s0", "target_s1")):
        vals = np.array([np.array(r["recovery"][positive]) -
                         np.array(r["recovery"][negative]) for r in rows])
        boot = np.array([vals[index].mean(axis=0) for index in draws])
        result[label] = {"mean": vals.mean(axis=0).tolist(),
                         "ci95_low": np.quantile(boot, 0.025, axis=0).tolist(),
                         "ci95_high": np.quantile(boot, 0.975, axis=0).tolist()}
    (ROOT / "transfer_combined.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("pairs_total", "adjacent_pairs", "curves")}))


if __name__ == "__main__":
    main()
