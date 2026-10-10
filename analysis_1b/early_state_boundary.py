"""Find how early the 1B s0 state can replace its state after a16.

The post-m0 and post-a8 endpoints already have full 128-sequence results in
state_interventions.json and middle_preservation.json. This script tests three
intermediate snapshots with the same intervention, sample, and metric.
"""

import numpy as np
import torch
import torch.nn.functional as F

from audit import CONFIG, ROOT, batches, load_model, make_split, paired_delta, save_json, sequence_ce


REFERENCES = {"m1": 3, "m4": 9, "m6": 13}
TARGET = 32  # after a16


def main():
    torch.set_num_threads(4)
    engine = load_model(CONFIG, "full_n2")
    assert engine.checkpoint_step == 124000 and len(engine.interfaces) == 64
    split = make_split(CONFIG, "test", 0.5)
    with np.load(ROOT / "intact.npz") as saved:
        intact = {key: saved[key] for key in saved.files}
    np.testing.assert_array_equal(intact["block_ids"], split["block_ids"])
    names = [(role, label, index) for role in ("masked", "visible")
             for label, index in REFERENCES.items()]
    sums = {(role, label): [] for role, label, _ in names}
    counts = {(role, label): [] for role, label, _ in names}
    stats = {(role, label): {"cosine_sum": 0.0, "norm_ratio_sum": 0.0, "n": 0}
             for role, label, _ in names}
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        for _, end, _, _, corrupted, ids, mask in batches(split, CONFIG["batch_size"]):
            _, cache = engine.run(corrupted, mask, cache=True)
            current = cache["states"][TARGET]
            for role, label, index in names:
                selected = mask if role == "masked" else ~mask
                reference = cache["states"][index]
                source = reference[..., 0, :].float()[selected]
                destination = current[..., 0, :].float()[selected]
                key = role, label
                stats[key]["cosine_sum"] += float(F.cosine_similarity(source, destination, dim=-1).sum())
                stats[key]["norm_ratio_sum"] += float((destination.norm(dim=-1) /
                    source.norm(dim=-1).clamp_min(1e-12)).sum())
                stats[key]["n"] += len(source)
                patched = current.clone()
                patched[..., 0, :] = torch.where(
                    selected[..., None], reference[..., 0, :], patched[..., 0, :])
                final, _ = engine.run(corrupted, mask, initial_state=patched, start_after=TARGET)
                local_sum, local_count = sequence_ce(engine, final, ids, mask)
                sums[key].append(local_sum)
                counts[key].append(local_count)
            if end % 16 == 0:
                print("BOUNDARY", end, "/", len(split["ids"]), flush=True)
    result = {"checkpoint_step": 124000, "ratio": 0.5,
              "test_sequences": len(split["ids"]), "target": "after a16, stream 0",
              "references": REFERENCES, "conditions": {}}
    for role, label, _ in names:
        key = role, label
        pack = {"seq_ce_sum": np.concatenate(sums[key]),
                "seq_mask_count": np.concatenate(counts[key]),
                "block_ids": split["block_ids"]}
        np.testing.assert_array_equal(pack["seq_mask_count"], intact["seq_mask_count"])
        np.savez_compressed(ROOT / f"early_state_{role}_{label}_to_a16.npz", **pack)
        value = stats[key]
        result["conditions"][f"{role}_{label}"] = {
            "mean_cosine": value["cosine_sum"] / value["n"],
            "mean_norm_ratio": value["norm_ratio_sum"] / value["n"],
            "positions": value["n"],
            "delta_ce": paired_delta(pack, intact),
        }
    save_json(ROOT / "early_state_boundary.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
