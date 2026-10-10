"""Test when each 1B stream alone suffices at matched relative depths.

At a chosen post-MLP boundary, replace the other stream at selected positions
with a copy of the tested stream, then rerun the remaining interfaces. This is
the same state surgery used by state_interventions.py/causal.py and the 337M
stream-collapse sweep. All effects are masked-target CE on the 128-sequence
test split at corruption ratio 0.5.
"""

import numpy as np
import torch

from audit import CONFIG, ROOT, batches, load_model, make_split, paired_delta, save_json, sequence_ce


DEPTHS = {f"m{block}": 2 * block + 1 for block in (8, 12, 16, 20, 24, 28)}


def main():
    torch.set_num_threads(4)
    engine = load_model(CONFIG, "full_n2")
    assert engine.checkpoint_step == 124000 and len(engine.interfaces) == 64
    split = make_split(CONFIG, "test", 0.5)
    with np.load(ROOT / "intact.npz") as saved:
        intact = {key: saved[key] for key in saved.files}
    np.testing.assert_array_equal(intact["block_ids"], split["block_ids"])
    names = [(role, label, index, stream)
             for role in ("masked", "visible")
             for label, index in DEPTHS.items()
             for stream in (0, 1)]
    sums = {tuple(name): [] for name in names}
    counts = {tuple(name): [] for name in names}
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        for _, end, _, _, corrupted, ids, mask in batches(split, CONFIG["batch_size"]):
            _, cache = engine.run(corrupted, mask, cache=True)
            for role, label, index, stream in names:
                selected = mask if role == "masked" else ~mask
                current = cache["states"][index]
                patched = current.clone()
                patched[..., 1 - stream, :] = torch.where(
                    selected[..., None], current[..., stream, :],
                    patched[..., 1 - stream, :])
                final, _ = engine.run(corrupted, mask, initial_state=patched,
                                      start_after=index)
                local_sum, local_count = sequence_ce(engine, final, ids, mask)
                key = (role, label, index, stream)
                sums[key].append(local_sum)
                counts[key].append(local_count)
            if end % 16 == 0:
                print("SUFFICIENCY", end, "/", len(split["ids"]), flush=True)
    result = {"checkpoint_step": 124000, "ratio": 0.5,
              "test_sequences": len(split["ids"]), "depths": DEPTHS,
              "intervention": "duplicate one selected stream at selected-role positions",
              "conditions": {}}
    for role, label, index, stream in names:
        key = (role, label, index, stream)
        pack = {"seq_ce_sum": np.concatenate(sums[key]),
                "seq_mask_count": np.concatenate(counts[key]),
                "block_ids": split["block_ids"]}
        np.testing.assert_array_equal(pack["seq_mask_count"], intact["seq_mask_count"])
        result["conditions"][f"{role}_{label}_only_s{stream}"] = paired_delta(pack, intact)
    save_json(ROOT / "stream_sufficiency_curve.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
