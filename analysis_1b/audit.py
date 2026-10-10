"""Frozen 1B checks of the arXiv article.

The training checkpoint is never modified.  All analyses use the disjoint
calibration and test blocks from the 337M protocol, with the 1B architecture.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(
    MODEL_DIM="1536", MODEL_LAYERS="32", MODEL_HEADS="24",
    MODEL_KV_HEADS="6", MODEL_INTERMEDIATE="4096", MHC_STREAMS="2",
    MHC_RESMIX="learned", MHC_PAPER="1", MHC_PROPER_INIT="1",
)

import numpy as np
import torch
import torch.nn.functional as F

from mechanism_audit_n2.engine import batches, load_model, make_split
from mechanism_audit_n2.role_interventions import RoleFactorHooks, calibration_means

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "mhc_step124000_model_only.pt"
DATA = Path(os.environ.get("PRETRAIN_DATA", "data/pretrain_fwe_3600")) / "all_tokens.npy"
CONFIG = {
    "models": {"full_n2": {"arm": "mhc", "streams": 2, "checkpoint": str(CHECKPOINT)}},
    "dataset": str(DATA),
    "mask_seed": 782,
    "ratio": 0.5,
    "splits": {"calibration": list(range(128, 192)),
               "test": list(range(1024, 1152))},
    "batch_size": 2,
    "head_chunk": 128,
}


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sequence_ce(engine, state, ids, mask, chunk=128):
    """Actual-mask CE, preserving a separate summed loss for each sequence."""
    hidden = state.mean(-2)[mask]
    labels = ids[mask]
    pieces = []
    for begin in range(0, len(labels), chunk):
        logits = engine.model.head(engine.model.norm(hidden[begin:begin+chunk])).float()
        pieces.append(F.cross_entropy(logits, labels[begin:begin+chunk], reduction="none"))
    loss = torch.cat(pieces)
    seq_index = torch.arange(ids.shape[0], device=ids.device)[:, None].expand_as(mask)[mask]
    sums = torch.zeros(ids.shape[0], device=ids.device, dtype=torch.float64)
    sums.index_add_(0, seq_index, loss.double())
    return sums.cpu().numpy(), mask.sum(-1).cpu().numpy()


def evaluate(engine, split, name, hooks=None, fixed=(), policy="dynamic"):
    path = ROOT / (name + ".npz")
    if path.exists():
        with np.load(path) as saved:
            return {key: saved[key] for key in saved.files}
    sums, counts = [], []
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        for begin, end, _, _, corrupted, ids, mask in batches(split, CONFIG["batch_size"]):
            if hooks is not None:
                hooks.configure(mask, fixed, policy=policy, layers=range(len(engine.interfaces)))
            state, _ = engine.run(corrupted, mask)
            local_sum, local_count = sequence_ce(engine, state, ids, mask)
            sums.append(local_sum); counts.append(local_count)
            if end % 16 == 0:
                print("EVAL", name, end, "/", len(split["ids"]), flush=True)
    result = {"seq_ce_sum": np.concatenate(sums), "seq_mask_count": np.concatenate(counts),
              "block_ids": split["block_ids"], "mask_sha256": np.array(split["mask_sha256"])}
    np.savez_compressed(path, **result)
    print("DONE", name, float(result["seq_ce_sum"].sum()/result["seq_mask_count"].sum()), flush=True)
    return result


def paired_delta(condition, intact, repeats=5000):
    np.testing.assert_array_equal(condition["block_ids"], intact["block_ids"])
    np.testing.assert_array_equal(condition["seq_mask_count"], intact["seq_mask_count"])
    diff = condition["seq_ce_sum"] - intact["seq_ce_sum"]
    counts = intact["seq_mask_count"]
    rng = np.random.default_rng(130914)
    draws = rng.integers(len(diff), size=(repeats, len(diff)))
    estimates = diff[draws].sum(1) / counts[draws].sum(1)
    return {"delta_ce": float(diff.sum()/counts.sum()),
            "ci95": np.quantile(estimates, [.025, .975]).tolist()}


def calibrate(engine):
    path = ROOT / "calibration_routes.npz"
    split = make_split(CONFIG, "calibration", .5)
    if path.exists():
        with np.load(path) as saved:
            router = saved["router"]
            np.testing.assert_array_equal(saved["block_ids"], split["block_ids"])
            np.testing.assert_array_equal(saved["mask"], split["mask"])
    else:
        captures = []
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            for begin, end, _, _, corrupted, ids, mask in batches(split, CONFIG["batch_size"]):
                local = []
                def capture(index, before, after, route, routed, delta):
                    local.append(route.flatten(0, 1).float().cpu().numpy())
                engine.run(corrupted, mask, callback=capture)
                captures.append(np.stack(local))
                if end % 16 == 0:
                    print("CALIBRATION", end, "/", len(split["ids"]), flush=True)
        router = np.concatenate(captures, axis=1)
        np.savez_compressed(path, router=router, mask=split["mask"], block_ids=split["block_ids"])
    assert router.shape == (64, len(split["ids"])*split["ids"].shape[1], 6)
    means = calibration_means(router, split["mask"].ravel(),
                              np.ones(split["mask"].size, dtype=bool))
    summary = {"checkpoint_step": engine.checkpoint_step,
               "calibration_blocks": [int(x) for x in split["block_ids"]],
               "mask_sha256": split["mask_sha256"],
               "means": {policy: {factor: values.tolist() for factor, values in factors.items()}
                         for policy, factors in means.items()},
               "router_shape": list(router.shape)}
    save_json(ROOT / "calibration.json", summary)
    return means, router, split


def roles():
    torch.set_num_threads(4)
    engine = load_model(CONFIG, "full_n2")
    assert engine.checkpoint_step == 124000 and len(engine.interfaces) == 64
    means, router, calibration_split = calibrate(engine)
    test = make_split(CONFIG, "test", .5)
    hooks = RoleFactorHooks(engine, means)
    conditions = [("read", "global"), ("read", "role"), ("read", "role_swap"),
                  ("mixer", "global")]
    try:
        intact = evaluate(engine, test, "intact")
        summary = {"checkpoint_step": engine.checkpoint_step,
                   "test_sequences": len(test["ids"]), "ratio": .5,
                   "intact_ce": float(intact["seq_ce_sum"].sum()/intact["seq_mask_count"].sum()),
                   "conditions": {}}
        for factor, policy in conditions:
            name = f"{factor}_{policy}"
            pack = evaluate(engine, test, name, hooks, (factor,), policy)
            summary["conditions"][name] = paired_delta(pack, intact)
            save_json(ROOT / "roles.json", summary)
    finally:
        hooks.close()
    print(json.dumps(summary), flush=True)


def smoke():
    torch.set_num_threads(4)
    engine = load_model(CONFIG, "full_n2")
    assert engine.checkpoint_step == 124000
    assert len(engine.interfaces) == 64
    split = make_split(CONFIG, "test", 0.5)
    _, _, _, _, corrupted, ids, mask = next(batches(split, 1))
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        direct = engine.model(corrupted)
        state, _ = engine.run(corrupted, mask)
        instrumented = engine.original_logits(state)
        torch.testing.assert_close(direct, instrumented, atol=0, rtol=0)
        values = engine.final_stats(state, ids, mask, CONFIG["head_chunk"])
        local_sum, local_count = sequence_ce(engine, state, ids, mask)
        assert abs(local_sum.sum()/local_count.sum() - values[:, 0].mean()) < 1e-6
    result = {"checkpoint_step": engine.checkpoint_step,
              "interfaces": len(engine.interfaces),
              "mask_count": int(mask.sum()),
              "instrumented_logits_exact": True,
              "masked_ce": float(values[:, 0].mean())}
    (ROOT / "smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["smoke", "roles"])
    args = parser.parse_args()
    globals()[args.stage]()
