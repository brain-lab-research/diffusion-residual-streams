"""Describe both 1B residual streams without assuming the 337M stream labels."""
import json

import numpy as np
import torch
import torch.nn.functional as F

from audit import CONFIG, ROOT, batches, load_model, make_split, save_json


def router_profile():
    with np.load(ROOT / "calibration_routes.npz") as saved:
        routes = saved["router"].astype(np.float64)
        mask = saved["mask"].ravel()
    result = {"checkpoint_step": 124000, "sequences": 64, "ratio": 0.5,
              "roles": {}}
    for name, select in (("masked", mask), ("visible", ~mask)):
        mean = routes[:, select].mean(1)
        read = mean[:, :2]
        gains = mean[:, 2:4]
        result["roles"][name] = {
            "read": read.tolist(), "read_share": (read/read.sum(-1, keepdims=True)).tolist(),
            "gains": gains.tolist(), "write_share": (gains/gains.sum(-1, keepdims=True)).tolist(),
            "volume": gains.sum(-1).tolist(),
            "mixer_swap_probability": mean[:, 5].tolist(),
        }
    save_json(ROOT / "router_profile.json", result)
    for name, role in result["roles"].items():
        r = np.array(role["read_share"]); g = np.array(role["write_share"])
        print(name, "a0 write_share", g[0], "mid read_share", r[8:40].mean(0),
              "late read_share", r[48:64].mean(0), flush=True)


def state_profile():
    torch.set_num_threads(4)
    engine = load_model(CONFIG, "full_n2")
    split = make_split(CONFIG, "test", .5)
    split = {**split, "ids": split["ids"][:32], "mask": split["mask"][:32],
             "block_ids": split["block_ids"][:32]}
    totals = np.zeros((2, 2, 64), dtype=np.float64)
    norm_ratios = np.zeros_like(totals)
    counts = np.zeros(2, dtype=np.int64)
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        for _, end, _, _, corrupted, ids, mask in batches(split, CONFIG["batch_size"]):
            _, cache = engine.run(corrupted, mask, cache=True)
            reference = cache["states"][1]
            for role, selected in enumerate((mask, ~mask)):
                counts[role] += int(selected.sum())
                for stream in (0, 1):
                    baseline = reference[..., stream, :].float()
                    baseline_norm = baseline.norm(dim=-1).clamp_min(1e-9)
                    for layer, state in enumerate(cache["states"]):
                        current = state[..., stream, :].float()
                        totals[role, stream, layer] += float(F.cosine_similarity(current, baseline, dim=-1)[selected].sum())
                        norm_ratios[role, stream, layer] += float((current.norm(dim=-1)/baseline_norm)[selected].sum())
            if end % 8 == 0:
                print("STATES", end, "/", len(split["ids"]), flush=True)
    result = {"checkpoint_step": engine.checkpoint_step, "sequences": len(split["ids"]),
              "ratio": 0.5, "reference": "post-m0 at the same position",
              "interface_labels": [f"{kind}{i}" for i in range(32) for kind in ("a", "m")],
              "roles": {}}
    for role, name in enumerate(("masked", "visible")):
        result["roles"][name] = {
            "positions": int(counts[role]),
            "cosine_to_m0": (totals[role]/counts[role]).tolist(),
            "norm_ratio_to_m0": (norm_ratios[role]/counts[role]).tolist(),
        }
    save_json(ROOT / "state_profile.json", result)
    for name, role in result["roles"].items():
        c = np.asarray(role["cosine_to_m0"])
        print(name, "cosine at a16", c[:, 32], "at m24", c[:, 49],
              "at m31", c[:, 63], flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["routes", "states", "all"])
    stage = parser.parse_args().stage
    if stage in ("routes", "all"):
        router_profile()
    if stage in ("states", "all"):
        state_profile()
