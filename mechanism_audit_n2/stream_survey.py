"""Is one residual stream parked?  Same measurement across n and across objectives.

Runs the three descriptive families — routing coefficients, per-step geometry and the
frozen-head lens — for any mHC-lite arm, diffusion or AR, with any stream count.  The
parking statistic is deliberately the one locked for the n=2 diffusion audit: inside the
a2..a8 window, the least-read stream's read share and its median per-step cosine.

python -m mechanism_audit_n2.stream_survey --model full_n3 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from .ar_volume import batches_ar, split_for
from .engine import batches, head_stats, load_model, make_split
from .factors import ROOT
from .primary import write_json
from .safety import GPUReservation
from .volume_noise_sweep import BoundedSharedGPUReservation

OUTPUT = ROOT / "stream_survey"
INTERFACES = tuple(f"{kind}{block}" for block in range(16) for kind in ("a", "m"))
PARKED_WINDOW = list(range(4, 17))     # a2..a8, as locked for the n=2 diffusion audit
STRIDE = 4
RATIO = .5
MAX_POSITIONS = 256
NLL, ENTROPY, TOP1, CORRECT = 0, 2, 4, 5

EXTRA_ARMS = {
    "full_n1": ("diff36_lit_s1", 1, False), "full_n3": ("diff36_lit_s3", 3, False),
    "ar_n1": ("ar36_lit_s1", 1, True), "ar_n3": ("ar36_lit_s3", 3, True),
    "ar_n4": ("ar36_lit", 4, True),
}


def model_config():
    config = json.loads((ROOT / "mechanism_audit_n2" / "config.json").read_text())
    for name, (directory, streams, causal) in EXTRA_ARMS.items():
        config["models"].setdefault(name, {
            "arm": "mhc", "streams": streams, "causal": causal,
            "checkpoint": str(ROOT / "runs_pt" / directory / "ckpt_53399.pt")})
    return config


def cosine(left, right):
    return torch.nn.functional.cosine_similarity(left.float(), right.float(), dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", default='0')
    parser.add_argument("--allow-external-mib", type=float, default=0)
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    config = model_config()
    spec = config["models"][args.model]
    streams, causal = spec["streams"], bool(spec.get("causal", False))
    permutations = math.factorial(streams)
    policy = dict(config["gpu_policy"]); policy["never_use_training_gpus"] = []
    reservation = (BoundedSharedGPUReservation(args.gpu, args.allow_external_mib)
                   if args.allow_external_mib else GPUReservation(args.gpu, policy))
    engine = load_model(config, args.model)

    roles = ("all",) if causal else ("masked", "visible")
    views = tuple(f"stream{k}" for k in range(streams)) + ("mean_readout",)
    routing = {role: {"read": np.zeros((32, streams)), "gain": np.zeros((32, streams)),
                      "p_identity": np.zeros(32), "p_nonidentity": np.zeros(32)} for role in roles}
    geometry = {role: {"step_cos": np.zeros((32, streams)), "norm": np.zeros((32, streams)),
                       "cos_pairs": np.zeros(32)} for role in roles}
    lens = {role: {view: {k: np.zeros(32) for k in ("ce", "entropy", "accuracy")}
                   for view in views} for role in roles}
    counts = {role: 0 for role in roles}
    weights = {role: 0 for role in roles}

    if causal:
        split = split_for(config, "calibration")
        source = ((ids, valid, ids[:, 1:], None) for _, _, ids, valid
                  in batches_ar(split, config["batch_size"]))
        total = len(split["ids"])
    else:
        split = make_split(config, "calibration", RATIO)
        source = ((corrupted, mask, ids, mask) for _, _, _, _, corrupted, ids, mask
                  in batches(split, config["batch_size"]))
        total = len(split["ids"])

    done = 0
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        for corrupted, flags, targets, mask in source:
            reservation.check()
            if causal:
                selectors = {"all": torch.arange(0, corrupted.shape[1] - 1,
                                                 STRIDE, device=corrupted.device)}
            else:
                selectors = {}
            state = engine.initial(corrupted)
            previous = None
            collected = {role: [] for role in roles}
            for index in range(32):
                state, _, router, _, _ = engine.interface(index, state, flags)
                coefficients = router.float()
                for role in roles:
                    if causal:
                        picked = state[:, selectors["all"]]
                        route = coefficients[:, selectors["all"]].reshape(-1, coefficients.shape[-1])
                        labels = targets[:, selectors["all"]].reshape(-1)
                        previous_picked = previous[:, selectors["all"]] if previous is not None else None
                    else:
                        chosen = mask if role == "masked" else ~mask
                        picked = state[chosen]
                        route = coefficients[chosen]
                        labels = targets[chosen]
                        previous_picked = previous[chosen] if previous is not None else None
                    weight = picked.shape[0] * (picked.shape[1] if causal else 1)
                    flat = picked.reshape(-1, streams, picked.shape[-1])
                    routing[role]["read"][index] += route[:, :streams].sum(0).cpu().numpy()
                    routing[role]["gain"][index] += route[:, streams:2 * streams].sum(0).cpu().numpy()
                    routing[role]["p_identity"][index] += float(route[:, 2 * streams].sum())
                    if permutations > 1:
                        routing[role]["p_nonidentity"][index] += float(route[:, 2 * streams + 1:].sum())
                    geometry[role]["norm"][index] += flat.float().norm(dim=-1).sum(0).cpu().numpy()
                    if previous_picked is not None:
                        old = previous_picked.reshape(-1, streams, picked.shape[-1])
                        geometry[role]["step_cos"][index] += cosine(flat, old).sum(0).cpu().numpy()
                    pairs = [float(cosine(flat[:, a], flat[:, b]).sum())
                             for a in range(streams) for b in range(a + 1, streams)]
                    geometry[role]["cos_pairs"][index] += sum(pairs) / max(len(pairs), 1)
                    stack = torch.stack([flat[:, k] for k in range(streams)] + [flat.mean(1)])
                    stats, _ = head_stats(engine.model, stack, labels, config["head_chunk"])
                    collected[role].append(stats)
                    if index == 0:
                        counts[role] += flat.shape[0]
                        weights[role] += weight
                previous = state.detach().clone()

            for role in roles:
                stacked = np.stack(collected[role])
                for slot, view in enumerate(views):
                    local = stacked[:, slot]
                    lens[role][view]["ce"] += local[:, :, NLL].sum(1)
                    lens[role][view]["entropy"] += local[:, :, ENTROPY].sum(1)
                    lens[role][view]["accuracy"] += local[:, :, CORRECT].sum(1)
            done += corrupted.shape[0]
            print("SURVEY", args.model, done, "/", total, flush=True)

    results = {"model": args.model, "streams": streams, "causal": causal,
               "checkpoint": spec["checkpoint"], "sequences": total,
               "interfaces": list(INTERFACES), "views": list(views), "roles": {}}
    for role in roles:
        n = counts[role]
        read = routing[role]["read"] / n
        share = read / np.maximum(read.sum(-1, keepdims=True), 1e-12)
        step = geometry[role]["step_cos"] / n
        step[0] = np.nan
        window_share = share[PARKED_WINDOW].mean(0)
        quiet = int(window_share.argmin())
        results["roles"][role] = {
            "tokens": int(n),
            "read_share": share.tolist(), "read": read.tolist(),
            "gain": (routing[role]["gain"] / n).tolist(),
            "p_nonidentity": (routing[role]["p_nonidentity"] / n).tolist(),
            "step_cos": [[None if np.isnan(v) else float(v) for v in row] for row in step],
            "norm": (geometry[role]["norm"] / n).tolist(),
            "cos_pairs_mean": (geometry[role]["cos_pairs"] / n).tolist(),
            "parking": {
                "window": f"{INTERFACES[PARKED_WINDOW[0]]}-{INTERFACES[PARKED_WINDOW[-1]]}",
                "quietest_stream": quiet,
                "read_share_mean": float(window_share[quiet]),
                "read_share_max": float(share[PARKED_WINDOW, quiet].max()),
                "step_cos_median": float(np.nanmedian(step[PARKED_WINDOW, quiet])),
                "step_cos_min": float(np.nanmin(step[PARKED_WINDOW, quiet]))},
            "lens": {view: {k: (v / n).tolist() for k, v in local.items()}
                     for view, local in lens[role].items()}}

    write_json(OUTPUT / f"{args.model}.json", results)
    for role in roles:
        p = results["roles"][role]["parking"]
        final = results["roles"][role]["lens"]["mean_readout"]["ce"][31]
        print(f"{args.model:<10} n={streams} {'AR' if causal else 'diff':<5} role={role:<8}"
              f" самый тихий поток {p['quietest_stream']}:"
              f" доля чтения {p['read_share_mean']:.3f} (макс {p['read_share_max']:.3f}),"
              f" step cos медиана {p['step_cos_median']:.3f} мин {p['step_cos_min']:.3f}"
              f" | финальный CE {final:.3f}", flush=True)
    print("WROTE", OUTPUT / f"{args.model}.json", flush=True)


if __name__ == "__main__":
    main()
