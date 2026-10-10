"""Does the parked side stream actually keep its early content?

The routing profile only says the side stream is rarely *read* in the middle
blocks.  This measures the state itself: how far the side vector drifts from
its a0/m0 value, and how much active-stream content leaks into it.

python -m mechanism_audit_n2.side_drift --gpu 0
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .engine import batches, load_model, make_split
from .factors import ROOT
from .primary import write_json
from .safety import GPUReservation
from .volume_noise_sweep import BoundedSharedGPUReservation

OUTPUT = ROOT / "side_retrieval_n2"
INTERFACES = tuple(f"{kind}{block}" for block in range(16) for kind in ("a", "m"))
RATIO = .5


def cosine(left, right):
    left = left.float(); right = right.float()
    return torch.nn.functional.cosine_similarity(left, right, dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default='0')
    parser.add_argument("--allow-external-mib", type=float, default=0)
    args = parser.parse_args()
    config = json.loads((ROOT / "mechanism_audit_n2" / "config.json").read_text())
    policy = dict(config["gpu_policy"]); policy["never_use_training_gpus"] = []
    reservation = (BoundedSharedGPUReservation(args.gpu, args.allow_external_mib)
                   if args.allow_external_mib else GPUReservation(args.gpu, policy))
    engine = load_model(config, "full_n2")
    split = make_split(config, "discovery", RATIO)

    # role -> metric -> per-interface accumulator.  Step cosines and norms are
    # tracked for both streams; the a0/m0 references only for the side stream.
    keys = ("cos_a0", "cos_m0", "cos_prev", "cos_active", "norm_ratio_m0", "norm",
            "cos_prev_s0", "cos_prev_s1", "norm_s0", "norm_s1")
    totals = {role: {key: np.zeros(32) for key in keys} for role in ("masked", "visible")}
    counts = {role: 0 for role in ("masked", "visible")}

    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        for begin, end, _, _, corrupted, ids, mask in batches(split, config["batch_size"]):
            reservation.check()
            state = engine.initial(corrupted)
            reference = {}
            previous = None
            previous_streams = None
            per_interface = []
            for index in range(32):
                state, _, _, _, _ = engine.interface(index, state, mask)
                side = state[..., 1, :]
                active = state[..., 0, :]
                if index in (0, 1):
                    reference[index] = side.detach().clone()
                # a0 predates the m0 reference, so its m0 columns are undefined.
                blank = torch.full_like(side[..., 0], float("nan"))
                row = {
                    "cos_a0": cosine(side, reference[0]),
                    "cos_m0": cosine(side, reference[1]) if 1 in reference else blank,
                    "cos_prev": cosine(side, previous) if previous is not None else blank,
                    "cos_active": cosine(side, active),
                    "norm": side.float().norm(dim=-1),
                    "norm_ratio_m0": (side.float().norm(dim=-1)
                                      / reference[1].float().norm(dim=-1).clamp_min(1e-9))
                                     if 1 in reference else blank,
                    "cos_prev_s0": cosine(active, previous_streams[0])
                                   if previous_streams is not None else blank,
                    "cos_prev_s1": cosine(side, previous_streams[1])
                                   if previous_streams is not None else blank,
                    "norm_s0": active.float().norm(dim=-1),
                    "norm_s1": side.float().norm(dim=-1),
                }
                per_interface.append(row)
                previous = side.detach().clone()
                previous_streams = (active.detach().clone(), side.detach().clone())

            for role, selected in (("masked", mask), ("visible", ~mask)):
                counts[role] += int(selected.sum())
                for index, row in enumerate(per_interface):
                    for key, value in row.items():
                        totals[role][key][index] += float(value[selected].sum())
            print("DRIFT", end, "/", len(split["ids"]), flush=True)

    results = {"ratio": RATIO, "checkpoint_step": engine.checkpoint_step,
               "sequences": len(split["ids"]), "interfaces": list(INTERFACES),
               "side_stream": 1, "active_stream": 0, "roles": {}}
    for role in ("masked", "visible"):
        results["roles"][role] = {"tokens": counts[role],
                                  **{key: [None if np.isnan(item) else float(item)
                                           for item in value / counts[role]]
                                     for key, value in totals[role].items()}}
    write_json(OUTPUT / "side_drift.json", results)

    for role in ("masked", "visible"):
        local = results["roles"][role]
        print(f"\n===== {role} side stream =====")
        print(f"{'iface':<6}{'step cos s0':>13}{'step cos s1':>13}{'|s0|':>9}{'|s1|':>9}"
              f"{'cos(a0)':>9}{'cos(m0)':>9}{'cos(s0,s1)':>12}")
        for index, label in enumerate(INTERFACES):
            def cell(key, width):
                item = local[key][index]
                return ("n/a" if item is None else f"{item:.3f}").rjust(width)
            print(f"{label:<6}{cell('cos_prev_s0', 13)}{cell('cos_prev_s1', 13)}"
                  f"{cell('norm_s0', 9)}{cell('norm_s1', 9)}"
                  f"{cell('cos_a0', 9)}{cell('cos_m0', 9)}{cell('cos_active', 12)}")
    print("WROTE", OUTPUT / "side_drift.json", flush=True)


if __name__ == "__main__":
    main()
