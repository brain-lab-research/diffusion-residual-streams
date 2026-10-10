"""Does the AR n=2 arm park one residual stream the way the diffusion arm does?

Same three measurement families as the diffusion audit, on the matched AR checkpoint
(same size, same step, same data, causal attention, next-token objective):

  routing   read / write / mixer coefficients per interface
  geometry  per-step cosine of each stream, norms, angle between streams
  lens      the frozen head applied to stream 0, stream 1 and their mean

AR has no masked/visible split, so every scored position is one population.

python -m mechanism_audit_n2.ar_streams --model ar_n2 --gpu 0
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .ar_volume import batches_ar, split_for
from .engine import head_stats, load_model
from .factors import ROOT
from .primary import write_json
from .safety import GPUReservation
from .volume_noise_sweep import BoundedSharedGPUReservation

OUTPUT = ROOT / "ar_streams_n2"
INTERFACES = tuple(f"{kind}{block}" for block in range(16) for kind in ("a", "m"))
VIEWS = ("stream0", "stream1", "mean_readout")
STRIDE = 4          # scored positions per sequence: 0, 4, 8, ... keeps the lens affordable
NLL, ENTROPY, TOP1, CORRECT = 0, 2, 4, 5


def cosine(left, right):
    return torch.nn.functional.cosine_similarity(left.float(), right.float(), dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ar_n2")
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--gpu", default='0')
    parser.add_argument("--allow-external-mib", type=float, default=0)
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    config = json.loads((ROOT / "mechanism_audit_n2" / "config.json").read_text())
    policy = dict(config["gpu_policy"]); policy["never_use_training_gpus"] = []
    reservation = (BoundedSharedGPUReservation(args.gpu, args.allow_external_mib)
                   if args.allow_external_mib else GPUReservation(args.gpu, policy))
    engine = load_model(config, args.model)
    split = split_for(config, args.split)

    routing = {name: np.zeros(32) for name in
               ("read_s0", "read_s1", "gain_s0", "gain_s1", "p_identity", "p_swap")}
    geometry = {name: np.zeros(32) for name in
                ("cos_prev_s0", "cos_prev_s1", "norm_s0", "norm_s1", "cos_s0_s1",
                 "cos_a0_s1", "cos_m0_s1", "cos_a0_s0", "cos_m0_s0")}
    lens = {view: {name: np.zeros(32) for name in ("ce", "entropy", "accuracy", "agrees_with_final")}
            for view in VIEWS}
    tokens = 0

    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        for begin, end, ids, valid in batches_ar(split, config["batch_size"]):
            reservation.check()
            positions = torch.arange(0, ids.shape[1] - 1, STRIDE, device=ids.device)
            labels = ids[:, 1:][:, positions].reshape(-1)
            state = engine.initial(ids)
            reference, previous = {}, None
            collected = []
            for index in range(32):
                state, _, router, _, _ = engine.interface(index, state, valid)
                coefficients = router.float().reshape(-1, 6).mean(0).cpu().numpy()
                for slot, name in enumerate(("read_s0", "read_s1", "gain_s0", "gain_s1",
                                             "p_identity", "p_swap")):
                    routing[name][index] += float(coefficients[slot]) * ids.shape[0]

                scored = state[:, positions]                        # [batch, pos, streams, hidden]
                active, side = scored[..., 0, :], scored[..., 1, :]
                if index in (0, 1):
                    reference[index] = (active.detach().clone(), side.detach().clone())
                blank = torch.full_like(active[..., 0], float("nan"))
                row = {
                    "cos_prev_s0": cosine(active, previous[0]) if previous else blank,
                    "cos_prev_s1": cosine(side, previous[1]) if previous else blank,
                    "norm_s0": active.float().norm(dim=-1),
                    "norm_s1": side.float().norm(dim=-1),
                    "cos_s0_s1": cosine(active, side),
                    "cos_a0_s0": cosine(active, reference[0][0]),
                    "cos_a0_s1": cosine(side, reference[0][1]),
                    "cos_m0_s0": cosine(active, reference[1][0]) if 1 in reference else blank,
                    "cos_m0_s1": cosine(side, reference[1][1]) if 1 in reference else blank,
                }
                for name, value in row.items():
                    geometry[name][index] += float(value.float().sum())
                previous = (active.detach().clone(), side.detach().clone())

                flat = scored.reshape(-1, scored.shape[2], scored.shape[3])
                views = torch.stack([flat[:, 0], flat[:, 1], flat.mean(1)])
                stats, _ = head_stats(engine.model, views, labels, config["head_chunk"])
                collected.append(stats)

            stacked = np.stack(collected)                            # [32, views, tokens, 8]
            final = stacked[31, VIEWS.index("mean_readout"), :, TOP1]
            tokens += stacked.shape[2]
            for slot, view in enumerate(VIEWS):
                local = stacked[:, slot]
                lens[view]["ce"] += local[:, :, NLL].sum(1)
                lens[view]["entropy"] += local[:, :, ENTROPY].sum(1)
                lens[view]["accuracy"] += local[:, :, CORRECT].sum(1)
                lens[view]["agrees_with_final"] += (local[:, :, TOP1] == final[None, :]).sum(1)
            print("AR", end, "/", len(split["ids"]), flush=True)

    sequences = len(split["ids"])
    results = {"model": args.model, "arm": config["models"][args.model],
               "split": args.split, "sequences": sequences, "scored_tokens": int(tokens),
               "stride": STRIDE, "interfaces": list(INTERFACES),
               "objective": "teacher-forced next token; no input corruption",
               "routing": {k: (v / sequences).tolist() for k, v in routing.items()},
               "geometry": {k: [None if np.isnan(x) else float(x)
                                for x in (v / tokens)] for k, v in geometry.items()},
               "lens": {view: {k: (v / tokens).tolist() for k, v in local.items()}
                        for view, local in lens.items()}}
    share = np.array(results["routing"]["read_s1"]) / np.maximum(
        np.array(results["routing"]["read_s0"]) + np.array(results["routing"]["read_s1"]), 1e-12)
    results["routing"]["side_read_share"] = share.tolist()
    write_json(OUTPUT / f"{args.model}_streams.json", results)

    print(f"\n===== {args.model}: {tokens} scored tokens =====")
    print(f"{'iface':<6}{'read s0':>9}{'read s1':>9}{'share s1':>10}{'gain s0':>9}{'gain s1':>9}"
          f"{'step cos s0':>13}{'step cos s1':>13}{'|s0|':>9}{'|s1|':>9}{'cos(s0,s1)':>12}"
          f"{'CE s0':>9}{'CE s1':>9}{'CE mean':>9}")
    for index, label in enumerate(INTERFACES):
        def cell(source, key, width, digits=3):
            item = source[key][index]
            return ("n/a" if item is None else f"{item:.{digits}f}").rjust(width)
        print(f"{label:<6}"
              f"{cell(results['routing'],'read_s0',9)}{cell(results['routing'],'read_s1',9)}"
              f"{cell(results['routing'],'side_read_share',10)}"
              f"{cell(results['routing'],'gain_s0',9)}{cell(results['routing'],'gain_s1',9)}"
              f"{cell(results['geometry'],'cos_prev_s0',13)}{cell(results['geometry'],'cos_prev_s1',13)}"
              f"{cell(results['geometry'],'norm_s0',9,2)}{cell(results['geometry'],'norm_s1',9,2)}"
              f"{cell(results['geometry'],'cos_s0_s1',12)}"
              f"{cell(results['lens']['stream0'],'ce',9,2)}{cell(results['lens']['stream1'],'ce',9,2)}"
              f"{cell(results['lens']['mean_readout'],'ce',9,2)}")
    print("WROTE", OUTPUT / f"{args.model}_streams.json", flush=True)


if __name__ == "__main__":
    main()
