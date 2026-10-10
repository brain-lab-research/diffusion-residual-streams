"""Is the parked side state functionally just its early (a0 / after-m0) content?

The main audit patched early states immediately before a11.  That point sits
past the end of the parked phase: the side stream is rewritten during the
m8..m10 transition (cos to m0 falls .949 -> .672, norm grows 1.17x -> 2.55x),
so an early raw state is off-manifold there and the test cannot separate
"wrong content" from "wrong depth".

Here the substitution happens at the parked boundary itself (after a8) on the
otherwise clean trajectory, and the network then runs m8..m15 normally.  A
patch that leaves CE at intact level means the parked state carries no
functionally relevant information beyond the patched early state.

python -m mechanism_audit_n2.side_identity --role masked --gpu 0
python -m mechanism_audit_n2.side_identity --render
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .engine import aggregate, batches, load_model, make_split, summarize_tokens
from .factors import ROOT
from .primary import load_pack, save_pack, write_json
from .safety import GPUReservation
from .side_retrieval import (CHECKPOINT_META, DISRUPT_AFTER, clean_trajectory, paired_effect,
                             patch_role_stream, run_range, selected_positions,
                             shuffle_role_stream, selection_path)
from .volume_noise_sweep import BoundedSharedGPUReservation

OUTPUT = ROOT / "side_retrieval_n2"
EFFECTS = OUTPUT / "identity"
RATIO = .5
ROLES = ("masked", "visible")
# a0 and m0 snapshots, patched at the parked boundary (after a8).
CONDITIONS = ("intact", "a8_shuffle", "a8_patch_a0_raw", "a8_patch_m0_raw",
              "a8_patch_a0_normmatched", "a8_patch_m0_normmatched")
SOURCES = {"a0": 0, "m0": 1}


def pack_path(role: str, condition: str):
    path = EFFECTS / role / f"{condition}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def evaluate(engine, split, config, reservation, role: str, selection: dict):
    if all(pack_path(role, name).exists() for name in CONDITIONS):
        print("CACHED", role, flush=True)
        return
    side = selection["roles"][role]["side_stream"]
    partial = EFFECTS / role / "worker.partial.npz"
    completed = 0
    values = {name: [] for name in CONDITIONS}
    diagnostics = {"shuffle": {}, "patch_norm_errors": {}}
    if partial.exists():
        saved = load_pack(partial)
        assert json.loads(str(saved["conditions_json"])) == list(CONDITIONS)
        completed = int(saved["completed"])
        values = {name: [saved[name]] for name in CONDITIONS}
        diagnostics = json.loads(str(saved["diagnostics_json"]))
        print("RESUME", role, completed, "/", len(split["ids"]), flush=True)

    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        for begin, end, _, _, corrupted, ids, mask in batches(split, config["batch_size"]):
            if end <= completed:
                continue
            assert begin >= completed
            reservation.check()
            chosen = selected_positions(mask, role)

            intact_state, snapshots = clean_trajectory(engine, corrupted, mask)
            values["intact"].append(engine.final_stats(intact_state, ids, mask, config["head_chunk"]))

            boundary = snapshots[DISRUPT_AFTER]
            disrupted, local = shuffle_role_stream(boundary, chosen, side)
            for key, value in local.items():
                if key.startswith("minimum"):
                    diagnostics["shuffle"][key] = min(diagnostics["shuffle"].get(key, value), value)
                else:
                    diagnostics["shuffle"][key] = max(diagnostics["shuffle"].get(key, 0.), value)
            state = run_range(engine, disrupted, mask, DISRUPT_AFTER + 1, 31)
            values["a8_shuffle"].append(engine.final_stats(state, ids, mask, config["head_chunk"]))

            for source, index in SOURCES.items():
                for suffix, normmatched in (("raw", False), ("normmatched", True)):
                    patched, error = patch_role_stream(boundary, snapshots[index], chosen,
                                                       side, normmatched=normmatched)
                    key = f"{source}_{suffix}"
                    diagnostics["patch_norm_errors"][key] = max(
                        diagnostics["patch_norm_errors"].get(key, 0.), error)
                    state = run_range(engine, patched, mask, DISRUPT_AFTER + 1, 31)
                    values[f"a8_patch_{source}_{suffix}"].append(
                        engine.final_stats(state, ids, mask, config["head_chunk"]))

            if end % 16 == 0 or end == len(split["ids"]):
                payload = {"completed": np.array(end),
                           "conditions_json": np.array(json.dumps(list(CONDITIONS))),
                           "diagnostics_json": np.array(json.dumps(diagnostics))}
                payload.update({name: np.concatenate(items) for name, items in values.items()})
                temporary = partial.with_name(partial.name + ".tmp")
                with temporary.open("wb") as target:
                    np.savez_compressed(target, **payload)
                temporary.replace(partial)
                print("EVALUATE", role, end, "/", len(split["ids"]), flush=True)
            del intact_state, snapshots, boundary, disrupted, patched, state

    intact = summarize_tokens(split, np.concatenate(values["intact"]))
    intact["diagnostics_json"] = np.array(json.dumps(diagnostics))
    save_pack(pack_path(role, "intact"), intact)
    for name in CONDITIONS:
        if name == "intact":
            continue
        pack = summarize_tokens(split, np.concatenate(values[name]))
        pack["diagnostics_json"] = np.array(json.dumps(diagnostics))
        save_pack(pack_path(role, name), pack, intact)
    partial.unlink(missing_ok=True)
    print("COMPLETE", role, flush=True)


def render(config: dict):
    results = {"ratio": RATIO, "patch_point": "after a8 (parked boundary), network reruns m8..m15",
               "bootstrap": {"unit": "sequence-paired", "repeats": config["bootstrap_repeats"],
                             "seed": config["bootstrap_seed"]}, "roles": {}}
    for role in ROLES:
        if not all(pack_path(role, name).exists() for name in CONDITIONS):
            continue
        packs = {name: load_pack(pack_path(role, name)) for name in CONDITIONS}
        intact = packs["intact"]
        effects = {name: {**aggregate(pack), **paired_effect(pack, intact, config)}
                   for name, pack in packs.items()}
        contrasts = {}
        for suffix in ("raw", "normmatched"):
            contrasts[f"m0_minus_a0_{suffix}"] = paired_effect(
                packs[f"a8_patch_m0_{suffix}"], packs[f"a8_patch_a0_{suffix}"], config)
            for source in SOURCES:
                contrasts[f"a8_patch_{source}_{suffix}_vs_shuffle"] = paired_effect(
                    packs[f"a8_patch_{source}_{suffix}"], packs["a8_shuffle"], config)
        results["roles"][role] = {"effects_vs_intact": effects, "contrasts": contrasts}
    write_json(OUTPUT / "side_identity.json", results)

    def fmt(value):
        return f'{value["delta_ce"]:+.5f} [{value["ci95"][0]:+.5f}, {value["ci95"][1]:+.5f}]'
    for role, local in results["roles"].items():
        print(f"\n===== {role}: patch at the parked boundary (after a8) =====")
        for name in CONDITIONS:
            if name == "intact":
                continue
            print(f"  {name:<28}{fmt(local['effects_vs_intact'][name])}")
        print("  -- contrasts --")
        for name, value in local["contrasts"].items():
            print(f"  {name:<28}{fmt(value)}")
    print("WROTE", OUTPUT / "side_identity.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gpu", default='0')
    parser.add_argument("--allow-external-mib", type=float, default=0)
    args = parser.parse_args()
    config = json.loads((ROOT / "mechanism_audit_n2" / "config.json").read_text())
    if args.render:
        render(config)
        return
    if args.role is None:
        raise ValueError("--role is required unless --render is used")
    selection = json.loads(selection_path().read_text())
    split = make_split(config, "discovery", RATIO)
    policy = dict(config["gpu_policy"]); policy["never_use_training_gpus"] = []
    reservation = (BoundedSharedGPUReservation(args.gpu, args.allow_external_mib)
                   if args.allow_external_mib else GPUReservation(args.gpu, policy))
    engine = load_model(config, "full_n2")
    if engine.checkpoint_step != CHECKPOINT_META["checkpoint_step"]:
        raise AssertionError((engine.checkpoint_step, CHECKPOINT_META["checkpoint_step"]))
    evaluate(engine, split, config, reservation, args.role, selection)


if __name__ == "__main__":
    main()
