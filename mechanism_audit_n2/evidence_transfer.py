"""Where along depth does stream 0 carry evidence from a visible source into a masked target?

For a masked target i and a causally important visible token j, two runs differ only
in whether j is visible.  Restoring the clean stream-0 state of ONE position at depth
L and finishing the forward pass answers two different questions:

  source patch  restore s0 at j  -> can the evidence still leave its origin at depth L?
  target patch  restore s0 at i  -> has the evidence already arrived inside the target?

Recovery is measured on the target's own log-probability:
    R = (logp_patched - logp_corrupt) / (logp_clean - logp_corrupt)

python -m mechanism_audit_n2.evidence_transfer --gpu 0
python -m mechanism_audit_n2.evidence_transfer --render
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .engine import batches, load_model, make_split, training
from .factors import ROOT
from .primary import write_json
from .safety import GPUReservation
from .side_retrieval import CHECKPOINT_META, run_range
from .volume_noise_sweep import BoundedSharedGPUReservation

OUTPUT = ROOT / "side_retrieval_n2"
RATIO = .5
INTERFACES = tuple(f"{kind}{block}" for block in range(16) for kind in ("a", "m"))
# Masking the source corrupts BOTH of its streams, so restoring only one can look weak for
# the wrong reason.  Every patch is therefore run three ways, at the source and at the target,
# plus a specificity control that restores an irrelevant visible position instead.
STREAM_SETS = {"sE": (0,), "sP": (1,), "both": (0, 1)}
CONDITIONS = tuple(f"{place}_{streams}" for place in ("source", "target")
                   for streams in STREAM_SETS) + ("control_irrelevant_both",)
TARGETS_PER_SEQUENCE = 6  # several targets per sequence, so the pair count is not 1 per block
NEAR_CANDIDATES = 3       # nearest visible positions on either side of each target
FAR_CANDIDATES = 2        # uniformly sampled visible positions per target
# Masking one candidate changes the whole sequence, so a single forward scores every target
# at once: each target ends up choosing from the pooled candidate columns, not just its own.
MIN_DROP = .5             # nats; below this the recovery denominator is noise
SELECTION_SEED = 130914


def target_logprob(engine, state, ids, positions):
    """log p(true token) at one position per sequence."""
    rows = torch.arange(state.shape[0], device=state.device)
    picked = state[rows, positions].mean(-2)
    logits = engine.model.head(engine.model.norm(picked)).float()
    labels = ids[rows, positions]
    return logits.log_softmax(-1).gather(1, labels[:, None]).squeeze(1)


def forward(engine, corrupted, mask, keep=None):
    state = engine.initial(corrupted)
    snapshots = {}
    for index in range(32):
        state, _, _, _, _ = engine.interface(index, state, mask)
        if keep is not None and index in keep:
            snapshots[index] = state.detach().clone()
    return state, snapshots


def choose_targets(engine, corrupted, mask, ids, config):
    """Several masked targets per sequence, ranked by clean confidence; no intervention seen."""
    state, _ = forward(engine, corrupted, mask)
    collapsed = state.mean(-2)
    picks, valid = [], []
    for row in range(state.shape[0]):
        positions = mask[row].nonzero(as_tuple=True)[0]
        scores = torch.full((positions.numel(),), -1e9, device=state.device)
        for begin in range(0, positions.numel(), config["head_chunk"]):
            chunk = positions[begin:begin + config["head_chunk"]]
            logits = engine.model.head(engine.model.norm(collapsed[row, chunk])).float()
            labels = ids[row, chunk]
            value = logits.log_softmax(-1).gather(1, labels[:, None]).squeeze(1)
            correct = logits.argmax(-1).eq(labels)
            scores[begin:begin + chunk.numel()] = torch.where(
                correct, value, torch.full_like(value, -1e9))
        count = min(TARGETS_PER_SEQUENCE, positions.numel())
        top = scores.topk(count).indices
        chosen = positions[top]
        if chosen.numel() < TARGETS_PER_SEQUENCE:
            chosen = torch.cat([chosen, chosen[-1:].repeat(TARGETS_PER_SEQUENCE - chosen.numel())])
            top = torch.cat([top, top[-1:].repeat(TARGETS_PER_SEQUENCE - top.numel())])
        picks.append(chosen)
        valid.append(scores[top] > -1e8)
    return torch.stack(picks), torch.stack(valid)


def candidate_sources(mask, targets, generator):
    """Pooled visible candidates for the batch: neighbours of every target plus random picks."""
    per_row = []
    for row in range(mask.shape[0]):
        visible = (~mask[row]).nonzero(as_tuple=True)[0]
        columns = []
        for slot in range(targets.shape[1]):
            distance = (visible - targets[row, slot]).abs()
            order = distance.argsort()
            columns.append(visible[order[:2 * NEAR_CANDIDATES]])
            rest = visible[order[2 * NEAR_CANDIDATES:]]
            if rest.numel() > FAR_CANDIDATES:
                index = torch.randperm(rest.numel(), generator=generator).to(rest.device)
                rest = rest[index[:FAR_CANDIDATES]]
            columns.append(rest)
        per_row.append(torch.cat(columns))
    width = min(item.numel() for item in per_row)
    return torch.stack([item[:width] for item in per_row])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default='0')
    parser.add_argument("--allow-external-mib", type=float, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "mechanism_audit_n2" / "config.json").read_text())
    if args.render:
        render()
        return
    policy = dict(config["gpu_policy"]); policy["never_use_training_gpus"] = []
    reservation = (BoundedSharedGPUReservation(args.gpu, args.allow_external_mib)
                   if args.allow_external_mib else GPUReservation(args.gpu, policy))
    engine = load_model(config, "full_n2")
    if engine.checkpoint_step != CHECKPOINT_META["checkpoint_step"]:
        raise AssertionError((engine.checkpoint_step, CHECKPOINT_META["checkpoint_step"]))
    split = make_split(config, "discovery", RATIO)
    generator = torch.Generator().manual_seed(SELECTION_SEED)

    records = []
    partial = OUTPUT / "evidence_transfer.partial.json"
    completed = 0
    if partial.exists():
        saved = json.loads(partial.read_text())
        records, completed = saved["records"], saved["completed"]
        print("RESUME", completed, "/", len(split["ids"]), flush=True)

    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        for begin, end, _, _, corrupted, ids, mask in batches(split, config["batch_size"]):
            if end <= completed:
                continue
            reservation.check()
            rows = torch.arange(corrupted.shape[0], device=corrupted.device)
            targets, valid = choose_targets(engine, corrupted, mask, ids, config)

            clean_state, clean_snapshots = forward(engine, corrupted, mask, keep=set(range(32)))
            clean_logp = torch.stack([target_logprob(engine, clean_state, ids, targets[:, slot])
                                      for slot in range(targets.shape[1])], dim=1)

            # One forward per pooled candidate scores every target at once.
            pool = candidate_sources(mask, targets, generator)
            drops = torch.full((corrupted.shape[0], targets.shape[1], pool.shape[1]),
                               -1e9, device=corrupted.device)
            for column in range(pool.shape[1]):
                candidate = pool[:, column]
                variant = corrupted.clone(); variant[rows, candidate] = training.MASK_ID
                variant_mask = mask.clone(); variant_mask[rows, candidate] = True
                state, _ = forward(engine, variant, variant_mask)
                for slot in range(targets.shape[1]):
                    hit = candidate.eq(targets[:, slot])   # never a target, but stay safe
                    drop = clean_logp[:, slot] - target_logprob(engine, state, ids, targets[:, slot])
                    drops[:, slot, column] = torch.where(hit, torch.full_like(drop, -1e9), drop)
            best = drops.argmax(-1)
            best_source = torch.gather(pool, 1, best)
            # least causally important candidate: the specificity control patches this one
            quiet = drops.abs().argmin(-1)
            quiet_source = torch.gather(pool, 1, quiet)

            for slot in range(targets.shape[1]):
                position_i = targets[:, slot]
                position_j = best_source[:, slot]
                variant = corrupted.clone(); variant[rows, position_j] = training.MASK_ID
                variant_mask = mask.clone(); variant_mask[rows, position_j] = True
                corrupt_state, corrupt_snapshots = forward(engine, variant, variant_mask,
                                                           keep=set(range(32)))
                corrupt_logp = target_logprob(engine, corrupt_state, ids, position_i)

                position_q = quiet_source[:, slot]
                curves = {name: np.zeros((corrupted.shape[0], 32)) for name in CONDITIONS}
                plan = [(f"source_{k}", position_j, v) for k, v in STREAM_SETS.items()]
                plan += [(f"target_{k}", position_i, v) for k, v in STREAM_SETS.items()]
                plan += [("control_irrelevant_both", position_q, (0, 1))]
                for depth in range(32):
                    for name, position, streams in plan:
                        patched = corrupt_snapshots[depth].clone()
                        for stream in streams:
                            patched[rows, position, stream] = \
                                clean_snapshots[depth][rows, position, stream]
                        state = (run_range(engine, patched, variant_mask, depth + 1, 31)
                                 if depth < 31 else patched)
                        curves[name][:, depth] = (
                            target_logprob(engine, state, ids, position_i).float().cpu().numpy())

                for row in range(corrupted.shape[0]):
                    records.append({
                        "block": int(split["block_ids"][begin + row]), "slot": slot,
                        "valid": bool(valid[row, slot]),
                        "target": int(position_i[row]), "source": int(position_j[row]),
                        "distance": int(position_j[row]) - int(position_i[row]),
                        "control_source": int(position_q[row]),
                        "control_drop": float(drops[row, slot, quiet[row, slot]]),
                        "clean_logp": float(clean_logp[row, slot]),
                        "corrupt_logp": float(corrupt_logp[row]),
                        **{f"{name}_logp": curves[name][row].tolist() for name in CONDITIONS}})
                del corrupt_snapshots
            partial.write_text(json.dumps({"records": records, "completed": end}))
            print("EVIDENCE", end, "/", len(split["ids"]),
                  f"pairs={len(records)}", flush=True)
            del clean_snapshots

    write_json(OUTPUT / "evidence_transfer_raw.json",
               {"ratio": RATIO, "conditions": list(CONDITIONS), "min_drop": MIN_DROP,
                "checkpoint_step": engine.checkpoint_step,
                "interfaces": list(INTERFACES), "records": records})
    partial.unlink(missing_ok=True)
    render()


def render():
    raw = json.loads((OUTPUT / "evidence_transfer_raw.json").read_text())
    records = raw["records"]
    kept = [r for r in records
            if r.get("valid", True) and r["clean_logp"] - r["corrupt_logp"] >= MIN_DROP]
    if not kept:
        raise AssertionError("no pair cleared the minimum-drop threshold")
    denominator = np.array([r["clean_logp"] - r["corrupt_logp"] for r in kept])[:, None]
    corrupt = np.array([r["corrupt_logp"] for r in kept])[:, None]
    curves = {name: (np.array([r[f"{name}_logp"] for r in kept]) - corrupt) / denominator
              for name in raw["conditions"]}

    # Pairs are not independent: up to TARGETS_PER_SEQUENCE of them come from one sequence.
    # The bootstrap therefore resamples whole sequences and takes all of their pairs.
    blocks = np.array([r["block"] for r in kept])
    unique = np.unique(blocks)
    members = [np.flatnonzero(blocks == block) for block in unique]
    rng = np.random.default_rng(SELECTION_SEED)
    draws = [np.concatenate([members[i] for i in rng.integers(len(unique), size=len(unique))])
             for _ in range(4000)]

    results = {
        "ratio": raw["ratio"],
        "selection": {
            "targets_per_sequence": TARGETS_PER_SEQUENCE,
            "target_rule": "masked positions ranked by clean log-probability among those the "
                           "clean run already predicts correctly; no intervention is consulted",
            "source_rule": f"of {NEAR_CANDIDATES * 2} nearest visible positions per target plus "
                           f"{FAR_CANDIDATES} random ones, pooled over the batch, the candidate "
                           "whose masking costs the target the most log-probability",
            "control_rule": "same pool, the candidate whose masking costs the target the least",
            "kept_rule": f"pairs whose clean-minus-corrupt gap reaches {MIN_DROP} nats",
            "pairs_measured": len(records), "pairs_kept": len(kept),
            "sequences_kept": int(len(unique)),
            "median_drop_nats": float(np.median(denominator)),
            "median_distance": float(np.median([abs(r["distance"]) for r in kept])),
            "median_control_drop_nats": float(np.median([abs(r["control_drop"]) for r in kept])),
        },
        "bootstrap": {"unit": "sequence (cluster bootstrap over all pairs of a sequence)",
                      "draws": len(draws), "seed": SELECTION_SEED},
        "interfaces": raw["interfaces"], "curves": {}}
    for name, values in curves.items():
        boot = np.array([values[index].mean(0) for index in draws])
        results["curves"][name] = {
            "mean": values.mean(0).tolist(),
            "ci95_low": np.quantile(boot, .025, axis=0).tolist(),
            "ci95_high": np.quantile(boot, .975, axis=0).tolist(),
            "median": np.median(values, 0).tolist()}
    write_json(OUTPUT / "evidence_transfer.json", results)

    info = results["selection"]
    print(f"\npairs {info['pairs_kept']}/{info['pairs_measured']} из "
          f"{info['sequences_kept']} последовательностей · медианный урон "
          f"{info['median_drop_nats']:.2f} нат · медианное |i-j| {info['median_distance']:.0f} · "
          f"контрольный источник стоит {info['median_control_drop_nats']:.2f} нат")
    print(f"бутстрэп: {results['bootstrap']['unit']}, {results['bootstrap']['draws']} draws\n")
    header = "".join(f"{name:>26}" for name in raw["conditions"])
    print(f"{'iface':<7}{header}")
    for index, label in enumerate(raw["interfaces"]):
        line = f"{label:<7}"
        for name in raw["conditions"]:
            c = results["curves"][name]
            line += f'{c["mean"][index]:+.3f}'.rjust(26)
        print(line)
    print("WROTE", OUTPUT / "evidence_transfer.json", flush=True)


if __name__ == "__main__":
    main()
