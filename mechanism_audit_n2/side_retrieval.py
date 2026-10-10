"""Causal audit of late retrieval from the parked residual side stream.

Candidate interfaces are selected from the t=.5 calibration routing capture,
before any intervention CE is read.  The main audit runs at t=.5; only the
retrieval triad is repeated at t=.1 and t=.9.

Examples
--------
python -m mechanism_audit_n2.side_retrieval --prepare
python -m mechanism_audit_n2.side_retrieval --ratio .5 --role masked --gpu 0
python -m mechanism_audit_n2.side_retrieval --ratio .5 --role visible --gpu 0
python -m mechanism_audit_n2.side_retrieval --ratio .1 --role masked --gpu 0
python -m mechanism_audit_n2.side_retrieval --render
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .engine import aggregate, batches, load_model, make_split, summarize_tokens
from .factors import ROOT
from .primary import load_pack, save_pack, write_json
from .safety import GPUReservation
from .volume_noise_sweep import BoundedSharedGPUReservation
from .volume_roles import assert_aligned


OUTPUT = ROOT / "side_retrieval_n2"
EFFECTS = OUTPUT / "effects"
CAPTURE = ROOT / "residual_circuits_n2" / "captures" / "r50_all_positions.npz"
CHECKPOINT_META = json.loads((ROOT / "residual_circuits_n2" / "config.json").read_text())
RATIOS = (.1, .5, .9)
ROLES = ("masked", "visible")
INTERFACES = tuple(f"{kind}{block}" for block in range(16) for kind in ("a", "m"))

# These architectural regions are declared without looking at causal outcomes.
# a2..a8 is the long low-side-read plateau; a11..m15 is the final five blocks.
PARKED_INTERFACES = tuple(range(4, 17))
DISRUPT_AFTER = 16                 # after a8
PRE_LATE_AFTER = 21                # after m10; a11 is next
LATE_REGION = tuple(range(22, 32)) # a11..m15
REBOUND_MARGIN = .15

KEY_CONDITIONS = (
    "intact",
    "late_cut_side",
    "mid_shuffle_side",
    "mid_shuffle_side_rescue",
)
PRIMARY_CONTROLS = (
    "late_cut_active",
    "mid_shuffle_active",
    "prehead_zero_side",
)
MASKED_PATCHES = (
    "mid_shuffle_patch_a0_raw",
    "mid_shuffle_patch_m0_raw",
    "mid_shuffle_patch_a0_normmatched",
    "mid_shuffle_patch_m0_normmatched",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ratio_key(ratio: float) -> str:
    return f"r{round(100 * ratio):02d}"


def selection_path() -> Path:
    return OUTPUT / "candidate_selection.json"


def pack_path(ratio: float, role: str, condition: str) -> Path:
    path = EFFECTS / ratio_key(ratio) / role / f"{condition}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def conditions_for(ratio: float, role: str) -> tuple[str, ...]:
    conditions = list(KEY_CONDITIONS)
    if ratio == .5:
        conditions.extend(PRIMARY_CONTROLS)
        if role == "masked":
            conditions.extend(MASKED_PATCHES)
    return tuple(conditions)


def choose_candidates(config: dict) -> dict:
    """Select roles/streams/candidates from router profiles only."""
    with np.load(CAPTURE) as saved:
        router = saved["router"].astype(np.float64)
        mask = saved["mask"].reshape(-1)
        capture_blocks = saved["block_ids"]
        capture_mask_hash = str(saved["mask_sha256"])
    calibration = make_split(config, "calibration", .5)
    np.testing.assert_array_equal(capture_blocks, calibration["block_ids"])
    assert capture_mask_hash == calibration["mask_sha256"]
    assert router.shape == (32, mask.size, 6)

    result = {
        "selection_status": "locked_before_causal_outcomes",
        "selection_source": str(CAPTURE),
        "selection_source_sha256": sha256(CAPTURE),
        "selection_uses_ce": False,
        "interface_order": list(INTERFACES),
        "parked_interfaces": list(PARKED_INTERFACES),
        "parked_labels": [INTERFACES[i] for i in PARKED_INTERFACES],
        "disrupt_after": DISRUPT_AFTER,
        "disrupt_after_label": INTERFACES[DISRUPT_AFTER],
        "transition_interfaces": list(range(DISRUPT_AFTER + 1, PRE_LATE_AFTER + 1)),
        "transition_labels": [INTERFACES[i] for i in range(DISRUPT_AFTER + 1, PRE_LATE_AFTER + 1)],
        "late_region": list(LATE_REGION),
        "late_region_labels": [INTERFACES[i] for i in LATE_REGION],
        "selection_rule": (
            "Within each role, active is the stream with larger mean read over a2..a8; side is the other. "
            "A final-five-block interface is a candidate when its side share r_side/(r0+r1) is at least "
            "the maximum parked side share plus 0.15."
        ),
        "rebound_margin": REBOUND_MARGIN,
        "roles": {},
    }
    for role, selected in (("masked", mask), ("visible", ~mask)):
        read = router[:, selected, :2].mean(1)
        parked_mean = read[list(PARKED_INTERFACES)].mean(0)
        active = int(np.argmax(parked_mean)); side = 1 - active
        share = read[:, side] / np.maximum(read.sum(-1), 1e-12)
        parked_max = float(share[list(PARKED_INTERFACES)].max())
        threshold = parked_max + REBOUND_MARGIN
        candidates = [i for i in LATE_REGION if share[i] >= threshold]
        if not candidates:
            raise AssertionError(f"No routing-profile candidates for {role}")
        result["roles"][role] = {
            "active_stream": active,
            "side_stream": side,
            "parked_mean_read": parked_mean.tolist(),
            "parked_side_share_mean": float(share[list(PARKED_INTERFACES)].mean()),
            "parked_side_share_max": parked_max,
            "candidate_threshold": threshold,
            "candidates": candidates,
            "candidate_labels": [INTERFACES[i] for i in candidates],
            "late_side_share": {INTERFACES[i]: float(share[i]) for i in LATE_REGION},
            "all_side_share": share.tolist(),
            "all_read_mean": read.tolist(),
        }
    return result


def prepare(config: dict) -> tuple[dict, dict]:
    OUTPUT.mkdir(exist_ok=True)
    selection = choose_candidates(config)
    if selection_path().exists():
        prior = json.loads(selection_path().read_text())
        if prior != selection:
            raise AssertionError("Candidate selection changed after it was locked")
    else:
        write_json(selection_path(), selection)

    protocol = {
        "checkpoint": config["models"]["full_n2"],
        "checkpoint_step": CHECKPOINT_META["checkpoint_step"],
        "training_input_tokens": CHECKPOINT_META["training_input_tokens"],
        "ratios": list(RATIOS),
        "primary_ratio": .5,
        "evaluation_split": "discovery",
        "evaluation_sequences_per_ratio": 128,
        "outcome": "cross-entropy on the original masked targets",
        "candidate_selection": selection,
        "late_read_cut": (
            "For selected-role tokens at candidate interfaces, preserve total read mass exactly while setting "
            "r_side=0 and adding its original mass to r_active. Gains and mixer probabilities remain live."
        ),
        "parked_state_disruption": (
            "Immediately after a8, cyclically permute whole vectors in one stream among tokens of the same role "
            "within each sequence. This exactly preserves the per-sequence vector multiset and every donor norm."
        ),
        "rescue": (
            "Continue the disrupted trajectory through m10, then replace only its selected-role side stream with "
            "the clean same-example side state immediately before a11; active state remains from the disrupted run."
        ),
        "controls": {
            "late_cut_active": "Same mass-preserving read cut with active and side interchanged.",
            "mid_shuffle_active": "Same within-sequence role-preserving state shuffle in the active stream.",
            "prehead_zero_side": "Set only the selected-role side state to zero after m15, immediately before mean readout.",
        },
        "masked_identity_patches": (
            "In the masked disrupted run immediately before a11, patch side with the same token's clean state "
            "after a0 or after m0. Report both raw and per-token norm-matched-to-clean-a11-input variants."
        ),
        "noise_scope": "At t=.1 and t=.9 repeat only intact, side late-read cut, side disruption, and clean-state rescue.",
        "statistics": f"{config['bootstrap_repeats']} paired sequence-bootstrap draws; seed {config['bootstrap_seed']}.",
        "retrieval_criterion": (
            "At t=.5 for masked interventions: positive-CI harm from late_cut_side and mid_shuffle_side, "
            "plus positive-CI rescue benefit mid_shuffle_side - mid_shuffle_side_rescue."
        ),
        "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                          ("side_retrieval.py", "engine.py", "primary.py")},
    }
    protocol_path = OUTPUT / "protocol.json"
    if protocol_path.exists():
        prior = json.loads(protocol_path.read_text())
        # The implementation hash is allowed to change only before any effects exist.
        if prior != protocol and any(EFFECTS.glob("**/*.npz")):
            raise AssertionError("Protocol/source changed after causal effects were written")
    write_json(protocol_path, protocol)
    return selection, protocol


class ReadCutHooks:
    def __init__(self, engine):
        self.mode = None
        self.selected = None
        self.layers: set[int] = set()
        self.active = 0
        self.side = 1
        self.maxima = {}
        self.handles = []
        for index, interface in enumerate(engine.interfaces):
            def hook(module, args, outputs, layer=index):
                return self.transform(layer, outputs)
            self.handles.append(interface.router.register_forward_hook(hook))

    def configure(self, mode=None, selected=None, layers=(), active=0, side=1):
        assert mode in (None, "side", "active")
        self.mode = mode
        self.selected = selected
        self.layers = set(layers)
        self.active, self.side = active, side
        self.maxima = {key: 0.0 for key in
                       ("read_sum_error", "removed_read_error", "redirect_error",
                        "unselected_read_error", "other_router_coordinate_error")}

    def transform(self, layer, outputs):
        if self.mode is None or layer not in self.layers:
            return outputs
        read, gains, probabilities, mixer = outputs
        old = read
        changed = old.clone()
        removed = self.side if self.mode == "side" else self.active
        destination = self.active if self.mode == "side" else self.side
        changed[..., destination] = old[..., destination] + old[..., removed]
        changed[..., removed] = 0
        selected = self.selected
        rebuilt = torch.where(selected[..., None], changed, old)
        if selected.any():
            self.maxima["read_sum_error"] = max(self.maxima["read_sum_error"],
                float((rebuilt[selected].sum(-1) - old[selected].sum(-1)).abs().max()))
            self.maxima["removed_read_error"] = max(self.maxima["removed_read_error"],
                float(rebuilt[..., removed][selected].abs().max()))
            self.maxima["redirect_error"] = max(self.maxima["redirect_error"],
                float((rebuilt[..., destination][selected] - old[..., destination][selected]
                       - old[..., removed][selected]).abs().max()))
        if (~selected).any():
            self.maxima["unselected_read_error"] = max(self.maxima["unselected_read_error"],
                float((rebuilt[~selected] - old[~selected]).abs().max()))
        # These are returned by identity, so the exact error is definitionally zero.
        self.maxima["other_router_coordinate_error"] = max(
            self.maxima["other_router_coordinate_error"], 0.0)
        return rebuilt, gains, probabilities, mixer

    def close(self):
        for handle in self.handles:
            handle.remove()


def selected_positions(mask: torch.Tensor, role: str) -> torch.Tensor:
    return mask if role == "masked" else ~mask


def run_range(engine, state, mask, first: int, last: int):
    value = state.clone()
    for index in range(first, last + 1):
        value, _, _, _, _ = engine.interface(index, value, mask)
    return value


def clean_trajectory(engine, corrupted, mask):
    state = engine.initial(corrupted)
    snapshots = {}
    for index in range(32):
        state, _, _, _, _ = engine.interface(index, state, mask)
        if index in (0, 1, DISRUPT_AFTER, PRE_LATE_AFTER):
            snapshots[index] = state.detach().clone()
    return state, snapshots


def shuffle_role_stream(state, selected, stream: int):
    changed = state.clone()
    maximum_multiset_error = 0.0
    minimum_count = state.shape[1]
    fixed_points = 0
    for sequence in range(state.shape[0]):
        indices = selected[sequence].nonzero(as_tuple=True)[0]
        minimum_count = min(minimum_count, int(indices.numel()))
        if indices.numel() <= 1:
            continue
        donors = indices.roll(1)
        before = state[sequence, indices, stream]
        replacement = state[sequence, donors, stream]
        changed[sequence, indices, stream] = replacement
        before_norm = before.float().norm(dim=-1).sort().values
        after_norm = changed[sequence, indices, stream].float().norm(dim=-1).sort().values
        maximum_multiset_error = max(maximum_multiset_error,
                                     float((before_norm - after_norm).abs().max()))
        fixed_points += int((indices == donors).sum())
    return changed, {"maximum_norm_multiset_error": maximum_multiset_error,
                     "minimum_role_tokens_per_sequence": minimum_count,
                     "fixed_points": fixed_points}


def patch_role_stream(base, replacement, selected, stream: int, normmatched=False):
    changed = base.clone()
    value = replacement[..., stream, :]
    maximum_norm_error = 0.0
    if normmatched:
        reference_norm = base[..., stream, :].float().norm(dim=-1, keepdim=True)
        value = value.float()
        value = value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12) * reference_norm
        value = value.to(base.dtype)
    changed[..., stream, :] = torch.where(selected[..., None], value, changed[..., stream, :])
    if normmatched and selected.any():
        wanted = base[..., stream, :].float().norm(dim=-1)[selected]
        actual = changed[..., stream, :].float().norm(dim=-1)[selected]
        maximum_norm_error = float((wanted - actual).abs().max())
    return changed, maximum_norm_error


def zero_role_stream(state, selected, stream: int):
    changed = state.clone()
    changed[..., stream, :] = torch.where(
        selected[..., None], torch.zeros_like(changed[..., stream, :]), changed[..., stream, :])
    return changed


def evaluate(engine, split, config, reservation, ratio: float, role: str, selection: dict):
    conditions = conditions_for(ratio, role)
    existing = [pack_path(ratio, role, name).exists() for name in conditions]
    if all(existing):
        print("CACHED", ratio, role, flush=True)
        return
    if any(existing):
        raise RuntimeError("A role/ratio bundle is only resumable through its partial file; remove no individual packs")

    role_selection = selection["roles"][role]
    active, side = role_selection["active_stream"], role_selection["side_stream"]
    candidates = role_selection["candidates"]
    hooks = ReadCutHooks(engine)
    partial = EFFECTS / ratio_key(ratio) / role / "worker.partial.npz"
    completed = 0
    values = {name: [] for name in conditions}
    diagnostics = {"shuffle_side": {}, "shuffle_active": {}, "read_cut_side": {},
                   "read_cut_active": {}, "patch_norm_errors": {}}
    if partial.exists():
        saved = load_pack(partial)
        assert json.loads(str(saved["conditions_json"])) == list(conditions)
        completed = int(saved["completed"])
        values = {name: [saved[name]] for name in conditions}
        diagnostics = json.loads(str(saved["diagnostics_json"]))
        print("RESUME", ratio, role, completed, "/", len(split["ids"]), flush=True)

    try:
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            for begin, end, _, _, corrupted, ids, mask in batches(split, config["batch_size"]):
                if end <= completed:
                    continue
                assert begin >= completed
                reservation.check()
                chosen = selected_positions(mask, role)

                hooks.configure()
                intact_state, snapshots = clean_trajectory(engine, corrupted, mask)
                values["intact"].append(engine.final_stats(intact_state, ids, mask, config["head_chunk"]))

                hooks.configure("side", chosen, candidates, active, side)
                state = run_range(engine, snapshots[PRE_LATE_AFTER], mask,
                                  PRE_LATE_AFTER + 1, 31)
                values["late_cut_side"].append(engine.final_stats(state, ids, mask, config["head_chunk"]))
                for key, value in hooks.maxima.items():
                    diagnostics["read_cut_side"][key] = max(
                        diagnostics["read_cut_side"].get(key, 0.0), value)

                hooks.configure()
                disrupted, local = shuffle_role_stream(snapshots[DISRUPT_AFTER], chosen, side)
                for key, value in local.items():
                    if key.startswith("minimum"):
                        diagnostics["shuffle_side"][key] = min(
                            diagnostics["shuffle_side"].get(key, value), value)
                    else:
                        diagnostics["shuffle_side"][key] = max(
                            diagnostics["shuffle_side"].get(key, 0.0), value)
                disrupted_pre_late = run_range(engine, disrupted, mask,
                                                DISRUPT_AFTER + 1, PRE_LATE_AFTER)
                state = run_range(engine, disrupted_pre_late, mask,
                                  PRE_LATE_AFTER + 1, 31)
                values["mid_shuffle_side"].append(engine.final_stats(state, ids, mask, config["head_chunk"]))

                rescued, _ = patch_role_stream(disrupted_pre_late, snapshots[PRE_LATE_AFTER],
                                                chosen, side)
                state = run_range(engine, rescued, mask, PRE_LATE_AFTER + 1, 31)
                values["mid_shuffle_side_rescue"].append(
                    engine.final_stats(state, ids, mask, config["head_chunk"]))

                if ratio == .5:
                    hooks.configure("active", chosen, candidates, active, side)
                    state = run_range(engine, snapshots[PRE_LATE_AFTER], mask,
                                      PRE_LATE_AFTER + 1, 31)
                    values["late_cut_active"].append(
                        engine.final_stats(state, ids, mask, config["head_chunk"]))
                    for key, value in hooks.maxima.items():
                        diagnostics["read_cut_active"][key] = max(
                            diagnostics["read_cut_active"].get(key, 0.0), value)

                    hooks.configure()
                    active_disrupted, local = shuffle_role_stream(
                        snapshots[DISRUPT_AFTER], chosen, active)
                    for key, value in local.items():
                        if key.startswith("minimum"):
                            diagnostics["shuffle_active"][key] = min(
                                diagnostics["shuffle_active"].get(key, value), value)
                        else:
                            diagnostics["shuffle_active"][key] = max(
                                diagnostics["shuffle_active"].get(key, 0.0), value)
                    state = run_range(engine, active_disrupted, mask, DISRUPT_AFTER + 1, 31)
                    values["mid_shuffle_active"].append(
                        engine.final_stats(state, ids, mask, config["head_chunk"]))

                    state = zero_role_stream(intact_state, chosen, side)
                    values["prehead_zero_side"].append(
                        engine.final_stats(state, ids, mask, config["head_chunk"]))

                    if role == "masked":
                        for source_index, source_label in ((0, "a0"), (1, "m0")):
                            for normmatched, suffix in ((False, "raw"), (True, "normmatched")):
                                name = f"mid_shuffle_patch_{source_label}_{suffix}"
                                patched, error = patch_role_stream(
                                    disrupted_pre_late, snapshots[source_index], chosen, side,
                                    normmatched=normmatched)
                                diagnostics["patch_norm_errors"][name] = max(
                                    diagnostics["patch_norm_errors"].get(name, 0.0), error)
                                state = run_range(engine, patched, mask, PRE_LATE_AFTER + 1, 31)
                                values[name].append(
                                    engine.final_stats(state, ids, mask, config["head_chunk"]))

                if end % 16 == 0 or end == len(split["ids"]):
                    payload = {"completed": np.array(end),
                               "conditions_json": np.array(json.dumps(list(conditions))),
                               "diagnostics_json": np.array(json.dumps(diagnostics))}
                    payload.update({name: np.concatenate(items) for name, items in values.items()})
                    temporary = partial.with_name(partial.name + ".tmp")
                    with temporary.open("wb") as target:
                        np.savez_compressed(target, **payload)
                    temporary.replace(partial)
                    print("EVALUATE", ratio, role, end, "/", len(split["ids"]), flush=True)
                del intact_state, snapshots, disrupted, disrupted_pre_late, rescued, state
    finally:
        hooks.close()

    intact = summarize_tokens(split, np.concatenate(values["intact"]))
    intact["diagnostics_json"] = np.array(json.dumps(diagnostics))
    save_pack(pack_path(ratio, role, "intact"), intact)
    for name in conditions:
        if name == "intact":
            continue
        pack = summarize_tokens(split, np.concatenate(values[name]))
        pack["diagnostics_json"] = np.array(json.dumps(diagnostics))
        save_pack(pack_path(ratio, role, name), pack, intact)
    partial.unlink(missing_ok=True)
    print("COMPLETE", ratio, role, {name: aggregate(load_pack(pack_path(ratio, role, name)))["ce"]
                                     for name in conditions}, flush=True)


def paired_effect(left: dict, right: dict, config: dict) -> dict:
    assert_aligned(left, right)
    counts = left["seq_mask_count"]
    difference = left["seq_ce_sum"] - right["seq_ce_sum"]
    point = float(difference.sum() / counts.sum())
    rng = np.random.default_rng(config["bootstrap_seed"])
    draws = []
    n = len(counts)
    for begin in range(0, config["bootstrap_repeats"], 250):
        indices = rng.integers(n, size=(min(250, config["bootstrap_repeats"] - begin), n))
        draws.extend(difference[indices].sum(-1) / counts[indices].sum(-1))
    return {"delta_ce": point,
            "ci95": list(map(float, np.quantile(draws, [.025, .975])))}


def rescue_fraction(disrupted: dict, rescued: dict, intact: dict, config: dict) -> dict:
    for pack in (rescued, intact):
        assert_aligned(disrupted, pack)
    counts = disrupted["seq_mask_count"]
    numerator = disrupted["seq_ce_sum"] - rescued["seq_ce_sum"]
    denominator = disrupted["seq_ce_sum"] - intact["seq_ce_sum"]
    point = float(numerator.sum() / denominator.sum())
    rng = np.random.default_rng(config["bootstrap_seed"])
    draws = []
    n = len(counts)
    for begin in range(0, config["bootstrap_repeats"], 250):
        indices = rng.integers(n, size=(min(250, config["bootstrap_repeats"] - begin), n))
        top = numerator[indices].sum(-1)
        bottom = denominator[indices].sum(-1)
        good = np.abs(bottom) > 1e-12
        draws.extend((top[good] / bottom[good]).tolist())
    return {"fraction": point, "ci95": list(map(float, np.quantile(draws, [.025, .975]))),
            "masked_token_weighting": True}


def analyze(config: dict, selection: dict) -> dict:
    results = {"ratios": {}, "bootstrap": {"unit": "sequence-paired",
               "repeats": config["bootstrap_repeats"], "seed": config["bootstrap_seed"]}}
    for ratio in RATIOS:
        rkey = ratio_key(ratio)
        results["ratios"][rkey] = {}
        for role in ROLES:
            conditions = conditions_for(ratio, role)
            packs = {name: load_pack(pack_path(ratio, role, name)) for name in conditions}
            intact = packs["intact"]
            effects = {}
            for name, pack in packs.items():
                effects[name] = {**aggregate(pack), **paired_effect(pack, intact, config)}
            contrasts = {
                "rescue_benefit": paired_effect(packs["mid_shuffle_side"],
                                                packs["mid_shuffle_side_rescue"], config),
                "rescue_residual": paired_effect(packs["mid_shuffle_side_rescue"], intact, config),
                "rescue_fraction": rescue_fraction(packs["mid_shuffle_side"],
                                                   packs["mid_shuffle_side_rescue"], intact, config),
            }
            if ratio == .5 and role == "masked":
                for suffix in ("raw", "normmatched"):
                    contrasts[f"m0_minus_a0_{suffix}"] = paired_effect(
                        packs[f"mid_shuffle_patch_m0_{suffix}"],
                        packs[f"mid_shuffle_patch_a0_{suffix}"], config)
                    for source in ("a0", "m0"):
                        name = f"mid_shuffle_patch_{source}_{suffix}"
                        contrasts[f"{name}_vs_disrupted"] = paired_effect(
                            packs[name], packs["mid_shuffle_side"], config)
            results["ratios"][rkey][role] = {"effects_vs_intact": effects,
                                              "contrasts": contrasts}

    masked = results["ratios"]["r50"]["masked"]
    criterion = {
        "late_read_cut_harm": masked["effects_vs_intact"]["late_cut_side"]["ci95"][0] > 0,
        "parked_state_disruption_harm": masked["effects_vs_intact"]["mid_shuffle_side"]["ci95"][0] > 0,
        "clean_state_rescue_benefit": masked["contrasts"]["rescue_benefit"]["ci95"][0] > 0,
    }
    criterion["retrieval_confirmed"] = all(criterion.values())
    results["retrieval_criterion"] = criterion
    results["candidate_selection_sha256"] = sha256(selection_path())
    write_json(OUTPUT / "final_results.json", results)
    return results


def fmt_effect(value: dict) -> str:
    return f'{value["delta_ce"]:+.5f} [{value["ci95"][0]:+.5f}, {value["ci95"][1]:+.5f}]'


def render(config: dict, selection: dict, protocol: dict, results: dict):
    rows = []
    with (OUTPUT / "effects.csv").open("w", newline="") as target:
        fields = ["ratio", "role", "condition", "ce", "delta_ce", "ci95_low", "ci95_high"]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for ratio in RATIOS:
            for role in ROLES:
                local = results["ratios"][ratio_key(ratio)][role]["effects_vs_intact"]
                for name, value in local.items():
                    writer.writerow({"ratio": ratio, "role": role, "condition": name,
                        "ce": value["ce"], "delta_ce": value["delta_ce"],
                        "ci95_low": value["ci95"][0], "ci95_high": value["ci95"][1]})
                    rows.append((ratio, role, name, value))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), layout="constrained")
    colors = {"masked": "#b43c5a", "visible": "#247f91"}
    labels = {"late_cut_side": "late side-read cut", "mid_shuffle_side": "mid side shuffle",
              "mid_shuffle_side_rescue": "shuffle + clean rescue"}
    x = np.arange(3)
    for offset, role in ((-.12, "masked"), (.12, "visible")):
        local = results["ratios"]["r50"][role]["effects_vs_intact"]
        y = np.array([local[name]["delta_ce"] for name in labels])
        ci = np.array([local[name]["ci95"] for name in labels])
        axes[0].errorbar(x + offset, y, yerr=np.vstack([y-ci[:, 0], ci[:, 1]-y]),
                         marker="o", capsize=3, lw=2, color=colors[role], label=role)
    axes[0].set_xticks(x, [labels[name] for name in labels], rotation=20, ha="right")
    axes[0].set_ylabel("Δ masked-target CE")
    axes[0].set_title("A · Retrieval triad at t=.5", loc="left", fontweight="bold")
    axes[0].legend(frameon=False); axes[0].axhline(0, color="#555", lw=.8); axes[0].grid(alpha=.15)

    control_names = ("late_cut_side", "late_cut_active", "mid_shuffle_side",
                     "mid_shuffle_active", "prehead_zero_side")
    width = .36
    for j, role in enumerate(ROLES):
        local = results["ratios"]["r50"][role]["effects_vs_intact"]
        y = np.array([local[name]["delta_ce"] for name in control_names])
        axes[1].bar(np.arange(len(y)) + (j-.5)*width, y, width=width,
                    color=colors[role], alpha=.88, label=role)
    axes[1].set_xticks(np.arange(len(control_names)),
                       ["side read", "active read", "side shuffle", "active shuffle", "side→0\nprehead"],
                       rotation=20, ha="right")
    axes[1].set_ylabel("Δ masked-target CE")
    axes[1].set_title("B · Active and direct-readout controls", loc="left", fontweight="bold")
    axes[1].axhline(0, color="#555", lw=.8); axes[1].grid(axis="y", alpha=.15)

    for role, marker in (("masked", "o"), ("visible", "s")):
        y, lo, hi = [], [], []
        for ratio in RATIOS:
            value = results["ratios"][ratio_key(ratio)][role]["effects_vs_intact"]["late_cut_side"]
            y.append(value["delta_ce"]); lo.append(value["ci95"][0]); hi.append(value["ci95"][1])
        y, lo, hi = map(np.array, (y, lo, hi))
        axes[2].errorbar(RATIOS, y, yerr=np.vstack([y-lo, hi-y]), marker=marker,
                         capsize=3, lw=2, color=colors[role], label=role)
    axes[2].set_xticks(RATIOS); axes[2].set_xlabel("corruption t")
    axes[2].set_ylabel("late side-read cut ΔCE")
    axes[2].set_title("C · Noise check", loc="left", fontweight="bold")
    axes[2].axhline(0, color="#555", lw=.8); axes[2].grid(alpha=.15); axes[2].legend(frameon=False)
    fig.suptitle("Does the model causally retrieve a parked residual side stream?", fontsize=15)
    fig.savefig(OUTPUT / "side_retrieval.svg")
    fig.savefig(OUTPUT / "side_retrieval.png", dpi=190)
    plt.close(fig)

    main_rows = []
    for ratio in RATIOS:
        for role in ROLES:
            local = results["ratios"][ratio_key(ratio)][role]
            effects = local["effects_vs_intact"]
            main_rows.append(
                f"<tr><td>{ratio:.1f}</td><td>{role}</td>"
                f"<td>{fmt_effect(effects['late_cut_side'])}</td>"
                f"<td>{fmt_effect(effects['mid_shuffle_side'])}</td>"
                f"<td>{fmt_effect(effects['mid_shuffle_side_rescue'])}</td>"
                f"<td>{fmt_effect(local['contrasts']['rescue_benefit'])}</td>"
                f"<td>{local['contrasts']['rescue_fraction']['fraction']:.1%} "
                f"[{local['contrasts']['rescue_fraction']['ci95'][0]:.1%}, "
                f"{local['contrasts']['rescue_fraction']['ci95'][1]:.1%}]</td></tr>")

    controls_rows = []
    for role in ROLES:
        effects = results["ratios"]["r50"][role]["effects_vs_intact"]
        controls_rows.append(
            f"<tr><td>{role}</td><td>{fmt_effect(effects['late_cut_side'])}</td>"
            f"<td>{fmt_effect(effects['late_cut_active'])}</td>"
            f"<td>{fmt_effect(effects['mid_shuffle_side'])}</td>"
            f"<td>{fmt_effect(effects['mid_shuffle_active'])}</td>"
            f"<td>{fmt_effect(effects['prehead_zero_side'])}</td></tr>")

    patch = results["ratios"]["r50"]["masked"]
    patch_rows = []
    for suffix in ("raw", "normmatched"):
        a = patch["effects_vs_intact"][f"mid_shuffle_patch_a0_{suffix}"]
        m = patch["effects_vs_intact"][f"mid_shuffle_patch_m0_{suffix}"]
        d = patch["contrasts"][f"m0_minus_a0_{suffix}"]
        patch_rows.append(f"<tr><td>{suffix}</td><td>{fmt_effect(a)}</td>"
                          f"<td>{fmt_effect(m)}</td><td>{fmt_effect(d)}</td></tr>")

    candidate_rows = []
    for role in ROLES:
        item = selection["roles"][role]
        late_min = min(item["late_side_share"].values())
        late_max = max(item["late_side_share"].values())
        candidate_rows.append(f"<tr><td>{role}</td><td>{item['active_stream']}</td>"
            f"<td>{item['side_stream']}</td><td>{item['parked_side_share_mean']:.3f}</td>"
            f"<td>{item['parked_side_share_max']:.3f}</td><td>{late_min:.3f}–{late_max:.3f}</td>"
            f"<td>{', '.join(item['candidate_labels'])}</td></tr>")

    criterion = results["retrieval_criterion"]
    verdict = ("Поздний retrieval side stream причинно подтверждён для masked positions при t=.5."
               if criterion["retrieval_confirmed"] else
               "Функционально необходимый поздний retrieval для masked positions не подтверждён.")
    statement = ("The frozen model causally reuses a role-specific residual side stream after a middle-depth "
                 "parking phase: blocking late side reads and disrupting the parked state both raise masked-target "
                 "loss, while restoring the clean side state before reopening rescues the loss."
                 if criterion["retrieval_confirmed"] else
                 "The model physically preserves a side stream, but this audit does not establish functionally "
                 "necessary late retrieval from it.")
    html = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Late side-stream retrieval audit</title><style>
body{{font:15px/1.55 system-ui;max-width:1220px;margin:auto;padding:32px 24px;color:#202632}}h1,h2{{line-height:1.2}}code{{background:#f1f3f5;padding:2px 4px}}.verdict{{border-left:5px solid {'#247f91' if criterion['retrieval_confirmed'] else '#b43c5a'};background:#f4f8fa;padding:15px 19px}}.note{{border-left:4px solid #d28d27;background:#fcf8ef;padding:12px 16px}}img{{width:100%;border:1px solid #dfe4ea;border-radius:10px}}.table{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}th,td{{padding:8px;border-bottom:1px solid #dfe4ea;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}.scheme{{font:14px/1.7 ui-monospace,monospace;background:#f7f7fa;padding:14px;overflow:auto}}
</style><body><h1>Используется ли позднее parked side state?</h1>
<div class="verdict"><b>{escape(verdict)}</b><br>{escape(statement)}</div>
<h2>Кандидаты зафиксированы до CE</h2><p>Calibration routing profile при <code>t=.5</code>, 64 последовательности. Active/side определены по read в <code>a2…a8</code>; кандидат проходит порог <code>max parked side share + .15</code> внутри заранее заданных последних пяти блоков. Ни один intervention CE в selection не участвовал.</p>
<div class="table"><table><thead><tr><th>role</th><th>active</th><th>side</th><th>parked mean</th><th>parked max</th><th>late range</th><th>candidate reads</th></tr></thead><tbody>{''.join(candidate_rows)}</tbody></table></div>
<h2>Схема вмешательств</h2><div class="scheme">clean: a0 → m0 → … → a8 │ transition m8…m10 │ candidate reads a11…m15 → mean readout<br>
shuffle:                    [permute side within sequence+role] ───────────────────────────────→<br>
rescue:                     [same disruption] ─────────────── [patch clean side before a11] →<br>
read cut:                                                     [r_side→0; mass→r_active]     →<br>
head control:                                                                 [side→0] → head</div>
<img src="side_retrieval.svg" alt="Causal side-stream retrieval results">
<h2>Основной retrieval-тест и noise check</h2><p>Все числа — ΔCE относительно intact и paired 95% sequence-bootstrap CI. Rescue benefit = CE(disruption) − CE(rescue), поэтому положительное значение означает восстановление.</p>
<div class="table"><table><thead><tr><th>t</th><th>intervened role</th><th>late side-read cut</th><th>mid side shuffle</th><th>shuffle + rescue</th><th>rescue benefit</th><th>damage rescued</th></tr></thead><tbody>{''.join(main_rows)}</tbody></table></div>
<h2>Active-stream и direct-readout controls при t=.5</h2><div class="table"><table><thead><tr><th>role</th><th>side read cut</th><th>active read cut</th><th>side shuffle</th><th>active shuffle</th><th>side→0 prehead</th></tr></thead><tbody>{''.join(controls_rows)}</tbody></table></div>
<h2>Masked identity test: a0 или after-m0 state?</h2><p>Здесь patch ставится в side stream непосредственно перед <code>a11</code> поверх trajectory с mid-side disruption. Raw — буквальный ранний state; normmatched сохраняет позднюю clean side-норму каждого токена. Последний столб — CE(m0 patch) − CE(a0 patch), отрицательное значение означает преимущество after-m0.</p>
<div class="table"><table><thead><tr><th>patch</th><th>a0-state ΔCE</th><th>after-m0-state ΔCE</th><th>m0−a0</th></tr></thead><tbody>{''.join(patch_rows)}</tbody></table></div>
<h2>Что доказано / не доказано / формулировка для статьи</h2>
<p><b>Доказано:</b> verdict определяется ровно preregistered тройкой masked tests: cut harm={criterion['late_read_cut_harm']}, disruption harm={criterion['parked_state_disruption_harm']}, rescue={criterion['clean_state_rescue_benefit']}.</p>
<p><b>Не следует автоматически:</b> сохранённый tracer не равен памяти raw <code>a0</code>; это отдельно проверяют a0/m0 patches. Тест ничего не говорит о выборе attention keys и не является noise-dependent attention-source-switch анализом.</p>
<p class="note"><b>Article-safe:</b> {escape(statement)} Интерпретацию как память именно исходного a0-вектора следует добавлять только если identity patches поддерживают её без конфликта с after-m0 patch.</p>
<p>Frozen full-n=2 checkpoint; 128 одинаковых paired sequences/masks на condition внутри каждого t; 5,000 bootstrap draws. Полные machine-readable результаты: <code>final_results.json</code>; таблица: <code>effects.csv</code>; protocol и locked selection сохранены отдельно.</p></body></html>'''
    (OUTPUT / "report.html").write_text(html)


def verify(config: dict, selection: dict, protocol: dict, results: dict):
    checks = {
        "selection_locked_without_ce": selection["selection_uses_ce"] is False,
        "candidate_before_first_effect": selection["selection_status"] == "locked_before_causal_outcomes",
        "all_candidates_profile_selected": all(
            selection["roles"][role]["candidates"] and
            set(selection["roles"][role]["candidates"]).issubset(LATE_REGION)
            for role in ROLES),
        "all_effect_packs_present": True,
        "all_128_sequences": True,
        "all_masks_and_blocks_paired": True,
        "shuffle_norm_multiset_exact": True,
        "shuffle_has_no_fixed_points": True,
        "read_mass_preserved": True,
        "other_router_coordinates_dynamic": True,
        "patch_norm_tolerance": True,
    }
    for ratio in RATIOS:
        split = make_split(config, "discovery", ratio)
        for role in ROLES:
            conditions = conditions_for(ratio, role)
            packs = {}
            for name in conditions:
                path = pack_path(ratio, role, name)
                checks["all_effect_packs_present"] &= path.exists()
                pack = load_pack(path); packs[name] = pack
                checks["all_128_sequences"] &= len(pack["block_ids"]) == 128
                checks["all_masks_and_blocks_paired"] &= (
                    np.array_equal(pack["block_ids"], split["block_ids"]) and
                    np.array_equal(pack["seq_mask_count"], split["mask"].sum(1)))
            diagnostics = json.loads(str(packs["intact"]["diagnostics_json"]))
            checks["shuffle_norm_multiset_exact"] &= diagnostics["shuffle_side"].get(
                "maximum_norm_multiset_error", 1) == 0
            checks["shuffle_has_no_fixed_points"] &= diagnostics["shuffle_side"].get("fixed_points", 1) == 0
            side_cut = diagnostics["read_cut_side"]
            checks["read_mass_preserved"] &= side_cut.get("read_sum_error", 1) < 1e-6
            checks["other_router_coordinates_dynamic"] &= side_cut.get(
                "other_router_coordinate_error", 1) == 0
            if ratio == .5:
                checks["shuffle_norm_multiset_exact"] &= diagnostics["shuffle_active"].get(
                    "maximum_norm_multiset_error", 1) == 0
                checks["shuffle_has_no_fixed_points"] &= diagnostics["shuffle_active"].get(
                    "fixed_points", 1) == 0
                active_cut = diagnostics["read_cut_active"]
                checks["read_mass_preserved"] &= active_cut.get("read_sum_error", 1) < 1e-6
                if role == "masked":
                    checks["patch_norm_tolerance"] &= all(
                        value < .15 for key, value in diagnostics["patch_norm_errors"].items()
                        if key.endswith("normmatched"))
    checks["passed"] = all(checks.values())
    required = ("candidate_selection.json", "protocol.json", "final_results.json", "effects.csv",
                "side_retrieval.svg", "side_retrieval.png", "report.html")
    checks["artifact_sha256"] = {name: sha256(OUTPUT / name) for name in required}
    write_json(OUTPUT / "verification.json", checks)
    if not checks["passed"]:
        raise AssertionError(checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--ratio", type=float, choices=RATIOS)
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--gpu", default='0')
    parser.add_argument("--allow-external-mib", type=float, default=0)
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    selection, protocol = prepare(config)
    if args.prepare:
        print("LOCKED", selection_path(), flush=True)
        return
    if args.render:
        results = analyze(config, selection)
        render(config, selection, protocol, results)
        verify(config, selection, protocol, results)
        print("REPORT COMPLETE", OUTPUT / "report.html", flush=True)
        return
    if args.ratio is None or args.role is None:
        raise ValueError("--ratio and --role are required unless --prepare or --render is used")
    split = make_split(config, "discovery", args.ratio)
    policy = dict(config["gpu_policy"]); policy["never_use_training_gpus"] = []
    reservation = (BoundedSharedGPUReservation(args.gpu, args.allow_external_mib)
                   if args.allow_external_mib else GPUReservation(args.gpu, policy))
    engine = load_model(config, "full_n2")
    if engine.checkpoint_step != CHECKPOINT_META["checkpoint_step"]:
        raise AssertionError((engine.checkpoint_step, CHECKPOINT_META["checkpoint_step"]))
    evaluate(engine, split, config, reservation, args.ratio, args.role, selection)


if __name__ == "__main__":
    main()
