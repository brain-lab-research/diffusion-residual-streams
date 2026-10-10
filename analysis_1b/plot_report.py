"""Standalone figures for the frozen 1B mHC step-124000 audit."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
BLUE = "#087da9"
RED = "#b03959"
GOLD = "#bd8420"
GRAY = "#788694"


def read(name):
    return json.loads((ROOT / f"{name}.json").read_text())


def save(fig, name):
    fig.savefig(ROOT / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(ROOT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d9e0e5", linewidth=0.65)
    ax.set_axisbelow(True)


def delta(row):
    return row["delta_ce"]


def circuit():
    early = read("early_state_boundary")["conditions"]
    middle = read("middle_preservation")["conditions"]
    late = read("state_interventions")["conditions"]
    fig, axes = plt.subplots(1, 3, figsize=(12.1, 3.6))
    for role, color in (("masked", RED), ("visible", BLUE)):
        values = [delta(early[f"{role}_m{i}"]["delta_ce"]) for i in (1, 4, 6)]
        values += [delta(middle[f"{role}_s0"]["restore_a8_delta_ce"])]
        axes[0].plot([1, 4, 6, 8], values, "o-", color=color, label=role)
        for stream, marker in ((0, "o"), (1, "s")):
            value = delta(middle[f"{role}_s{stream}"]["restore_a8_delta_ce"])
            axes[1].scatter(stream + (-0.08 if role == "masked" else 0.08),
                            value, marker=marker, s=70, color=color,
                            label=role if stream == 0 else None)
    for stream in (0, 1):
        value = delta(late[f"late_only_visible_s{stream}"])
        axes[2].bar(stream, value, width=0.48, color=(BLUE if stream == 0 else GRAY))
    axes[0].set(title="Preparation of s0", xlabel="Source state", ylabel="ΔCE at post-a16")
    axes[0].set_xticks([1, 4, 6, 8], ["m1", "m4", "m6", "a8"])
    axes[1].set(title="Middle-state restoration", xlabel="Stream restored from a8",
                ylabel="ΔCE at post-a16")
    axes[1].set_xticks([0, 1], ["s0", "s1"])
    axes[2].set(title="Late visible-token sufficiency", ylabel="ΔCE at post-m24")
    axes[2].set_xticks([0, 1], ["only s0", "only s1"])
    for ax in axes:
        style(ax)
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.suptitle("Early preparation, middle preservation, late reuse · 1B mHC, step 124,000")
    fig.tight_layout()
    save(fig, "circuit_stages")


def first_block():
    profile = read("router_profile")["roles"]
    scheme = read("first_block_scheme")["conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.7))
    x = np.arange(2)
    shares = np.array([profile[role]["write_share"][0]
                       for role in ("masked", "visible")])
    axes[0].bar(x - 0.17, shares[:, 0], width=0.33, color=BLUE, label="s0")
    axes[0].bar(x + 0.17, shares[:, 1], width=0.33, color=GRAY, label="s1")
    axes[0].set_xticks(x, ["masked", "visible"])
    axes[0].set(ylabel="Share of physical a0 write", ylim=(0, 1.08),
                title="First attention write")
    axes[0].legend(frameon=False)
    keys = ["a0_masked_zero_both", "a0_zero_both_restore_s0_after_m0",
            "a0_zero_both_restore_s1_after_m0"]
    values = [scheme[key]["delta_ce"] for key in keys]
    axes[1].bar(range(3), values, color=[RED, BLUE, GRAY])
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(3), ["delete both", "restore s0", "restore s1"])
    axes[1].set(ylabel="ΔCE after deleting masked a0 write", title="State rescue after m0")
    for ax in axes:
        style(ax)
    fig.suptitle("The first masked write is addressed to s0 · step 124,000")
    fig.tight_layout()
    save(fig, "first_block")


def sufficiency():
    rows = read("stream_sufficiency_curve")["conditions"]
    blocks = [8, 12, 16, 20, 24, 28]
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.7), sharex=True)
    for ax, role in zip(axes, ("visible", "masked")):
        for stream, color in ((0, BLUE), (1, GRAY)):
            keys = [f"{role}_m{block}_only_s{stream}" for block in blocks]
            values = np.asarray([rows[key]["delta_ce"] for key in keys])
            ci = np.asarray([rows[key]["ci95"] for key in keys])
            ax.plot(blocks, values, "o-", color=color, label=f"only s{stream}")
            ax.fill_between(blocks, ci[:, 0], ci[:, 1], color=color, alpha=0.13)
        ax.set(xlabel="Boundary after MLP block", ylabel="ΔCE", title=role.capitalize())
        ax.set_xticks(blocks)
        style(ax)
    axes[0].legend(frameon=False)
    fig.suptitle("Which stream is sufficient at each late boundary · step 124,000")
    fig.tight_layout()
    save(fig, "stream_sufficiency")


def transfer():
    data = read("transfer_combined")
    x = (np.asarray(data["depths"]) + 1) / 64 * 100
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7), sharex=True)
    for key, color, label in (("source_both", BLUE, "restore visible source"),
                              ("target_both", RED, "restore masked target")):
        row = data["curves"][key]
        y = np.asarray(row["mean"])
        axes[0].plot(x, y, "o-", color=color, label=label)
        axes[0].fill_between(x, row["ci95_low"], row["ci95_high"], color=color, alpha=0.15)
    diff = data["target_minus_source"]
    axes[1].plot(x, diff["mean"], "o-", color=GOLD)
    axes[1].fill_between(x, diff["ci95_low"], diff["ci95_high"], color=GOLD, alpha=0.2)
    axes[1].axhline(0, color="#444c54", linewidth=0.8)
    axes[0].set(title="Where missing evidence can be restored",
                ylabel="Fraction of lost target log-prob restored", ylim=(-0.1, 1.18))
    axes[1].set(title="Target minus source recovery", ylabel="Paired difference")
    for ax in axes:
        ax.set(xlabel="Depth (% of interfaces)", xlim=(0, 103))
        style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Visible-source influence moves to the masked target · {data['pairs_total']} selected pairs")
    fig.tight_layout()
    save(fig, "evidence_transfer")


def factors():
    rows = dict(read("roles")["conditions"])
    rows.update(read("roles_extended")["conditions"])
    physical = ROOT.parent / "volume_analysis/physical_destination_joint"
    for policy in ("global", "role"):
        rows[f"destination_{policy}"] = json.loads(
            (physical / f"1b_destination_{policy}.json").read_text())
    rows["volume_global"] = read("volume_remove_exchange")["conditions"]["mean_global"]["effect_vs_intact"]
    rows["volume_role"] = read("volume_physical_control")["conditions"]["mean_both"]["effect_vs_intact"]
    rows["volume_role_swap"] = read("volume_remove_exchange")["conditions"]["mean_exchange"]["effect_vs_intact"]
    labels = ["read", "destination", "mixer", "volume"]
    names = ["Read", "Write destination", "Mixer", "Write volume"]
    fig, ax = plt.subplots(figsize=(9.8, 4.5))
    x = np.arange(len(labels))
    for offset, policy, color in ((-0.25, "global", GRAY), (0, "role", BLUE),
                                  (0.25, "role_swap", RED)):
        present = [(i, delta(rows[f"{factor}_{policy}"]))
                   for i, factor in enumerate(labels) if f"{factor}_{policy}" in rows]
        ax.bar([i + offset for i, _ in present], [value for _, value in present],
               width=0.23, color=color,
               label={"global": "one global mean", "role": "masked / visible means",
                      "role_swap": "roles swapped"}[policy])
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_xticks(x, names)
    ax.set_ylabel("ΔCE after router-factor replacement")
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncol=3, fontsize=8,
               loc="center", bbox_to_anchor=(0.5, 0.87))
    style(ax)
    fig.suptitle("Router controls depend on token role · step 124,000", y=0.97)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.14, top=0.78)
    save(fig, "role_controls")


def training():
    rows = read("training_gain")["rows"]
    common = np.asarray([row["step"] for row in rows])
    if len(common) == 0 or common[-1] != 124000:
        raise ValueError("Matched validation at step 124000 is missing")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for field, name, color in (("mhc", "mHC n=2", BLUE),
                               ("baseline", "standard residual", GRAY)):
        axes[0].plot(common / 1000, [row[field] for row in rows],
                     color=color, label=name, linewidth=1.6)
    gain = np.asarray([row["gain_ce"] for row in rows])
    axes[1].plot(common / 1000, gain, color=BLUE, linewidth=1.6)
    axes[1].axhline(0, color="#444c54", linewidth=0.8)
    for ax in axes:
        ax.axvline(125, color=RED, linestyle="--", linewidth=1)
        ax.set_xlabel("Training step (thousands)")
        style(ax)
    axes[0].set(ylabel="Validation CE", title="Matched validation")
    axes[1].set(ylabel="Baseline CE − mHC CE", title="Two-stream gain")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("1B training through the frozen step-124,000 checkpoint")
    fig.tight_layout()
    save(fig, "training")


if __name__ == "__main__":
    for fn in (first_block, circuit, sufficiency, transfer, factors, training):
        fn()
