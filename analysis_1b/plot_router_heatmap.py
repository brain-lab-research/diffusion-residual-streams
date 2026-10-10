"""Plot the measured 1B routing profile in the style of the 337M atlas.

Run: python analysis_1b/plot_router_heatmap.py
Inputs are the role-conditioned means in router_profile.json; no model run is needed.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent
PROFILE = ROOT / "router_profile.json"
STEMS = ("a", "m")
LABELS = [f"{kind}{block}" for block in range(32) for kind in STEMS]

# The middle tile contains the interfaces between the two state snapshots:
# immediately after a8 (index 16) and immediately after a16 (index 32).
PHASES = (
    ("early · through a8", 0, 17),
    ("middle · a8→a16", 17, 33),
    ("late · after a16", 33, 64),
)
GAP = 1.25
STREAM_COLORS = ("#c28b20", "#7052a0", "#303947")
ROLE_COLORS = {"visible": "#057fae", "masked": "#ae3554"}
CMAPS = {
    "visible": LinearSegmentedColormap.from_list(
        "visible_atlas", ("#f7fbfd", "#7bc6e2", "#087daa", "#073e55")
    ),
    "masked": LinearSegmentedColormap.from_list(
        "masked_atlas", ("#fdf8f9", "#e4a6b4", "#ad3857", "#591a2c")
    ),
}
ROWS = (
    ("Read share", "read_share", 1.0),
    ("Write share", "write_share", 1.0),
    ("Write gain", "gains", 2.0),
)


def x_for(index):
    return index + GAP * sum(index >= end for _, _, end in PHASES[:-1])


def draw_tile(axis, values, cmap, row, count, maximum):
    for slot, (_, begin, end) in enumerate(PHASES):
        offset = slot * GAP
        axis.imshow(
            (values[begin:end] / maximum)[None, :],
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
            aspect="auto",
            origin="lower",
            extent=(begin - 0.5 + offset, end - 0.5 + offset,
                    count - row - 1, count - row),
        )


def panel(fig, y, height, role, group, values, show_x=False):
    axis = fig.add_axes((0.155, y, 0.705, height), zorder=2)
    count = len(values)
    cmap = CMAPS[role]
    for row, (name, data, maximum) in enumerate(values):
        draw_tile(axis, data, cmap, row, count, maximum)
    for slot, (_, begin, end) in enumerate(PHASES):
        axis.add_patch(Rectangle(
            (begin - 0.5 + slot * GAP, 0), end - begin, count,
            fill=False, edgecolor=STREAM_COLORS[group], linewidth=0.9,
            clip_on=False, zorder=4,
        ))
    for level in range(1, count):
        axis.axhline(level, color="white", linewidth=0.8)
    axis.set_yticks([count - row - 0.5 for row in range(count)],
                    [item[0] for item in values], fontsize=9.6)
    axis.set_xlim(-0.65, x_for(63) + 0.65)
    axis.set_ylim(0, count)
    axis.tick_params(axis="y", length=0, pad=7)
    if show_x:
        axis.set_xticks([x_for(i) for i in range(64)], LABELS,
                        rotation=90, fontsize=6.3)
        axis.tick_params(axis="x", length=0, pad=2)
        axis.set_xlabel("Interface", fontsize=11, labelpad=4)
    else:
        axis.set_xticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    # Pale row-group panels and a colour key for the stream index.
    fig.add_artist(FancyBboxPatch(
        (0.042, y - 0.006), 0.834, height + 0.012,
        boxstyle="round,pad=0.004,rounding_size=0.009",
        transform=fig.transFigure, facecolor=STREAM_COLORS[group], alpha=0.045,
        edgecolor="none", zorder=0,
    ))
    fig.lines.append(plt.Line2D(
        (0.069, 0.069), (y + 0.007, y + height - 0.007),
        transform=fig.transFigure, color=STREAM_COLORS[group], linewidth=2.1,
        solid_capstyle="butt", zorder=3,
    ))
    fig.text(0.057, y + height / 2,
             ("s₀", "s₁", "both")[group],
             rotation=90, va="center", ha="center", fontsize=10.3,
             color=STREAM_COLORS[group], fontweight="bold")
    return axis


def main():
    profile = json.loads(PROFILE.read_text())
    assert profile["checkpoint_step"] == 124000 and profile["sequences"] == 64
    for role in ("visible", "masked"):
        for key in ("read_share", "write_share", "gains", "mixer_swap_probability"):
            assert len(profile["roles"][role][key]) == 64

    plt.rcParams.update({"font.family": "DejaVu Serif", "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig = plt.figure(figsize=(18.5, 9.0), facecolor="white")
    fig.text(0.155, 0.97, "1B mHC routing atlas", fontsize=16,
             fontweight="bold", va="top", color="#202936")
    fig.text(0.155, 0.935,
             "step 124,000 · 32 blocks / 64 interfaces · mask ratio 0.5 · 64 calibration sequences",
             fontsize=9.5, va="top", color="#596473")

    # Two roles, with three stream groups of relative heights 3:3:1.
    specs = (("visible", 0.745), ("masked", 0.377))
    for role, top in specs:
        d = profile["roles"][role]
        data = {key: np.asarray(d[key], dtype=float)
                for key in ("read_share", "write_share", "gains")}
        mixer = np.asarray(d["mixer_swap_probability"], dtype=float)
        group_specs = (
            (0, top, 0.120, [(name, data[key][:, 0], maximum)
                            for name, key, maximum in ROWS]),
            (1, top - 0.146, 0.120,
             [(name, data[key][:, 1], maximum) for name, key, maximum in ROWS]),
            (2, top - 0.230, 0.040, [("Mixer swap", mixer, 0.25)]),
        )
        for group, y, height, values in group_specs:
            panel(fig, y, height, role, group, values,
                  show_x=(role == "masked" and group == 2))
        low, high = top - 0.230, top + 0.120
        cb_axis = fig.add_axes((0.881, low, 0.009, high - low))
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1),
                                                  cmap=CMAPS[role]),
                           cax=cb_axis, ticks=(0, 0.5, 1))
        bar.ax.set_yticklabels(("0", "½", "full"), fontsize=8.2)
        bar.outline.set_linewidth(0.6)
        bar.ax.tick_params(length=2, width=0.6)
        fig.text(0.924, (low + high) / 2, role, rotation=270,
                 color=ROLE_COLORS[role], fontsize=13, fontweight="bold",
                 ha="center", va="center")

    # Phase boundaries reflect the measured a8-to-a16 state comparison, not
    # claims that the 337M construction and reopening circuit has transferred.
    for title, begin, end in PHASES:
        left, right = x_for(begin), x_for(end - 1)
        x = 0.155 + 0.705 * ((left + right) / 2 + 0.65) / (x_for(63) + 1.3)
        fig.text(x, 0.893, title, ha="center", va="bottom", fontsize=10,
                 color="#26313f", fontweight="bold")
        x0 = 0.155 + 0.705 * (left + 0.15) / (x_for(63) + 1.3)
        x1 = 0.155 + 0.705 * (right + 1.15) / (x_for(63) + 1.3)
        fig.lines.append(plt.Line2D((x0, x1), (0.888, 0.888),
                                    transform=fig.transFigure, color="#35404e",
                                    linewidth=1.2))

    fig.text(0.155, 0.055,
             "Colour = fraction of row scale: shares 0–1; gain 0–2; mixer swap 0–0.25. "
             "Shares are ratios of role-mean gates; gain and swap are role means.",
             fontsize=8.9, color="#495564")
    fig.text(0.155, 0.033,
             "Phase labels mark the post-a8 → post-a16 comparison; routing colour alone does not measure state preservation.",
             fontsize=8.5, color="#67717e")
    for ext, options in (("png", {"dpi": 220}), ("pdf", {}), ("svg", {})):
        fig.savefig(ROOT / f"router_heatmap.{ext}", **options)
    plt.close(fig)


if __name__ == "__main__":
    main()
