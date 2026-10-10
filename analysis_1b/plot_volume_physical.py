"""Render the physical write-volume interventions on the frozen 1B model."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
means = json.loads((ROOT / 'volume_physical_control.json').read_text())['conditions']
shuffles = json.loads((ROOT / 'volume_pair_swap.json').read_text())['conditions']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 160})
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
for ax, role in zip(axes, ('masked', 'visible')):
    rows = [means[f'mean_{role}'], shuffles[f'shuffle_{role}_seed130921']]
    for x, (row, color) in enumerate(zip(rows, ('#1a7494', '#2f8063'))):
        value = row['effect_vs_intact']
        center = value['delta_ce']
        low, high = value['ci95']
        ax.errorbar(x, center, yerr=[[center-low], [high-center]], marker='o',
                    capsize=3, lw=1.2, markersize=5, color=color)
        coverage = row['diagnostics']['all']['feasible_fraction']
        ax.annotate(f'{coverage:.1%} accepted', (x, center), xytext=(0, 9),
                    textcoords='offset points', ha='center', fontsize=8)
    ax.set_xticks((0, 1), ('role mean', 'pairwise shuffle'))
    ax.set_title(f'{role.capitalize()} positions', loc='left', fontweight='bold')
    ax.set_ylabel('Δ masked CE vs intact')
    ax.axhline(0, color='#333', lw=.7)
    ax.grid(axis='y', alpha=.15)
fig.suptitle('Physical write-volume interventions · 1B step 124,000', fontweight='bold')
for suffix in ('png', 'pdf'):
    fig.savefig(ROOT / f'volume_physical_control.{suffix}', bbox_inches='tight')
plt.close(fig)
