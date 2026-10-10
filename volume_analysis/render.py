"""Render the physical write-volume results from analysis.json."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run import ROOT


data = json.loads((ROOT / 'analysis.json').read_text())
core = data['confirmation']
colors = {'role mean': '#1a7494', 'pairwise shuffle': '#2f8063'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 160})


def effect(row):
    return row['effect_vs_intact']


def interval(ax, x, row, color, label=None):
    item = effect(row)
    center = item['delta_ce']
    low, high = item['ci95']
    ax.errorbar(x, center, yerr=[[center-low], [high-center]], marker='o',
                capsize=3, lw=1.2, markersize=5, color=color, label=label)


fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
for ax, role in zip(axes, ('masked', 'visible')):
    rows = [core['mean'][role], core['shuffle'][f'{role}_seed130921']]
    for i, (label, row) in enumerate(zip(colors, rows)):
        interval(ax, i, row, colors[label])
    ax.set_xticks(range(2), list(colors))
    ax.set_title(f'{role.capitalize()} positions', loc='left', fontweight='bold')
    ax.set_ylabel('Δ masked CE vs intact')
    ax.axhline(0, color='#333', lw=.7)
    ax.grid(axis='y', alpha=.15)
    for i, row in enumerate(rows):
        coverage = row['diagnostics']['all']['feasible_fraction']
        ax.annotate(f'{coverage:.1%} accepted', (i, effect(row)['delta_ce']),
                    xytext=(0, 9), textcoords='offset points', ha='center', fontsize=8)
fig.suptitle('Physical write-volume interventions · 337M confirmation split', fontweight='bold')
for suffix in ('png', 'pdf'):
    fig.savefig(ROOT / f'volume_physical_control.{suffix}', bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
for ax, role, blocks in ((axes[0], 'visible', (1, 13, 16)),
                         (axes[1], 'masked', (1, 10, 16))):
    for x, block in enumerate(blocks):
        interval(ax, x, core['depth'][f'block{block:02d}_{role}'], '#1a7494')
    ax.set_xticks(range(len(blocks)), [f'block {block}' for block in blocks])
    ax.set_title(f'{role.capitalize()} positions', loc='left', fontweight='bold')
    ax.set_ylabel('Δ masked CE vs intact')
    ax.axhline(0, color='#333', lw=.7)
    ax.grid(axis='y', alpha=.15)
ratios = (.1, .3, .5, .7, .9)
for role, color in (('masked', '#1a7494'), ('visible', '#d39b52'), ('both', '#2f8063')):
    for ratio in ratios:
        interval(axes[2], ratio, data['noise'][f'r{round(100*ratio):02d}']['roles'][role],
                 color, role if ratio == ratios[0] else None)
axes[2].set_xticks(ratios)
axes[2].set_xlabel('Mask fraction')
axes[2].set_ylabel('Δ masked CE vs intact')
axes[2].set_title('Discovery split', loc='left', fontweight='bold')
axes[2].axhline(0, color='#333', lw=.7)
axes[2].grid(axis='y', alpha=.15)
axes[2].legend(frameon=False, fontsize=8)
fig.suptitle('Depth and corruption dependence · physical write shares', fontweight='bold')
for suffix in ('png', 'pdf'):
    fig.savefig(ROOT / f'volume_physical_schedule.{suffix}', bbox_inches='tight')
plt.close(fig)
print('FIGURES COMPLETE')
