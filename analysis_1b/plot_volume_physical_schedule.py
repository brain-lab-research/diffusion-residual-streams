"""Plot 1B physical write-volume sensitivity by block and mask ratio."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
followup = json.loads((ROOT / 'volume_physical_followup.json').read_text())['conditions']
main = json.loads((ROOT / 'volume_physical_control.json').read_text())['conditions']
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 160})
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
for ax, role, color in ((axes[0], 'visible', '#1a7494'),
                        (axes[1], 'masked', '#b03959')):
    rows = [followup[f'r50_mean_{role}_{2*block}_{2*block+1}']['effect_vs_intact']
            for block in range(32)]
    y = np.asarray([row['delta_ce'] for row in rows])
    ci = np.asarray([row['ci95'] for row in rows])
    ax.errorbar(range(32), y, yerr=[y-ci[:, 0], ci[:, 1]-y], marker='o',
                markersize=2.5, lw=1, capsize=1.5, color=color)
    ax.set_xticks((0, 8, 16, 24, 31))
    ax.set_xlabel('Block')
    ax.set_ylabel('Δ masked CE vs intact')
    ax.set_title(f'{role.capitalize()} positions', loc='left', fontweight='bold')
    ax.axhline(0, color='#333', lw=.7)
    ax.grid(axis='y', alpha=.15)
ratios = (.1, .3, .5, .7, .9)
rows = [(main['mean_both']['effect_vs_intact'] if ratio == .5 else
         followup[f'r{round(100*ratio):02d}_mean_both_all']['effect_vs_intact'])
        for ratio in ratios]
y = np.asarray([row['delta_ce'] for row in rows])
ci = np.asarray([row['ci95'] for row in rows])
axes[2].errorbar(ratios, y, yerr=[y-ci[:, 0], ci[:, 1]-y], marker='o',
                 capsize=2.5, lw=1.2, color='#2f8063')
axes[2].set_xticks(ratios)
axes[2].set_xlabel('Mask fraction')
axes[2].set_ylabel('Δ masked CE vs intact')
axes[2].set_title('All interfaces', loc='left', fontweight='bold')
axes[2].axhline(0, color='#333', lw=.7)
axes[2].grid(axis='y', alpha=.15)
fig.suptitle('Physical write-volume sensitivity · 1B step 124,000', fontweight='bold')
for suffix in ('png', 'pdf'):
    fig.savefig(ROOT / f'volume_physical_schedule.{suffix}', bbox_inches='tight')
plt.close(fig)
