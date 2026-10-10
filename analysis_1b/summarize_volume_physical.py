"""Paired sequence-bootstrap contrasts for the physical-volume report."""
import json

import numpy as np

from audit import ROOT, paired_delta, save_json


def load(name):
    with np.load(ROOT / name) as data:
        return {key: data[key] for key in data.files}


summary = {'checkpoint_step': 124000, 'bootstrap_unit': 'sequence',
           'bootstrap_repeats': 5000, 'conditions': {}}
for role in ('masked', 'visible'):
    mean = load(f'volume_physical_proportional_mean_{role}_r50.npz')
    for seed in (130921, 130922):
        suffix = f'shuffle_{role}_seed{seed}_r50.npz'
        exact = load('volume_physical_pairwise_proportional_' + suffix)
        summary['conditions'][f'{role}_seed{seed}'] = {
            'pairwise_shuffle_minus_mean': paired_delta(exact, mean),
        }
save_json(ROOT / 'volume_physical_comparisons.json', summary)
print(json.dumps(summary, indent=2))
