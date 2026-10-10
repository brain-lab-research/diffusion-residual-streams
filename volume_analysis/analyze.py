"""Summarize the physical write-volume interventions on sealed splits."""
import json

import numpy as np

from run import BASE, CONFIG, ROOT, SEEDS, atomic_json, effect_name, reference_for


def load(mode, policy='proportional', ratio=.5, split='confirmation', layers=None, seed=None):
    name = effect_name(mode, policy, ratio, layers, seed, split)
    with np.load(ROOT / 'effects' / (name + '.npz')) as saved:
        pack = {key: saved[key] for key in saved.files}
    diag = json.loads((ROOT / 'effects' / (name + '.json')).read_text())
    return pack, diag


def contrast(left, right):
    np.testing.assert_array_equal(left['block_ids'], right['block_ids'])
    np.testing.assert_array_equal(left['seq_mask_count'], right['seq_mask_count'])
    assert len(left['block_ids']) == len(np.unique(left['block_ids']))
    delta = left['seq_ce_sum'] - right['seq_ce_sum']
    count = left['seq_mask_count']
    rng = np.random.default_rng(CONFIG['bootstrap_seed'])
    draws = []
    for start in range(0, CONFIG['bootstrap_repeats'], 250):
        pick = rng.integers(len(count), size=(min(250, CONFIG['bootstrap_repeats']-start), len(count)))
        draws.extend(delta[pick].sum(-1) / count[pick].sum(-1))
    return {'delta_ce': float(delta.sum()/count.sum()),
            'ci95': list(map(float, np.quantile(draws, [.025,.975])))}


def row(mode, intact, ratio=.5, split='confirmation', layers=None, seed=None,
        policy='proportional'):
    pack, diag = load(mode, policy, ratio, split, layers, seed)
    np.testing.assert_array_equal(pack['block_ids'], intact['block_ids'])
    np.testing.assert_array_equal(pack['seq_mask_count'], intact['seq_mask_count'])
    artifact = ROOT / 'effects' / (effect_name(mode, policy, ratio, layers, seed, split) + '.npz')
    return {'effect_vs_intact': contrast(pack, intact), 'diagnostics': diag,
            'artifact': str(artifact.relative_to(BASE))}


def main():
    intact = reference_for('confirmation', .5)
    result = {'checkpoint_step': 53399, 'model': CONFIG['models']['full_n2'],
              'protocol': 'physical gains: g_i = V q_i; feasible proportional replacement only',
              'confirmation': {'sequences': len(intact['block_ids']),
                               'masked_tokens': int(intact['seq_mask_count'].sum()),
                               'intact_ce': float(intact['seq_ce_sum'].sum()/intact['seq_mask_count'].sum()),
                               'mean': {}, 'remove_exchange': {}, 'shuffle': {}, 'depth': {}},
              'noise': {}, 'statistics': {'unit': 'paired sequence',
                                        'bootstrap_repeats': CONFIG['bootstrap_repeats'],
                                        'bootstrap_seed': CONFIG['bootstrap_seed']}}
    core = result['confirmation']
    for role in ('masked', 'visible', 'both'):
        core['mean'][role] = row('mean_' + role, intact)

    role_pack, _ = load('mean_both')
    for label, mode in (('remove', 'mean_global'), ('exchange', 'mean_exchange')):
        effect = row(mode, intact)
        pack, _ = load(mode)
        effect['minus_role_mean'] = contrast(pack, role_pack)
        core['remove_exchange'][label] = effect
    exchange_pack, _ = load('mean_exchange')
    remove_pack, _ = load('mean_global')
    core['remove_exchange']['exchange_minus_remove'] = contrast(exchange_pack, remove_pack)

    for role in ('masked', 'visible'):
        mean_pack, _ = load('mean_' + role)
        mode = 'shuffle_' + role
        for seed in SEEDS:
            key = f'{role}_seed{seed}'
            effect = row(mode, intact, seed=seed, policy='pairwise_proportional')
            pack, _ = load(mode, 'pairwise_proportional', seed=seed)
            effect['minus_role_mean'] = contrast(pack, mean_pack)
            core['shuffle'][key] = effect

    for block in range(1, 17):
        layers = (2*(block-1), 2*(block-1)+1)
        for role in ('masked', 'visible'):
            core['depth'][f'block{block:02d}_{role}'] = row('mean_' + role, intact, layers=layers)

    for ratio in (.1, .3, .5, .7, .9):
        key = f'r{round(100*ratio):02d}'
        local_intact = reference_for('discovery', ratio)
        result['noise'][key] = {
            'sequences': len(local_intact['block_ids']),
            'intact_ce': float(local_intact['seq_ce_sum'].sum()/local_intact['seq_mask_count'].sum()),
            'roles': {role: row('mean_' + role, local_intact, ratio=ratio, split='discovery')
                      for role in ('masked', 'visible', 'both')}}

    atomic_json(ROOT / 'analysis.json', result)
    print('ANALYSIS COMPLETE', ROOT / 'analysis.json')


if __name__ == '__main__':
    main()
