"""Exact-multiset, physical-share-preserving within-role volume swaps."""
import json

import numpy as np
import torch

from audit import CONFIG, ROOT, load_model, make_split, paired_delta, save_json
from volume_calibration import mean_volume_for_ratio
from volume_physical_control import PhysicalShareHooks, evaluate


def main():
    torch.set_num_threads(4)
    engine = load_model(CONFIG, 'full_n2')
    assert engine.checkpoint_step == 124000
    split = make_split(CONFIG, 'test', .5)
    means = mean_volume_for_ratio(engine, .5)
    with np.load(ROOT/'intact.npz') as saved:
        intact = {k: saved[k] for k in saved.files}
    hooks = PhysicalShareHooks(engine, means)
    summary = {'checkpoint_step': 124000, 'conditions': {},
               'definition': 'Disjoint within-sequence, within-role, within-interface pairs '
               'swap total volume only when both proposed gains remain in [0,2]. '
               'Each original physical share g_i/V is held fixed under proportional scaling. '
               'The exact multiset of V is preserved at every modified interface.'}
    try:
        for seed in (130921, 130922):
            for mode in ('shuffle_masked', 'shuffle_visible'):
                key = f'{mode}_seed{seed}'
                pack, diag = evaluate(engine, split, hooks, mode,
                                      'pairwise_proportional', seed=seed)
                summary['conditions'][key] = {'effect_vs_intact': paired_delta(pack, intact),
                                              'diagnostics': diag}
                save_json(ROOT/'volume_pair_swap.json', summary)
    finally:
        hooks.close()
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
