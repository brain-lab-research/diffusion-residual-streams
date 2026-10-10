"""Clean, physically share preserving Remove and Exchange at t=0.5."""
import json

import numpy as np
import torch

from audit import CONFIG, ROOT, load_model, make_split, paired_delta, save_json
from volume_physical_control import PhysicalShareHooks, evaluate


def load_pack(path):
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def main():
    torch.set_num_threads(4)
    calibration = json.loads((ROOT/'calibration.json').read_text())
    role_means = np.asarray(calibration['means']['role']['volume'], dtype=np.float64)
    global_means = np.asarray(calibration['means']['global']['volume'], dtype=np.float64)
    engine = load_model(CONFIG, 'full_n2')
    assert engine.checkpoint_step == 124000 and len(engine.interfaces) == 64
    split = make_split(CONFIG, 'test', .5)
    assert not set(calibration['calibration_blocks']) & set(map(int, split['block_ids']))
    intact = load_pack(ROOT/'intact.npz')
    role = load_pack(ROOT/'volume_physical_proportional_mean_both_r50.npz')
    for pack in (intact, role):
        np.testing.assert_array_equal(pack['block_ids'], split['block_ids'])
        np.testing.assert_array_equal(pack['seq_mask_count'], split['mask'].sum(1))
        assert str(pack['mask_sha256']) == split['mask_sha256']
    hooks = PhysicalShareHooks(engine, role_means, global_means)
    summary = {'checkpoint_step': 124000, 'ratio': .5, 'sequences': len(split['ids']),
               'mask_sha256': split['mask_sha256'], 'calibration_blocks': calibration['calibration_blocks'],
               'definition': 'Remove assigns the all-token calibration mean V at each interface; '
               'Exchange assigns masked positions the visible mean and vice versa. '
               'Apply only when target V times current max(g_i/V) <= 2; otherwise leave gains unchanged.',
               'conditions': {}}
    try:
        for mode in ('mean_global', 'mean_exchange'):
            pack, diag = evaluate(engine, split, hooks, mode, 'proportional')
            assert diag['max_abs_physical_share_drift'] < 1e-6
            summary['conditions'][mode] = {
                'effect_vs_intact': paired_delta(pack, intact),
                'effect_vs_role_mean': paired_delta(pack, role),
                'diagnostics': diag}
            save_json(ROOT/'volume_remove_exchange.json', summary)
    finally:
        hooks.close()
    exchange = load_pack(ROOT/'volume_physical_proportional_mean_exchange_r50.npz')
    remove = load_pack(ROOT/'volume_physical_proportional_mean_global_r50.npz')
    summary['exchange_minus_remove'] = paired_delta(exchange, remove)
    save_json(ROOT/'volume_remove_exchange.json', summary)
    print('REMOVE/EXCHANGE COMPLETE', ROOT/'volume_remove_exchange.json', flush=True)


if __name__ == '__main__':
    main()
