"""Verify the saved clean Remove/Exchange effects against the frozen 1B split."""
import json

import numpy as np

from audit import CONFIG, ROOT, make_split, save_json


def main():
    split = make_split(CONFIG, 'test', .5)
    calibration = json.loads((ROOT/'calibration.json').read_text())
    summary = json.loads((ROOT/'volume_remove_exchange.json').read_text())
    assert not set(calibration['calibration_blocks']) & set(map(int, split['block_ids']))
    assert summary['mask_sha256'] == split['mask_sha256']
    with np.load(ROOT/'intact.npz') as saved:
        intact = {key: saved[key] for key in saved.files}
    checks = {'checkpoint_step': 124000, 'sequences': len(split['ids']),
              'masked_targets': int(split['mask'].sum()), 'mask_sha256': split['mask_sha256'],
              'effects': 0, 'max_physical_share_drift': 0., 'max_gain_bound_violation': 0.}
    for mode in ('mean_global', 'mean_exchange'):
        path = ROOT/f'volume_physical_proportional_{mode}_r50.npz'
        with np.load(path) as saved:
            pack = {key: saved[key] for key in saved.files}
        np.testing.assert_array_equal(pack['block_ids'], split['block_ids'])
        np.testing.assert_array_equal(pack['seq_mask_count'], split['mask'].sum(1))
        assert str(pack['mask_sha256']) == split['mask_sha256']
        delta = (pack['seq_ce_sum']-intact['seq_ce_sum']).sum()/pack['seq_mask_count'].sum()
        np.testing.assert_allclose(delta, summary['conditions'][mode]
                                   ['effect_vs_intact']['delta_ce'], atol=1e-12, rtol=0)
        diag = json.loads(path.with_suffix('.json').read_text())
        checks['max_gain_bound_violation'] = max(checks['max_gain_bound_violation'],
                                                  diag['max_gain_bound_violation'])
        checks['max_physical_share_drift'] = max(
            checks['max_physical_share_drift'], diag['max_abs_physical_share_drift'])
        checks['effects'] += 1
    assert checks['effects'] == 2
    assert checks['max_gain_bound_violation'] == 0
    assert checks['max_physical_share_drift'] < 3e-8
    checks['passed'] = True
    save_json(ROOT/'volume_remove_exchange_verification.json', checks)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
