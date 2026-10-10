"""Check the saved physical interventions against sealed splits and gain invariants."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechanism_audit_n2.engine import make_split
from run import CONFIG, ROOT, means_for_ratio, global_means_for_ratio, atomic_json


def reference_path(name, ratio):
    return ROOT / 'references' / f'{name}_r{round(100*ratio):02d}.npz'


def main():
    checks = {'checkpoint_step': 53399, 'splits': {}, 'effects': 0,
              'max_physical_share_drift': 0., 'max_gain_bound_violation': 0.}
    unit = json.loads((ROOT / 'unit.json').read_text())
    assert unit['intact_seq_ce_sum_max_abs_error'] == 0
    assert unit['max_proposed_volume_multiset_error'] == 0
    for name, ratios in (('confirmation', (.5,)), ('discovery', (.1, .3, .5, .7, .9))):
        for ratio in ratios:
            split = make_split(CONFIG, name, ratio)
            tag = f'{name}_r{round(100*ratio):02d}'
            with np.load(reference_path(name, ratio)) as intact:
                np.testing.assert_array_equal(intact['block_ids'], split['block_ids'])
                np.testing.assert_array_equal(intact['seq_mask_count'], split['mask'].sum(1))
                np.testing.assert_array_equal(intact['positions'], np.where(split['mask'])[1])
                np.testing.assert_array_equal(intact['token_sequence'],
                    np.repeat(np.arange(len(split['ids'])), split['mask'].sum(1)))
            checks['splits'][tag] = {'sequences': len(split['ids']),
                                     'masked_targets': int(split['mask'].sum()),
                                     'mask_sha256': split['mask_sha256']}
            for path in sorted((ROOT / 'effects').glob(tag + '*.npz')):
                if '.partial.' in path.name:
                    raise RuntimeError(f'unfinished effect: {path}')
                assert 'bounded' not in path.name, path
                with np.load(path) as effect:
                    np.testing.assert_array_equal(effect['block_ids'], split['block_ids'])
                    np.testing.assert_array_equal(effect['seq_mask_count'], split['mask'].sum(1))
                    assert str(effect['mask_sha256']) == split['mask_sha256']
                diagnostic = json.loads(path.with_suffix('.json').read_text())
                checks['max_gain_bound_violation'] = max(
                    checks['max_gain_bound_violation'], diagnostic['max_gain_bound_violation'])
                checks['max_physical_share_drift'] = max(
                    checks['max_physical_share_drift'], diagnostic['max_abs_physical_share_drift'])
                checks['effects'] += 1
    role = np.asarray(json.loads((ROOT / 'calibration_means.json').read_text())
                      ['means']['role']['volume'])
    np.testing.assert_array_equal(role, means_for_ratio(.5))
    global_volume = np.asarray(json.loads((ROOT / 'calibration_means.json').read_text())
                               ['means']['global']['volume'])
    np.testing.assert_array_equal(global_volume, global_means_for_ratio(.5))
    assert checks['effects'] == 56, checks['effects']
    assert checks['max_gain_bound_violation'] == 0
    assert checks['max_physical_share_drift'] < 1e-6
    checks['passed'] = True
    atomic_json(ROOT / 'verification.json', checks)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
