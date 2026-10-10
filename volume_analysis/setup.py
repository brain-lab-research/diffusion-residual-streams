"""Prepare disjoint physical-factor calibration and intact 337M references.

Run after scripts.configure_analysis and before run.py.
"""
import json
import numpy as np
import torch

from run import BASE, CONFIG, ROOT, atomic_json, sequence_ce
from mechanism_audit_n2.engine import batches, load_model, make_split


def capture_routes(engine, split):
    chunks = []
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        for _, end, _, _, corrupted, ids, mask in batches(split, CONFIG['batch_size']):
            local = []
            def collect(index, before, after, route, routed, delta):
                local.append(route.flatten(0, 1).float().cpu().numpy())
            engine.run(corrupted, mask, callback=collect)
            chunks.append(np.stack(local))
            print('CALIBRATE', split['ratio'], end, '/', len(split['ids']), flush=True)
    return np.concatenate(chunks, axis=1).astype(np.float64)


def mean_by_role(values, mask):
    return {'global': values.mean(1, dtype=np.float64),
            'role': np.stack((values[:, mask].mean(1, dtype=np.float64),
                              values[:, ~mask].mean(1, dtype=np.float64)), axis=1)}


def calibrate(engine, ratio):
    split = make_split(CONFIG, 'calibration', ratio)
    path = ROOT / f'calibration_r{round(100*ratio):02d}.json'
    capture = ROOT / 'calibration_all_positions.npz'
    if path.exists() and (ratio != .5 or capture.exists()):
        existing = json.loads(path.read_text())
        assert existing['mask_sha256'] == split['mask_sha256']
        assert existing['checkpoint_step'] == engine.checkpoint_step
        print('CACHED CALIBRATION', ratio, flush=True)
        return
    source = BASE / 'residual_circuits_n2/captures' / f'r{round(100*ratio):02d}_all_positions.npz'
    if source.exists():
        with np.load(source) as saved:
            np.testing.assert_array_equal(saved['block_ids'], split['block_ids'])
            np.testing.assert_array_equal(saved['mask'], split['mask'])
            assert str(saved['mask_sha256']) == split['mask_sha256']
            assert str(saved['data_sha256']) == split['data_sha256']
            routes = saved['router'].astype(np.float64)
    else:
        routes = capture_routes(engine, split)
    assert routes.shape == (32, split['mask'].size, 6)
    gains = routes[..., 2:4]
    volume = gains.sum(-1, keepdims=True)
    assert np.isfinite(routes).all() and (volume > 0).all()
    means = {'volume': mean_by_role(volume, split['mask'].ravel())}
    record = {'checkpoint_step': engine.checkpoint_step, 'ratio': ratio,
              'mask_sha256': split['mask_sha256'], 'data_sha256': split['data_sha256'],
              'calibration_blocks': list(map(int, split['block_ids'])),
              'volume_role_means': means['volume']['role'].tolist()}
    atomic_json(path, record)
    if ratio == .5:
        np.savez_compressed(capture, router=routes.astype(np.float32), mask=split['mask'],
                            valid=np.ones_like(split['mask']), block_ids=split['block_ids'],
                            mask_sha256=np.array(split['mask_sha256']))
        raw = {'read': routes[..., :2], 'volume': volume,
               'destination': gains / volume, 'mixer': routes[..., 4:]}
        full = {'global': {}, 'role': {}}
        for factor, values in raw.items():
            grouped = mean_by_role(values, split['mask'].ravel())
            for policy in full:
                mean = grouped[policy]
                if factor in ('destination', 'mixer'):
                    mean = mean / mean.sum(-1, keepdims=True)
                full[policy][factor] = mean.tolist()
        atomic_json(ROOT / 'calibration_means.json',
                    {'checkpoint_step': engine.checkpoint_step,
                     'mask_sha256': split['mask_sha256'],
                     'data_sha256': split['data_sha256'], 'means': full})


def intact(engine, name, ratio):
    split = make_split(CONFIG, name, ratio)
    directory = ROOT / 'references'
    directory.mkdir(exist_ok=True)
    path = directory / f'{name}_r{round(100*ratio):02d}.npz'
    if path.exists():
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved['block_ids'], split['block_ids'])
            assert str(saved['mask_sha256']) == split['mask_sha256']
        print('CACHED INTACT', name, ratio, flush=True)
        return
    sums, counts = [], []
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        for _, end, _, _, corrupted, ids, mask in batches(split, CONFIG['batch_size']):
            state, _ = engine.run(corrupted, mask)
            local_sums, local_counts = sequence_ce(engine, state, ids, mask)
            sums.append(local_sums); counts.append(local_counts)
            print('INTACT', name, ratio, end, '/', len(split['ids']), flush=True)
    mask = split['mask']
    np.savez_compressed(path, seq_ce_sum=np.concatenate(sums),
                        seq_mask_count=np.concatenate(counts), block_ids=split['block_ids'],
                        mask_sha256=np.array(split['mask_sha256']),
                        positions=np.where(mask)[1],
                        token_sequence=np.repeat(np.arange(len(split['ids'])), mask.sum(1)))


def main():
    torch.set_num_threads(4)
    engine = load_model(CONFIG, 'full_n2')
    assert engine.checkpoint_step == 53399 and len(engine.interfaces) == 32
    for ratio in (.1, .3, .5, .7, .9):
        calibrate(engine, ratio)
        intact(engine, 'discovery', ratio)
    intact(engine, 'confirmation', .5)
    print('PHYSICAL REFERENCES COMPLETE', ROOT, flush=True)


if __name__ == '__main__':
    main()
