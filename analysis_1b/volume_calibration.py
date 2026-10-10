"""Calibration and pair maps shared by the physical 1B write-volume controls."""
import json

import numpy as np
import torch

from audit import CONFIG, ROOT, batches, evaluate, make_split, save_json
from mechanism_audit_n2.volume_roles import VolumeRoleHooks


class VolumeHooks64(VolumeRoleHooks):
    def configure(self, mask, valid, mode, block_ids=None, seed=None, layers=None):
        assert mode in ('dynamic', 'mean_masked', 'mean_visible', 'mean_both',
                        'shuffle_masked', 'shuffle_visible')
        n = len(self.engine.interfaces)
        self.mask, self.valid, self.mode = mask, valid, mode
        self.active_layers = set(range(n)) if layers is None else set(layers)
        self.maps = None
        if mode.startswith('shuffle'):
            assert block_ids is not None and seed is not None
            selected = (mask if mode == 'shuffle_masked' else ~mask) & valid
            role_np = selected.cpu().numpy()
            maps = np.broadcast_to(np.arange(mask.shape[1], dtype=np.int64),
                                   (n, mask.shape[0], mask.shape[1])).copy()
            code = 0 if mode == 'shuffle_masked' else 1
            for layer in range(n):
                for b, block_id in enumerate(block_ids):
                    positions = np.flatnonzero(role_np[b])
                    rng = np.random.default_rng(np.random.SeedSequence(
                        [int(seed), int(block_id), layer, code]))
                    maps[layer, b, positions] = rng.permutation(positions)
            self.maps = torch.from_numpy(maps).to('cuda')


def mean_volume_for_ratio(engine, ratio):
    path = ROOT / f'volume_calibration_r{round(100*ratio):02d}.json'
    if path.exists():
        return np.asarray(json.loads(path.read_text())['means'], dtype=np.float64)
    split = make_split(CONFIG, 'calibration', ratio)
    sums = np.zeros((64, 2), dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        for _, end, _, _, corrupted, ids, mask in batches(split, CONFIG['batch_size']):
            for role, selected in enumerate((mask, ~mask)):
                counts[role] += int(selected.sum())
            def collect(index, before, after, router, routed, delta):
                volume = router[..., 2:4].double().sum(-1)
                for role, selected in enumerate((mask, ~mask)):
                    sums[index, role] += float(volume[selected].sum())
            engine.run(corrupted, mask, callback=collect)
            if end % 16 == 0:
                print('CALIBRATE V', ratio, end, '/', len(split['ids']), flush=True)
    means = (sums / counts[None]).reshape(64, 2, 1)
    save_json(path, {'ratio': ratio, 'checkpoint_step': engine.checkpoint_step,
                     'calibration_sequences': len(split['ids']), 'counts': counts.tolist(),
                     'means': means.tolist(), 'mask_sha256': split['mask_sha256']})
    return means


def intact_for_ratio(engine, split, ratio):
    name = 'intact' if ratio == .5 else f'volume_intact_r{round(100*ratio):02d}'
    return evaluate(engine, split, name)
