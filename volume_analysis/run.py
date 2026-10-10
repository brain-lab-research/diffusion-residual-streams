"""Physical-share-preserving write-volume controls on the final 337M n=2 model.

CUDA_VISIBLE_DEVICES=2 python volume_analysis/run.py --group means
Groups: means, shuffle_masked, shuffle_visible, pairwise, depth, noise, smoke.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from mechanism_audit_n2.engine import batches, load_model, make_split
from mechanism_audit_n2.volume_roles import VolumeRoleHooks


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
CONFIG = json.loads((BASE / 'mechanism_audit_n2/config.json').read_text())
SEEDS = (130921, 130922)


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def means_for_ratio(ratio):
    if ratio == .5:
        return np.asarray(json.loads((ROOT / 'calibration_means.json').read_text())
                          ['means']['role']['volume'], dtype=np.float64)
    return np.asarray(json.loads((ROOT / f'calibration_r{round(100*ratio):02d}.json').read_text())
                      ['volume_role_means'], dtype=np.float64)


def global_means_for_ratio(ratio):
    if ratio != .5:
        raise ValueError('Global calibration was frozen only for ratio 0.5')
    return np.asarray(json.loads((ROOT / 'calibration_means.json').read_text())
                      ['means']['global']['volume'], dtype=np.float64)


def reference_for(split_name, ratio):
    if split_name not in ('confirmation', 'discovery') or (split_name == 'confirmation' and ratio != .5):
        raise ValueError((split_name, ratio))
    path = ROOT / 'references' / f'{split_name}_r{round(100*ratio):02d}.npz'
    with np.load(path) as saved:
        return {key: saved[key] for key in ('seq_ce_sum', 'seq_mask_count', 'block_ids')}


def sequence_ce(engine, state, ids, mask, chunk=128):
    hidden = state.mean(-2)[mask]
    labels = ids[mask]
    pieces = []
    for start in range(0, len(labels), chunk):
        logits = engine.model.head(engine.model.norm(hidden[start:start+chunk])).float()
        pieces.append(F.cross_entropy(logits, labels[start:start+chunk], reduction='none'))
    losses = torch.cat(pieces)
    seq = torch.arange(ids.shape[0], device=ids.device)[:, None].expand_as(mask)[mask]
    totals = torch.zeros(ids.shape[0], device=ids.device, dtype=torch.float64)
    totals.index_add_(0, seq, losses.double())
    return totals.cpu().numpy(), mask.sum(-1).cpu().numpy()


class PhysicalVolumeHooks(VolumeRoleHooks):
    def __init__(self, engine, means, global_means=None):
        super().__init__(engine, means)
        self.global_means = (None if global_means is None else
                             torch.from_numpy(np.asarray(global_means, dtype=np.float64).copy()).to('cuda'))
        self.policy = 'proportional'
        # selected, feasible, nontrivial proposal, applied, old V>2, proposed V>2,
        # absolute physical-share drift, absolute applied change of V.
        self.stats = torch.zeros((32, 2, 8), dtype=torch.float64, device='cuda')
        self.p_error_max = torch.zeros((), dtype=torch.float64, device='cuda')
        self.bound_error_max = torch.zeros((), dtype=torch.float64, device='cuda')
        self.check_multiset = False
        self.multiset_error_max = torch.zeros((), dtype=torch.float64, device='cuda')

    def reset_diagnostics(self):
        self.stats.zero_(); self.p_error_max.zero_(); self.bound_error_max.zero_()
        self.multiset_error_max.zero_()

    def configure(self, mask, valid, mode, block_ids=None, seed=None, layers=None):
        if mode in ('mean_global', 'mean_exchange'):
            assert self.global_means is not None
            super().configure(mask, valid, 'mean_both', block_ids=block_ids,
                              seed=seed, layers=layers)
            self.mode = mode
        else:
            super().configure(mask, valid, mode, block_ids=block_ids,
                              seed=seed, layers=layers)

    def transform(self, layer, outputs):
        if self.mode == 'dynamic' or layer not in self.active_layers:
            return outputs
        read, gains, probabilities, mixer = outputs
        old_v = gains.double().sum(-1, keepdim=True)
        if self.mode.startswith('mean'):
            selected = (self.valid & self.mask if self.mode == 'mean_masked' else
                        self.valid & ~self.mask if self.mode == 'mean_visible' else self.valid)
            role = (~self.mask).long()
            if self.mode == 'mean_global':
                proposal = self.global_means[layer].expand_as(old_v)
            elif self.mode == 'mean_exchange':
                proposal = self.means[layer][1-role]
            else:
                proposal = self.means[layer][role]
        else:
            selected = (self.mask if self.mode == 'shuffle_masked' else ~self.mask) & self.valid
            proposal = torch.take_along_dim(old_v, self.maps[layer][..., None], dim=1)

        p = gains.double() / old_v.clamp_min(1e-12)
        if self.policy == 'pairwise_proportional':
            assert self.mode.startswith('shuffle')
            proposal = old_v.clone()
            feasible = torch.zeros_like(selected)
            pmax = p.max(-1).values
            for b in range(selected.shape[0]):
                selected_positions = torch.nonzero(selected[b], as_tuple=True)[0]
                order = self.maps[layer, b, selected_positions]
                pairs = order[:2 * (len(order) // 2)].reshape(-1, 2)
                left, right = pairs[:, 0], pairs[:, 1]
                mutual = torch.isfinite(old_v[b, right, 0]) & torch.isfinite(old_v[b, left, 0]) & (
                    old_v[b, right, 0] * pmax[b, left] <= 2 + 1e-10) & (
                    old_v[b, left, 0] * pmax[b, right] <= 2 + 1e-10) & (
                    old_v[b, left, 0] > 1e-12) & (old_v[b, right, 0] > 1e-12)
                left, right = left[mutual], right[mutual]
                proposal[b, left] = old_v[b, right]
                proposal[b, right] = old_v[b, left]
                feasible[b, left] = True
                feasible[b, right] = True
            if self.check_multiset:
                for b in range(selected.shape[0]):
                    original = old_v[b, selected[b], 0].sort().values
                    proposed = proposal[b, selected[b], 0].sort().values
                    self.multiset_error_max.copy_(torch.maximum(
                        self.multiset_error_max, (original-proposed).abs().max()))
        else:
            feasible = torch.isfinite(proposal[..., 0]) & (proposal[..., 0] > 0) & (
                old_v[..., 0] > 1e-12) & (
                proposal[..., 0] * p.max(-1).values <= 2 + 1e-10)

        applied = selected & feasible
        if self.policy not in ('proportional', 'pairwise_proportional'):
            raise ValueError(self.policy)
        rebuilt = (p * proposal).to(gains.dtype)
        changed = torch.where(applied[..., None], rebuilt, gains)
        new_v = changed.double().sum(-1, keepdim=True)
        p_after = changed.double() / new_v.clamp_min(1e-12)
        drift = (p_after[..., 0] - p[..., 0]).abs()
        self.p_error_max.copy_(torch.maximum(
            self.p_error_max, torch.where(applied, drift, 0).max()))
        self.bound_error_max.copy_(torch.maximum(
            self.bound_error_max,
            torch.maximum((-changed.double()).clamp_min(0).max(),
                          (changed.double() - 2).clamp_min(0).max())))
        delta_v = (proposal[..., 0] - old_v[..., 0]).abs()
        for role, role_mask in enumerate((self.mask, ~self.mask)):
            s = selected & role_mask
            a = applied & role_mask
            self.stats[layer, role] += torch.stack((
                s.sum(), (s & feasible).sum(), (s & (delta_v > 1e-6)).sum(), a.sum(),
                (s & (old_v[..., 0] > 2)).sum(), (s & (proposal[..., 0] > 2)).sum(),
                (drift * a).sum(), (delta_v * a).sum()))
        return read, changed, probabilities, mixer

    def diagnostics(self):
        values = self.stats.cpu().numpy()
        result = {}
        for label, role in (('masked', 0), ('visible', 1), ('all', None)):
            row = values[:, role].sum(0) if role is not None else values.sum((0, 1))
            n = max(row[0], 1); a = max(row[3], 1)
            result[label] = {
                'selected': int(row[0]), 'feasible': int(row[1]),
                'feasible_fraction': float(row[1] / n),
                'nontrivial_proposal_fraction': float(row[2] / n),
                'original_volume_above_2_fraction': float(row[4] / n),
                'proposed_volume_above_2_fraction': float(row[5] / n),
                'mean_abs_physical_share_drift': float(row[6] / a),
                'mean_abs_applied_volume_change': float(row[7] / a),
            }
        result['by_interface_feasible_fraction'] = (
            values[:, :, 1] / np.maximum(values[:, :, 0], 1)).tolist()
        result['max_abs_physical_share_drift'] = float(self.p_error_max.cpu())
        result['max_gain_bound_violation'] = float(self.bound_error_max.cpu())
        result['max_proposed_volume_multiset_error'] = float(self.multiset_error_max.cpu())
        return result


def unit_check(engine, split, means):
    hooks = PhysicalVolumeHooks(engine, means)
    try:
        begin, end, _, _, corrupted, ids, mask = next(batches(split, CONFIG['batch_size']))
        with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
            hooks.configure(mask, torch.ones_like(mask), 'dynamic')
            state, _ = engine.run(corrupted, mask)
            calculated, counts = sequence_ce(engine, state, ids, mask)
            reference = reference_for(split['name'], split['ratio'])
            np.testing.assert_array_equal(counts, reference['seq_mask_count'][begin:end])
            np.testing.assert_allclose(calculated, reference['seq_ce_sum'][begin:end], rtol=2e-5, atol=1e-3)
            intact_error = float(np.max(np.abs(calculated - reference['seq_ce_sum'][begin:end])))
            hooks.policy = 'pairwise_proportional'
            hooks.check_multiset = True
            hooks.configure(mask, torch.ones_like(mask), 'shuffle_masked',
                            block_ids=split['block_ids'][begin:end], seed=SEEDS[0])
            hooks.reset_diagnostics()
            engine.run(corrupted, mask)
            diagnostic = hooks.diagnostics()
            assert diagnostic['max_abs_physical_share_drift'] < 1e-6
            assert diagnostic['max_gain_bound_violation'] == 0
            assert float(hooks.multiset_error_max.cpu()) == 0
        result = {'checkpoint_step': engine.checkpoint_step,
                  'intact_seq_ce_sum_max_abs_error': intact_error,
                  'max_proposed_volume_multiset_error': float(hooks.multiset_error_max.cpu()),
                  'pairwise_diagnostics': diagnostic}
        atomic_json(ROOT / 'unit.json', result)
        print('UNIT OK', intact_error, flush=True)
    finally:
        hooks.close()


def effect_name(mode, policy, ratio, layers=None, seed=None, split_name='confirmation'):
    name = f'{split_name}_r{round(100*ratio):02d}_{mode}_{policy}'
    if seed is not None:
        name += f'_seed{seed}'
    if layers is not None:
        name += '_layers' + '_'.join(f'{index:02d}' for index in layers)
    return name


def evaluate(engine, split, hooks, mode, policy, layers=None, seed=None):
    ratio = split['ratio']
    name = effect_name(mode, policy, ratio, layers, seed, split['name'])
    path = ROOT / 'effects' / (name + '.npz')
    diag_path = ROOT / 'effects' / (name + '.json')
    if path.exists() and diag_path.exists():
        print('CACHED', name, flush=True)
        return
    path.parent.mkdir(exist_ok=True)
    partial = path.with_suffix('.partial.npz')
    hooks.policy = policy
    hooks.reset_diagnostics()
    hooks.check_multiset = policy == 'pairwise_proportional'
    sums, counts, completed = [], [], 0
    if partial.exists():
        with np.load(partial) as saved:
            completed = int(saved['completed'])
            sums = [saved['seq_ce_sum']]; counts = [saved['seq_mask_count']]
            hooks.stats.copy_(torch.from_numpy(saved['stats']).to('cuda'))
            hooks.p_error_max.fill_(float(saved['p_error_max']))
            hooks.bound_error_max.fill_(float(saved['bound_error_max']))
        print('RESUME', name, completed, '/', len(split['ids']), flush=True)
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        for begin, end, _, _, corrupted, ids, mask in batches(split, CONFIG['batch_size']):
            if end <= completed:
                continue
            kwargs = {'block_ids': split['block_ids'][begin:end], 'seed': seed} if seed else {}
            hooks.configure(mask, torch.ones_like(mask), mode, layers=layers, **kwargs)
            state, _ = engine.run(corrupted, mask)
            local_sum, local_count = sequence_ce(engine, state, ids, mask)
            sums.append(local_sum); counts.append(local_count)
            if end % 64 == 0:
                temporary = partial.with_suffix('.tmp.npz')
                np.savez_compressed(temporary, completed=end,
                                    seq_ce_sum=np.concatenate(sums),
                                    seq_mask_count=np.concatenate(counts),
                                    stats=hooks.stats.cpu().numpy(),
                                    p_error_max=float(hooks.p_error_max.cpu()),
                                    bound_error_max=float(hooks.bound_error_max.cpu()))
                temporary.replace(partial)
            if end % 128 == 0 or end == len(split['ids']):
                print('EVAL', name, end, '/', len(split['ids']), flush=True)
    pack = {'seq_ce_sum': np.concatenate(sums),
            'seq_mask_count': np.concatenate(counts),
            'block_ids': split['block_ids'],
            'mask_sha256': np.array(split['mask_sha256'])}
    reference = reference_for(split['name'], ratio)
    np.testing.assert_array_equal(pack['block_ids'], reference['block_ids'])
    np.testing.assert_array_equal(pack['seq_mask_count'], reference['seq_mask_count'])
    diag = hooks.diagnostics()
    assert diag['max_gain_bound_violation'] == 0, diag['max_gain_bound_violation']
    assert diag['max_abs_physical_share_drift'] < 1e-6
    if policy == 'pairwise_proportional':
        assert diag['max_proposed_volume_multiset_error'] == 0
    np.savez_compressed(path, **pack)
    atomic_json(diag_path, diag)
    partial.unlink(missing_ok=True)
    print('DONE', name, 'CE', float(pack['seq_ce_sum'].sum()/pack['seq_mask_count'].sum()),
          'coverage', diag['all']['feasible_fraction'], flush=True)


def group_cases(group):
    if group == 'remove_exchange':
        return [(mode, 'proportional', None, None, .5, 'confirmation')
                for mode in ('mean_global', 'mean_exchange')]
    if group == 'means':
        return [('mean_'+role, 'proportional', None, None, .5, 'confirmation')
                for role in ('masked','visible','both')]
    if group in ('shuffle_masked','shuffle_visible'):
        return [(group, 'pairwise_proportional', None, seed, .5, 'confirmation')
                for seed in SEEDS]
    if group == 'depth':
        return [('mean_'+role, 'proportional', tuple(2*(block-1)+offset for offset in (0,1)),
                 None, .5, 'confirmation')
                for block, role in ((1,'visible'),(1,'masked'),(16,'visible'),(16,'masked'),
                                    (13,'visible'),(10,'masked'))]
    if group == 'depth_full':
        return [('mean_'+role, 'proportional', (2*block, 2*block+1),
                 None, .5, 'confirmation')
                for block in range(16) for role in ('masked', 'visible')]
    if group == 'depth_late':
        return [('mean_'+role, 'proportional', (2*block, 2*block+1),
                 None, .5, 'confirmation')
                for block in range(8, 16) for role in ('masked', 'visible')]
    if group == 'depth_middle':
        return [('mean_'+role, 'proportional', (2*block, 2*block+1),
                 None, .5, 'confirmation')
                for block in range(5, 8) for role in ('masked', 'visible')]
    if group == 'noise':
        return [('mean_both', 'proportional', None, None, ratio, 'discovery')
                for ratio in (.1,.3,.5,.7,.9)]
    if group == 'noise_roles':
        return [('mean_'+role, 'proportional', None, None, ratio, 'discovery')
                for ratio in (.1,.3,.5,.7,.9) for role in ('masked','visible','both')]
    raise ValueError(group)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=('means','shuffle_masked','shuffle_visible',
                                            'depth','depth_full','depth_middle',
                                            'depth_late','noise',
                                            'noise_roles','remove_exchange','smoke'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    engine = load_model(CONFIG, 'full_n2')
    assert engine.checkpoint_step == 53399 and len(engine.interfaces) == 32
    if args.group == 'smoke':
        split = make_split(CONFIG, 'confirmation', .5)
        unit_check(engine, split, means_for_ratio(.5))
        return
    active = None
    try:
        for mode, policy, layers, seed, ratio, split_name in group_cases(args.group):
            split = make_split(CONFIG, split_name, ratio)
            means = means_for_ratio(ratio)
            if active is None:
                active = PhysicalVolumeHooks(engine, means, global_means_for_ratio(.5))
            else:
                active.means = torch.from_numpy(means.copy()).to('cuda')
            evaluate(engine, split, active, mode, policy, layers, seed)
    finally:
        if active is not None:
            active.close()


if __name__ == '__main__':
    main()
