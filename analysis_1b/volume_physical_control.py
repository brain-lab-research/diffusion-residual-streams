"""Physical-share-preserving write-volume controls for the 1B checkpoint.

Run with CUDA_VISIBLE_DEVICES=<free GPU> python volume_physical_control.py.
Every condition uses the frozen step-124000 model, the existing disjoint
calibration means and the same 128 held-out test blocks.
"""
import json

import numpy as np
import torch

from audit import CONFIG, ROOT, batches, load_model, make_split, paired_delta, save_json, sequence_ce
from volume_calibration import VolumeHooks64, mean_volume_for_ratio


class PhysicalShareHooks(VolumeHooks64):
    def __init__(self, engine, role_volume_means, global_volume_means=None):
        super().__init__(engine, role_volume_means)
        self.global_means = (None if global_volume_means is None else
                             torch.from_numpy(np.asarray(global_volume_means, dtype=np.float64).copy()).to('cuda'))
        self.policy = 'proportional'
        # selected, feasible, changed proposal, applied, V>2, target>2,
        # absolute physical-share drift, absolute proposed volume change
        self.stats = torch.zeros((64, 2, 8), device='cuda', dtype=torch.float64)
        self.p_error_max = torch.zeros((), device='cuda', dtype=torch.float64)
        self.bound_error_max = torch.zeros((), device='cuda', dtype=torch.float64)
        self.multiset_error_max = torch.zeros((), device='cuda', dtype=torch.float64)

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
        old_volume = gains.double().sum(-1, keepdim=True)
        if self.mode.startswith('mean'):
            if self.mode == 'mean_masked':
                selected = self.mask & self.valid
            elif self.mode == 'mean_visible':
                selected = ~self.mask & self.valid
            else:
                selected = self.valid
            role = (~self.mask).long()
            if self.mode == 'mean_global':
                proposal = self.global_means[layer].expand_as(old_volume)
            elif self.mode == 'mean_exchange':
                proposal = self.means[layer][1-role]
            else:
                proposal = self.means[layer][role]
        else:
            selected = (self.mask if self.mode == 'shuffle_masked' else ~self.mask) & self.valid
            proposal = torch.take_along_dim(old_volume, self.maps[layer][..., None], dim=1)

        p = gains.double() / old_volume.clamp_min(1e-12)
        if self.policy == 'pairwise_proportional':
            assert self.mode.startswith('shuffle')
            # Disjoint, mutually feasible swaps preserve the exact volume
            # multiset within each sequence, interface, and token role.
            proposal = old_volume.clone()
            feasible = torch.zeros_like(selected)
            pmax = p.max(-1).values
            for b in range(selected.shape[0]):
                positions = torch.nonzero(selected[b], as_tuple=True)[0]
                order = self.maps[layer, b, positions]
                pairs = order[:2*(len(order)//2)].reshape(-1, 2)
                left, right = pairs[:, 0], pairs[:, 1]
                mutual = torch.isfinite(old_volume[b, right, 0]) & torch.isfinite(old_volume[b, left, 0]) & (
                    old_volume[b, right, 0] * pmax[b, left] <= 2.0 + 1e-10) & (
                    old_volume[b, left, 0] * pmax[b, right] <= 2.0 + 1e-10) & (
                    old_volume[b, left, 0] > 1e-12) & (old_volume[b, right, 0] > 1e-12)
                left, right = left[mutual], right[mutual]
                proposal[b, left] = old_volume[b, right]
                proposal[b, right] = old_volume[b, left]
                feasible[b, left] = True
                feasible[b, right] = True
                original = old_volume[b, selected[b], 0].sort().values
                exchanged = proposal[b, selected[b], 0].sort().values
                self.multiset_error_max.copy_(torch.maximum(
                    self.multiset_error_max, (original - exchanged).abs().max()))
        else:
            feasible = torch.isfinite(proposal[..., 0]) & (proposal[..., 0] > 0) & (
                old_volume[..., 0] > 1e-12) & (
                proposal[..., 0] * p.max(-1).values <= 2.0 + 1e-10)
        applied = selected & feasible
        if self.policy not in ('proportional', 'pairwise_proportional'):
            raise ValueError(self.policy)
        rebuilt = (p * proposal).to(gains.dtype)
        changed_gains = torch.where(applied[..., None], rebuilt, gains)
        new_volume = changed_gains.double().sum(-1, keepdim=True)
        p_new = changed_gains.double() / new_volume.clamp_min(1e-12)
        p_drift = (p_new[..., 0] - p[..., 0]).abs()
        self.p_error_max.copy_(torch.maximum(self.p_error_max, p_drift[applied].max()
                              if applied.any() else self.p_error_max))
        self.bound_error_max.copy_(torch.maximum(
            self.bound_error_max,
            torch.maximum((-changed_gains).clamp_min(0).max(),
                          (changed_gains.double()-2).clamp_min(0).max())))
        delta_v = (proposal[..., 0] - old_volume[..., 0]).abs()
        for role, role_mask in enumerate((self.mask, ~self.mask)):
            s = selected & role_mask
            a = applied & role_mask
            row = torch.stack((s.sum(), (s & feasible).sum(),
                               (s & (delta_v > 1e-6)).sum(), a.sum(),
                               (s & (old_volume[..., 0] > 2)).sum(),
                               (s & (proposal[..., 0] > 2)).sum(),
                               p_drift[a].sum(), delta_v[a].sum()))
            self.stats[layer, role] += row
        return read, changed_gains, probabilities, mixer

    def diagnostics(self):
        s = self.stats.cpu().numpy()
        output = {}
        for role, i in [('masked', 0), ('visible', 1), ('all', None)]:
            row = s[:, i, :].sum(0) if i is not None else s.sum((0, 1))
            n = max(row[0], 1)
            a = max(row[3], 1)
            output[role] = dict(selected=int(row[0]), feasible=int(row[1]),
                                feasible_fraction=float(row[1]/n),
                                changed_proposal_fraction=float(row[2]/n),
                                original_volume_above_2_fraction=float(row[4]/n),
                                proposed_volume_above_2_fraction=float(row[5]/n),
                                mean_abs_physical_share_drift=float(row[6]/a),
                                mean_abs_proposed_volume_change=float(row[7]/a))
        output['max_abs_physical_share_drift'] = float(self.p_error_max.cpu())
        output['max_gain_bound_violation'] = float(self.bound_error_max.cpu())
        output['max_proposed_volume_multiset_error'] = float(self.multiset_error_max.cpu())
        output['by_interface_feasible_fraction'] = (
            s[:, :, 1] / np.maximum(s[:, :, 0], 1)).tolist()
        return output


def evaluate(engine, split, hooks, mode, policy, seed=None, ratio=.5, layers=None):
    label = f'{policy}_{mode}' + (f'_seed{seed}' if seed is not None else '')
    if layers is not None:
        label += '_layers' + '_'.join(f'{layer:02d}' for layer in layers)
    tag = f'r{round(100*ratio):02d}'
    path = ROOT / f'volume_physical_{label}_{tag}.npz'
    diagnostic_path = ROOT / f'volume_physical_{label}_{tag}.json'
    if path.exists() and diagnostic_path.exists():
        with np.load(path) as saved:
            return {k: saved[k] for k in saved.files}, json.loads(diagnostic_path.read_text())
    hooks.policy = policy
    hooks.stats.zero_(); hooks.p_error_max.zero_(); hooks.bound_error_max.zero_()
    hooks.multiset_error_max.zero_()
    sums, counts = [], []
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        for begin, end, _, _, corrupted, ids, mask in batches(split, CONFIG['batch_size']):
            kwargs = {'block_ids': split['block_ids'][begin:end], 'seed': seed} if seed else {}
            hooks.configure(mask, torch.ones_like(mask), mode, layers=layers, **kwargs)
            state, _ = engine.run(corrupted, mask)
            local_sum, local_count = sequence_ce(engine, state, ids, mask)
            sums.append(local_sum); counts.append(local_count)
            if end % 32 == 0:
                print('PHYSICAL', label, end, '/', len(split['ids']), flush=True)
    pack = dict(seq_ce_sum=np.concatenate(sums), seq_mask_count=np.concatenate(counts),
                block_ids=split['block_ids'], mask_sha256=np.array(split['mask_sha256']))
    diag = hooks.diagnostics()
    assert diag['max_gain_bound_violation'] < 1e-9, diag['max_gain_bound_violation']
    assert diag['max_abs_physical_share_drift'] < 1e-6
    if policy == 'pairwise_proportional':
        assert diag['max_proposed_volume_multiset_error'] == 0
    np.savez_compressed(path, **pack)
    save_json(diagnostic_path, diag)
    print('DONE', label, float(pack['seq_ce_sum'].sum()/pack['seq_mask_count'].sum()),
          'coverage', diag['all']['feasible_fraction'], flush=True)
    return pack, diag


def main():
    torch.set_num_threads(4)
    engine = load_model(CONFIG, 'full_n2')
    assert engine.checkpoint_step == 124000 and len(engine.interfaces) == 64
    means = mean_volume_for_ratio(engine, .5)
    split = make_split(CONFIG, 'test', .5)
    with np.load(ROOT/'intact.npz') as saved:
        intact = {k: saved[k] for k in saved.files}
    hooks = PhysicalShareHooks(engine, means)
    summary = {'checkpoint_step': 124000, 'mask_sha256': split['mask_sha256'],
               'sequences': len(split['ids']), 'definition':
               'Reconstruct g_i as target V times current g_i/V only when both gains are feasible. '
               'Infeasible positions keep their original gains.', 'conditions': {}}
    try:
        cases = [('mean_masked', None), ('mean_visible', None), ('mean_both', None)]
        for mode, seed in cases:
            name = mode + (f'_seed{seed}' if seed is not None else '')
            pack, diag = evaluate(engine, split, hooks, mode, 'proportional', seed)
            summary['conditions'][name] = {
                'effect_vs_intact': paired_delta(pack, intact), 'diagnostics': diag}
            save_json(ROOT/'volume_physical_control.json', summary)
    finally:
        hooks.close()
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
