"""Replay the arxiv physical-destination interventions with live acceptance counts.

CUDA_VISIBLE_DEVICES=1 python volume_analysis/physical_destination_joint.py --model 337m
CUDA_VISIBLE_DEVICES=2 python volume_analysis/physical_destination_joint.py --model 1b

Each condition uses the original checkpoint, mask, BF16 forward, and disjoint
calibration capture. Destination is E_cal[g / sum(g)], never E[g] / E[sum(g)].
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'physical_destination_joint'
sys.path.insert(0, str(BASE))

from volume_analysis.physical_protocol import cases, settings


def sequence_ce(engine, state, ids, mask, chunk=128):
    hidden = state.mean(-2)[mask]
    labels = ids[mask]
    parts = []
    for begin in range(0, len(labels), chunk):
        logits = engine.model.head(engine.model.norm(hidden[begin:begin + chunk])).float()
        parts.append(F.cross_entropy(logits, labels[begin:begin + chunk], reduction='none'))
    losses = torch.cat(parts)
    seq = torch.arange(ids.shape[0], device=ids.device)[:, None].expand_as(mask)[mask]
    totals = torch.zeros(ids.shape[0], device=ids.device, dtype=torch.float64)
    totals.index_add_(0, seq, losses.double())
    return totals.cpu().numpy(), mask.sum(-1).cpu().numpy()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def write_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def physical_means(model, cal_json_path, cal_split, n):
    capture_path = (BASE / 'volume_analysis/calibration_all_positions.npz' if model == '337m'
                    else BASE / 'analysis_1b/calibration_routes.npz')
    with np.load(capture_path) as saved:
        np.testing.assert_array_equal(saved['block_ids'], cal_split['block_ids'])
        np.testing.assert_array_equal(saved['mask'], cal_split['mask'])
        valid = saved['valid'].ravel() if 'valid' in saved else np.ones(cal_split['mask'].size, bool)
        router = saved['router'].astype(np.float64)
    assert router.shape == (n, cal_split['mask'].size, 6)
    assert valid.all(), 'Original packed blocks must have no padding'
    gains = router[..., 2:4]
    volume = gains.sum(-1, keepdims=True)
    assert np.all(volume > 0) and np.all(np.isfinite(volume))
    destination = gains / volume
    roles = (cal_split['mask'].ravel(), ~cal_split['mask'].ravel())
    old = json.loads(cal_json_path.read_text())['means']
    means = {policy: {factor: np.asarray(value, dtype=np.float64)
                      for factor, value in factors.items()}
             for policy, factors in old.items()}
    means['global']['destination'] = destination[:, valid].mean(1, dtype=np.float64)
    means['role']['destination'] = np.stack(
        [destination[:, valid & role].mean(1, dtype=np.float64) for role in roles], 1)
    for policy in means:
        means[policy]['destination'] /= means[policy]['destination'].sum(-1, keepdims=True)
    # The other three controls must agree with the archived calibration.
    for factor, values in [('read', router[..., :2]), ('volume', volume),
                           ('mixer', router[..., 4:])]:
        np.testing.assert_allclose(means['global'][factor], values[:, valid].mean(1), atol=1e-7, rtol=0)
        for role in (0, 1):
            np.testing.assert_allclose(means['role'][factor][:, role],
                                       values[:, valid & roles[role]].mean(1), atol=1e-7, rtol=0)
    record = {'model': model, 'capture': str(capture_path), 'calibration_mask_sha256': cal_split['mask_sha256'],
              'calibration_data_sha256': cal_split['data_sha256'],
              'calibration_positions': int(valid.sum()), 'calibration_role_positions': [int(x.sum()) for x in roles],
              'definition': 'float64 token-weighted E_cal[g_i / (g_0+g_1)] per interface and role',
              'means': {policy: {factor: values.tolist() for factor, values in factors.items()}
                        for policy, factors in means.items()}}
    write_json(OUT / f'{model}_calibration.json', record)
    return means


def reference(path, split):
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved['block_ids'], split['block_ids'])
        np.testing.assert_array_equal(saved['seq_mask_count'], split['mask'].sum(1))
        if 'positions' in saved:
            np.testing.assert_array_equal(saved['positions'], np.where(split['mask'])[1])
        if 'mask_sha256' in saved:
            assert str(saved['mask_sha256']) == split['mask_sha256']
        return saved['seq_ce_sum'].copy(), saved['seq_mask_count'].copy()


def coverage(counts):
    assert np.all(counts[:, :, 0] > 0)
    assert np.all(counts[:, :, 1] <= counts[:, :, 0])
    per_role = 100 * counts[:, :, 1] / counts[:, :, 0]
    per_all = 100 * counts[:, :, 1].sum(1) / counts[:, :, 0].sum(1)
    return {'masked': float(per_role[:, 0].mean()), 'visible': float(per_role[:, 1].mean()),
            'total': float(per_all.mean())}, per_role, per_all


def paired_interval(sums, intact_sums, counts, seed=130914, repeats=5000):
    rng = np.random.default_rng(seed)
    diff = sums - intact_sums
    samples = rng.integers(len(diff), size=(repeats, len(diff)))
    draws = diff[samples].sum(1) / counts[samples].sum(1)
    return np.quantile(draws, [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('337m', '1b'), required=True)
    args = parser.parse_args()
    config, cal_path, ref_dir, split_name, step, n, batches, load_model, make_split = settings(args.model)
    from mechanism_audit_n2.role_interventions import RoleFactorHooks

    cal_split = make_split(config, 'calibration', .5)
    means = physical_means(args.model, cal_path, cal_split, n)
    split = make_split(config, split_name, .5)
    intact_path = (ref_dir / 'confirmation_r50.npz' if args.model == '337m'
                   else ref_dir / 'intact.npz')
    intact_sums, mask_counts = reference(intact_path, split)

    class PhysicalHooks(RoleFactorHooks):
        def __init__(self, engine):
            super().__init__(engine, means)
            self.counts = torch.zeros((n, 2, 2), dtype=torch.int64, device='cuda')
            self.max_applied_gain_error = torch.zeros((), dtype=torch.float64, device='cuda')
            self.max_applied_volume_error = torch.zeros((), dtype=torch.float64, device='cuda')

        def transform(self, layer, module, state, outputs):
            read, gains, probabilities, mixer = outputs
            selected = self.valid
            current = gains.double()
            current_volume = current.sum(-1, keepdim=True)
            assert bool((current_volume[selected] > 0).all())
            if 'read' in self.fixed:
                read = torch.where(selected[..., None], self.replacement('read', layer).to(read.dtype), read)

            if any(factor in self.fixed for factor in ('volume', 'destination')):
                proposed_volume = (self.replacement('volume', layer) if 'volume' in self.fixed
                                   else current_volume)
                proposed_q = (self.replacement('destination', layer) if 'destination' in self.fixed
                              else current / current_volume)
                proposal = proposed_volume * proposed_q
                accepted = (torch.isfinite(proposal).all(-1) & (proposal >= 0).all(-1)
                            & (proposal <= 2).all(-1))
                apply = selected & accepted
                gains = torch.where(apply[..., None], proposal.to(gains.dtype), gains)
            else:
                proposed_volume = current_volume
                proposal = current
                accepted = torch.ones_like(selected)
                apply = selected
            for role_id, role_mask in enumerate((self.mask, ~self.mask)):
                chosen = selected & role_mask
                self.counts[layer, role_id, 0] += chosen.sum()
                self.counts[layer, role_id, 1] += (chosen & accepted).sum()
            applied_error = (gains.double() - proposal).abs().amax(-1)
            volume_error = (gains.double().sum(-1, keepdim=True) - proposed_volume).abs().squeeze(-1)
            self.max_applied_gain_error.copy_(torch.maximum(self.max_applied_gain_error,
                torch.where(apply, applied_error, 0).max()))
            self.max_applied_volume_error.copy_(torch.maximum(self.max_applied_volume_error,
                torch.where(apply, volume_error, 0).max()))

            if 'mixer' in self.fixed:
                probabilities = torch.where(selected[..., None],
                                            self.replacement('mixer', layer).to(probabilities.dtype),
                                            probabilities)
                new_mixer = torch.einsum('btp,pij->btij', probabilities,
                                        module.permutations.to(probabilities.dtype)).to(state.dtype)
                mixer = torch.where(selected[..., None, None], new_mixer, mixer)
            return read, gains, probabilities, mixer

    torch.set_num_threads(4)
    engine = load_model(config, 'full_n2')
    assert engine.checkpoint_step == step and len(engine.interfaces) == n
    hooks = PhysicalHooks(engine)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        for label, fixed, policy in cases(args.model):
            destination = OUT / f'{args.model}_{label}.npz'
            report_path = OUT / f'{args.model}_{label}.json'
            if destination.exists() and report_path.exists():
                row = json.loads(report_path.read_text())
                assert row['code_sha256'] == source_sha and row['mask_sha256'] == split['mask_sha256']
                print('CACHED', args.model, label, flush=True)
                continue
            hooks.counts.zero_(); hooks.max_applied_gain_error.zero_(); hooks.max_applied_volume_error.zero_()
            partial = OUT / f'{args.model}_{label}.partial.npz'
            all_sums = np.empty(len(split['ids']), dtype=np.float64)
            start = 0
            if partial.exists():
                with np.load(partial) as saved:
                    assert str(saved['code_sha256']) == source_sha
                    assert str(saved['mask_sha256']) == split['mask_sha256']
                    start = int(saved['end'])
                    all_sums[:start] = saved['seq_ce_sum']
                    hooks.counts.copy_(torch.from_numpy(saved['counts']).to('cuda'))
                    hooks.max_applied_gain_error.copy_(torch.tensor(float(saved['gain_error']), device='cuda'))
                    hooks.max_applied_volume_error.copy_(torch.tensor(float(saved['volume_error']), device='cuda'))
                print('RESUME', args.model, label, start, flush=True)
            with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
                for begin, end, _, _, corrupted, ids, mask in batches(split, config['batch_size']):
                    if end <= start:
                        continue
                    assert begin == start or begin > start
                    hooks.configure(mask, fixed, valid=torch.ones_like(mask), policy=policy, layers=range(n))
                    state, _ = engine.run(corrupted, mask)
                    sums, counts = sequence_ce(engine, state, ids, mask)
                    np.testing.assert_array_equal(counts, mask_counts[begin:end])
                    all_sums[begin:end] = sums
                    if end % (128 if args.model == '337m' else 16) == 0 or end == len(split['ids']):
                        write_npz(partial, end=np.array(end), seq_ce_sum=all_sums[:end],
                                  counts=hooks.counts.cpu().numpy(),
                                  gain_error=np.array(float(hooks.max_applied_gain_error)),
                                  volume_error=np.array(float(hooks.max_applied_volume_error)),
                                  code_sha256=np.array(source_sha), mask_sha256=np.array(split['mask_sha256']))
                        print('EVAL', args.model, label, end, '/', len(split['ids']), flush=True)
            raw = hooks.counts.cpu().numpy()
            expected_selected = np.stack([split['mask'].sum(), (~split['mask']).sum()])
            np.testing.assert_array_equal(raw[:, :, 0], np.broadcast_to(expected_selected, (n, 2)))
            changes_gains = any(factor in fixed for factor in ('volume', 'destination'))
            summary, per_role, per_all = coverage(raw)
            new_ce = float(all_sums.sum() / mask_counts.sum())
            intact_ce = float(intact_sums.sum() / mask_counts.sum())
            row = {'model': args.model, 'condition': label, 'checkpoint_step': step,
                   'sequences': len(split['ids']), 'interfaces': n, 'precision': 'BF16 autocast',
                   'mask_sha256': split['mask_sha256'], 'data_sha256': split['data_sha256'],
                   'calibration_mask_sha256': cal_split['mask_sha256'],
                   'code_sha256': source_sha, 'fixed_factors': list(fixed), 'policy': policy,
                   'destination_definition': 'q_i = g_i / (g_0 + g_1)',
                   'acceptance': ('finite proposed gains, both in [0,2]; rejected pair retains live gains'
                                  if changes_gains else 'write gains unchanged'),
                   'coverage_percent': summary if changes_gains else None,
                   'counts_columns': ['selected_count', 'accepted_count'],
                   'counts_role_order': ['masked', 'visible'],
                   'by_interface_counts': raw.tolist(),
                   'by_interface_coverage_percent': per_role.tolist() if changes_gains else None,
                   'by_interface_total_coverage_percent': per_all.tolist() if changes_gains else None,
                   'ce': new_ce, 'intact_ce': intact_ce, 'delta_ce': new_ce-intact_ce,
                   'delta_ce_ci95': paired_interval(all_sums, intact_sums, mask_counts),
                   'max_applied_gain_rounding_error': float(hooks.max_applied_gain_error),
                   'max_applied_volume_rounding_error': float(hooks.max_applied_volume_error)}
            write_npz(destination, seq_ce_sum=all_sums, seq_mask_count=mask_counts,
                      block_ids=split['block_ids'], mask_sha256=np.array(split['mask_sha256']),
                      selected_count=raw[:, :, 0], accepted_count=raw[:, :, 1])
            write_json(report_path, row)
            partial.unlink(missing_ok=True)
            print('DONE', args.model, label, 'delta', row['delta_ce'], 'coverage', summary, flush=True)
    finally:
        hooks.close()


if __name__ == '__main__':
    main()
