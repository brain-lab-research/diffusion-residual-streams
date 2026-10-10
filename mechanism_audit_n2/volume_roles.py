"""Role-specific write-volume interventions for the final mHC-lite n=2 model.

The primary comparison is at corruption ratio 0.5 on the sealed 1024-sequence
confirmation split.  Existing intact, baseline, masked-mean, and both-role-mean
packs are reused only after exact split/protocol checks.  New inference covers
visible-role mean replacement and within-sequence, within-role volume shuffles.

python -m mechanism_audit_n2.volume_roles --gpu 0
python -m mechanism_audit_n2.volume_roles --stage render
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
from html import escape
import json
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .engine import aggregate, batches, load_model, make_split, summarize_tokens
from .factors import ROOT, gpu_factors, reconstruct
from .primary import load_pack, save_pack, write_json
from .role_interventions import calibration_means


OUTPUT = ROOT / 'volume_role_assignment_n2'
EFFECTS = OUTPUT / 'effects'
RATIO = .5
PERMUTATION_SEEDS = (130921, 130922, 130923)
REFERENCE = {
    'dynamic_dynamic': ROOT / 'router_factor_confirmation/effects/r50/intact.npz',
    'baseline': ROOT / 'router_factor_confirmation/effects/r50/baseline.npz',
    'mean_dynamic': ROOT / 'router_factor_confirmation/effects/r50/mean_volume.npz',
    'mean_mean': ROOT / 'role_ladder_n2/effects/volume_role.npz',
}


class SmallContextReservation:
    """Allow a changing low-memory context while rejecting actual GPU work."""
    def __init__(self, gpu, policy, maximum_external_mib):
        from .safety import query
        self.gpu, self.policy = str(gpu), policy
        self.maximum_external_mib = float(maximum_external_mib)
        if self.maximum_external_mib <= 0:
            raise ValueError('Explicit positive co-occupant memory bound required')
        maximum_external_mib = self.maximum_external_mib
        self.lock = Path('/tmp/mhc_mechanism_gpu' + self.gpu + '.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.observations = []; self.observed_external_max_mib = {}; self.last_checked = 0
        for index in range(1):
            rows = query(['--query-gpu=index,uuid,utilization.gpu,memory.used', '--format=csv,nounits'])
            row = next(item for item in rows if item['index'].strip() == self.gpu)
            self.uuid = row['uuid'].strip()
            external = self._external(query)
            if (sum(external.values()) > maximum_external_mib or
                    float(row['memory.used [MiB]']) > maximum_external_mib + 64):
                raise RuntimeError('Requested GPU exceeds the allowed co-occupant memory limit')
            self._record(external); self.observations.append({**row, 'allowed_external_processes': external})
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
        os.environ.setdefault('OMP_NUM_THREADS', '4')

    def _external(self, query):
        applications = query(['--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,nounits'])
        return {int(item['pid']): float(item['used_gpu_memory [MiB]']) for item in applications
                if item['gpu_uuid'].strip() == self.uuid and int(item['pid']) != os.getpid()}

    def _record(self, external):
        for pid, memory in external.items():
            self.observed_external_max_mib[pid] = max(memory, self.observed_external_max_mib.get(pid, 0))

    def check(self, force=False):
        from .safety import query
        if not force and time.monotonic() - self.last_checked < 5: return
        external = self._external(query); self._record(external)
        if sum(external.values()) > self.maximum_external_mib:
            raise RuntimeError('External GPU activity exceeded the allowed small-context limit')
        self.last_checked = time.monotonic()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def effect_path(name):
    EFFECTS.mkdir(parents=True, exist_ok=True)
    return EFFECTS / (name + '.npz')


def assert_aligned(left, right):
    for key in ('block_ids', 'seq_mask_count', 'positions', 'token_sequence'):
        np.testing.assert_array_equal(left[key], right[key])


def prepare(config):
    OUTPUT.mkdir(exist_ok=True); EFFECTS.mkdir(exist_ok=True)
    calibration_split = make_split(config, 'calibration', RATIO)
    evaluation_split = make_split(config, 'confirmation', RATIO)
    overlap = sorted(set(map(int, calibration_split['block_ids'])) &
                     set(map(int, evaluation_split['block_ids'])))
    assert not overlap
    captured = load_pack(ROOT / 'role_ladder_n2/calibration_all_positions.npz')
    np.testing.assert_array_equal(captured['block_ids'], calibration_split['block_ids'])
    np.testing.assert_array_equal(captured['mask'], calibration_split['mask'])
    assert captured['valid'].all()
    means = calibration_means(captured['router'].astype(np.float64),
                              captured['mask'].ravel(), captured['valid'].ravel())
    saved_means = json.loads((ROOT / 'role_ladder_n2/calibration_means.json').read_text())
    for policy in ('global', 'role'):
        for factor, value in means[policy].items():
            np.testing.assert_allclose(value, np.asarray(saved_means['means'][policy][factor]),
                                       rtol=0, atol=2e-14)
    old_masked = np.asarray(json.loads((ROOT / 'router_factor_confirmation/calibration/r50_bounded_means.json').read_text())['means']['volume'])
    masked_mean_error = float(np.max(np.abs(old_masked - means['role']['volume'][:, 0])))
    assert masked_mean_error < 2e-14

    references = {name: load_pack(path) for name, path in REFERENCE.items()}
    for pack in references.values():
        assert_aligned(pack, references['dynamic_dynamic'])
        np.testing.assert_array_equal(pack['block_ids'], evaluation_split['block_ids'])
        np.testing.assert_array_equal(pack['seq_mask_count'], evaluation_split['mask'].sum(1))
    role_dynamic = load_pack(ROOT / 'role_ladder_n2/effects/dynamic.npz')
    assert_aligned(role_dynamic, references['dynamic_dynamic'])
    np.testing.assert_array_equal(role_dynamic['token_stats'], references['dynamic_dynamic']['token_stats'])
    for name, pack in references.items():
        save_pack(effect_path(name), pack)

    source_names = ['volume_roles.py', 'engine.py', 'factors.py', 'primary.py',
                    'role_interventions.py']
    protocol = {
        'checkpoint': config['models']['full_n2'], 'baseline': config['models']['baseline'],
        'ratio': RATIO, 'evaluation_sequences': 1024, 'calibration_sequences': 64,
        'calibration_blocks': list(map(int, calibration_split['block_ids'])),
        'evaluation_blocks': list(map(int, evaluation_split['block_ids'])),
        'calibration_evaluation_overlap': overlap,
        'mask_seed': config['mask_seed'], 'bootstrap_seed': config['bootstrap_seed'],
        'bootstrap_repeats': config['bootstrap_repeats'],
        'permutation_seeds': list(PERMUTATION_SEEDS),
        'role_definition': 'Observed corruption mask only: MASKED=mask, VISIBLE=valid & ~mask.',
        'target': 'Masked-token CE only; visible interventions act indirectly through context.',
        'means': 'Per-interface, per-role token-weighted float64 means on the separate calibration split.',
        'volume_coordinate': 'V=g0+g1; q_i=(g_i-L)/(U-L), L=max(0,V-2), U=min(2,V).',
        'controls': 'Read, bounded write destination q, and residual mixer remain current-router dynamic.',
        'permutation': ('Separately within each sequence, selected role, interface (attention/MLP included), '
                        'and seed; preserves the exact multiset of V and changes only position assignment.'),
        'padding': 'Dense 1024-token packed blocks have no padding; valid mask is all true, as in the source pipeline.',
        'conditions': ['dynamic_dynamic', 'mean_dynamic', 'dynamic_mean', 'mean_mean',
                       'shuffle_masked_seed*', 'shuffle_visible_seed*'],
        'source_sha256': {name: sha256(Path(__file__).with_name(name)) for name in source_names},
        'frozen_source_sha256': {path: sha256(path) for path in config['source_files']},
        'reference_sha256': {name: sha256(path) for name, path in REFERENCE.items()},
        'masked_calibration_mean_max_abs_error_vs_prior': masked_mean_error,
    }
    config_path = OUTPUT / 'config.json'
    if not config_path.exists() or json.loads(config_path.read_text()) != protocol:
        write_json(config_path, protocol)
    return evaluation_split, means, references, protocol


class VolumeRoleHooks:
    def __init__(self, engine, role_volume_means):
        self.engine = engine
        self.means = torch.from_numpy(np.asarray(role_volume_means, dtype=np.float64).copy()).to('cuda')
        self.mode = 'dynamic'; self.mask = None; self.valid = None
        self.active_layers = set(range(32))
        self.maps = None; self.handles = []
        self.maxima = {}
        for index, interface in enumerate(engine.interfaces):
            def hook(module, args, outputs, layer=index):
                return self.transform(layer, outputs)
            self.handles.append(interface.router.register_forward_hook(hook))

    def reset_maxima(self):
        self.maxima = {key: 0.0 for key in [
            'read_error', 'mixer_probability_error', 'mixer_matrix_error',
            'unselected_volume_error', 'destination_error', 'assigned_volume_error',
            'shuffle_multiset_error', 'gain_bound_error']}

    def configure(self, mask, valid, mode, block_ids=None, seed=None, layers=None):
        assert mode in ('dynamic', 'mean_masked', 'mean_visible', 'mean_both',
                        'shuffle_masked', 'shuffle_visible')
        self.mask, self.valid, self.mode = mask, valid, mode
        self.active_layers = set(range(32)) if layers is None else set(layers)
        self.maps = None
        if mode.startswith('shuffle'):
            assert block_ids is not None and seed is not None
            role_mask = (mask if mode == 'shuffle_masked' else ~mask) & valid
            role_np = role_mask.cpu().numpy()
            maps = np.broadcast_to(np.arange(mask.shape[1], dtype=np.int64),
                                   (32, mask.shape[0], mask.shape[1])).copy()
            role_code = 0 if mode == 'shuffle_masked' else 1
            for layer in range(32):
                for batch_index, block_id in enumerate(block_ids):
                    positions = np.flatnonzero(role_np[batch_index])
                    rng = np.random.default_rng(np.random.SeedSequence(
                        [int(seed), int(block_id), layer, role_code]))
                    maps[layer, batch_index, positions] = rng.permutation(positions)
            self.maps = torch.from_numpy(maps).to('cuda')

    def transform(self, layer, outputs):
        read, gains, probabilities, mixer = outputs
        if self.mode == 'dynamic' or layer not in self.active_layers:
            return outputs
        old_volume, destination = gpu_factors(gains, 'bounded')
        changed_volume = old_volume.clone()
        if self.mode.startswith('mean'):
            if self.mode == 'mean_masked':
                selected = self.valid & self.mask
            elif self.mode == 'mean_visible':
                selected = self.valid & ~self.mask
            else:
                selected = self.valid
            role = (~self.mask).long()
            replacement = self.means[layer][role]
            changed_volume = torch.where(selected[..., None], replacement, old_volume)
        else:
            selected = ((self.mask if self.mode == 'shuffle_masked' else ~self.mask) & self.valid)
            changed_volume = torch.take_along_dim(old_volume, self.maps[layer][..., None], dim=1)
            changed_volume = torch.where(selected[..., None], changed_volume, old_volume)
            for b in range(old_volume.shape[0]):
                local = selected[b]
                if local.any():
                    error = float((old_volume[b, local, 0].sort().values -
                                   changed_volume[b, local, 0].sort().values).abs().max())
                    self.maxima['shuffle_multiset_error'] = max(self.maxima['shuffle_multiset_error'], error)
        rebuilt = reconstruct(changed_volume, destination, 'bounded', gains.dtype)
        changed_gains = torch.where(selected[..., None], rebuilt, gains)
        actual_volume, actual_destination = gpu_factors(changed_gains, 'bounded')
        self.maxima['unselected_volume_error'] = max(self.maxima['unselected_volume_error'],
            float((actual_volume[~selected] - old_volume[~selected]).abs().max()) if (~selected).any() else 0.)
        self.maxima['assigned_volume_error'] = max(self.maxima['assigned_volume_error'],
            float((actual_volume[selected] - changed_volume[selected]).abs().max()) if selected.any() else 0.)
        identifiable = self.valid & (actual_volume[..., 0] > 1e-3) & (actual_volume[..., 0] < 4 - 1e-3)
        self.maxima['destination_error'] = max(self.maxima['destination_error'],
            float((actual_destination[identifiable] - destination[identifiable]).abs().max()) if identifiable.any() else 0.)
        bound = max(float((-changed_gains).clamp_min(0).max()), float((changed_gains - 2).clamp_min(0).max()))
        self.maxima['gain_bound_error'] = max(self.maxima['gain_bound_error'], bound)
        return read, changed_gains, probabilities, mixer

    def close(self):
        for handle in self.handles:
            handle.remove()


def unit_check(engine, hooks, split, config, reservation):
    begin, end, _, _, corrupted, ids, mask = next(batches(split, 1))
    valid = torch.ones_like(mask)
    result = {'modes': {}, 'instrumented_intact_exact': False}
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        reservation.check(force=True)
        hooks.configure(mask, valid, 'dynamic')
        direct = engine.model(corrupted)
        state, _ = engine.run(corrupted, mask)
        instrumented = engine.original_logits(state)
        torch.testing.assert_close(direct, instrumented, atol=0, rtol=0)
        result['instrumented_intact_exact'] = True
        for mode in ('mean_masked', 'mean_visible', 'mean_both', 'shuffle_masked', 'shuffle_visible'):
            hooks.reset_maxima()
            kwargs = {}
            if mode.startswith('shuffle'):
                kwargs = {'block_ids': split['block_ids'][begin:end], 'seed': PERMUTATION_SEEDS[0]}
            hooks.configure(mask, valid, mode, **kwargs)
            engine.run(corrupted, mask)
            result['modes'][mode] = dict(hooks.maxima)
            assert hooks.maxima['read_error'] == 0
            assert hooks.maxima['mixer_probability_error'] == 0
            assert hooks.maxima['mixer_matrix_error'] == 0
            assert hooks.maxima['unselected_volume_error'] == 0
            assert hooks.maxima['shuffle_multiset_error'] == 0
            assert hooks.maxima['gain_bound_error'] == 0
            assert hooks.maxima['destination_error'] < 1e-4
            assert hooks.maxima['assigned_volume_error'] < 1e-4
    write_json(OUTPUT / 'unit.json', result)
    return result


def evaluate(engine, hooks, split, config, reservation, name, mode, seed=None):
    path = effect_path(name)
    if path.exists():
        print('CACHED', name, flush=True); return load_pack(path)
    partial = EFFECTS / (name + '.partial.npz')
    values = []; completed = 0; hooks.reset_maxima()
    if partial.exists():
        saved = load_pack(partial)
        completed = int(saved['completed'])
        values = [saved['token_stats']]
        hooks.maxima.update(json.loads(str(saved['maximum_invariants_json'])))
        print('RESUME', name, completed, '/', len(split['ids']), flush=True)
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        for begin, end, _, _, corrupted, ids, mask in batches(split, config['batch_size']):
            if end <= completed:
                continue
            assert begin >= completed
            reservation.check()
            valid = torch.ones_like(mask)
            kwargs = {'block_ids': split['block_ids'][begin:end], 'seed': seed} if mode.startswith('shuffle') else {}
            hooks.configure(mask, valid, mode, **kwargs)
            state, _ = engine.run(corrupted, mask)
            values.append(engine.final_stats(state, ids, mask, config['head_chunk']))
            if end % 16 == 0:
                temporary = partial.with_name(partial.name + '.tmp')
                with temporary.open('wb') as target:
                    np.savez_compressed(target, completed=np.array(end), token_stats=np.concatenate(values),
                                        maximum_invariants_json=np.array(json.dumps(hooks.maxima)))
                temporary.replace(partial)
            if end % 128 == 0:
                print('EVALUATION', name, end, '/', len(split['ids']), flush=True)
    pack = summarize_tokens(split, np.concatenate(values))
    pack['maximum_invariants_json'] = np.array(json.dumps(hooks.maxima))
    save_pack(path, pack, load_pack(effect_path('dynamic_dynamic')))
    partial.unlink(missing_ok=True)
    print('COMPLETE', name, aggregate(pack)['ce'], json.dumps(hooks.maxima), flush=True)
    return pack


def bootstrap(condition, intact, baseline, config):
    assert_aligned(condition, intact); assert_aligned(condition, baseline)
    counts = condition['seq_mask_count']; n = len(counts)
    rng = np.random.default_rng(config['bootstrap_seed'])
    point = aggregate(condition); intact_point = aggregate(intact); baseline_point = aggregate(baseline)
    draws = {key: [] for key in ('ce', 'delta_intact', 'delta_baseline', 'fraction_gain_lost')}
    for start in range(0, config['bootstrap_repeats'], 250):
        index = rng.integers(n, size=(min(250, config['bootstrap_repeats'] - start), n))
        denominator = counts[index].sum(-1)
        value = condition['seq_ce_sum'][index].sum(-1)
        original = intact['seq_ce_sum'][index].sum(-1)
        base = baseline['seq_ce_sum'][index].sum(-1)
        draws['ce'].extend(value / denominator)
        draws['delta_intact'].extend((value - original) / denominator)
        draws['delta_baseline'].extend((value - base) / denominator)
        draws['fraction_gain_lost'].extend((value - original) / (base - original))
    delta = point['ce'] - intact_point['ce']
    gain = baseline_point['ce'] - intact_point['ce']
    result = dict(point)
    result.update(delta_ce_intact=delta, delta_ce_baseline=point['ce'] - baseline_point['ce'],
                  fraction_gain_lost=delta / gain, intact_gain_ce=gain,
                  delta_accuracy_intact=point['accuracy'] - intact_point['accuracy'],
                  delta_target_logit_intact=point['target_logit'] - intact_point['target_logit'],
                  delta_entropy_intact=point['entropy'] - intact_point['entropy'])
    for key, values in draws.items():
        result[key + '_ci95'] = list(map(float, np.quantile(values, [.025, .975])))
    return result


def contrast(left, right, config):
    assert_aligned(left, right)
    counts = left['seq_mask_count']; n = len(counts)
    rng = np.random.default_rng(config['bootstrap_seed'])
    point = float((left['seq_ce_sum'] - right['seq_ce_sum']).sum() / counts.sum())
    values = []
    for start in range(0, config['bootstrap_repeats'], 250):
        index = rng.integers(n, size=(min(250, config['bootstrap_repeats'] - start), n))
        values.extend((left['seq_ce_sum'][index].sum(-1) - right['seq_ce_sum'][index].sum(-1)) /
                      counts[index].sum(-1))
    return {'delta_ce': point, 'ci95': list(map(float, np.quantile(values, [.025, .975])))}


def ensemble(packs):
    first = packs[0]
    for pack in packs[1:]: assert_aligned(pack, first)
    result = {key: first[key].copy() for key in ('block_ids', 'seq_mask_count', 'positions', 'token_sequence')}
    result['seq_ce_sum'] = np.stack([p['seq_ce_sum'] for p in packs]).mean(0)
    result['token_stats'] = np.stack([p['token_stats'] for p in packs]).mean(0)
    return result


def analyze(config):
    packs = {path.stem: load_pack(path) for path in EFFECTS.glob('*.npz')}
    intact, baseline = packs['dynamic_dynamic'], packs['baseline']
    masked_seeds = [packs[f'shuffle_masked_seed{seed}'] for seed in PERMUTATION_SEEDS]
    visible_seeds = [packs[f'shuffle_visible_seed{seed}'] for seed in PERMUTATION_SEEDS]
    packs['shuffle_masked_mean'] = ensemble(masked_seeds)
    packs['shuffle_visible_mean'] = ensemble(visible_seeds)
    order = ['dynamic_dynamic', 'mean_dynamic', 'dynamic_mean', 'mean_mean',
             'shuffle_masked_mean', 'shuffle_visible_mean']
    results = {name: bootstrap(packs[name], intact, baseline, config) for name in order}
    seed_results = {role: {str(seed): bootstrap(packs[f'shuffle_{role}_seed{seed}'], intact, baseline, config)
                           for seed in PERMUTATION_SEEDS} for role in ('masked', 'visible')}
    contrasts = {
        'masked_mean_minus_visible_mean': contrast(packs['mean_dynamic'], packs['dynamic_mean'], config),
        'masked_shuffle_minus_visible_shuffle': contrast(packs['shuffle_masked_mean'], packs['shuffle_visible_mean'], config),
        'masked_shuffle_minus_masked_mean': contrast(packs['shuffle_masked_mean'], packs['mean_dynamic'], config),
        'visible_shuffle_minus_visible_mean': contrast(packs['shuffle_visible_mean'], packs['dynamic_mean'], config),
    }
    final = {'conditions': results, 'permutation_seed_results': seed_results, 'contrasts': contrasts,
             'bootstrap': {'repeats': config['bootstrap_repeats'], 'seed': config['bootstrap_seed'],
                           'unit': 'sequence-paired; masks, means and permutation seeds fixed'}}
    write_json(OUTPUT / 'final_results.json', final)

    fields = ['condition', 'ce', 'ce_ci_low', 'ce_ci_high', 'delta_ce_intact', 'delta_ci_low',
              'delta_ci_high', 'fraction_gain_lost', 'fraction_ci_low', 'fraction_ci_high',
              'delta_ce_baseline', 'accuracy', 'target_logit', 'entropy', 'masked_tokens', 'sequences']
    with (OUTPUT / 'aggregate.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fields); writer.writeheader()
        for name in order:
            row = results[name]
            writer.writerow({'condition': name, 'ce': row['ce'], 'ce_ci_low': row['ce_ci95'][0],
                'ce_ci_high': row['ce_ci95'][1], 'delta_ce_intact': row['delta_ce_intact'],
                'delta_ci_low': row['delta_intact_ci95'][0], 'delta_ci_high': row['delta_intact_ci95'][1],
                'fraction_gain_lost': row['fraction_gain_lost'],
                'fraction_ci_low': row['fraction_gain_lost_ci95'][0],
                'fraction_ci_high': row['fraction_gain_lost_ci95'][1],
                'delta_ce_baseline': row['delta_ce_baseline'], 'accuracy': row['accuracy'],
                'target_logit': row['target_logit'], 'entropy': row['entropy'],
                'masked_tokens': row['masked_tokens'], 'sequences': row['sequences']})
    raw_names = order[:4] + [f'shuffle_{role}_seed{seed}' for role in ('masked', 'visible') for seed in PERMUTATION_SEEDS]
    with (OUTPUT / 'per_sequence.jsonl').open('w') as target:
        for i, block in enumerate(intact['block_ids']):
            count = int(intact['seq_mask_count'][i])
            row = {'sequence_index': i, 'block_id': int(block), 'masked_tokens': count,
                   'ce': {name: float(packs[name]['seq_ce_sum'][i] / count) for name in raw_names}}
            target.write(json.dumps(row, ensure_ascii=False) + '\n')
    return final


def render(config, final):
    labels = ['dynamic / dynamic', 'mean / dynamic', 'dynamic / mean', 'mean / mean',
              'shuffle masked', 'shuffle visible']
    names = list(final['conditions'])
    values = np.array([final['conditions'][n]['delta_ce_intact'] for n in names])
    ci = np.array([final['conditions'][n]['delta_intact_ci95'] for n in names])
    fig, ax = plt.subplots(figsize=(10, 5.2))
    colors = ['#9aa5ad', '#d65f5f', '#4c90c0', '#7c66b2', '#e49a45', '#45a18b']
    ax.bar(np.arange(len(values)), values, color=colors, width=.72)
    ax.errorbar(np.arange(len(values)), values, yerr=np.vstack([values-ci[:,0], ci[:,1]-values]),
                fmt='none', ecolor='#172033', capsize=4, lw=1.3)
    ax.axhline(0, color='#172033', lw=.8)
    ax.set_xticks(np.arange(len(values)), labels, rotation=18, ha='right')
    ax.set_ylabel('ΔCE relative to intact n=2')
    ax.set_title('Which role needs token-specific write volume?  t=0.5')
    for x, value in enumerate(values): ax.text(x, value + max(values)*.025, f'{value:+.4f}', ha='center', fontsize=9)
    fig.tight_layout(); fig.savefig(OUTPUT / 'volume_role_effects.png', dpi=180); fig.savefig(OUTPUT / 'volume_role_effects.svg')
    plt.close(fig)

    rows = []
    for name, label in zip(names, labels):
        r = final['conditions'][name]
        rows.append(f'<tr><td>{escape(label)}</td><td>{r["ce"]:.6f}</td>'
                    f'<td>{r["delta_ce_intact"]:+.6f} [{r["delta_intact_ci95"][0]:+.6f}, {r["delta_intact_ci95"][1]:+.6f}]</td>'
                    f'<td>{100*r["fraction_gain_lost"]:.1f}% [{100*r["fraction_gain_lost_ci95"][0]:.1f}, {100*r["fraction_gain_lost_ci95"][1]:.1f}]</td>'
                    f'<td>{r["delta_ce_baseline"]:+.6f}</td><td>{r["accuracy"]:.4f}</td>'
                    f'<td>{r["target_logit"]:.4f}</td><td>{r["entropy"]:.4f}</td></tr>')
    c = final['contrasts']; masked = final['conditions']['mean_dynamic']; visible = final['conditions']['dynamic_mean']
    if c['masked_mean_minus_visible_mean']['ci95'][0] > 0:
        interpretation = ('Masked-only replacement is significantly more destructive than visible-only: '
                          'dynamic volume predominantly regulates construction of masked prediction states at t=0.5.')
    elif c['masked_mean_minus_visible_mean']['ci95'][1] < 0:
        interpretation = ('Both replacements are destructive, and visible-only is significantly more destructive than '
                          'masked-only: dynamic volume is not masked-specific and has an especially strong role in '
                          'processing visible evidence at t=0.5 (measured indirectly through masked-target CE).')
    elif visible['delta_intact_ci95'][0] > 0:
        interpretation = ('Both role replacements are destructive without a resolved role difference: volume has a broader token-specific function.')
    else:
        interpretation = ('Only masked-role replacement is detectably destructive; visible-volume variability is not sufficient evidence of functional assignment.')
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>n=2 write volume by token role</title>
<style>body{{font:15px/1.5 system-ui;max-width:1250px;margin:auto;padding:28px;color:#172033}}h1,h2{{line-height:1.2}}.result{{background:#edf6f8;border-left:4px solid #267c90;padding:12px 18px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #d8e3e8;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%}}code{{background:#eef1f5;padding:2px 4px}}</style><body>
<h1>Final n=2: role-specific write-volume assignment</h1>
<p class="result"><b>Established result.</b> {escape(interpretation)} The masked-minus-visible mean-replacement contrast is {c['masked_mean_minus_visible_mean']['delta_ce']:+.6f} CE, 95% CI [{c['masked_mean_minus_visible_mean']['ci95'][0]:+.6f}, {c['masked_mean_minus_visible_mean']['ci95'][1]:+.6f}].</p>
<img src="volume_role_effects.png" alt="Write volume intervention effects">
<h2>Results</h2><table><thead><tr><th>Condition</th><th>Masked CE</th><th>ΔCE vs intact [95% CI]</th><th>Gain lost [95% CI]</th><th>ΔCE vs baseline</th><th>Accuracy</th><th>Target logit</th><th>Entropy</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Implementation and checks</h2><p>All conditions use checkpoint <code>ckpt_53399.pt</code>, the same 1024 held-out sequences, corruption masks and actual-mask CE as the earlier factor intervention. Role means are accumulated in float64 on 64 disjoint calibration sequences. Role is read exclusively from the observed corruption mask. Read, bounded write destination and stream mixer remain dynamic at every attention and MLP interface.</p>
<p>Permutation conditions shuffle V independently within each sequence, role and interface for seeds {', '.join(map(str, PERMUTATION_SEEDS))}. They preserve each selected volume multiset. Visible-position effects are measured only indirectly through masked-target CE.</p>
<h2>Interpretation limits</h2><p>Gain-loss percentages compare a frozen intervention in the co-adapted n=2 checkpoint with a separately trained residual baseline; they are not an additive decomposition. Permutation confidence intervals are sequence-paired intervals for the seed-averaged result and are conditional on the three fixed seeds.</p>
</body></html>'''
    (OUTPUT / 'report.html').write_text(html)


def verify(config, protocol):
    final = json.loads((OUTPUT / 'final_results.json').read_text())
    prior_masked = aggregate(load_pack(REFERENCE['mean_dynamic']))['ce']
    prior_both = aggregate(load_pack(REFERENCE['mean_mean']))['ce']
    checks = {
        'intact_ce_exact': final['conditions']['dynamic_dynamic']['ce'] == 3.20422060546099,
        'masked_only_ce_exact_prior': final['conditions']['mean_dynamic']['ce'] == prior_masked,
        'both_role_mean_ce_exact_prior': final['conditions']['mean_mean']['ce'] == prior_both,
        'calibration_evaluation_disjoint': not protocol['calibration_evaluation_overlap'],
        'three_permutation_seeds_each_role': all(len(final['permutation_seed_results'][r]) == 3 for r in ('masked','visible')),
    }
    checks['passed'] = all(checks.values())
    artifacts = ['config.json', 'unit.json', 'final_results.json', 'aggregate.csv',
                 'per_sequence.jsonl', 'volume_role_effects.png', 'volume_role_effects.svg', 'report.html']
    checks['artifact_sha256'] = {name: sha256(OUTPUT / name) for name in artifacts}
    write_json(OUTPUT / 'verification.json', checks)
    assert checks['passed'], checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('full', 'render'), default='full')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--allow-small-cooccupant-mib', type=float, default=0)
    parser.add_argument('--only', help='Run one new condition and exit; used by parallel workers.')
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name('config.json').read_text())
    split, means, references, protocol = prepare(config)
    if args.stage == 'full':
        policy = dict(config['gpu_policy']); policy['never_use_training_gpus'] = []
        from .safety import GPUReservation
        reservation = (SmallContextReservation(args.gpu, policy, args.allow_small_cooccupant_mib)
                       if args.allow_small_cooccupant_mib else GPUReservation(args.gpu, policy))
        write_json(OUTPUT / 'execution.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
                   'gpu': args.gpu, 'observations': reservation.observations,
                   'allowed_external_max_mib': getattr(reservation, 'observed_external_max_mib', {})})
        engine = load_model(config, 'full_n2'); assert engine.checkpoint_step == 53399
        hooks = VolumeRoleHooks(engine, means['role']['volume'])
        if not (OUTPUT / 'unit.json').exists():
            unit_check(engine, hooks, split, config, reservation)
        tasks = [('dynamic_mean', 'mean_visible', None)] + [
            (f'shuffle_{role}_seed{seed}', f'shuffle_{role}', seed)
            for role in ('masked', 'visible') for seed in PERMUTATION_SEEDS]
        if args.only:
            tasks = [task for task in tasks if task[0] == args.only]
            if not tasks: raise ValueError('Unknown --only condition: ' + args.only)
        for name, mode, seed in tasks:
            evaluate(engine, hooks, split, config, reservation, name, mode, seed)
        hooks.close(); del hooks, engine; torch.cuda.empty_cache()
        if args.only:
            print('WORKER COMPLETE', args.only, flush=True)
            return
    final = analyze(config)
    render(config, final)
    verify(config, protocol)
    if args.stage == 'full':
        execution = json.loads((OUTPUT / 'execution.json').read_text())
        execution['completed_utc'] = datetime.now(timezone.utc).isoformat()
        execution['external_compute_context_max_mib'] = getattr(reservation, 'observed_external_max_mib', {})
        write_json(OUTPUT / 'execution.json', execution)
    print('VOLUME ROLE ASSIGNMENT COMPLETE', OUTPUT / 'report.html', flush=True)


if __name__ == '__main__':
    main()
