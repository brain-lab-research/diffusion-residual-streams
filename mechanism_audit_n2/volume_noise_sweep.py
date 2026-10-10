"""Minimal causal sweep of role-specific write volume across diffusion noise.

The runner uses the final 3.5B n=2 checkpoint and a 128-sequence discovery
split.  Calibration means are recomputed independently at each corruption
ratio from 64 disjoint captured sequences.  This isolates within-role dynamic
volume from the change in the role mean itself across noise.

Examples
--------
python -m mechanism_audit_n2.volume_noise_sweep --ratio .1 --gpu 0
python -m mechanism_audit_n2.volume_noise_sweep --ratio .5 --gpu 0
python -m mechanism_audit_n2.volume_noise_sweep --render
"""
import argparse
import csv
import fcntl
import hashlib
from html import escape
import json
import os
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

from .engine import aggregate, batches, load_model, make_split, summarize_tokens
from .factors import ROOT
from .primary import load_pack, save_pack, write_json
from .role_interventions import calibration_means
from .safety import GPUReservation, query
from .volume_roles import VolumeRoleHooks, assert_aligned, sha256


OUTPUT = ROOT / 'volume_noise_sweep_n2'
EFFECTS = OUTPUT / 'effects'
RATIOS = (.1, .3, .5, .7, .9)
SHUFFLE_RATIOS = (.1, .5, .9)
SHUFFLE_SEED = 130921
STAGES = {
    'early': tuple(range(0, 10)),       # blocks 1--5, attention + MLP
    'middle': tuple(range(10, 22)),     # blocks 6--11
    'late': tuple(range(22, 32)),       # blocks 12--16
}
ROLE_MODE = {'masked': 'mean_masked', 'visible': 'mean_visible'}


class BoundedSharedGPUReservation:
    """Use an explicitly allowed small co-occupant set, never an unbounded busy GPU."""
    def __init__(self, gpu, maximum_external_mib):
        self.gpu = str(gpu); self.maximum_external_mib = float(maximum_external_mib)
        assert self.maximum_external_mib > 0
        self.lock = Path('/tmp/mhc_mechanism_gpu' + self.gpu + '.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = query(['--query-gpu=index,uuid,utilization.gpu,memory.used', '--format=csv,nounits'])
        row = next(item for item in rows if item['index'].strip() == self.gpu)
        self.uuid = row['uuid'].strip()
        external = self._external()
        if sum(external.values()) > self.maximum_external_mib:
            raise RuntimeError('Existing processes exceed the explicitly allowed external-memory bound')
        self.allowed_pids = set(external)
        self.observations = [{**row, 'allowed_external_processes': external,
                              'maximum_external_mib': self.maximum_external_mib}]
        self.observed_external_max_mib = dict(external); self.last_checked = 0.; self.query_failures = 0
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
        os.environ.setdefault('OMP_NUM_THREADS', '4')

    def _external(self):
        rows = query(['--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,nounits'])
        return {int(row['pid']): float(row['used_gpu_memory [MiB]']) for row in rows
                if row['gpu_uuid'].strip() == self.uuid and int(row['pid']) != os.getpid()}

    def check(self, force=False):
        if not force and time.monotonic() - self.last_checked < 5: return
        try:
            external = self._external()
        except subprocess.TimeoutExpired:
            self.query_failures += 1; self.last_checked = time.monotonic()
            if self.query_failures >= 3:
                raise RuntimeError('Three consecutive GPU-safety queries timed out')
            return
        self.query_failures = 0
        if sum(external.values()) > self.maximum_external_mib:
            raise RuntimeError('External GPU memory exceeded the explicit bound')
        for pid, value in external.items():
            self.observed_external_max_mib[pid] = max(value, self.observed_external_max_mib.get(pid, 0.))
        self.last_checked = time.monotonic()


def ratio_key(ratio):
    return f'r{round(100 * ratio):02d}'


def condition_specs(ratio):
    specs = {
        'intact': ('dynamic', None, None),
        'mean_masked_all': ('mean_masked', None, None),
        'mean_visible_all': ('mean_visible', None, None),
        'mean_both_all': ('mean_both', None, None),
    }
    for role, mode in ROLE_MODE.items():
        for stage, layers in STAGES.items():
            specs[f'mean_{role}_{stage}'] = (mode, layers, None)
    if ratio in SHUFFLE_RATIOS:
        specs['shuffle_masked_all'] = ('shuffle_masked', None, SHUFFLE_SEED)
        specs['shuffle_visible_all'] = ('shuffle_visible', None, SHUFFLE_SEED)
    return specs


def effect_path(ratio, condition):
    path = EFFECTS / ratio_key(ratio) / f'{condition}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_ratio_means(config, ratio):
    """Load an independently captured calibration split and derive V means."""
    path = ROOT / 'residual_circuits_n2' / 'captures' / f'{ratio_key(ratio)}_all_positions.npz'
    with np.load(path) as saved:
        captured = {key: saved[key] for key in saved.files}
    split = make_split(config, 'calibration', ratio)
    np.testing.assert_array_equal(captured['block_ids'], split['block_ids'])
    np.testing.assert_array_equal(captured['mask'], split['mask'])
    assert float(captured['ratio']) == ratio
    assert str(captured['mask_sha256']) == split['mask_sha256']
    assert str(captured['data_sha256']) == split['data_sha256']
    valid = np.ones(captured['mask'].size, dtype=bool)
    means = calibration_means(captured['router'].astype(np.float64),
                              captured['mask'].ravel(), valid)
    volume = np.asarray(means['role']['volume'], dtype=np.float64)
    assert volume.shape == (32, 2, 1)
    return volume, split, path


def prepare(config):
    OUTPUT.mkdir(exist_ok=True); EFFECTS.mkdir(exist_ok=True)
    calibration = {}; evaluation = {}; means = {}; captures = {}
    for ratio in RATIOS:
        means[ratio], calibration[ratio], captures[ratio] = load_ratio_means(config, ratio)
        evaluation[ratio] = make_split(config, 'discovery', ratio)
        overlap = set(map(int, calibration[ratio]['block_ids'])) & set(map(int, evaluation[ratio]['block_ids']))
        assert not overlap
    protocol = {
        'checkpoint': config['models']['full_n2'],
        'ratios': list(RATIOS),
        'evaluation_split': 'discovery',
        'evaluation_sequences_per_ratio': len(evaluation[RATIOS[0]]['ids']),
        'calibration_sequences_per_ratio': len(calibration[RATIOS[0]]['ids']),
        'calibration_evaluation_disjoint': True,
        'role_definition': 'MASKED=observed corruption mask; VISIBLE=valid & not mask.',
        'target': 'Masked-token CE; visible interventions affect it indirectly through context.',
        'volume_coordinate': 'V=g0+g1; bounded write destination is reconstructed unchanged.',
        'ratio_specific_means': ('At every ratio, per-interface and per-role V means are token-weighted '
                                 'float64 means on its own disjoint 64-sequence calibration mask.'),
        'controls': 'Read, bounded destination, mixer, and all non-selected interfaces stay dynamic.',
        'stages': {name: {'blocks': [layers[0] // 2 + 1, layers[-1] // 2 + 1],
                          'interfaces_zero_based': [layers[0], layers[-1]]}
                   for name, layers in STAGES.items()},
        'conditions': {ratio_key(r): list(condition_specs(r)) for r in RATIOS},
        'shuffle': ('One prespecified seed; independently within each sequence, role, and interface; '
                    'exactly preserves the selected V multiset.'),
        'shuffle_seed': SHUFFLE_SEED,
        'statistics': ('5000 sequence-paired bootstrap draws. Cross-noise contrasts resample the same '
                       'block IDs jointly but retain each ratio-specific denominator.'),
        'primary_intermediate_peak': ('mean_both_all: mean of effects at t=.5,.7 minus mean of effects '
                                      'at t=.1,.9; t=.3 is trajectory evidence, not in the contrast.'),
        'primary_handoff': ('early-stage (masked minus visible) effect at t=.9 minus the same gap at t=.1. '
                            'A strong handoff also requires visible>masked at .1 and masked>visible at .9.'),
        'nonadditivity': 'Stage and all-layer intervention effects are not additive attributions.',
        'mask_seed': config['mask_seed'],
        'bootstrap_seed': config['bootstrap_seed'],
        'bootstrap_repeats': config['bootstrap_repeats'],
        'capture_sha256': {ratio_key(r): sha256(captures[r]) for r in RATIOS},
        'source_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
                          ['volume_noise_sweep.py', 'volume_roles.py', 'engine.py', 'factors.py',
                           'primary.py', 'role_interventions.py']},
    }
    config_path = OUTPUT / 'config.json'
    if not config_path.exists() or json.loads(config_path.read_text()) != protocol:
        write_json(config_path, protocol)
    means_document = {
        ratio_key(r): {'masked': means[r][:, 0, 0].tolist(),
                       'visible': means[r][:, 1, 0].tolist()}
        for r in RATIOS}
    means_path = OUTPUT / 'calibration_means.json'
    if not means_path.exists() or json.loads(means_path.read_text()) != means_document:
        write_json(means_path, means_document)
    return evaluation, means, protocol


def merge_maxima(old, new):
    return {key: max(float(old.get(key, 0.)), float(value)) for key, value in new.items()}


def unit_check(engine, hooks, split, reservation):
    begin, end, _, _, corrupted, ids, mask = next(batches(split, 1))
    valid = torch.ones_like(mask)
    result = {'instrumented_intact_exact': False, 'conditions': {}}
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        reservation.check(force=True)
        hooks.configure(mask, valid, 'dynamic')
        direct = engine.model(corrupted)
        state, _ = engine.run(corrupted, mask)
        torch.testing.assert_close(direct, engine.original_logits(state), rtol=0, atol=0)
        result['instrumented_intact_exact'] = True
        tests = {
            'mean_masked_early': ('mean_masked', STAGES['early'], None),
            'mean_visible_middle': ('mean_visible', STAGES['middle'], None),
            'mean_both_all': ('mean_both', None, None),
            'shuffle_masked_all': ('shuffle_masked', None, SHUFFLE_SEED),
            'shuffle_visible_all': ('shuffle_visible', None, SHUFFLE_SEED),
        }
        for name, (mode, layers, seed) in tests.items():
            hooks.reset_maxima()
            kwargs = ({'block_ids': split['block_ids'][begin:end], 'seed': seed}
                      if mode.startswith('shuffle') else {})
            hooks.configure(mask, valid, mode, layers=layers, **kwargs)
            engine.run(corrupted, mask)
            result['conditions'][name] = dict(hooks.maxima)
            invariant_check(hooks.maxima)
    return result


def invariant_check(maxima):
    for key in ('read_error', 'mixer_probability_error', 'mixer_matrix_error',
                'unselected_volume_error', 'shuffle_multiset_error', 'gain_bound_error'):
        assert maxima[key] == 0, (key, maxima[key])
    assert maxima['destination_error'] < 1e-4, maxima
    assert maxima['assigned_volume_error'] < 1e-4, maxima


def evaluate_ratio(engine, hooks, split, config, reservation, ratio):
    """Evaluate all missing conditions together, caching each dynamic prefix once."""
    specs = condition_specs(ratio)
    missing = [name for name in specs if not effect_path(ratio, name).exists()]
    if not missing:
        print('RATIO CACHED', ratio, flush=True); return
    signature = hashlib.sha256('\n'.join(missing).encode()).hexdigest()[:10]
    partial = effect_path(ratio, f'worker_{signature}').with_suffix('.partial.npz')
    collected = {name: [] for name in missing}
    maxima = {name: {} for name in missing}
    completed = 0
    if partial.exists():
        saved = load_pack(partial)
        assert json.loads(str(saved['conditions_json'])) == missing
        completed = int(saved['completed'])
        for name in missing: collected[name] = [saved[name]]
        maxima = json.loads(str(saved['maxima_json']))
        print('RESUME', ratio, completed, '/', len(split['ids']), flush=True)
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        for begin, end, _, _, corrupted, ids, mask in batches(split, config['batch_size']):
            if end <= completed: continue
            assert begin >= completed
            reservation.check(); valid = torch.ones_like(mask)
            hooks.configure(mask, valid, 'dynamic')
            intact_state, cache = engine.run(corrupted, mask, cache=True)
            initial = engine.initial(corrupted)
            if 'intact' in missing:
                collected['intact'].append(engine.final_stats(intact_state, ids, mask, config['head_chunk']))
            for name in missing:
                if name == 'intact': continue
                mode, layers, seed = specs[name]
                start = 0 if layers is None else layers[0]
                before = initial if start == 0 else cache['states'][start - 1]
                kwargs = ({'block_ids': split['block_ids'][begin:end], 'seed': seed}
                          if mode.startswith('shuffle') else {})
                hooks.reset_maxima()
                hooks.configure(mask, valid, mode, layers=layers, **kwargs)
                state, _ = engine.run(corrupted, mask, initial_state=before, start_after=start - 1)
                collected[name].append(engine.final_stats(state, ids, mask, config['head_chunk']))
                maxima[name] = merge_maxima(maxima[name], hooks.maxima)
                del state
            if end % 16 == 0 or end == len(split['ids']):
                payload = {'completed': np.array(end), 'conditions_json': np.array(json.dumps(missing)),
                           'maxima_json': np.array(json.dumps(maxima))}
                payload.update({name: np.concatenate(values) for name, values in collected.items()})
                temporary = partial.with_name(partial.name + '.tmp')
                with temporary.open('wb') as target: np.savez_compressed(target, **payload)
                temporary.replace(partial)
                print('RATIO', ratio, end, '/', len(split['ids']), flush=True)
            del intact_state, cache, initial
    intact = (summarize_tokens(split, np.concatenate(collected['intact']))
              if 'intact' in missing else load_pack(effect_path(ratio, 'intact')))
    for name in missing:
        pack = intact if name == 'intact' else summarize_tokens(split, np.concatenate(collected[name]))
        pack = dict(pack)
        pack['maximum_invariants_json'] = np.array(json.dumps(maxima[name]))
        if name != 'intact': invariant_check(maxima[name])
        save_pack(effect_path(ratio, name), pack, None if name == 'intact' else intact)
        print('COMPLETE', ratio, name, aggregate(pack)['ce'], flush=True)
    partial.unlink(missing_ok=True)


def condition_effect(pack, intact, config):
    assert_aligned(pack, intact)
    counts = pack['seq_mask_count']; n = len(counts)
    point = aggregate(pack)
    original = aggregate(intact)
    rng = np.random.default_rng(config['bootstrap_seed'])
    ce_draws = []; delta_draws = []
    for begin in range(0, config['bootstrap_repeats'], 250):
        indices = rng.integers(n, size=(min(250, config['bootstrap_repeats'] - begin), n))
        denominator = counts[indices].sum(-1)
        value = pack['seq_ce_sum'][indices].sum(-1) / denominator
        base = intact['seq_ce_sum'][indices].sum(-1) / denominator
        ce_draws.extend(value); delta_draws.extend(value - base)
    result = dict(point)
    result.update(delta_ce_intact=point['ce'] - original['ce'],
                  delta_accuracy_intact=point['accuracy'] - original['accuracy'],
                  delta_target_logit_intact=point['target_logit'] - original['target_logit'],
                  delta_entropy_intact=point['entropy'] - original['entropy'],
                  ce_ci95=list(map(float, np.quantile(ce_draws, [.025, .975]))),
                  delta_intact_ci95=list(map(float, np.quantile(delta_draws, [.025, .975]))))
    return result


def bootstrap_cross_ratio(packs, coefficients, config):
    """Linear contrast of CE effects; jointly resample shared sequence IDs."""
    assert set(packs) == set(coefficients)
    first = next(iter(packs.values()))[0]
    n = len(first['block_ids'])
    for condition, intact in packs.values():
        np.testing.assert_array_equal(condition['block_ids'], first['block_ids'])
        np.testing.assert_array_equal(intact['block_ids'], first['block_ids'])
    def value(indices=None):
        total = 0.
        for ratio, (condition, intact) in packs.items():
            if indices is None:
                denominator = condition['seq_mask_count'].sum()
                delta = (condition['seq_ce_sum'].sum() - intact['seq_ce_sum'].sum()) / denominator
            else:
                denominator = condition['seq_mask_count'][indices].sum(-1)
                delta = ((condition['seq_ce_sum'][indices].sum(-1) -
                          intact['seq_ce_sum'][indices].sum(-1)) / denominator)
            total = total + coefficients[ratio] * delta
        return total
    rng = np.random.default_rng(config['bootstrap_seed']); draws = []
    for begin in range(0, config['bootstrap_repeats'], 250):
        indices = rng.integers(n, size=(min(250, config['bootstrap_repeats'] - begin), n))
        draws.extend(value(indices))
    return {'delta_ce': float(value()), 'ci95': list(map(float, np.quantile(draws, [.025, .975])))}


def within_ratio_contrast(left, right, config):
    assert_aligned(left, right)
    counts = left['seq_mask_count']; n = len(counts)
    point = float((left['seq_ce_sum'] - right['seq_ce_sum']).sum() / counts.sum())
    rng = np.random.default_rng(config['bootstrap_seed']); draws = []
    for begin in range(0, config['bootstrap_repeats'], 250):
        indices = rng.integers(n, size=(min(250, config['bootstrap_repeats'] - begin), n))
        draws.extend((left['seq_ce_sum'][indices].sum(-1) - right['seq_ce_sum'][indices].sum(-1)) /
                     counts[indices].sum(-1))
    return {'delta_ce': point, 'ci95': list(map(float, np.quantile(draws, [.025, .975])))}


def analyze(config):
    packs = {ratio: {name: load_pack(effect_path(ratio, name))
                     for name in condition_specs(ratio)} for ratio in RATIOS}
    conditions = {}
    role_gaps = {}
    for ratio in RATIOS:
        intact = packs[ratio]['intact']
        conditions[ratio_key(ratio)] = {
            name: condition_effect(pack, intact, config) for name, pack in packs[ratio].items()}
        role_gaps[ratio_key(ratio)] = {
            stage: within_ratio_contrast(packs[ratio][f'mean_masked_{stage}'],
                                         packs[ratio][f'mean_visible_{stage}'], config)
            for stage in STAGES}
        role_gaps[ratio_key(ratio)]['all'] = within_ratio_contrast(
            packs[ratio]['mean_masked_all'], packs[ratio]['mean_visible_all'], config)

    peak_coefficients = {.1: -.5, .5: .5, .7: .5, .9: -.5}
    peak_packs = {r: (packs[r]['mean_both_all'], packs[r]['intact']) for r in peak_coefficients}
    intermediate_peak = bootstrap_cross_ratio(peak_packs, peak_coefficients, config)
    central = {r: (packs[r]['mean_both_all'], packs[r]['intact']) for r in (.5, .7)}
    peak_vs_low = bootstrap_cross_ratio({**central, .1: (packs[.1]['mean_both_all'], packs[.1]['intact'])},
                                        {.5: .5, .7: .5, .1: -1.}, config)
    peak_vs_high = bootstrap_cross_ratio({**central, .9: (packs[.9]['mean_both_all'], packs[.9]['intact'])},
                                         {.5: .5, .7: .5, .9: -1.}, config)

    # Difference of early role gaps: (M9-V9) - (M1-V1).
    handoff_terms = {
        (.9, 'm'): (packs[.9]['mean_masked_early'], packs[.9]['intact']),
        (.9, 'v'): (packs[.9]['mean_visible_early'], packs[.9]['intact']),
        (.1, 'm'): (packs[.1]['mean_masked_early'], packs[.1]['intact']),
        (.1, 'v'): (packs[.1]['mean_visible_early'], packs[.1]['intact']),
    }
    # bootstrap_cross_ratio keys need only be unique; its arithmetic is agnostic to key type.
    handoff = bootstrap_cross_ratio(handoff_terms,
                                    {(.9, 'm'): 1., (.9, 'v'): -1.,
                                     (.1, 'm'): -1., (.1, 'v'): 1.}, config)
    low_gap = role_gaps['r10']['early']; high_gap = role_gaps['r90']['early']
    strong_handoff = (handoff['ci95'][0] > 0 and low_gap['ci95'][1] < 0 and high_gap['ci95'][0] > 0)
    final = {
        'conditions': conditions,
        'masked_minus_visible': role_gaps,
        'hypotheses': {
            'intermediate_noise_peak': {
                'primary_contrast': intermediate_peak,
                'central_minus_t01': peak_vs_low,
                'central_minus_t09': peak_vs_high,
                'supported': (intermediate_peak['ci95'][0] > 0 and
                              peak_vs_low['ci95'][0] > 0 and peak_vs_high['ci95'][0] > 0),
            },
            'early_noise_dependent_handoff': {
                'primary_contrast': handoff,
                'early_gap_t01': low_gap,
                'early_gap_t09': high_gap,
                'strong_handoff_supported': strong_handoff,
                'positive_shift_supported': handoff['ci95'][0] > 0,
            },
        },
        'bootstrap': {'repeats': config['bootstrap_repeats'], 'seed': config['bootstrap_seed'],
                      'unit': 'sequence paired; cross-noise draws share block IDs'},
    }
    write_json(OUTPUT / 'final_results.json', final)
    fields = ['ratio', 'condition', 'ce', 'delta_ce_intact', 'delta_ci_low', 'delta_ci_high',
              'accuracy', 'target_logit', 'entropy', 'masked_tokens', 'sequences']
    with (OUTPUT / 'aggregate.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fields); writer.writeheader()
        for ratio in RATIOS:
            for name, result in conditions[ratio_key(ratio)].items():
                writer.writerow({'ratio': ratio, 'condition': name, 'ce': result['ce'],
                    'delta_ce_intact': result['delta_ce_intact'],
                    'delta_ci_low': result['delta_intact_ci95'][0],
                    'delta_ci_high': result['delta_intact_ci95'][1],
                    'accuracy': result['accuracy'], 'target_logit': result['target_logit'],
                    'entropy': result['entropy'], 'masked_tokens': result['masked_tokens'],
                    'sequences': result['sequences']})
    return final, packs


def render(config, protocol, final, packs):
    ratios = np.array(RATIOS)
    conditions = final['conditions']
    colors = {'masked': '#d45b69', 'visible': '#337fb5', 'both': '#7056a8'}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    ax = axes[0, 0]
    for role in ('masked', 'visible', 'both'):
        name = f'mean_{role}_all'
        y = np.array([conditions[ratio_key(r)][name]['delta_ce_intact'] for r in ratios])
        ci = np.array([conditions[ratio_key(r)][name]['delta_intact_ci95'] for r in ratios])
        ax.errorbar(ratios, y, yerr=np.vstack([y-ci[:, 0], ci[:, 1]-y]), marker='o',
                    lw=2, capsize=3, label=role, color=colors[role])
    ax.axhline(0, color='#29313b', lw=.8); ax.set_title('All-depth dynamic-volume necessity')
    ax.set_ylabel('Δ masked-token CE vs intact'); ax.legend(frameon=False)

    ax = axes[0, 1]
    for role in ('masked', 'visible'):
        name = f'mean_{role}_early'
        y = np.array([conditions[ratio_key(r)][name]['delta_ce_intact'] for r in ratios])
        ci = np.array([conditions[ratio_key(r)][name]['delta_intact_ci95'] for r in ratios])
        ax.errorbar(ratios, y, yerr=np.vstack([y-ci[:, 0], ci[:, 1]-y]), marker='o',
                    lw=2, capsize=3, label=role, color=colors[role])
    ax.axhline(0, color='#29313b', lw=.8); ax.set_title('Early blocks 1–5: proposed handoff')
    ax.legend(frameon=False)

    for column, role in enumerate(('masked', 'visible')):
        ax = axes[1, column]
        matrix = np.array([[conditions[ratio_key(r)][f'mean_{role}_{stage}']['delta_ce_intact']
                            for stage in STAGES] for r in ratios])
        bound = max(abs(float(matrix.min())), abs(float(matrix.max())), 1e-8)
        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', norm=TwoSlopeNorm(0, -bound, bound))
        for y in range(len(ratios)):
            for x in range(len(STAGES)):
                ax.text(x, y, f'{matrix[y, x]:+.3f}', ha='center', va='center', fontsize=9,
                        color='white' if abs(matrix[y, x]) > .58 * bound else '#172033')
        ax.set_xticks(range(3), STAGES); ax.set_yticks(range(5), [f'{r:.1f}' for r in ratios])
        ax.set_xlabel('depth stage'); ax.set_ylabel('corruption t')
        ax.set_title(f'{role.capitalize()} role by depth')
        fig.colorbar(im, ax=ax, fraction=.046, label='ΔCE')
    for ax in axes.flat[:2]: ax.set_xticks(ratios); ax.set_xlabel('corruption ratio t')
    fig.suptitle('Final 3.5B checkpoint: write volume across noise and depth', fontsize=15)
    fig.tight_layout(); fig.savefig(OUTPUT / 'volume_noise_sweep.png', dpi=190)
    fig.savefig(OUTPUT / 'volume_noise_sweep.svg'); plt.close(fig)

    shuffle_rows = []
    for ratio in SHUFFLE_RATIOS:
        for role in ('masked', 'visible'):
            result = conditions[ratio_key(ratio)][f'shuffle_{role}_all']
            shuffle_rows.append(f'<tr><td>{ratio:.1f}</td><td>{role}</td><td>{result["delta_ce_intact"]:+.6f} '
                                f'[{result["delta_intact_ci95"][0]:+.6f}, {result["delta_intact_ci95"][1]:+.6f}]</td></tr>')
    stage_rows = []
    for ratio in RATIOS:
        for role in ('masked', 'visible'):
            values = [conditions[ratio_key(ratio)][f'mean_{role}_{stage}']['delta_ce_intact'] for stage in STAGES]
            stage_rows.append('<tr><td>{:.1f}</td><td>{}</td>{}</tr>'.format(
                ratio, role, ''.join(f'<td>{value:+.6f}</td>' for value in values)))
    h = final['hypotheses']; peak = h['intermediate_noise_peak']; handoff = h['early_noise_dependent_handoff']
    peak_label = 'SUPPORTED' if peak['supported'] else 'NOT ESTABLISHED'
    handoff_label = 'STRONGLY SUPPORTED' if handoff['strong_handoff_supported'] else (
        'POSITIVE SHIFT ONLY' if handoff['positive_shift_supported'] else 'NOT ESTABLISHED')
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Write volume across diffusion noise</title>
<style>body{{font:15px/1.55 system-ui;max-width:1250px;margin:auto;padding:28px;color:#172033}}h1,h2{{line-height:1.2}}.claim{{background:#edf6f8;border-left:4px solid #267c90;padding:12px 18px;margin:14px 0}}.warn{{background:#fff5de;border-left-color:#d89423}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #d8e3e8;text-align:right}}th:nth-child(-n+2),td:nth-child(-n+2){{text-align:left}}img{{max-width:100%}}code{{background:#eef1f5;padding:2px 4px}}</style><body>
<h1>Write volume across noise and depth — final 3.5B checkpoint</h1>
<div class="claim"><b>Intermediate-noise peak: {peak_label}.</b> Prespecified central-minus-extremes contrast = {peak['primary_contrast']['delta_ce']:+.6f} CE, 95% CI [{peak['primary_contrast']['ci95'][0]:+.6f}, {peak['primary_contrast']['ci95'][1]:+.6f}]. Support additionally requires the central average to exceed both t=.1 and t=.9 separately.</div>
<div class="claim"><b>Early-role handoff: {handoff_label}.</b> Change in masked-minus-visible gap, t=.9 minus t=.1 = {handoff['primary_contrast']['delta_ce']:+.6f} CE, 95% CI [{handoff['primary_contrast']['ci95'][0]:+.6f}, {handoff['primary_contrast']['ci95'][1]:+.6f}]. Early gap at .1: {handoff['early_gap_t01']['delta_ce']:+.6f}; at .9: {handoff['early_gap_t09']['delta_ce']:+.6f}.</div>
<img src="volume_noise_sweep.png" alt="Write-volume effects across noise and depth">
<h2>Stage interventions</h2><table><thead><tr><th>t</th><th>role</th><th>early (blocks 1–5)</th><th>middle (6–11)</th><th>late (12–16)</th></tr></thead><tbody>{''.join(stage_rows)}</tbody></table>
<h2>Within-role shuffle controls</h2><table><thead><tr><th>t</th><th>role</th><th>ΔCE vs intact [95% CI]</th></tr></thead><tbody>{''.join(shuffle_rows)}</tbody></table>
<h2>What was held fixed</h2><p>At each t, mean replacement uses a separate 64-sequence calibration split and its own per-interface, per-role mean. It therefore removes token-specific volume variation while retaining the noise-dependent mean program. Read weights, bounded write destination, stream mixer, and every unselected interface remain dynamic. Evaluation uses 128 disjoint discovery sequences from the final <code>ckpt_53399.pt</code>.</p>
<p class="claim warn"><b>Scope.</b> This is a prespecified minimal discovery sweep, not the final confirmatory estimate. Stage effects are interventions on a nonlinear recurrent computation and must not be added. The single shuffle seed is a mechanism control, not a seed-marginalized estimate.</p>
</body></html>'''
    (OUTPUT / 'report.html').write_text(html)

    with (OUTPUT / 'per_sequence.jsonl').open('w') as target:
        for i, block_id in enumerate(packs[RATIOS[0]]['intact']['block_ids']):
            row = {'sequence_index': i, 'block_id': int(block_id), 'ratios': {}}
            for ratio in RATIOS:
                local = {}
                for name, pack in packs[ratio].items():
                    local[name] = float(pack['seq_ce_sum'][i] / pack['seq_mask_count'][i])
                row['ratios'][str(ratio)] = {'masked_tokens': int(packs[ratio]['intact']['seq_mask_count'][i]),
                                             'ce': local}
            target.write(json.dumps(row) + '\n')


def verify(config, protocol, final):
    checks = {
        'all_ratios_present': set(final['conditions']) == {ratio_key(r) for r in RATIOS},
        'all_conditions_present': all(set(final['conditions'][ratio_key(r)]) == set(condition_specs(r)) for r in RATIOS),
        'all_128_sequences': all(result['sequences'] == 128 for conditions in final['conditions'].values()
                                 for result in conditions.values()),
        'calibration_evaluation_disjoint': protocol['calibration_evaluation_disjoint'],
        'all_controls_exact': True,
        'maximum_destination_error': 0.,
        'maximum_assigned_volume_error': 0.,
    }
    for ratio in RATIOS:
        intact = load_pack(effect_path(ratio, 'intact'))
        for name in condition_specs(ratio):
            pack = load_pack(effect_path(ratio, name))
            assert_aligned(pack, intact)
            if name == 'intact': continue
            maxima = json.loads(str(pack['maximum_invariants_json']))
            exact = all(maxima[key] == 0 for key in ('read_error', 'mixer_probability_error',
                        'mixer_matrix_error', 'unselected_volume_error', 'shuffle_multiset_error',
                        'gain_bound_error'))
            checks['all_controls_exact'] &= exact
            checks['maximum_destination_error'] = max(checks['maximum_destination_error'], maxima['destination_error'])
            checks['maximum_assigned_volume_error'] = max(checks['maximum_assigned_volume_error'], maxima['assigned_volume_error'])
    checks['bounded_reconstruction_within_tolerance'] = (
        checks['maximum_destination_error'] < 1e-4 and checks['maximum_assigned_volume_error'] < 1e-4)
    checks['passed'] = all(value for key, value in checks.items() if key not in
                           ('maximum_destination_error', 'maximum_assigned_volume_error'))
    artifacts = ['config.json', 'calibration_means.json'] + [f'unit_{ratio_key(r)}.json' for r in RATIOS] + ['final_results.json',
                 'aggregate.csv', 'per_sequence.jsonl', 'volume_noise_sweep.png',
                 'volume_noise_sweep.svg', 'report.html']
    checks['artifact_sha256'] = {name: sha256(OUTPUT / name) for name in artifacts}
    write_json(OUTPUT / 'verification.json', checks)
    assert checks['passed'], checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratio', type=float, choices=RATIOS)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--allow-external-mib', type=float, default=0,
                        help='Explicitly permit only the processes already on this GPU up to this total memory.')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name('config.json').read_text())
    evaluation, means, protocol = prepare(config)
    if args.render:
        final, packs = analyze(config); render(config, protocol, final, packs)
        verify(config, protocol, final)
        print('REPORT COMPLETE', OUTPUT / 'report.html', flush=True); return
    if args.ratio is None: raise ValueError('--ratio is required unless --render is used')
    ratio = float(args.ratio)
    policy = dict(config['gpu_policy']); policy['never_use_training_gpus'] = []
    reservation = (BoundedSharedGPUReservation(args.gpu, args.allow_external_mib)
                   if args.allow_external_mib else GPUReservation(args.gpu, policy))
    write_json(OUTPUT / f'gpu_observation_{ratio_key(ratio)}.json', reservation.observations)
    engine = load_model(config, 'full_n2')
    hooks = VolumeRoleHooks(engine, means[ratio])
    unit_path = OUTPUT / f'unit_{ratio_key(ratio)}.json'
    if not unit_path.exists(): write_json(unit_path, unit_check(engine, hooks, evaluation[ratio], reservation))
    evaluate_ratio(engine, hooks, evaluation[ratio], config, reservation, ratio)
    hooks.close()
    print('WORKER COMPLETE', ratio, flush=True)


if __name__ == '__main__':
    main()
