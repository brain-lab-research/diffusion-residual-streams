"""Frozen AR n=2 volume interventions; uncorrupted inputs and shifted targets.

python -m mechanism_audit_n2.ar_volume --stage full --gpu 0
python -m mechanism_audit_n2.ar_volume --stage render
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .engine import ROOT, aggregate, head_stats, load_model, summarize_tokens
from .factors import FactorHooks, QUARTILES, gpu_factors, reconstruct
from .primary import load_pack, save_pack, write_json

OUTPUT = ROOT / 'ar_volume_n2'
CONDITIONS = ('dynamic', 'volume_global', 'volume_context_quartile', 'volume_position',
              'volume_Q1', 'volume_Q2', 'volume_Q3', 'volume_Q4',
              'volume_attention', 'volume_mlp', 'baseline')
POSITION_BINS = ((0, 256), (256, 512), (512, 768), (768, 1023))


def sources():
    paths = [Path(__file__).with_name(n) for n in
             ('ar_volume.py', 'engine.py', 'factors.py', 'primary.py', 'safety.py')]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def split_for(config, name):
    blocks = np.asarray(config['splits'][name], dtype=np.int32)
    data = np.load(config['dataset'], mmap_mode='r')
    ids = np.asarray(data[blocks], dtype=np.int64).copy()
    assert ids.shape[1] == 1024
    # This mask selects source positions with a next-token target; NEVER corruption.
    return {'name': name, 'ids': ids, 'block_ids': blocks,
            'mask': np.ones((len(ids), 1023), dtype=bool),
            'data_sha256': hashlib.sha256(ids.tobytes()).hexdigest()}


def batches_ar(split, size):
    for start in range(0, len(split['ids']), size):
        end = min(start + size, len(split['ids']))
        ids = torch.from_numpy(split['ids'][start:end]).to('cuda')
        valid = torch.ones_like(ids, dtype=torch.bool)
        valid[:, -1] = False
        yield start, end, ids, valid


def prepare(config):
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / 'effects').mkdir(exist_ok=True)
    metadata = json.loads(Path(config['data_meta']).read_text())
    assert metadata['n_val_blocks'] > max(config['splits']['confirmation'])
    assert len(config['splits']['calibration']) == 64
    assert len(config['splits']['confirmation']) == 1024
    assert not set(config['splits']['calibration']) & set(config['splits']['confirmation'])
    protocol = {'objective': 'Teacher-forced next-token CE: logits[:, :-1], ids[:, 1:]; no input masking.',
        'intervention_positions': 'Source positions 0..1022; last source position has no scored target and is untouched.',
        'means': 'Intact independent calibration; token-weighted float64, per interface.',
        'factorization': 'V=g0+g1; L=max(0,V-2); U=min(2,V); q=(g-L)/(U-L); gprime=L(Vprime)+(U(Vprime)-L(Vprime))*q.',
        'other_factors': 'Current live read, bounded destination q, and residual mixer; no gain clipping.',
        'position_bins': POSITION_BINS, 'depth_quartiles': QUARTILES,
        'conditions': CONDITIONS, 'calibration_blocks': config['splits']['calibration'],
        'test_blocks': config['splits']['confirmation'],
        'models': {n: config['models'][n] for n in ('ar_n2', 'ar_baseline')},
        'dataset': config['dataset'], 'metadata': metadata,
        'bootstrap_seed': config['bootstrap_seed'], 'bootstrap_repeats': config['bootstrap_repeats'],
        'batch_size': config['batch_size'], 'head_chunk': config['head_chunk'],
        'computation_sha256': sources(), 'frozen_source_sha256': config['source_files'],
        'scope': 'Final existing AR n=2 and matched baseline; no training, probes, regressions or lenses.'}
    protocol = json.loads(json.dumps(protocol))
    path = OUTPUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Use a new directory for a changed protocol.'
    else:
        write_json(path, protocol)
    for name in ('calibration', 'confirmation'):
        split = split_for(config, name)
        path = OUTPUT / ('split_' + name + '.npz')
        if path.exists():
            saved = load_pack(path)
            np.testing.assert_array_equal(saved['ids'], split['ids'])
        else:
            np.savez_compressed(path, **split)
    return protocol


def calibrate(engine, config, reservation):
    split = split_for(config, 'calibration')
    path = OUTPUT / 'calibration.npz'
    if path.exists():
        saved = load_pack(path)
        assert str(saved['data_sha256']) == split['data_sha256']
        volume, ratios = saved['volume'], saved['mean_write_to_residual_norm']
    else:
        volumes, norms = [], []
        with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
            for _, _, ids, valid in batches_ar(split, config['batch_size']):
                reservation.check()
                local_volume, local_ratio = [], []
                def capture(index, before, after, routes, routed, delta):
                    v = routes[..., 2:4].double().sum(-1)
                    local_volume.append(v[:, :-1].cpu().numpy())
                    # For a doubly stochastic mixer, mean-stream write is V/2 * delta.
                    denominator = before.float().mean(-2).norm(dim=-1).clamp_min(1e-12)
                    ratio = .5 * v * delta.float().norm(dim=-1) / denominator
                    local_ratio.append(ratio[:, :-1].float().cpu().numpy())
                engine.run(ids, valid, callback=capture)
                volumes.append(np.stack(local_volume)); norms.append(np.stack(local_ratio))
        volume, ratios = np.concatenate(volumes, axis=1), np.concatenate(norms, axis=1)
        np.savez_compressed(path, volume=volume, mean_write_to_residual_norm=ratios,
                            data_sha256=split['data_sha256'], block_ids=split['block_ids'])
    assert volume.dtype == np.float64 and volume.shape == (32, 64, 1023)
    means = {'global': volume.mean((1, 2), dtype=np.float64)[:, None],
             'position': volume.mean(1, dtype=np.float64)[..., None],
             'context_quartile': np.stack([volume[..., lo:hi].mean((1, 2), dtype=np.float64)
                                          for lo, hi in POSITION_BINS], axis=1)[..., None]}
    assert all(np.all((v >= 0) & (v <= 4)) for v in means.values())
    summary = {'means': {k: v.tolist() for k, v in means.items()}, 'accumulator': 'float64',
        'tokens': 64 * 1023, 'data_sha256': split['data_sha256'],
        'volume_mean': volume.mean((1, 2), dtype=np.float64).tolist(),
        'volume_sd': volume.std((1, 2), dtype=np.float64).tolist(),
        'volume_quantiles': np.quantile(volume, [.05, .5, .95], axis=(1, 2)).tolist(),
        'position_bin_volume_mean': means['context_quartile'][..., 0].tolist(),
        'relative_mean_write_norm_median': np.median(ratios, axis=(1, 2)).tolist(),
        'relative_mean_write_norm_position_bins': np.stack([
            np.median(ratios[..., lo:hi], axis=(1, 2)) for lo, hi in POSITION_BINS], axis=1).tolist(),
        'exact_position_explained_variance': (1 - ((volume - means['position'][..., 0][:, None]) ** 2).mean((1, 2)) /
                                             np.maximum(volume.var((1, 2)), 1e-30)).tolist(),
        'descriptive_only': 'Calibration statistics; explained variance is a finite-calibration description, not a fitted probe or held-out predictive estimate.'}
    write_json(OUTPUT / 'calibration_summary.json', summary)
    print('AR CALIBRATION COMPLETE', flush=True)
    return means


class ARVolumeHooks(FactorHooks):
    def __init__(self, engine, means):
        super().__init__(engine, {'volume': means['global']}, 'bounded')
        self.profiles = {k: torch.from_numpy(v.copy()).to('cuda') for k, v in means.items()}
        self.policy = 'global'
        self.maximum_volume_error = torch.zeros((), device='cuda', dtype=torch.float64)
        self.maximum_q_error = torch.zeros((), device='cuda', dtype=torch.float64)

    def configure_condition(self, valid, condition):
        self.policy = condition.removeprefix('volume_')
        layers = None
        if self.policy in QUARTILES:
            layers = QUARTILES[self.policy]; self.policy = 'global'
        elif self.policy in ('attention', 'mlp'):
            layers = range(0 if self.policy == 'attention' else 1, 32, 2)
            self.policy = 'global'
        self.configure(valid, [] if condition == 'dynamic' else ['volume'], layers)
        self.maximum_volume_error.zero_(); self.maximum_q_error.zero_()

    def replacement(self, layer, length):
        if self.policy == 'global':
            return self.profiles['global'][layer]
        positions = torch.arange(length, device='cuda').clamp_max(1022)
        if self.policy == 'position':
            return self.profiles['position'][layer][positions][None]
        assert self.policy == 'context_quartile'
        return self.profiles['context_quartile'][layer][positions.div(256, rounding_mode='floor')][None]

    def transform(self, layer, module, state, outputs):
        read, gains, probabilities, mixer = outputs
        volume, q = gpu_factors(gains, 'bounded')
        wanted = torch.where(self.mask[..., None], self.replacement(layer, gains.shape[1]), volume)
        changed = torch.where(self.mask[..., None], reconstruct(wanted, q, 'bounded', gains.dtype), gains)
        actual_v, actual_q = gpu_factors(changed, 'bounded')
        self.maximum_volume_error.copy_(torch.maximum(self.maximum_volume_error, (actual_v - wanted).abs().max()))
        width = wanted.clamp_max(2) - (wanted - 2).clamp_min(0)
        q_error = torch.where(width > 1e-3, (actual_q - q).abs(), torch.zeros_like(q)).max()
        self.maximum_q_error.copy_(torch.maximum(self.maximum_q_error, q_error))
        violation = torch.stack([(-changed).clamp_min(0).double().max(),
                                  (changed - 2).clamp_min(0).double().max(),
                                  (actual_q.sum(-1) - 1).abs().max()]).max()
        self.bound_violation.copy_(torch.maximum(self.bound_violation, violation))
        return read, changed, probabilities, mixer


def unit_check(engine, hooks, config, reservation):
    _, _, ids, valid = next(batches_ar(split_for(config, 'confirmation'), 1))
    result = {'original_forward_error': {}, 'dynamic_hook_error': {}, 'future_prefix_error': {},
              'shifted_ce_error': {}, 'full_head_ce_rounding_error': {}, 'unchanged_outputs_exact': True,
              'unselected_gains_exact': True, 'maximum_volume_error': 0., 'maximum_q_error': 0.}
    assert all(b.attn.causal for b in engine.model.blocks)
    with torch.inference_mode():
        for dtype in (torch.float32, torch.bfloat16):
            reservation.check()
            with torch.autocast('cuda', torch.bfloat16, enabled=dtype == torch.bfloat16):
                hooks.configure_condition(valid, 'dynamic')
                reference = engine.model(ids)
                state, _ = engine.run(ids, valid)
                reconstructed = engine.original_logits(state)
                error = float((reference.float() - reconstructed.float()).abs().max())
                result['original_forward_error'][str(dtype)] = error
                assert error == 0
                handles = hooks.handles
                for handle in handles: handle.remove()
                hooks.handles = []
                unhooked, _ = engine.run(ids, valid)
                result['dynamic_hook_error'][str(dtype)] = float((unhooked - state).abs().max())
                assert torch.equal(unhooked, state)
                # Reinstall independent hooks after the identity comparison.
                for index, interface in enumerate(engine.interfaces):
                    def hook(module, args, outputs, layer=index):
                        return outputs if not hooks.fixed or layer not in hooks.layers else hooks.transform(layer, module, args[0], outputs)
                    hooks.handles.append(interface.router.register_forward_hook(hook))
                changed_ids = ids.clone(); changed_ids[:, 512:] = ids[:, 512:].roll(1, 1)
                changed_state, _ = engine.run(changed_ids, valid)
                result['future_prefix_error'][str(dtype)] = float((changed_state[:, :512] - state[:, :512]).abs().max())
                assert torch.equal(changed_state[:, :512], state[:, :512])
                positions = torch.tensor([0, 255, 511, 1022], device='cuda')
                labels = ids[:, 1:][:, positions].flatten()
                actual = head_stats(engine.model, state.mean(-2)[:, positions].flatten(0, 1)[None], labels, 4)[0][0, :, 0]
                full_expected = F.cross_entropy(reference[:, positions].float().flatten(0, 1), labels, reduction='none').cpu().numpy()
                # Identical head batch size avoids unrelated GEMM rounding differences.
                subset = unhooked.mean(-2)[:, positions].flatten(0, 1)
                subset_logits = engine.model.head(engine.model.norm(subset)).float()
                expected = F.cross_entropy(subset_logits, labels, reduction='none').cpu().numpy()
                result['full_head_ce_rounding_error'][str(dtype)] = float(np.max(np.abs(actual - full_expected)))
                result['shifted_ce_error'][str(dtype)] = float(np.max(np.abs(actual - expected)))
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-6)
                initial = engine.initial(ids)
                original = engine.interfaces[0].router(initial)
                selected = valid.clone(); selected[:, :8] = False
                for condition in ('volume_global', 'volume_context_quartile', 'volume_position'):
                    hooks.configure_condition(selected, condition)
                    changed = hooks.transform(0, engine.interfaces[0].router, initial, original)
                    assert all(torch.equal(original[i], changed[i]) for i in (0, 2, 3))
                    assert torch.equal(original[1][~selected], changed[1][~selected])
                    result['maximum_volume_error'] = max(result['maximum_volume_error'], float(hooks.maximum_volume_error))
                    result['maximum_q_error'] = max(result['maximum_q_error'], float(hooks.maximum_q_error))
                    assert float(hooks.bound_violation) <= 1e-6
                hooks.configure_condition(valid, 'dynamic')
                del reference, reconstructed, state, unhooked, changed_state
    assert result['maximum_volume_error'] < 1e-6 and result['maximum_q_error'] < 1e-4
    write_json(OUTPUT / 'unit_checks.json', result)
    print('AR UNIT CHECKS PASSED', result, flush=True)


def evaluate(engine, hooks, config, reservation, condition):
    split = split_for(config, 'confirmation')
    path = OUTPUT / 'effects' / (condition + '.npz')
    if path.exists():
        saved = load_pack(path)
        np.testing.assert_array_equal(saved['block_ids'], split['block_ids'])
        assert str(saved['data_sha256']) == split['data_sha256']
        print('REUSE', condition, aggregate(saved)['ce'], flush=True)
        return
    collected = []; started = time.monotonic()
    maxima = {'gain_simplex_bounds': 0., 'volume_reconstruction': 0., 'q_preservation': 0.}
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        for begin, end, ids, valid in batches_ar(split, config['batch_size']):
            reservation.check()
            if hooks is not None: hooks.configure_condition(valid, condition)
            state, _ = engine.run(ids, valid)
            # Source t predicts the untouched target token t+1.
            states = state[:, :-1].mean(-2).flatten(0, 1).unsqueeze(0)
            collected.append(head_stats(engine.model, states, ids[:, 1:].flatten(), config['head_chunk'])[0][0])
            if hooks is not None:
                maxima['gain_simplex_bounds'] = max(maxima['gain_simplex_bounds'], float(hooks.bound_violation))
                maxima['volume_reconstruction'] = max(maxima['volume_reconstruction'], float(hooks.maximum_volume_error))
                maxima['q_preservation'] = max(maxima['q_preservation'], float(hooks.maximum_q_error))
            if end % 128 == 0:
                print('PROGRESS', condition, end, '/1024', round(time.monotonic() - started, 1), 'seconds', flush=True)
    assert maxima['gain_simplex_bounds'] <= 1e-6 and maxima['volume_reconstruction'] <= 1e-6
    assert maxima['q_preservation'] <= 1e-4
    pack = summarize_tokens(split, np.concatenate(collected))
    pack['data_sha256'] = split['data_sha256']
    save_pack(path.with_suffix('.tmp.npz'), pack)
    path.with_suffix('.tmp.npz').replace(path)
    write_json(OUTPUT / (condition + '_execution.json'),
               {'condition': condition, 'model': engine.name, 'checkpoint_step': engine.checkpoint_step,
                'seconds': time.monotonic() - started, 'invariants': maxima, 'metrics': aggregate(pack)})
    print('COMPLETE', condition, aggregate(pack), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('full', 'render'), default='full')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name('config.json').read_text())
    if args.stage == 'render':
        from .ar_volume_report import render
        render(config); return
    protocol = prepare(config)
    for path, expected in config['source_files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    from .safety import GPUReservation
    reservation = GPUReservation(args.gpu, config['gpu_policy'])
    torch.set_num_threads(4)
    write_json(OUTPUT / 'execution.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
        'gpu': args.gpu, 'idle_observations': reservation.observations,
        'torch': torch.__version__, 'numpy': np.__version__})
    engine = load_model(config, 'ar_n2'); assert engine.checkpoint_step == 53399
    means = calibrate(engine, config, reservation)
    hooks = ARVolumeHooks(engine, means)
    unit_check(engine, hooks, config, reservation)
    for condition in CONDITIONS[:-1]: evaluate(engine, hooks, config, reservation, condition)
    hooks.close(); del hooks, engine; torch.cuda.empty_cache()
    engine = load_model(config, 'ar_baseline'); assert engine.checkpoint_step == 53399
    evaluate(engine, None, config, reservation, 'baseline')
    del engine; torch.cuda.empty_cache()
    assert sources() == protocol['computation_sha256']
    write_json(OUTPUT / 'completed.json', {'finished_utc': datetime.now(timezone.utc).isoformat(),
               'conditions': CONDITIONS, 'computation_sources_unchanged': True})
    from .ar_volume_report import render
    render(config)
    print('AR VOLUME STUDY COMPLETE', flush=True)


if __name__ == '__main__':
    main()
