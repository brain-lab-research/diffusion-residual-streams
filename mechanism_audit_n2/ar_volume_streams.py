"""AR n=3/n=4 volume confirmation with a common-factorization n=2 bridge.

python -m mechanism_audit_n2.ar_volume_streams --stage full --gpu 5
python -m mechanism_audit_n2.ar_volume_streams --stage render
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

from .ar_volume import POSITION_BINS, batches_ar, split_for
from .engine import ROOT, aggregate, head_stats, load_model, summarize_tokens
from .factors import QUARTILES
from .odds_volume import ERROR_NAMES, OddsVolumeHooks, shifted_gains
from .primary import load_pack, save_pack, write_json

OUTPUT = ROOT / 'ar_volume_streams'
MAIN_CONDITIONS = ('dynamic', 'volume_global', 'volume_context_quartile', 'volume_position')
FULL_CONDITIONS = (*MAIN_CONDITIONS, 'volume_Q1', 'volume_Q2', 'volume_Q3', 'volume_Q4',
                   'volume_attention', 'volume_mlp')


def sources():
    names = ('ar_volume_streams.py', 'odds_volume.py', 'ar_volume.py', 'engine.py',
             'factors.py', 'primary.py', 'safety.py')
    return {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}


def model_config():
    config = json.loads(Path(__file__).with_name('config.json').read_text())
    for n, directory in ((3, 'ar36_lit_s3'), (4, 'ar36_lit')):
        config['models']['ar_n' + str(n)] = {'arm': 'mhc', 'streams': n, 'causal': True,
            'checkpoint': str(ROOT / 'runs_pt' / directory / 'ckpt_53399.pt')}
    return config


def prepare(config):
    OUTPUT.mkdir(exist_ok=True)
    metadata = json.loads(Path(config['data_meta']).read_text())
    assert metadata['n_val_blocks'] > max(config['splits']['confirmation'])
    assert not set(config['splits']['calibration']) & set(config['splits']['confirmation'])
    assert len(config['splits']['calibration']) == 64 and len(config['splits']['confirmation']) == 1024
    previous = json.loads((ROOT / 'ar_volume_n2' / 'protocol.json').read_text())
    for key in ('calibration_blocks', 'test_blocks'):
        expected = config['splits']['calibration' if key == 'calibration_blocks' else 'confirmation']
        assert previous[key] == expected
    protocol = {'objective': 'Uncorrupted AR input; logits[:, :-1] predict ids[:, 1:].',
        'factorization': 'Physical V=sum_i(g_i), q=softmax(write_logits)=normalized gain odds; solve sum_i 2sigmoid(z_i+c)=Vprime, preserving q.',
        'root_solver': 'Float64 analytical bracket and 40 bisections in a fused GPU kernel; no gains clipping.',
        'intervention': 'Only volume changes; current live read and residual mixer exact; relative write odds retained.',
        'source_positions': [0, 1022], 'final_source': 'Position 1023 has no next-token target and remains untouched.',
        'conditions': {'2': MAIN_CONDITIONS, '3': FULL_CONDITIONS, '4': FULL_CONDITIONS},
        'models': {name: config['models'][name] for name in ('ar_n2', 'ar_n3', 'ar_n4', 'ar_baseline')},
        'calibration_blocks': config['splits']['calibration'], 'test_blocks': config['splits']['confirmation'],
        'dataset': config['dataset'], 'metadata': metadata, 'position_bins': POSITION_BINS,
        'depth_quartiles': QUARTILES, 'bootstrap_repeats': config['bootstrap_repeats'],
        'bootstrap_seed': config['bootstrap_seed'], 'batch_size': config['batch_size'],
        'head_chunk': config['head_chunk'], 'computation_sha256': sources(),
        'frozen_source_sha256': config['source_files'],
        'n2_bridge': 'Fresh intact/global/context-quartile/exact-position n=2 under the same odds factorization; prior bounded binary report remains unchanged.',
        'baseline_reuse': 'Same exact held-out inputs and original frozen AR baseline; previous NPZ checksum sealed below.',
        'baseline_pack_sha256': hashlib.sha256((ROOT / 'ar_volume_n2/effects/baseline.npz').read_bytes()).hexdigest(),
        'scope': 'Inference only on existing final checkpoints; no training or new probes.'}
    protocol = json.loads(json.dumps(protocol))
    path = OUTPUT / 'protocol.json'
    if path.exists(): assert json.loads(path.read_text()) == protocol
    else: write_json(path, protocol)
    for name in ('calibration', 'confirmation'):
        split = split_for(config, name)
        path = OUTPUT / ('split_' + name + '.npz')
        if not path.exists(): np.savez_compressed(path, **split)
        else: np.testing.assert_array_equal(load_pack(path)['ids'], split['ids'])
    return protocol


def calibration(engine, config, reservation):
    output = OUTPUT / ('n' + str(engine.streams)); output.mkdir(exist_ok=True)
    (output / 'effects').mkdir(exist_ok=True)
    split = split_for(config, 'calibration')
    path = output / 'calibration.npz'
    if path.exists():
        saved = load_pack(path); assert str(saved['data_sha256']) == split['data_sha256']
        volume = saved['volume']
    else:
        collected = []
        with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
            for _, _, ids, valid in batches_ar(split, config['batch_size']):
                reservation.check(); local = []
                def capture(index, before, after, routes, routed, delta):
                    local.append(routes[:, :-1, engine.streams:2 * engine.streams].double().sum(-1).cpu().numpy())
                engine.run(ids, valid, callback=capture); collected.append(np.stack(local))
        volume = np.concatenate(collected, axis=1)
        np.savez_compressed(path, volume=volume, block_ids=split['block_ids'], data_sha256=split['data_sha256'])
    assert volume.dtype == np.float64 and volume.shape == (32, 64, 1023)
    means = {'global': volume.mean((1, 2), dtype=np.float64)[:, None],
        'position': volume.mean(1, dtype=np.float64)[..., None],
        'context_quartile': np.stack([volume[..., lo:hi].mean((1, 2), dtype=np.float64)
                                     for lo, hi in POSITION_BINS], axis=1)[..., None]}
    assert all(np.all((v > 0) & (v < 2 * engine.streams)) for v in means.values())
    write_json(output / 'calibration_means.json', {'means': {k: v.tolist() for k, v in means.items()},
        'dtype': 'float64', 'tokens': 64 * 1023, 'streams': engine.streams,
        'volume_mean': volume.mean((1, 2), dtype=np.float64).tolist(),
        'volume_sd': volume.std((1, 2), dtype=np.float64).tolist(), 'data_sha256': split['data_sha256']})
    print('CALIBRATION', engine.name, 'COMPLETE', flush=True)
    return means


def solver_check(reservation):
    result = {}
    generator = torch.Generator(device='cuda').manual_seed(130916)
    for n in (2, 3, 4):
        reservation.check()
        z = torch.randn((8, 129, n), generator=generator, device='cuda') * 12
        v = torch.rand((8, 129, 1), generator=generator, device='cuda', dtype=torch.float64) * (2 * n - 2e-5) + 1e-5
        v[:, 0] = 0.; v[:, 1] = 2 * n
        changed, shift = shifted_gains(z, v, torch.float32)
        error = float((changed.double().sum(-1, keepdim=True) - v).abs().max())
        lo = torch.full_like(v, -200); hi = torch.full_like(v, 200)
        for _ in range(90):
            mid = (lo + hi) / 2
            total = (2 * (z.double() + mid).sigmoid()).sum(-1, keepdim=True)
            lo = torch.where(total < v, mid, lo); hi = torch.where(total < v, hi, mid)
        reference = (2 * (z.double() + (lo + hi) / 2).sigmoid()).float()
        reference = torch.where(v <= 0, torch.zeros_like(reference),
                                torch.where(v >= 2 * n, torch.full_like(reference, 2), reference))
        reference_error = float((changed - reference).abs().max())
        permutation = torch.arange(n - 1, -1, -1, device='cuda')
        permuted, _ = shifted_gains(z[..., permutation], v, torch.float32)
        perm_error = float((changed[..., permutation] - permuted).abs().max())
        interior = (v > 0) & (v < 2 * n)
        q_error = float(torch.where(interior, ((z.double() + shift).softmax(-1) - z.double().softmax(-1)).abs(), 0).max())
        assert error <= 1e-6 and reference_error <= 3e-7 and perm_error <= 3e-7 and q_error <= 1e-12
        assert bool(((changed >= 0) & (changed <= 2)).all())
        result[str(n)] = {'volume_error': error, 'torch_float64_reference_error': reference_error,
                         'permutation_equivariance_error': perm_error, 'destination_error': q_error}
    write_json(OUTPUT / 'solver_checks.json', result)
    print('FUSED SOLVER CHECKS PASSED', result, flush=True)


def unit_check(engine, hooks, config, reservation):
    _, _, ids, valid = next(batches_ar(split_for(config, 'confirmation'), 1))
    result = {'original_forward_error': {}, 'future_prefix_error': {}, 'shifted_ce_error': {},
              'untouched_outputs_exact': True, 'unselected_gains_exact': True, 'maximum_errors': dict.fromkeys(ERROR_NAMES, 0.)}
    assert all(b.attn.causal for b in engine.model.blocks)
    with torch.inference_mode():
        for bf16 in (False, True):
            reservation.check(); name = 'bf16' if bf16 else 'fp32'
            with torch.autocast('cuda', torch.bfloat16, enabled=bf16):
                hooks.configure_condition(valid, 'dynamic')
                original_logits = engine.model(ids)
                state, _ = engine.run(ids, valid)
                error = float((original_logits.float() - engine.original_logits(state).float()).abs().max())
                assert error == 0; result['original_forward_error'][name] = error
                altered = ids.clone(); altered[:, 512:] = ids[:, 512:].roll(1, 1)
                future, _ = engine.run(altered, valid)
                prefix_error = float((future[:, :512] - state[:, :512]).abs().max())
                assert prefix_error == 0; result['future_prefix_error'][name] = prefix_error
                pos = torch.tensor([0, 255, 511, 1022], device='cuda'); labels = ids[:, 1:][:, pos].flatten()
                subset = state.mean(-2)[:, pos].flatten(0, 1)
                stats = head_stats(engine.model, subset[None], labels, 4)[0][0, :, 0]
                logits = engine.model.head(engine.model.norm(subset)).float()
                expected = F.cross_entropy(logits, labels, reduction='none').cpu().numpy()
                result['shifted_ce_error'][name] = float(np.max(np.abs(stats - expected)))
                np.testing.assert_allclose(stats, expected, atol=2e-6, rtol=1e-6)
                initial = engine.initial(ids); output = engine.interfaces[0].router(initial)
                selected = valid.clone(); selected[:, :8] = False
                for condition in MAIN_CONDITIONS[1:]:
                    hooks.configure_condition(selected, condition)
                    changed = hooks.transform(0, engine.interfaces[0].router, initial, output)
                    assert all(torch.equal(output[i], changed[i]) for i in (0, 2, 3))
                    assert torch.equal(output[1][~selected], changed[1][~selected])
                    for key, value in zip(ERROR_NAMES, hooks.maximum_errors.cpu().tolist()):
                        result['maximum_errors'][key] = max(result['maximum_errors'][key], value)
                hooks.configure_condition(valid, 'dynamic')
                del original_logits, state, future
    assert max(result['maximum_errors'].values()) <= 1e-6
    write_json(OUTPUT / ('n' + str(engine.streams)) / 'unit_checks.json', result)
    print('UNIT CHECKS', engine.name, 'PASSED', flush=True)


def evaluate(engine, hooks, config, reservation, condition):
    output = OUTPUT / ('n' + str(engine.streams))
    path = output / 'effects' / (condition + '.npz')
    split = split_for(config, 'confirmation')
    if path.exists():
        saved = load_pack(path)
        assert str(saved['data_sha256']) == split['data_sha256']
        np.testing.assert_array_equal(saved['block_ids'], split['block_ids'])
        print('REUSE', engine.name, condition, flush=True); return
    collected = []; maxima = np.zeros(len(ERROR_NAMES)); started = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        for begin, end, ids, valid in batches_ar(split, config['batch_size']):
            reservation.check(); hooks.configure_condition(valid, condition)
            state, _ = engine.run(ids, valid)
            states = state[:, :-1].mean(-2).flatten(0, 1).unsqueeze(0)
            collected.append(head_stats(engine.model, states, ids[:, 1:].flatten(), config['head_chunk'])[0][0])
            maxima = np.maximum(maxima, hooks.maximum_errors.cpu().numpy())
            if end % 128 == 0: print('PROGRESS', engine.name, condition, end, '/1024', round(time.monotonic() - started, 1), 'seconds', flush=True)
    assert maxima.max() <= 1e-6
    pack = summarize_tokens(split, np.concatenate(collected)); pack['data_sha256'] = split['data_sha256']
    if engine.streams == 2 and condition == 'dynamic':
        previous = load_pack(ROOT / 'ar_volume_n2/effects/dynamic.npz')
        np.testing.assert_array_equal(pack['token_stats'], previous['token_stats'])
        np.testing.assert_array_equal(pack['seq_ce_sum'], previous['seq_ce_sum'])
    save_pack(path.with_suffix('.tmp.npz'), pack); path.with_suffix('.tmp.npz').replace(path)
    write_json(output / (condition + '_execution.json'), {'condition': condition, 'model': engine.name,
        'checkpoint_step': engine.checkpoint_step, 'streams': engine.streams,
        'seconds': time.monotonic() - started, 'maximum_errors': dict(zip(ERROR_NAMES, maxima.tolist())),
        'metrics': aggregate(pack)})
    print('COMPLETE', engine.name, condition, aggregate(pack), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=('full', 'render'), default='full')
    parser.add_argument('--gpu', default='5'); args = parser.parse_args(); config = model_config()
    if args.stage == 'render':
        from .ar_volume_streams_report import render
        render(config); return
    protocol = prepare(config)
    for path, expected in config['source_files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    from .safety import GPUReservation
    reservation = GPUReservation(args.gpu, config['gpu_policy']); torch.set_num_threads(4)
    write_json(OUTPUT / 'execution.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
        'gpu': args.gpu, 'idle_observations': reservation.observations, 'torch': torch.__version__})
    solver_check(reservation)
    for n in (3, 4, 2):
        engine = load_model(config, 'ar_n' + str(n)); assert engine.checkpoint_step == 53399
        means = calibration(engine, config, reservation)
        hooks = OddsVolumeHooks(engine, means, POSITION_BINS, QUARTILES)
        unit_check(engine, hooks, config, reservation)
        for condition in (MAIN_CONDITIONS if n == 2 else FULL_CONDITIONS):
            evaluate(engine, hooks, config, reservation, condition)
        hooks.close(); del hooks, engine; torch.cuda.empty_cache()
    assert sources() == protocol['computation_sha256']
    assert hashlib.sha256((ROOT / 'ar_volume_n2/effects/baseline.npz').read_bytes()).hexdigest() == protocol['baseline_pack_sha256']
    write_json(OUTPUT / 'completed.json', {'finished_utc': datetime.now(timezone.utc).isoformat(),
               'computation_sources_unchanged': True, 'conditions': 24})
    from .ar_volume_streams_report import render
    render(config); print('AR VOLUME STREAMS COMPLETE', flush=True)


if __name__ == '__main__': main()
