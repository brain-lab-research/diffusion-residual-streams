"""Confirm factor-specific masked routing effects; no probes or new models.

python -m mechanism_audit_n2.factors --stage read_mixer --gpu 0
python -m mechanism_audit_n2.factors --stage full --coordinate bounded --gpu 0
python -m mechanism_audit_n2.factors --stage render
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .engine import aggregate, batches, load_model, make_split, summarize_tokens
from .primary import load_pack, pack_path, save_pack, write_json

FACTORS = ('read', 'volume', 'destination', 'mixer')
QUARTILES = {'Q1': list(range(0, 8)), 'Q2': list(range(8, 16)),
             'Q3': list(range(16, 24)), 'Q4': list(range(24, 32))}
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'router_factor_confirmation'


def factor_values(routes, coordinate):
    """Pure float64 analysis of original intact calibration coefficients."""
    r = np.asarray(routes, dtype=np.float64)
    gains = r[..., 2:4]
    total = gains.sum(-1, keepdims=True)
    if coordinate == 'bounded':
        lower = np.maximum(total - 2, 0); upper = np.minimum(total, 2)
        width = upper - lower
        destination = np.divide(gains - lower, width, out=np.full_like(gains, .5), where=width > 1e-12)
        volume = total
    elif coordinate == 'raw':
        destination = np.divide(gains, total, out=np.full_like(gains, .5), where=total > 1e-12)
        volume = total
    elif coordinate == 'logit':
        p = np.clip(gains / 2, 1e-7, 1 - 1e-7)
        logits = np.log(p) - np.log1p(-p)
        volume = logits.mean(-1, keepdims=True)
        destination = (logits[..., 1:2] - logits[..., 0:1]) / 2
    else:
        raise ValueError(coordinate)
    return {'read': r[..., :2], 'volume': volume, 'destination': destination, 'mixer': r[..., 4:]}


def mean_values(routes, coordinate):
    values = factor_values(routes, coordinate)
    means = {key: value.mean(1, dtype=np.float64) for key, value in values.items()}
    if coordinate in ['bounded', 'raw']:
        means['destination'] /= means['destination'].sum(-1, keepdims=True)
        assert np.all(means['destination'] >= 0) and np.all(means['destination'] <= 1)
    means['mixer'] /= means['mixer'].sum(-1, keepdims=True)
    assert np.all((means['read'] >= 0) & (means['read'] <= 1))
    assert np.allclose(means['mixer'].sum(-1), 1, rtol=0, atol=1e-12)
    return means


def gpu_factors(gains, coordinate):
    g = gains.double()
    total = g.sum(-1, keepdim=True)
    if coordinate == 'bounded':
        lower = (total - 2).clamp_min(0); upper = total.clamp_max(2)
        width = upper - lower
        destination = torch.where(width > 1e-12, (g - lower) / width.clamp_min(1e-12), torch.full_like(g, .5))
        return total, destination
    if coordinate == 'raw':
        return total, torch.where(total > 1e-12, g / total.clamp_min(1e-12), torch.full_like(g, .5))
    p = (g / 2).clamp(1e-7, 1 - 1e-7)
    logits = p.log() - torch.log1p(-p)
    return logits.mean(-1, keepdim=True), (logits[..., 1:2] - logits[..., 0:1]) / 2


def reconstruct(volume, destination, coordinate, dtype):
    if coordinate == 'bounded':
        lower = (volume - 2).clamp_min(0); upper = volume.clamp_max(2)
        value = lower + (upper - lower) * destination
    elif coordinate == 'raw':
        value = volume * destination
    else:
        value = 2 * torch.sigmoid(torch.cat([volume - destination, volume + destination], -1))
    return value.to(dtype)


class FactorHooks:
    """Outputs not intervened on are live outputs of the current router."""
    def __init__(self, engine, means, coordinate):
        self.engine, self.coordinate = engine, coordinate
        self.means = {key: torch.from_numpy(value.copy()).to('cuda') for key, value in means.items()}
        self.mask = None; self.fixed = frozenset(); self.layers = set(range(32))
        self.bound_violation = torch.zeros((), device='cuda', dtype=torch.float64)
        self.handles = []
        for index, interface in enumerate(engine.interfaces):
            def hook(module, args, outputs, layer=index):
                if not self.fixed or layer not in self.layers:
                    return outputs
                return self.transform(layer, module, args[0], outputs)
            self.handles.append(interface.router.register_forward_hook(hook))

    def transform(self, layer, module, state, outputs):
        read, gains, probabilities, mixer = outputs
        select = self.mask[..., None]
        if 'read' in self.fixed:
            read = torch.where(select, self.means['read'][layer].to(read.dtype), read)
        if {'volume', 'destination'} & self.fixed:
            volume, destination = gpu_factors(gains, self.coordinate)
            if 'volume' in self.fixed:
                volume = torch.where(select, self.means['volume'][layer], volume)
            if 'destination' in self.fixed:
                destination = torch.where(select, self.means['destination'][layer], destination)
            changed = reconstruct(volume, destination, self.coordinate, gains.dtype)
            gains = torch.where(select, changed, gains)
        if 'mixer' in self.fixed:
            probabilities = torch.where(select, self.means['mixer'][layer].to(probabilities.dtype), probabilities)
            replacement = torch.einsum('btp,pij->btij', probabilities,
                                       module.permutations.to(probabilities.dtype)).to(state.dtype)
            mixer = torch.where(select[..., None], replacement, mixer)
        violation = torch.stack([(-read).clamp_min(0).double().max(), (read - 1).clamp_min(0).double().max(),
            (-gains).clamp_min(0).double().max(), (gains - 2).clamp_min(0).double().max(),
            (-probabilities).clamp_min(0).double().max(),
            (probabilities.double().sum(-1) - 1).abs().max()]).max()
        self.bound_violation.copy_(torch.maximum(self.bound_violation, violation))
        return read, gains, probabilities, mixer

    def configure(self, mask, fixed, layers=None):
        self.mask = mask; self.fixed = frozenset(fixed)
        self.layers = set(range(32)) if layers is None else set(layers)
        self.bound_violation.zero_()

    def close(self):
        for handle in self.handles:
            handle.remove()


def calibration(engine, config, ratio, coordinate, reservation):
    name = 'r' + str(round(ratio * 100))
    path = OUTPUT / 'calibration' / (name + '_original_outputs.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    split = make_split(config, 'calibration', ratio)
    if path.exists():
        captured = load_pack(path)
        assert str(captured['mask_sha256']) == split['mask_sha256']
        routes = captured['router']
    else:
        collected = []
        with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
            for _, _, _, _, corrupted, ids, mask in batches(split, config['batch_size']):
                reservation.check()
                local = []
                def capture(index, before, after, router, routed, delta):
                    local.append(router[mask].float().cpu().numpy())
                engine.run(corrupted, mask, callback=capture)
                collected.append(np.stack(local))
        routes = np.concatenate(collected, axis=1)
        np.savez_compressed(path, router=routes, block_ids=split['block_ids'], mask_sha256=split['mask_sha256'])
    means = mean_values(routes, coordinate)
    write_json(OUTPUT / 'calibration' / (name + '_' + coordinate + '_means.json'),
               {'coordinate': coordinate, 'means': {key: value.tolist() for key, value in means.items()},
                'mask_sha256': split['mask_sha256'], 'data_sha256': split['data_sha256'],
                'blocks': list(map(int, split['block_ids'])), 'accumulator': 'float64'})
    return means


def effect_path(ratio, condition):
    path = OUTPUT / 'effects' / ('r' + str(round(ratio * 100))) / (condition + '.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def evaluate(engine, hooks, split, fixed, config, reservation, layers=None):
    values = []; maximum_violation = 0.
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        for begin, end, _, _, corrupted, ids, mask in batches(split, config['batch_size']):
            reservation.check()
            hooks.configure(mask, fixed, layers)
            state, _ = engine.run(corrupted, mask)
            values.append(engine.final_stats(state, ids, mask, config['head_chunk']))
            maximum_violation = max(maximum_violation, float(hooks.bound_violation))
            if end % 256 == 0:
                print('EVALUATION', split['ratio'], ','.join(fixed) or 'intact', layers or 'all', end, '/', len(split['ids']), flush=True)
    if hooks.coordinate != 'raw':
        assert maximum_violation <= 1e-6, maximum_violation
    pack = summarize_tokens(split, np.concatenate(values))
    pack['maximum_sigmoid_simplex_violation'] = np.array(maximum_violation)
    return pack


def bootstrap_metrics(pack, intact, baseline, config):
    assert np.array_equal(pack['seq_mask_count'], intact['seq_mask_count'])
    assert np.array_equal(pack['block_ids'], baseline['block_ids'])
    counts = pack['seq_mask_count']; n = len(counts)
    generator = np.random.default_rng(config['bootstrap_seed'])
    ci_draws = {key: [] for key in ['delta_intact', 'delta_baseline', 'gain_lost']}
    difference = pack['seq_ce_sum'] - intact['seq_ce_sum']
    base_difference = pack['seq_ce_sum'] - baseline['seq_ce_sum']
    gain = baseline['seq_ce_sum'] - intact['seq_ce_sum']
    for start in range(0, config['bootstrap_repeats'], 250):
        draw = generator.integers(n, size=(min(250, config['bootstrap_repeats'] - start), n))
        denominator = counts[draw].sum(-1)
        loss = difference[draw].sum(-1); drawn_gain = gain[draw].sum(-1)
        ci_draws['delta_intact'].extend(loss / denominator)
        ci_draws['delta_baseline'].extend(base_difference[draw].sum(-1) / denominator)
        valid = np.abs(drawn_gain) > 1e-10
        ci_draws['gain_lost'].extend(loss[valid] / drawn_gain[valid])
    result = aggregate(pack)
    result.update(delta_ce_intact=float(difference.sum() / counts.sum()),
                  delta_ce_baseline=float(base_difference.sum() / counts.sum()),
                  fraction_gain_lost=float(difference.sum() / gain.sum()) if abs(gain.sum()) > 1e-10 else None,
                  intact_gain_ce=float(gain.sum() / counts.sum()))
    for key, value in ci_draws.items():
        result[key + '_ci95'] = list(map(float, np.quantile(value, [.025, .975]))) if value else None
    for metric in ['accuracy', 'target_logit', 'entropy']:
        result['delta_' + metric + '_intact'] = result[metric] - aggregate(intact)[metric]
    return result


def base_references(engine, config, ratio, reservation):
    split = make_split(config, 'confirmation', ratio)
    np.savez_compressed(OUTPUT / ('split_r' + str(round(ratio * 100)) + '.npz'),
                        ids=split['ids'], mask=split['mask'], block_ids=split['block_ids'])
    if ratio == .5:
        for name, active in [('intact', engine), ('baseline', None), ('prior_joint_mean', engine)]:
            path = effect_path(ratio, name)
            if path.exists():
                continue
            if active is None:
                active = load_model(config, 'baseline')
            replacement = None
            if name == 'prior_joint_mean':
                captured = load_pack(OUTPUT / 'calibration/r50_original_outputs.npz')['router']
                replacement = torch.from_numpy(captured.mean(axis=1, dtype=np.float64)).to('cuda')[:, None, None, :]
            values = []
            with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
                for _, _, _, _, corrupted, ids, mask in batches(split, config['batch_size']):
                    reservation.check()
                    if replacement is None:
                        state, _ = active.run(corrupted, mask)
                    else:
                        state, _ = active.run(corrupted, mask, condition='router_mean',
                                              context={'router': replacement})
                    values.append(active.final_stats(state, ids, mask, config['head_chunk']))
            save_pack(path, summarize_tokens(split, np.concatenate(values)))
            if name == 'baseline':
                del active
                torch.cuda.empty_cache()
    return split


def condition(engine, hooks, split, name, fixed, config, reservation, layers=None):
    path = effect_path(split['ratio'], name)
    if not path.exists():
        pack = evaluate(engine, hooks, split, fixed, config, reservation, layers)
        intact = load_pack(effect_path(split['ratio'], 'intact')) if name != 'intact' else None
        save_pack(path, pack, intact)
    pack = load_pack(path)
    result = bootstrap_metrics(pack, load_pack(effect_path(split['ratio'], 'intact')),
                               load_pack(effect_path(split['ratio'], 'baseline')), config)
    result.update(fixed=list(fixed), layers=list(range(32)) if layers is None else list(layers), ratio=split['ratio'])
    print('RESULT', name, json.dumps(result), flush=True)
    write_json(path.with_suffix('.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['read_mixer', 'full', 'render'], default='full')
    parser.add_argument('--coordinate', choices=['bounded', 'raw', 'logit'], default='bounded')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name('config.json').read_text())
    OUTPUT.mkdir(exist_ok=True)
    for file, expected in config['source_files'].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == expected, file
    if args.stage == 'render':
        from .factor_report import render
        render(config, OUTPUT)
        return
    from .safety import GPUReservation
    reservation = GPUReservation(args.gpu, config['gpu_policy'])
    torch.set_num_threads(4)
    write_json(OUTPUT / ('reservation_' + args.stage + '.json'),
               {'pid': __import__('os').getpid(), 'gpu': args.gpu, 'observations': reservation.observations})
    execution = {'checkpoint': config['models']['full_n2'], 'baseline': config['models']['baseline'],
                 'dataset': config['dataset'], 'splits': {key: config['splits'][key] for key in ['calibration', 'confirmation']},
                 'mask_seed': config['mask_seed'], 'bootstrap_seed': config['bootstrap_seed'],
                 'bootstrap_repeats': config['bootstrap_repeats'], 'ratios': [.1, .3, .5, .7, .9],
                 'coordinate': args.coordinate, 'selection_threshold': .5, 'quartiles': QUARTILES,
                 'source_files': config['source_files'], 'started': datetime.now().isoformat()}
    write_json(OUTPUT / ('config_' + args.stage + '.json'), execution)
    engine = load_model(config, 'full_n2')
    means = calibration(engine, config, .5, args.coordinate, reservation)
    split = base_references(engine, config, .5, reservation)
    hooks = FactorHooks(engine, means, args.coordinate)
    if args.stage == 'read_mixer':
        for factor in ['read', 'mixer']:
            condition(engine, hooks, split, 'mean_' + factor, [factor], config, reservation)
        hooks.close()
        return
    from .factor_protocol import run_full
    run_full(engine, hooks, split, config, args, reservation, OUTPUT)


if __name__ == '__main__':
    main()
