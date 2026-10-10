"""Bounded confirmatory protocol, threshold selection and its limited extensions."""
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .engine import batches, load_model, make_split
from .factors import (FACTORS, QUARTILES, FactorHooks, base_references, calibration,
                      condition, effect_path, evaluate, gpu_factors)
from .primary import load_pack, save_pack, write_json


def equivalence_and_invariants(engine, hooks, split, config, reservation):
    begin, end, first, last, corrupted, ids, mask = next(batches(split, 1))
    result = {'precision': {}, 'maximum_nonintervened_factor_error': 0.,
              'maximum_visible_coefficient_error': 0., 'maximum_write_bound_error': 0.,
              'maximum_simplex_error': 0., 'coordinate': hooks.coordinate}
    for precision in ['fp32', 'bf16']:
        hooks.configure(mask, [])
        reservation.check()
        with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16, enabled=precision == 'bf16'):
            reference = engine.model(corrupted)
            state, cache = engine.run(corrupted, mask, cache=True)
            output = engine.original_logits(state)
            torch.testing.assert_close(output, reference, atol=0, rtol=0)
            result['precision'][precision] = {'maximum_logit_error': float((output.float() - reference.float()).abs().max())}
            before_states = [engine.initial(corrupted)] + cache['states'][:-1]
            for index, before in enumerate(before_states):
                module = engine.interfaces[index].router
                hooks.fixed = frozenset()
                original = module(before)
                original_volume, original_destination = gpu_factors(original[1], hooks.coordinate)
                for size in range(1, 5):
                    for subset in itertools.combinations(FACTORS, size):
                        hooks.configure(mask, subset)
                        changed = hooks.transform(index, module, before, original)
                        for old, new in zip(original, changed):
                            error = float((old[~mask].double() - new[~mask].double()).abs().max())
                            result['maximum_visible_coefficient_error'] = max(result['maximum_visible_coefficient_error'], error)
                            assert error == 0
                        if 'read' not in subset:
                            torch.testing.assert_close(changed[0], original[0], atol=0, rtol=0)
                        if 'mixer' not in subset:
                            torch.testing.assert_close(changed[2], original[2], atol=0, rtol=0)
                            torch.testing.assert_close(changed[3], original[3], atol=0, rtol=0)
                        v, d = gpu_factors(changed[1], hooks.coordinate)
                        for key, value, prior in [('volume', v, original_volume), ('destination', d, original_destination)]:
                            target = hooks.means[key][index].expand_as(value)[mask] if key in subset else prior[mask]
                            # At V=4/0 no free allocation exists: destination is unidentifiable there.
                            valid = torch.ones_like(value[..., 0][mask], dtype=torch.bool)
                            if key == 'destination' and hooks.coordinate == 'bounded':
                                visible_volume = v[mask][:, 0]
                                valid = (visible_volume > 1e-3) & (visible_volume < 4 - 1e-3)
                            if valid.any():
                                error = float((value[mask][valid] - target[valid]).abs().max())
                                result['maximum_nonintervened_factor_error'] = max(result['maximum_nonintervened_factor_error'], error)
                                torch.testing.assert_close(value[mask][valid], target[valid], atol=1e-4, rtol=2e-5)
                        for values, upper in [(changed[0], 1), (changed[1], 2), (changed[2], 1)]:
                            assert float(values.min()) >= 0 and float(values.max()) <= upper
                        simplex_error = float((changed[2].double().sum(-1) - 1).abs().max())
                        result['maximum_simplex_error'] = max(result['maximum_simplex_error'], simplex_error)
                        assert simplex_error < 1e-6
                hooks.configure(mask, [])
        del reference, output, state, cache, before_states
    # Cached primary references must match the current frozen model and mask policy.
    begin, end, first, last, corrupted, ids, mask = next(batches(split, config['batch_size']))
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        hooks.configure(mask, [])
        state, _ = engine.run(corrupted, mask)
        values = engine.final_stats(state, ids, mask, config['head_chunk'])
        cached = load_pack(effect_path(.5, 'intact'))['token_stats'][first:last]
        np.testing.assert_array_equal(values, cached)
    result['cached_intact_token_stats_exact'] = True
    result['destination_precision_note'] = 'Reconstructed gains cast back to router dtype. q invariant excludes free intervals <=1e-3, where finite gain precision cannot resolve q accurately; no gain clipping is used.'
    write_json(effect_path(.5, 'unit').with_suffix('.json'), result)
    print('FACTORIZATION UNIT PASSED', json.dumps(result), flush=True)
    return result


def interaction_contrast(names, name, config):
    intact = load_pack(effect_path(.5, 'intact'))
    joint = load_pack(effect_path(.5, name))
    singles = [load_pack(effect_path(.5, 'mean_' + factor)) for factor in names]
    effect = joint['seq_ce_sum'] - intact['seq_ce_sum']
    for pack in singles:
        effect -= pack['seq_ce_sum'] - intact['seq_ce_sum']
    count = intact['seq_mask_count']; n = len(count)
    rng = np.random.default_rng(config['bootstrap_seed'])
    values = []
    for start in range(0, config['bootstrap_repeats'], 250):
        draws = rng.integers(n, size=(min(250, config['bootstrap_repeats'] - start), n))
        values.extend(effect[draws].sum(-1) / count[draws].sum(-1))
    return {'joint_minus_sum_of_single_delta_ce': float(effect.sum() / count.sum()),
            'paired_ci95': list(map(float, np.quantile(values, [.025, .975]))),
            'interpretation': 'Positive interval establishes supra-additive CE damage for these mean interventions; it is not an additive decomposition of model ability.'}


def baseline_for_ratio(config, ratio, reservation):
    path = effect_path(ratio, 'baseline')
    if path.exists():
        return
    baseline = load_model(config, 'baseline')
    split = make_split(config, 'confirmation', ratio)
    values = []
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        for begin, end, _, _, corrupted, ids, mask in batches(split, config['batch_size']):
            reservation.check()
            state, _ = baseline.run(corrupted, mask)
            values.append(baseline.final_stats(state, ids, mask, config['head_chunk']))
    from .engine import summarize_tokens
    save_pack(path, summarize_tokens(split, np.concatenate(values)))
    del baseline; torch.cuda.empty_cache()


def run_full(engine, hooks, split, config, args, reservation, output):
    protocol = {'coordinate': args.coordinate, 'primary_ratio': .5,
        'primary_conditions': ['intact'] + ['mean_' + f for f in FACTORS] + ['only_' + f + '_dynamic' for f in FACTORS] + ['all_factor_mean', 'prior_joint_mean'],
        'selection_rule': 'Select the smallest fixed-factor set with point estimate >=50% gain loss at r=.5. Within the smallest cardinality choose the greatest observed loss; fixed lexicographic order breaks ties. If no singleton qualifies, evaluate the six prespecified pairs before using triples or all-mean.',
        'extension': 'Only the selected set: all five ratios with per-ratio separate calibration means; localize in four disjoint interface quartiles at the primary r=.5 only. No ratio-by-depth grid.',
        'quartiles': QUARTILES,
        'ci_scope': 'Nominal sequence-paired bootstrap CI conditional on trained models, masks, calibration means and selection; not training-seed variability or a new independent selection-validation split.'}
    write_json(output / 'protocol.json', protocol)
    unit_path = effect_path(.5, 'unit').with_suffix('.json')
    if not unit_path.exists():
        equivalence_and_invariants(engine, hooks, split, config, reservation)
    primary = {}
    for factor in FACTORS:
        primary['mean_' + factor] = condition(engine, hooks, split, 'mean_' + factor, [factor], config, reservation)
    for dynamic in FACTORS:
        fixed = [factor for factor in FACTORS if factor != dynamic]
        name = 'only_' + dynamic + '_dynamic'
        primary[name] = condition(engine, hooks, split, name, fixed, config, reservation)
    primary['all_factor_mean'] = condition(engine, hooks, split, 'all_factor_mean', FACTORS, config, reservation)
    primary['prior_joint_mean'] = condition(engine, hooks, split, 'prior_joint_mean', FACTORS, config, reservation)
    eligible = [(name, value) for name, value in primary.items() if name.startswith('mean_') and value['fraction_gain_lost'] >= .5]
    if not eligible:
        for pair in itertools.combinations(FACTORS, 2):
            name = 'mean_' + '_'.join(pair)
            primary[name] = condition(engine, hooks, split, name, pair, config, reservation)
        eligible = [(name, value) for name, value in primary.items()
                    if name.startswith('mean_') and len(value['fixed']) == 2 and value['fraction_gain_lost'] >= .5]
    if not eligible:
        eligible = [(name, value) for name, value in primary.items()
                    if name.startswith('only_') and value['fraction_gain_lost'] >= .5]
    if not eligible and primary['all_factor_mean']['fraction_gain_lost'] >= .5:
        eligible = [('all_factor_mean', primary['all_factor_mean'])]
    selected = sorted(eligible, key=lambda item: (-item[1]['fraction_gain_lost'], item[0]))[0] if eligible else None
    interactions = {'all_factor_mean': interaction_contrast(FACTORS, 'all_factor_mean', config)}
    if selected and len(selected[1]['fixed']) > 1:
        interactions['selected'] = interaction_contrast(selected[1]['fixed'], selected[0], config)
    selection = {'selected_condition': selected[0] if selected else None, 'selected_factors': selected[1]['fixed'] if selected else [],
                 'selection_metrics': selected[1] if selected else None,
                 'singletons_below50pct': all(primary['mean_' + factor]['fraction_gain_lost'] < .5 for factor in FACTORS),
                 'interaction_contrasts': interactions, 'rule': protocol['selection_rule']}
    write_json(output / 'primary.json', {'conditions': primary, 'selection': selection})
    hooks.close()
    extensions = {}
    if selected:
        fixed = selected[1]['fixed']
        for ratio in [.1, .3, .5, .7, .9]:
            reservation.check()
            if ratio != .5:
                baseline_for_ratio(config, ratio, reservation)
            current_split = base_references(engine, config, ratio, reservation)
            means = calibration(engine, config, ratio, args.coordinate, reservation)
            active = FactorHooks(engine, means, args.coordinate)
            if not effect_path(ratio, 'intact').exists():
                intact = evaluate(engine, active, current_split, [], config, reservation)
                save_pack(effect_path(ratio, 'intact'), intact)
            if ratio == .5 and not effect_path(ratio, 'selected_all_depths').exists():
                save_pack(effect_path(ratio, 'selected_all_depths'), load_pack(effect_path(.5, selected[0])))
            row = {'all_depths': condition(engine, active, current_split, 'selected_all_depths', fixed, config, reservation)}
            if ratio == .5:
                for quartile, layers in QUARTILES.items():
                    row[quartile] = condition(engine, active, current_split, 'selected_' + quartile, fixed, config, reservation, layers)
            active.close()
            extensions[str(ratio)] = row
            write_json(output / 'extensions.json', extensions)
    write_json(output / 'results.json', {'protocol': protocol, 'primary': primary, 'selection': selection,
                                       'extensions': extensions, 'unit': json.loads(unit_path.read_text())})
    del engine; torch.cuda.empty_cache()
    from .factor_report import render
    render(config, output)
    print('FACTOR CONFIRMATION COMPLETE', flush=True)
