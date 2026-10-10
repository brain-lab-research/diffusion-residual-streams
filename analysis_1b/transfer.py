"""Targeted source-to-target patching pilot for the frozen 1B checkpoint."""
import argparse
import json

import numpy as np
import torch

from audit import CONFIG, ROOT, load_model, make_split, save_json
from mechanism_audit_n2.engine import training

DEPTHS = (3, 15, 31, 47, 63)  # after m1, m7, m15, m23, m31
MIN_DROP = 0.5
SEQUENCES = 64
CONDITIONS = {'source_both': ('source', (0, 1)),
              'target_s0': ('target', (0,)),
              'target_s1': ('target', (1,)),
              'target_both': ('target', (0, 1))}


def target_logp(engine, state, ids, position):
    picked = state[:, position].mean(-2)
    logits = engine.model.head(engine.model.norm(picked)).float()
    return float(logits.log_softmax(-1)[0, int(ids[0, position])])


def choose_targets(engine, final_state, ids, mask, number=4):
    positions = torch.nonzero(mask[0], as_tuple=True)[0]
    hidden = final_state[0, positions].mean(-2)
    scores = []
    for begin in range(0, len(positions), 128):
        local = positions[begin:begin+128]
        logits = engine.model.head(engine.model.norm(hidden[begin:begin+128])).float()
        logp = logits.log_softmax(-1).gather(1, ids[0, local, None]).squeeze(-1)
        correct = logits.argmax(-1).eq(ids[0, local])
        scores.append(torch.where(correct, logp, torch.full_like(logp, -1e9)))
    score = torch.cat(scores)
    good = torch.nonzero(score > -1e8, as_tuple=True)[0]
    if not len(good):
        return []
    top = good[score[good].topk(min(number, len(good))).indices]
    return [int(positions[i]) for i in top]


def source_pool(mask, targets, block_id):
    visible = np.flatnonzero((~mask[0]).cpu().numpy())
    rng = np.random.default_rng(np.random.SeedSequence([130914, int(block_id)]))
    sources = set()
    for target in targets:
        order = np.argsort(np.abs(visible-target))
        sources.update(map(int, visible[order[:6]]))
        rest = visible[order[6:]]
        if len(rest):
            sources.update(map(int, rng.choice(rest, size=min(2, len(rest)), replace=False)))
    return sorted(sources)


def run(start=0, end=SEQUENCES, prefix='transfer'):
    torch.set_num_threads(4)
    engine = load_model(CONFIG, 'full_n2')
    split = make_split(CONFIG, 'test', .5)
    assert 0 <= start < end <= len(split['ids'])
    raw_path = ROOT/f'{prefix}_raw.json'
    records = json.loads(raw_path.read_text())['records'] if raw_path.exists() else []
    assert [row['block'] for row in records] == list(map(int, split['block_ids'][start:start+len(records)]))
    # One selected source-target pair per sequence limits dependence and cost.
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        for sequence in range(start+len(records), end):
            block_id = int(split['block_ids'][sequence])
            ids = torch.from_numpy(split['ids'][sequence:sequence+1]).to('cuda')
            mask = torch.from_numpy(split['mask'][sequence:sequence+1]).to('cuda')
            corrupted = ids.clone(); corrupted[mask] = training.MASK_ID
            clean_state, clean_cache = engine.run(corrupted, mask, cache=True)
            targets = choose_targets(engine, clean_state, ids, mask)
            if not targets:
                print('NO CORRECT TARGET', block_id, flush=True)
                records.append({'block': block_id, 'selected': False,
                                'reason': 'no_correct_target'})
                save_json(raw_path, {'checkpoint_step': 124000,
                          'depths': DEPTHS, 'min_drop': MIN_DROP, 'records': records})
                continue
            clean_logp = {position: target_logp(engine, clean_state, ids, position)
                          for position in targets}
            best = {position: (-float('inf'), None) for position in targets}
            for source in source_pool(mask, targets, block_id):
                variant = corrupted.clone(); variant[0, source] = training.MASK_ID
                variant_mask = mask.clone(); variant_mask[0, source] = True
                ablated, _ = engine.run(variant, variant_mask)
                for target in targets:
                    drop = clean_logp[target] - target_logp(engine, ablated, ids, target)
                    if drop > best[target][0]:
                        best[target] = (drop, source)
            target = max(targets, key=lambda position: best[position][0])
            drop, source = best[target]
            if drop < MIN_DROP:
                records.append({'block': block_id, 'selected': False, 'largest_drop': drop})
                print('NO PAIR', block_id, 'best_drop', round(drop, 3), flush=True)
                save_json(raw_path, {'checkpoint_step': 124000,
                          'depths': DEPTHS, 'min_drop': MIN_DROP, 'records': records})
                continue
            variant = corrupted.clone(); variant[0, source] = training.MASK_ID
            variant_mask = mask.clone(); variant_mask[0, source] = True
            ablated, ablated_cache = engine.run(variant, variant_mask, cache=True)
            ablated_logp = target_logp(engine, ablated, ids, target)
            record = {'block': block_id, 'selected': True, 'target': target, 'source': source,
                      'distance': source-target, 'clean_logp': clean_logp[target],
                      'ablated_logp': ablated_logp, 'drop': clean_logp[target]-ablated_logp,
                      'recovery': {name: [] for name in CONDITIONS}}
            for depth in DEPTHS:
                for name, (place, streams) in CONDITIONS.items():
                    position = source if place == 'source' else target
                    patched = ablated_cache['states'][depth].clone()
                    for stream in streams:
                        patched[0, position, stream] = clean_cache['states'][depth][0, position, stream]
                    final, _ = engine.run(variant, variant_mask, initial_state=patched,
                                          start_after=depth)
                    logp = target_logp(engine, final, ids, target)
                    record['recovery'][name].append((logp-ablated_logp)/record['drop'])
            records.append(record)
            save_json(raw_path, {'checkpoint_step': 124000,
                         'depths': DEPTHS, 'min_drop': MIN_DROP, 'records': records})
            print('TRANSFER', sequence-start+1, '/', end-start, 'drop', round(drop, 3),
                  'source', source, 'target', target, flush=True)
    kept = [row for row in records if row['selected']]
    summary = {'checkpoint_step': 124000, 'sequence_range': [start, end],
               'sequences_screened': end-start,
               'pairs_kept': len(kept), 'min_drop': MIN_DROP,
               'depths': DEPTHS, 'curves': {}}
    if kept:
        rng = np.random.default_rng(130914)
        draws = rng.integers(len(kept), size=(5000, len(kept)))
        for name in CONDITIONS:
            values = np.asarray([row['recovery'][name] for row in kept])
            boot = values[draws].mean(1)
            summary['curves'][name] = {'mean': values.mean(0).tolist(),
                                       'ci95_low': np.quantile(boot, .025, axis=0).tolist(),
                                       'ci95_high': np.quantile(boot, .975, axis=0).tolist()}
        difference = np.asarray([row['recovery']['target_s0'] for row in kept]) - np.asarray(
            [row['recovery']['target_s1'] for row in kept])
        boot = difference[draws].mean(1)
        summary['target_s0_minus_s1'] = {
            'mean': difference.mean(0).tolist(),
            'ci95_low': np.quantile(boot, .025, axis=0).tolist(),
            'ci95_high': np.quantile(boot, .975, axis=0).tolist()}
    save_json(ROOT/f'{prefix}.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=SEQUENCES)
    parser.add_argument('--prefix', default='transfer')
    args = parser.parse_args()
    run(args.start, args.end, args.prefix)
