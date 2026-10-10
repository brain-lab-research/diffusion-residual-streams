"""Test whether either 1B stream is nearly stationary between a8 and a16."""
import json

import numpy as np
import torch
import torch.nn.functional as F

from audit import CONFIG, ROOT, batches, load_model, make_split, paired_delta, save_json, sequence_ce

EARLY = 16  # after a8
MIDDLE = 32  # after a16


def run():
    torch.set_num_threads(4)
    engine = load_model(CONFIG, 'full_n2')
    split = make_split(CONFIG, 'test', .5)
    with np.load(ROOT/'intact.npz') as saved:
        intact = {key: saved[key] for key in saved.files}
    roles = ('masked', 'visible')
    names = [f'{role}_s{stream}' for role in roles for stream in (0, 1)]
    sums = {name: [] for name in names}
    counts = {name: [] for name in names}
    stats = {name: {'cos_sum': 0., 'ratio_sum': 0., 'rel_l2_sum': 0., 'n': 0}
             for name in names}
    intact_local = []
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        for begin, end, _, _, corrupted, ids, mask in batches(split, CONFIG['batch_size']):
            _, cache = engine.run(corrupted, mask, cache=True)
            early, middle = cache['states'][EARLY], cache['states'][MIDDLE]
            local_intact, _ = sequence_ce(engine, cache['states'][-1], ids, mask)
            intact_local.append(local_intact)
            for role in roles:
                selected = mask if role == 'masked' else ~mask
                for stream in (0, 1):
                    name = f'{role}_s{stream}'
                    first = early[..., stream, :].float()[selected]
                    second = middle[..., stream, :].float()[selected]
                    ref_norm = first.norm(dim=-1).clamp_min(1e-12)
                    stats[name]['cos_sum'] += float(F.cosine_similarity(first, second, dim=-1).sum())
                    stats[name]['ratio_sum'] += float((second.norm(dim=-1)/ref_norm).sum())
                    stats[name]['rel_l2_sum'] += float(((second-first).norm(dim=-1)/ref_norm).sum())
                    stats[name]['n'] += len(first)
                    patched = middle.clone()
                    patched[..., stream, :] = torch.where(
                        selected[..., None], early[..., stream, :], patched[..., stream, :])
                    final, _ = engine.run(corrupted, mask, initial_state=patched,
                                          start_after=MIDDLE)
                    local_sum, local_count = sequence_ce(engine, final, ids, mask)
                    sums[name].append(local_sum); counts[name].append(local_count)
            if end % 16 == 0:
                print('MIDDLE', end, '/', len(split['ids']), flush=True)
    np.testing.assert_allclose(np.concatenate(intact_local), intact['seq_ce_sum'],
                               rtol=0, atol=1e-5)
    summary = {'checkpoint_step': 124000, 'test_sequences': len(split['ids']),
               'earlier_interface': EARLY, 'later_interface': MIDDLE,
               'comparison': 'same position, after a8 versus after a16',
               'conditions': {}}
    for name in names:
        st = stats[name]
        pack = {'seq_ce_sum': np.concatenate(sums[name]),
                'seq_mask_count': np.concatenate(counts[name]),
                'block_ids': split['block_ids']}
        np.savez_compressed(ROOT/(f'middle_restore_a8_{name}.npz'), **pack)
        summary['conditions'][name] = {
            'positions': st['n'],
            'mean_cosine': st['cos_sum']/st['n'],
            'mean_norm_ratio': st['ratio_sum']/st['n'],
            'mean_relative_l2': st['rel_l2_sum']/st['n'],
            'restore_a8_delta_ce': paired_delta(pack, intact)}
    save_json(ROOT/'middle_preservation.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    run()
