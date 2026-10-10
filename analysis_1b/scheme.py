"""Locate the first masked a0 payload in the 1B residual streams."""
import json

import numpy as np
import torch

from audit import CONFIG, ROOT, batches, load_model, make_split, paired_delta, save_json, sequence_ce


def run():
    torch.set_num_threads(4)
    engine = load_model(CONFIG, 'full_n2')
    split = make_split(CONFIG, 'test', .5)
    with np.load(ROOT/'intact.npz') as saved:
        intact = {key: saved[key] for key in saved.files}
    names = ('a0_masked_zero_s0', 'a0_masked_zero_s1', 'a0_masked_zero_both',
             'a0_zero_both_restore_s0_after_m0',
             'a0_zero_both_restore_s1_after_m0',
             'a0_zero_both_restore_both_after_m0')
    sums = {name: [] for name in names}
    counts = {name: [] for name in names}
    intact_local = []
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        for begin, end, _, _, corrupted, ids, mask in batches(split, CONFIG['batch_size']):
            initial = engine.initial(corrupted)
            clean_a0, *_ = engine.interface(0, initial, mask)
            clean_m0, *_ = engine.interface(1, clean_a0, mask)
            clean_final, _ = engine.run(corrupted, mask, initial_state=clean_m0, start_after=1)
            local, local_counts = sequence_ce(engine, clean_final, ids, mask)
            intact_local.append(local)

            for stream in (0, 1, 'both'):
                def zero_router(module, args, outputs):
                    read, gains, probabilities, mixer = outputs
                    changed = gains.clone()
                    for lane in ((0, 1) if stream == 'both' else (stream,)):
                        changed[..., lane] = torch.where(mask, torch.zeros_like(gains[..., lane]),
                                                         gains[..., lane])
                    return read, changed, probabilities, mixer
                handle = engine.interfaces[0].router.register_forward_hook(zero_router)
                try:
                    ablated_a0, *_ = engine.interface(0, initial, mask)
                finally:
                    handle.remove()
                ablated_m0, *_ = engine.interface(1, ablated_a0, mask)
                final, _ = engine.run(corrupted, mask, initial_state=ablated_m0, start_after=1)
                name = f'a0_masked_zero_{"both" if stream == "both" else "s"+str(stream)}'
                local, count = sequence_ce(engine, final, ids, mask)
                sums[name].append(local); counts[name].append(count)

                if stream == 'both':
                    for restored in (0, 1, 'both'):
                        patched = ablated_m0.clone()
                        for lane in ((0, 1) if restored == 'both' else (restored,)):
                            patched[..., lane, :] = torch.where(
                                mask[..., None], clean_m0[..., lane, :], patched[..., lane, :])
                        result, _ = engine.run(corrupted, mask, initial_state=patched,
                                               start_after=1)
                        name = f'a0_zero_both_restore_{"both" if restored == "both" else "s"+str(restored)}_after_m0'
                        local, count = sequence_ce(engine, result, ids, mask)
                        sums[name].append(local); counts[name].append(count)
            if end % 16 == 0:
                print('SCHEME', end, '/', len(split['ids']), flush=True)

    np.testing.assert_allclose(np.concatenate(intact_local), intact['seq_ce_sum'],
                               rtol=0, atol=1e-5)
    summary = {'checkpoint_step': 124000, 'test_sequences': len(split['ids']),
               'intact_ce': float(intact['seq_ce_sum'].sum()/intact['seq_mask_count'].sum()),
               'conditions': {}}
    for name in names:
        pack = {'seq_ce_sum': np.concatenate(sums[name]),
                'seq_mask_count': np.concatenate(counts[name]),
                'block_ids': split['block_ids']}
        np.savez_compressed(ROOT/(name+'.npz'), **pack)
        summary['conditions'][name] = paired_delta(pack, intact)
    deleted = sums['a0_masked_zero_both']
    for restored in ('s0', 's1', 'both'):
        left = np.concatenate(sums[f'a0_zero_both_restore_{restored}_after_m0'])
        baseline = intact['seq_ce_sum']
        denominator = (np.concatenate(deleted)-baseline).sum()
        summary['conditions'][f'a0_zero_both_restore_{restored}_after_m0']['rescue_fraction'] = float(
            1-(left-baseline).sum()/denominator)
    save_json(ROOT/'first_block_scheme.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    run()
