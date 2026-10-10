"""Frozen-state tests for first-block routing and stream reuse in the 1B model."""
import json

import numpy as np
import torch

from audit import CONFIG, ROOT, batches, load_model, make_split, paired_delta, save_json, sequence_ce

MID = 32   # after a16; relative counterpart of a8 in the 16-block paper model
LATE = 49  # after m24; relative counterpart of m12


def patch_state(state, reference, selected, mode, stream, block_ids, depth):
    changed = state.clone()
    if mode == 'restore_early':
        changed[..., stream, :] = torch.where(
            selected[..., None], reference[..., stream, :], changed[..., stream, :])
    elif mode == 'shuffle':
        for b, block_id in enumerate(block_ids):
            positions = torch.nonzero(selected[b], as_tuple=True)[0]
            rng = np.random.default_rng(np.random.SeedSequence(
                [130923, int(block_id), depth, stream]))
            permutation = torch.from_numpy(rng.permutation(len(positions))).to(positions.device)
            changed[b, positions, stream, :] = state[b, positions[permutation], stream, :]
    elif mode == 'only':
        replacement = state[..., stream, :].unsqueeze(-2).expand_as(state)
        changed = torch.where(selected[..., None, None], replacement, state)
    else:
        raise ValueError(mode)
    return changed


def state_interventions():
    torch.set_num_threads(4)
    engine = load_model(CONFIG, 'full_n2')
    split = make_split(CONFIG, 'test', .5)
    with np.load(ROOT/'intact.npz') as saved:
        intact = {key: saved[key] for key in saved.files}
    names = [f'mid_{mode}_{role}_s{stream}'
             for mode in ('restore_early', 'shuffle')
             for role in ('masked', 'visible') for stream in (0, 1)]
    names += [f'late_only_{role}_s{stream}'
              for role in ('masked', 'visible') for stream in (0, 1)]
    names += [f'prehead_only_masked_s{stream}' for stream in (0, 1)]
    sums = {name: [] for name in names}
    counts = {name: [] for name in names}
    intact_local = []
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        for begin, end, _, _, corrupted, ids, mask in batches(split, CONFIG['batch_size']):
            _, cache = engine.run(corrupted, mask, cache=True)
            states = cache['states']
            local_intact, _ = sequence_ce(engine, states[-1], ids, mask)
            intact_local.append(local_intact)
            block_ids = split['block_ids'][begin:end]
            for name in names:
                if name.startswith('mid_'):
                    role = 'masked' if '_masked_' in name else 'visible'
                    mode = 'restore_early' if 'restore_early' in name else 'shuffle'
                    stream = int(name[-1])
                    selected = mask if role == 'masked' else ~mask
                    patched = patch_state(states[MID], states[1], selected, mode,
                                          stream, block_ids, MID)
                    result, _ = engine.run(corrupted, mask, initial_state=patched, start_after=MID)
                elif name.startswith('late_'):
                    role = 'masked' if '_masked_' in name else 'visible'
                    stream = int(name[-1]); selected = mask if role == 'masked' else ~mask
                    patched = patch_state(states[LATE], None, selected, 'only',
                                          stream, block_ids, LATE)
                    result, _ = engine.run(corrupted, mask, initial_state=patched, start_after=LATE)
                else:
                    stream = int(name[-1])
                    result = patch_state(states[-1], None, mask, 'only',
                                         stream, block_ids, 63)
                local_sum, local_count = sequence_ce(engine, result, ids, mask)
                sums[name].append(local_sum); counts[name].append(local_count)
            if end % 8 == 0:
                print('CAUSAL STATE', end, '/', len(split['ids']), flush=True)
    np.testing.assert_allclose(np.concatenate(intact_local), intact['seq_ce_sum'],
                               rtol=0, atol=1e-5)
    summary = {'checkpoint_step': 124000, 'test_sequences': len(split['ids']),
               'mid_interface': MID, 'late_interface': LATE, 'conditions': {}}
    for name in names:
        pack = {'seq_ce_sum': np.concatenate(sums[name]),
                'seq_mask_count': np.concatenate(counts[name]),
                'block_ids': split['block_ids']}
        np.savez_compressed(ROOT/(name+'.npz'), **pack)
        summary['conditions'][name] = paired_delta(pack, intact)
    save_json(ROOT/'state_interventions.json', summary)
    print(json.dumps(summary), flush=True)


def first_block():
    torch.set_num_threads(4)
    engine = load_model(CONFIG, 'full_n2')
    split = make_split(CONFIG, 'test', .5)
    with np.load(ROOT/'intact.npz') as saved:
        intact = {key: saved[key] for key in saved.files}
    names = [f'a0_{mode}_{role}' for mode in ('zero', 'swap')
             for role in ('masked', 'visible')]
    summary = {'checkpoint_step': 124000, 'test_sequences': len(split['ids']),
               'conditions': {}}
    for name in names:
        role = 'masked' if name.endswith('masked') else 'visible'
        mode = 'zero' if '_zero_' in name else 'swap'
        values, counts = [], []
        with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
            for begin, end, _, _, corrupted, ids, mask in batches(split, CONFIG['batch_size']):
                selected = mask if role == 'masked' else ~mask
                def hook(module, args, outputs):
                    read, gains, probabilities, mixer = outputs
                    changed = torch.zeros_like(gains) if mode == 'zero' else gains.flip(-1)
                    return read, torch.where(selected[..., None], changed, gains), probabilities, mixer
                handle = engine.interfaces[0].router.register_forward_hook(hook)
                try:
                    state, _ = engine.run(corrupted, mask)
                finally:
                    handle.remove()
                local_sum, local_count = sequence_ce(engine, state, ids, mask)
                values.append(local_sum); counts.append(local_count)
                if end % 32 == 0:
                    print('FIRST BLOCK', name, end, '/', len(split['ids']), flush=True)
        pack = {'seq_ce_sum': np.concatenate(values), 'seq_mask_count': np.concatenate(counts),
                'block_ids': split['block_ids']}
        np.savez_compressed(ROOT/(name+'.npz'), **pack)
        summary['conditions'][name] = paired_delta(pack, intact)
        save_json(ROOT/'first_block.json', summary)
    print(json.dumps(summary), flush=True)


def first_block_m0_rescue():
    """Restore the missing a0 contribution to m0 after deleting masked a0 writes."""
    torch.set_num_threads(4)
    engine = load_model(CONFIG, 'full_n2')
    split = make_split(CONFIG, 'test', .5)
    with np.load(ROOT/'intact.npz') as saved:
        intact = {key: saved[key] for key in saved.files}
    modes = ('attention_contribution', 'clean_edge_contribution', 'clean_m0_input')
    values = {mode: [] for mode in modes}; counts = {mode: [] for mode in modes}
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        for begin, end, _, _, corrupted, ids, mask in batches(split, CONFIG['batch_size']):
            clean_routed = []; clean_a0 = []; clean_a0_delta = []
            clean_a0_gains = []; clean_m0_read = []
            def capture(index, before, after, router, routed, delta):
                if index == 0:
                    clean_a0.append(after.detach().clone())
                    clean_a0_delta.append(delta.detach().clone())
                    clean_a0_gains.append(router[..., 2:4].detach().clone())
                if index == 1:
                    clean_routed.append(routed.detach().clone())
                    clean_m0_read.append(router[..., :2].detach().clone())
            engine.run(corrupted, mask, callback=capture)
            assert len(clean_routed) == len(clean_a0) == len(clean_a0_delta) == 1
            def delete_a0(module, args, outputs):
                read, gains, probabilities, mixer = outputs
                changed = torch.where(mask[..., None], torch.zeros_like(gains), gains)
                return read, changed, probabilities, mixer
            for mode in modes:
                ablated_a0 = []; ablated_read = []
                def capture_ablated(index, before, after, router, routed, delta):
                    if index == 0:
                        ablated_a0.append(after.detach().clone())
                def capture_read(module, args, outputs):
                    ablated_read.append(outputs[0].detach().clone())
                def restore_m0(module, args):
                    if mode == 'clean_m0_input':
                        replacement = clean_routed[0]
                    elif mode == 'clean_edge_contribution':
                        coefficient = (clean_m0_read[0].float() *
                                       clean_a0_gains[0].float()).sum(-1)
                        edge = coefficient.to(clean_a0_delta[0].dtype)[..., None] * clean_a0_delta[0]
                        replacement = args[0] + edge
                    else:
                        missing = clean_a0[0] - ablated_a0[0]
                        contribution = torch.einsum('bts,btsh->bth',
                            ablated_read[0].to(missing.dtype), missing)
                        replacement = args[0] + contribution
                    return (torch.where(mask[..., None], replacement, args[0]),)
                handle_a0 = engine.interfaces[0].router.register_forward_hook(delete_a0)
                handle_read = engine.interfaces[1].router.register_forward_hook(capture_read)
                handle_m0 = engine.interfaces[1].branch.register_forward_pre_hook(restore_m0)
                try:
                    state, _ = engine.run(corrupted, mask, callback=capture_ablated)
                finally:
                    handle_m0.remove(); handle_read.remove(); handle_a0.remove()
                local_sum, local_count = sequence_ce(engine, state, ids, mask)
                values[mode].append(local_sum); counts[mode].append(local_count)
            if end % 16 == 0:
                print('M0 RESCUE', end, '/', len(split['ids']), flush=True)
    with np.load(ROOT/'a0_zero_masked.npz') as saved:
        deleted = {key: saved[key] for key in saved.files}
    result = {'checkpoint_step': 124000, 'test_sequences': len(split['ids']),
              'deleted_delta_ce': paired_delta(deleted, intact),
              'modes': {}}
    for mode in modes:
        pack = {'seq_ce_sum': np.concatenate(values[mode]),
                'seq_mask_count': np.concatenate(counts[mode]),
                'block_ids': split['block_ids']}
        np.savez_compressed(ROOT/('a0_zero_masked_'+mode+'_rescue.npz'), **pack)
        result['modes'][mode] = {
            'delta_ce': paired_delta(pack, intact),
            'rescue_fraction': float(1 -
                (pack['seq_ce_sum']-intact['seq_ce_sum']).sum()/
                (deleted['seq_ce_sum']-intact['seq_ce_sum']).sum())}
    save_json(ROOT/'first_block_rescue.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['first', 'states', 'rescue'])
    stage = parser.parse_args().stage
    if stage == 'first': first_block()
    elif stage == 'states': state_interventions()
    else: first_block_m0_rescue()
