"""Local, non-invasive instrumentation; no frozen training source is modified."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(MHC_PAPER='1', MHC_PROPER_INIT='1', MHC_STREAMS='2',
                  MHC_RESMIX='learned')
sys.path.insert(0, str(ROOT))
import model_and_train as training

STAT_NAMES = ['ce', 'target_logit', 'entropy', 'target_rank', 'top1', 'correct',
              'target_margin', 'prediction_margin']


def load_model(config, name, checkpoint=None):
    spec = config['models'][name]
    training.STREAMS = spec['streams']
    torch.manual_seed(1337)
    model = training.Model(spec['arm'], causal=spec.get('causal', False))
    path = Path(checkpoint or spec['checkpoint'])
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(saved['model'], strict=True)
    model.requires_grad_(False)
    model = model.to('cuda').eval()
    return Engine(model, name, spec['streams'], saved['step'])


def make_split(config, name, ratio=None):
    ratio = config['ratio'] if ratio is None else ratio
    block_ids = np.array(config['splits'][name], dtype=np.int32)
    data = np.load(config['dataset'], mmap_mode='r')
    ids = np.asarray(data[block_ids], dtype=np.int64).copy()
    masks = []
    for block in block_ids:
        generator = torch.Generator().manual_seed(config['mask_seed'] + int(block) * 1009 + round(ratio * 100) * 10017)
        mask = torch.rand(ids.shape[1], generator=generator).numpy() < ratio
        if not mask.any():
            mask[0] = True
        masks.append(mask)
    masks = np.array(masks, dtype=bool)
    return {'name': name, 'ratio': ratio, 'block_ids': block_ids, 'ids': ids, 'mask': masks,
            'mask_sha256': hashlib.sha256(masks.tobytes()).hexdigest(),
            'data_sha256': hashlib.sha256(ids.tobytes()).hexdigest()}


def batches(split, batch_size):
    counts = split['mask'].sum(1)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    for begin in range(0, len(split['ids']), batch_size):
        end = min(begin + batch_size, len(split['ids']))
        ids = torch.from_numpy(split['ids'][begin:end]).to('cuda')
        mask = torch.from_numpy(split['mask'][begin:end]).to('cuda')
        corrupted = ids.clone()
        corrupted[mask] = training.MASK_ID
        yield begin, end, int(offsets[begin]), int(offsets[end]), corrupted, ids, mask


def head_stats(model, states, labels, chunk=128, pair_kl=False):
    """Actual frozen head; states is [streams, masked_tokens, hidden]."""
    outputs, kls = [], []
    for begin in range(0, labels.numel(), chunk):
        ys = labels[begin:begin + chunk]
        local, logs, probs, tops = [], [], [], []
        for stream in states:
            logits = model.head(model.norm(stream[begin:begin + chunk])).float()
            logp = logits.log_softmax(-1)
            probability = logp.exp()
            target = logits.gather(1, ys[:, None]).squeeze(1)
            nll = -logp.gather(1, ys[:, None]).squeeze(1)
            entropy = -(probability * logp).sum(-1)
            rank = (logits > target[:, None]).sum(-1).float() + 1
            top, topids = logits.topk(10, -1)
            correct = topids[:, 0].eq(ys)
            competing = torch.where(correct, top[:, 1], top[:, 0])
            local.append(torch.stack([nll, target, entropy, rank, topids[:, 0].float(),
                                      correct.float(), target - competing, top[:, 0] - top[:, 1]], -1).cpu().numpy())
            if pair_kl:
                logs.append(logp); probs.append(probability); tops.append(topids)
        outputs.append(np.stack(local))
        if pair_kl:
            assert len(local) == 2
            kl01 = (probs[0] * (logs[0] - logs[1])).sum(-1)
            kl10 = (probs[1] * (logs[1] - logs[0])).sum(-1)
            overlap = (tops[0][:, :, None] == tops[1][:, None, :]).any(-1).float().mean(-1)
            kls.append(torch.stack([kl01, kl10, overlap], -1).cpu().numpy())
    return np.concatenate(outputs, 1), np.concatenate(kls) if pair_kl else None


def norm_match(vector, reference):
    return vector.float() / vector.float().norm(dim=-1, keepdim=True).clamp_min(1e-12) * reference.float().norm(dim=-1, keepdim=True)


class Engine:
    def __init__(self, model, name, streams, step):
        self.model, self.name, self.streams, self.checkpoint_step = model, name, streams, step
        self.multi = model.arm == 'mhc'
        self.interfaces = [(block.wa, block.wm) if self.multi else (block.attn, block.mlp) for block in model.blocks]
        self.interfaces = [module for pair in self.interfaces for module in pair]
        self.mean_errors = {}

    def initial(self, corrupted):
        state = self.model.emb(corrupted)
        return self.model.expander(state) if self.multi else state.unsqueeze(-2)

    def interface(self, index, state, mask, condition='intact', context=None):
        context = context or {}
        module = self.interfaces[index]
        if self.multi:
            read, gains, probabilities, mixer = module.router(state)
            if condition.startswith('router_') and 'router' in context:
                value = context['router'][index]
                select = mask[..., None]
                if condition.startswith('router_interpolate'):
                    alpha = context['routing_alpha']
                    read = torch.where(select, (1 - alpha) * read + alpha * value[..., :self.streams], read)
                    gains = torch.where(select, (1 - alpha) * gains + alpha * value[..., self.streams:2 * self.streams], gains)
                    probabilities = torch.where(select, (1 - alpha) * probabilities + alpha * value[..., 2 * self.streams:], probabilities)
                else:
                    read = torch.where(select, value[..., :self.streams], read)
                    gains = torch.where(select, value[..., self.streams:2 * self.streams], gains)
                    probabilities = torch.where(select, value[..., 2 * self.streams:], probabilities)
                mixer = torch.einsum('btp,pij->btij', probabilities, module.router.permutations.to(probabilities.dtype)).to(state.dtype)
            routed = torch.einsum('bts,btsh->bth', read.to(state.dtype), state)
            delta = module.branch(routed)
            if isinstance(delta, tuple):
                delta = delta[0]
            mixed = torch.einsum('btij,btjh->btih', mixer, state)
            write = gains.to(state.dtype).unsqueeze(-1) * delta.unsqueeze(-2)
            original = mixed + write
            source = write[..., 1, :] if index == 0 else None
            if index == 0 and condition in ['zero', 'norm_random', 'role_mean', 'norm_role_mean',
                                            'within_shuffle', 'cross_shuffle', 'within_norm_shuffle', 'cross_norm_shuffle']:
                write = write.clone()
                replacement = torch.zeros_like(source) if condition == 'zero' else context['replacement']
                write[..., 1, :] = torch.where(mask[..., None], replacement.to(write.dtype), write[..., 1, :])
            if index == 0 and condition in ['swap', 'equal_split', 'single0', 'single1']:
                adjusted = gains.clone()
                if condition == 'swap':
                    values = gains.flip(-1)
                else:
                    total = gains.sum(-1)
                    values = torch.stack([total / 2, total / 2], -1) if condition == 'equal_split' else torch.stack(
                        [total if condition == 'single0' else torch.zeros_like(total),
                         total if condition == 'single1' else torch.zeros_like(total)], -1)
                adjusted = torch.where(mask[..., None], values, adjusted)
                write = adjusted.to(state.dtype).unsqueeze(-1) * delta.unsqueeze(-2)
            state = mixed + write
            router = torch.cat([read.float(), gains.float(), probabilities.float()], -1)
        else:
            routed = state[..., 0, :]
            delta = module(routed)
            source = delta if index == 0 else None
            if index == 0 and condition == 'zero':
                delta = torch.where(mask[..., None], torch.zeros_like(delta), delta)
            original = state + delta.unsqueeze(-2)
            state = original
            router = None
        if condition.startswith('collapse') and index == context.get('depth'):
            collapsed = state.mean(-2, keepdim=True).expand_as(state)
            state = torch.where(mask[..., None, None], collapsed, state) if context.get('masked_only') else collapsed.clone()
        if self.multi and (index == 0 and condition in ['swap', 'equal_split', 'single0', 'single1'] or
                           condition.startswith('collapse') and index == context.get('depth')):
            error = float((state.float().mean(-2) - original.float().mean(-2)).abs().max())
            self.mean_errors[condition] = max(self.mean_errors.get(condition, 0), error)
            torch.testing.assert_close(state.float().mean(-2), original.float().mean(-2), rtol=2e-5, atol=2e-5)
        return state, source, router, routed, delta

    def run(self, corrupted, mask, condition='intact', context=None, callback=None, cache=False,
            initial_state=None, start_after=-1):
        context = context or {}
        state = self.initial(corrupted) if initial_state is None else initial_state.clone()
        states, routers = [], []
        original_source = None
        for index in range(start_after + 1, len(self.interfaces)):
            input_state = state
            state, source, router, routed, delta = self.interface(index, state, mask, condition, context)
            if source is not None:
                original_source = source
            if cache:
                states.append(state.detach().clone())
                routers.append(router.detach().clone() if router is not None else None)
            if callback is not None:
                callback(index, input_state, state, router, routed, delta)
        return state, {'states': states, 'routers': routers, 'source': original_source}

    def final_stats(self, state, ids, mask, chunk=128):
        collapsed = state.mean(-2)
        return head_stats(self.model, collapsed[mask].unsqueeze(0), ids[mask], chunk)[0][0]

    def original_logits(self, state):
        return self.model.head(self.model.norm(state.mean(-2)))

    def patched(self, state, mask, addition=None, whole_state=None):
        patched = state.clone()
        destination = 1 if self.multi else 0
        if whole_state is not None:
            replacement = whole_state[..., destination, :]
        else:
            replacement = patched[..., destination, :] + addition
        patched[..., destination, :] = torch.where(mask[..., None], replacement, patched[..., destination, :])
        return patched

    def source_only(self, corrupted, mask):
        _, source, _, _, _ = self.interface(0, self.initial(corrupted), mask)
        return source


def summarize_tokens(split, tokens):
    counts = split['mask'].sum(1).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    assert len(tokens) == offsets[-1]
    sums = np.array([tokens[offsets[i]:offsets[i + 1], 0].sum(dtype=np.float64) for i in range(len(counts))])
    return {'seq_ce_sum': sums, 'seq_mask_count': counts, 'token_stats': tokens,
            'block_ids': split['block_ids'], 'positions': np.where(split['mask'])[1].astype(np.int16),
            'token_sequence': np.repeat(np.arange(len(counts)), counts).astype(np.int32)}


def aggregate(pack):
    tokens = pack['token_stats']
    return {'ce': float(pack['seq_ce_sum'].sum() / pack['seq_mask_count'].sum()),
            'masked_tokens': int(pack['seq_mask_count'].sum()), 'sequences': len(pack['seq_mask_count']),
            'accuracy': float(tokens[:, 5].mean()), 'target_logit': float(tokens[:, 1].mean()),
            'entropy': float(tokens[:, 2].mean()), 'target_margin': float(tokens[:, 6].mean())}


def paired_interval(left, right, repeats=5000, seed=130914):
    assert np.array_equal(left['seq_mask_count'], right['seq_mask_count'])
    assert np.array_equal(left['block_ids'], right['block_ids'])
    generator = np.random.default_rng(seed)
    n = len(left['seq_mask_count'])
    differences = []
    for begin in range(0, repeats, 250):
        draws = generator.integers(n, size=(min(250, repeats - begin), n))
        count = left['seq_mask_count'][draws].sum(-1)
        differences.extend((left['seq_ce_sum'][draws].sum(-1) - right['seq_ce_sum'][draws].sum(-1)) / count)
    return list(np.quantile(differences, [.025, .975]))
