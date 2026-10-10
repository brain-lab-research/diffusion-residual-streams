"""Role-factorized two-stream routing, initialized from paper mHC-lite n=2."""
import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ('role_matched', 'frr', 'frr_merged', 'frr_fixedvol')


class FRRInterface(nn.Module):
    def __init__(self, full, variant):
        super().__init__()
        if variant not in VARIANTS or full.router.num_streams != 2:
            raise ValueError('FRR requires a supported variant and exactly two streams')
        router = full.router
        if router.read_function != 'sigmoid' or hasattr(router, 'input_projection'):
            raise ValueError('FRR requires the direct paper router')
        self.variant = variant
        self.branch = full.branch
        self.hidden_size = router.hidden_size
        rows = 1 if variant == 'frr_merged' else 2
        self.read_role_logits = nn.Parameter(router.pre.bias.detach().repeat(rows, 1))
        direction = torch.sigmoid(router.post.bias.detach())
        self.write_role_logits = nn.Parameter(direction.log().repeat(rows, 1))
        if variant == 'frr_fixedvol':
            self.volume_role_logits = nn.Parameter(torch.zeros(2))
        else:
            self.norm = router.norm
            self.volume = nn.Linear(2 * self.hidden_size, 1)
            nn.init.zeros_(self.volume.weight)
            nn.init.zeros_(self.volume.bias)
            self.alpha_volume = nn.Parameter(torch.tensor(0.01))
        if variant == 'role_matched':
            self.res = router.res
            self.alpha_res = router.alpha_res
            self.register_buffer('permutations', router.permutations.detach().clone(), persistent=False)
        self.collect_stats = False
        self.last_stats = None

    def routes(self, streams, role_ids):
        if streams.ndim != 4 or streams.shape[-2:] != (2, self.hidden_size):
            raise ValueError('Expected [B,T,2,D] streams')
        if role_ids.shape != streams.shape[:2] or role_ids.dtype != torch.long:
            raise ValueError('Expected [B,T] long role IDs from corrupted input')
        rows = torch.zeros_like(role_ids) if self.variant == 'frr_merged' else role_ids
        read = torch.sigmoid(self.read_role_logits[rows])
        direction = torch.softmax(self.write_role_logits[rows], dim=-1)
        routed = None
        if self.variant == 'frr_fixedvol':
            volume = 2 * torch.sigmoid(self.volume_role_logits[role_ids]).unsqueeze(-1)
        else:
            routed = self.norm(streams.flatten(-2))
            volume = 2 * torch.sigmoid(self.alpha_volume * F.linear(routed, self.volume.weight) + self.volume.bias)
        mixer = None
        if self.variant == 'role_matched':
            probabilities = torch.softmax(self.alpha_res * F.linear(routed, self.res.weight) + self.res.bias, -1)
            mixer = torch.einsum('btp,pij->btij', probabilities, self.permutations.to(probabilities.dtype)).to(streams.dtype)
        return read, direction, volume, mixer

    def forward(self, streams, role_ids):
        read, direction, volume, mixer = self.routes(streams, role_ids)
        hidden = torch.einsum('btn,btnd->btd', read.to(streams.dtype), streams)
        delta = self.branch(hidden)
        residual = streams if mixer is None else torch.einsum('btij,btjd->btid', mixer, streams)
        updated = residual + (2 * volume * direction).to(streams.dtype).unsqueeze(-1) * delta.unsqueeze(-2)
        if self.collect_stats:
            with torch.no_grad():
                self.last_stats = {
                    'volume': volume.squeeze(-1).detach().float(),
                    'stream_norm': updated.detach().float().norm(dim=-1),
                    'stream_cosine': F.cosine_similarity(updated[..., 0, :].float(), updated[..., 1, :].float(), dim=-1).detach(),
                    'stream_difference': (updated[..., 0, :] - updated[..., 1, :]).detach().float().norm(dim=-1),
                }
        return updated


def convert_model(model, variant):
    for block in model.blocks:
        block.wa = FRRInterface(block.wa, variant)
        block.wm = FRRInterface(block.wm, variant)
        block.arm = 'frr'
    model.frr_variant = variant
    return model


def route_tables(model):
    result = []
    for name, module in model.named_modules():
        if isinstance(module, FRRInterface):
            read = torch.sigmoid(module.read_role_logits.detach()).float()
            write = torch.softmax(module.write_role_logits.detach(), -1).float()
            item = dict(interface=name, read=read.cpu().tolist(), write=write.cpu().tolist())
            for key, values in [('read', read), ('write', write)]:
                if values.shape[0] == 2:
                    item[key+'_l1'] = (values[0]-values[1]).abs().sum().item()
                    item[key+'_cosine'] = F.cosine_similarity(values[0], values[1], dim=0).item()
            if hasattr(module, 'volume_role_logits'):
                item['role_volume'] = (2*torch.sigmoid(module.volume_role_logits.detach())).cpu().tolist()
            result.append(item)
    return result
