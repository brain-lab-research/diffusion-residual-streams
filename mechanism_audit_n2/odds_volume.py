"""Physical volume + odds-simplex destination, valid for any stream count.

g_i=2 sigmoid(z_i); V=sum(g); q=softmax(z). A common shift of z changes
V while preserving q. The scalar root is bracketed analytically and solved
in float64 in one GPU kernel; gains are never clipped.
"""
import torch
from torch.nn import functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _shift_kernel(Z, V, G, C, ROWS: tl.constexpr, N: tl.constexpr,
                  PAD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    stream = tl.arange(0, PAD)
    z = tl.load(Z + row[:, None] * N + stream[None, :],
                (row[:, None] < ROWS) & (stream[None, :] < N), other=0).to(tl.float64)
    wanted = tl.load(V + row, row < ROWS, other=N).to(tl.float64)
    p = wanted / (2.0 * N)
    center = libdevice.log(p / (1.0 - p))
    maximum = tl.max(tl.where(stream[None, :] < N, z, float('-inf')), 1)
    minimum = tl.min(tl.where(stream[None, :] < N, z, float('inf')), 1)
    lo, hi = center - maximum, center - minimum
    # Width is max(z)-min(z); 40 bisections reduce a 100-logit span below 1e-10.
    for _ in range(40):
        mid = (lo + hi) * .5
        gains = 2.0 / (1.0 + libdevice.exp(-(z + mid[:, None])))
        total = tl.sum(tl.where(stream[None, :] < N, gains, 0.0), 1)
        lo = tl.where(total < wanted, mid, lo)
        hi = tl.where(total < wanted, hi, mid)
    shift = (lo + hi) * .5
    result = 2.0 / (1.0 + libdevice.exp(-(z + shift[:, None])))
    result = tl.where(wanted[:, None] <= 0, 0.0,
                      tl.where(wanted[:, None] >= 2 * N, 2.0, result))
    tl.store(G + row[:, None] * N + stream[None, :], result,
             (row[:, None] < ROWS) & (stream[None, :] < N))
    tl.store(C + row, shift, row < ROWS)


def shifted_gains(logits, volume, dtype):
    assert logits.is_cuda and volume.is_cuda
    logits, volume = logits.contiguous(), volume.contiguous()
    n = logits.shape[-1]; rows = logits.numel() // n
    assert volume.numel() == rows
    gains = torch.empty_like(logits, dtype=dtype)
    shift = torch.empty_like(volume, dtype=torch.float64)
    _shift_kernel[(triton.cdiv(rows, 64),)](logits, volume, gains, shift,
        ROWS=rows, N=n, PAD=triton.next_power_of_2(n), BLOCK=64, num_warps=4)
    return gains, shift


def write_logits(router, state):
    assert router.routing_mode == 'dynamic'
    flat = state.reshape(*state.shape[:2], router.num_streams * router.hidden_size)
    value = router.norm(flat)
    if hasattr(router, 'input_projection'):
        value = router.activation(router.input_projection(value))
    return router.alpha_post * F.linear(value, router.post.weight) + router.post.bias


class OddsVolumeHooks:
    def __init__(self, engine, means, position_bins, depth_quartiles):
        self.engine = engine
        self.means = {k: torch.from_numpy(v.copy()).to('cuda') for k, v in means.items()}
        self.depth_quartiles = depth_quartiles
        self.valid = None; self.layers = set(); self.policy = 'dynamic'; self.handles = []
        self.maximum_errors = torch.zeros(5, device='cuda', dtype=torch.float64)
        for index, interface in enumerate(engine.interfaces):
            def hook(module, args, outputs, layer=index):
                if self.policy == 'dynamic' or layer not in self.layers:
                    return outputs
                return self.transform(layer, module, args[0], outputs)
            self.handles.append(interface.router.register_forward_hook(hook))

    def configure_condition(self, valid, condition):
        self.valid = valid; self.policy = condition.removeprefix('volume_')
        self.layers = set(range(32))
        if self.policy in self.depth_quartiles:
            self.layers = set(self.depth_quartiles[self.policy]); self.policy = 'global'
        elif self.policy in ('attention', 'mlp'):
            self.layers = set(range(0 if self.policy == 'attention' else 1, 32, 2))
            self.policy = 'global'
        assert self.policy in ('dynamic', 'global', 'context_quartile', 'position')
        self.maximum_errors.zero_()

    def replacement(self, layer, length):
        if self.policy == 'global': return self.means['global'][layer]
        positions = torch.arange(length, device='cuda').clamp_max(1022)
        if self.policy == 'context_quartile':
            positions = positions.div(256, rounding_mode='floor')
        return self.means[self.policy][layer][positions][None]

    def transform(self, layer, module, state, outputs):
        read, gains, probabilities, mixer = outputs
        z = write_logits(module, state)
        volume = gains.double().sum(-1, keepdim=True)
        wanted = torch.where(self.valid[..., None], self.replacement(layer, gains.shape[1]), volume)
        replacement, shift = shifted_gains(z, wanted, gains.dtype)
        changed = torch.where(self.valid[..., None], replacement, gains)
        # Validate numerical reconstruction and q on the pre-rounding shifted logits.
        shifted = z.double() + shift
        q = z.double().softmax(-1); q_shifted = shifted.softmax(-1)
        errors = torch.stack([(changed.double().sum(-1, keepdim=True) - wanted).abs().max(),
            torch.maximum((-changed).clamp_min(0).double().max(), (changed - 2).clamp_min(0).double().max()),
            (q_shifted - q).abs().max(), (q_shifted.sum(-1) - 1).abs().max(),
            (2 * z.sigmoid() - gains).abs().double().max()])
        self.maximum_errors.copy_(torch.maximum(self.maximum_errors, errors))
        return read, changed, probabilities, mixer

    def close(self):
        for handle in self.handles: handle.remove()


ERROR_NAMES = ('volume_reconstruction', 'gain_bounds', 'destination_preservation',
               'destination_simplex', 'router_logits_reconstruction')
