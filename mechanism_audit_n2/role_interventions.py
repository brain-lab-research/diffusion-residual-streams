"""All-valid-position routing policies; roles come only from the corruption mask."""
import numpy as np
import torch

from .factors import FactorHooks, factor_values, gpu_factors, reconstruct

ROLE_NAMES = ('MASKED', 'VISIBLE')
INVARIANTS = ('read_bounds', 'gain_bounds', 'destination_bounds', 'destination_simplex',
              'mixer_bounds', 'mixer_simplex', 'volume_reconstruction')


def calibration_means(router, mask, valid):
    """router [32,N,6]; mask/valid [N]; exact float64 token-weighted means."""
    values = factor_values(router, 'bounded')
    selected = [valid & mask, valid & ~mask]
    assert valid.any() and all(indices.any() for indices in selected)
    means = {'global': {}, 'role': {}}
    for factor, array in values.items():
        means['global'][factor] = array[:, valid].mean(1, dtype=np.float64)
        means['role'][factor] = np.stack([array[:, indices].mean(1, dtype=np.float64)
                                         for indices in selected], axis=1)
        for policy in means:
            value = means[policy][factor]
            if factor in ['destination', 'mixer']:
                value /= value.sum(-1, keepdims=True)
                assert np.all((value >= 0) & (value <= 1))
                np.testing.assert_allclose(value.sum(-1), 1, atol=1e-12, rtol=0)
    return means


class RoleFactorHooks(FactorHooks):
    def __init__(self, engine, means):
        super().__init__(engine, means['global'], 'bounded')
        self.role_means = {key: torch.from_numpy(value.copy()).to('cuda')
                           for key, value in means['role'].items()}
        self.valid = None
        self.policy = 'dynamic'
        self.maximum_invariants = torch.zeros(len(INVARIANTS), device='cuda', dtype=torch.float64)

    def configure(self, mask, fixed, valid=None, policy='dynamic', layers=None):
        assert policy in ['global', 'role', 'role_swap', 'dynamic']
        super().configure(mask, fixed, layers)
        self.valid = torch.ones_like(mask, dtype=torch.bool) if valid is None else valid
        assert self.valid.shape == mask.shape
        self.policy = policy
        self.maximum_invariants.zero_()

    def replacement(self, factor, layer):
        if self.policy == 'global':
            return self.means[factor][layer]
        role = (~self.mask).long()  # MASKED=0, VISIBLE=1; never inspect token IDs.
        if self.policy == 'role_swap':
            role = 1 - role
        return self.role_means[factor][layer][role]

    def transform(self, layer, module, state, outputs):
        read, gains, probabilities, mixer = outputs
        select = self.valid[..., None]
        expected_volume, destination = gpu_factors(gains, 'bounded')
        if 'read' in self.fixed:
            read = torch.where(select, self.replacement('read', layer).to(read.dtype), read)
        if {'volume', 'destination'} & self.fixed:
            if 'volume' in self.fixed:
                expected_volume = torch.where(select, self.replacement('volume', layer), expected_volume)
            if 'destination' in self.fixed:
                destination = torch.where(select, self.replacement('destination', layer), destination)
            gains = torch.where(select, reconstruct(expected_volume, destination, 'bounded', gains.dtype), gains)
        if 'mixer' in self.fixed:
            probabilities = torch.where(select, self.replacement('mixer', layer).to(probabilities.dtype), probabilities)
            replacement = torch.einsum('btp,pij->btij', probabilities,
                                       module.permutations.to(probabilities.dtype)).to(state.dtype)
            mixer = torch.where(select[..., None], replacement, mixer)
        volume, actual_q = gpu_factors(gains, 'bounded')
        def bounds(value, upper):
            return torch.maximum((-value).clamp_min(0).double().max(),
                                 (value - upper).clamp_min(0).double().max())
        errors = torch.stack([bounds(read, 1), bounds(gains, 2), bounds(actual_q, 1),
            (actual_q.sum(-1) - 1).abs().max(), bounds(probabilities, 1),
            (probabilities.double().sum(-1) - 1).abs().max(), (volume - expected_volume).abs().max()])
        self.maximum_invariants.copy_(torch.maximum(self.maximum_invariants, errors))
        self.bound_violation.copy_(torch.maximum(self.bound_violation, errors.max()))
        return read, gains, probabilities, mixer
