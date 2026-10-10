from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


@dataclass
class MHCLiteStats:
    h_pre: torch.Tensor
    h_post: torch.Tensor
    permutation_probs: torch.Tensor
    h_res: torch.Tensor

    def summary(self) -> dict[str, float]:
        identity_prob = self.permutation_probs[..., 0].mean()
        entropy = -(self.permutation_probs * self.permutation_probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
        eye = torch.eye(self.h_res.shape[-1], dtype=self.h_res.dtype, device=self.h_res.device)
        distance = torch.linalg.matrix_norm(self.h_res - eye, ord="fro", dim=(-2, -1)).mean()
        row_error = (self.h_res.sum(dim=-1) - 1.0).abs().max()
        col_error = (self.h_res.sum(dim=-2) - 1.0).abs().max()
        return {
            "h_pre_mean": float(self.h_pre.mean().detach().cpu()),
            "h_post_mean": float(self.h_post.mean().detach().cpu()),
            "identity_prob": float(identity_prob.detach().cpu()),
            "permutation_entropy": float(entropy.detach().cpu()),
            "h_res_identity_fro": float(distance.detach().cpu()),
            "h_res_row_error": float(row_error.detach().cpu()),
            "h_res_col_error": float(col_error.detach().cpu()),
        }


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        normalized = x.float()
        variance = normalized.pow(2).mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * normalized).to(dtype=input_dtype)


def permutation_matrices(num_streams: int) -> torch.Tensor:
    matrices = []
    identity = tuple(range(num_streams))
    permutations = [identity]
    permutations.extend(perm for perm in itertools.permutations(range(num_streams)) if perm != identity)
    for perm in permutations:
        matrix = torch.zeros(num_streams, num_streams)
        for row, col in enumerate(perm):
            matrix[row, col] = 1.0
        matrices.append(matrix)
    return torch.stack(matrices, dim=0)


class MHCLiteRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_streams: int = 4,
        bottleneck_dim: int | None = None,
        read_function: str = "sigmoid",
        alpha_init: float = 0.01,
        non_identity_bias: float = -8.0,
        write_head_init_std: float = 0.0,
        residual_mixing: str = "learned",
        routing_mode: str = "dynamic",
    ) -> None:
        super().__init__()
        if num_streams < 1:
            raise ValueError("num_streams must be at least 1")
        # n=1 is the degenerate-but-meaningful case: the permutation set is {I}, so the
        # residual mixer is an exact no-op and the router reduces to a dynamic read gate
        # + write gain on a single stream (the n=1 rung of the streams ablation).
        if residual_mixing not in {"learned", "identity", "kromhc"}:
            raise ValueError("residual_mixing must be 'learned', 'identity', or 'kromhc'")
        if residual_mixing == "kromhc" and num_streams & (num_streams - 1):
            raise ValueError("kromhc requires num_streams to be a power of two")
        if routing_mode not in {"dynamic", "static"}:
            raise ValueError("routing_mode must be 'dynamic' or 'static'")
        if read_function not in {"sigmoid", "softmax"}:
            raise ValueError("read_function must be 'sigmoid' or 'softmax'")
        if write_head_init_std < 0.0:
            raise ValueError("write_head_init_std must be non-negative")

        self.hidden_size = hidden_size
        self.num_streams = num_streams
        self.bottleneck_dim = bottleneck_dim
        self.read_function = read_function
        self.write_head_init_std = float(write_head_init_std)
        self.residual_mixing = residual_mixing
        self.routing_mode = routing_mode
        flat_size = hidden_size * num_streams
        num_permutations = math.factorial(num_streams)
        self.kromhc_num_factors = int(math.log2(num_streams)) if residual_mixing == "kromhc" else 0

        if self.routing_mode == "dynamic":
            self.norm = RMSNorm(flat_size)
            router_input_size = flat_size
            if bottleneck_dim is not None:
                if bottleneck_dim < 1:
                    raise ValueError("bottleneck_dim must be positive when provided")
                self.input_projection = nn.Linear(flat_size, bottleneck_dim)
                self.activation = nn.SiLU()
                router_input_size = bottleneck_dim
            self.pre = nn.Linear(router_input_size, num_streams)
            self.post = nn.Linear(router_input_size, num_streams)
            if self.residual_mixing == "learned":
                self.res = nn.Linear(router_input_size, num_permutations)
            elif self.residual_mixing == "kromhc":
                self.res = nn.Linear(router_input_size, 2 * self.kromhc_num_factors)

            self.alpha_pre = nn.Parameter(torch.tensor(float(alpha_init)))
            self.alpha_post = nn.Parameter(torch.tensor(float(alpha_init)))
            if self.residual_mixing in {"learned", "kromhc"}:
                self.alpha_res = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.pre_logits = nn.Parameter(torch.empty(num_streams))
            self.post_logits = nn.Parameter(torch.empty(num_streams))
            if self.residual_mixing == "learned":
                self.res_logits = nn.Parameter(torch.empty(num_permutations))
            elif self.residual_mixing == "kromhc":
                self.res_logits = nn.Parameter(torch.empty(2 * self.kromhc_num_factors))

        self.register_buffer("permutations", permutation_matrices(num_streams), persistent=False)
        self.reset_parameters(non_identity_bias=non_identity_bias)

    def reset_parameters(self, non_identity_bias: float = -8.0) -> None:
        if self.routing_mode == "dynamic":
            if hasattr(self, "input_projection"):
                nn.init.xavier_uniform_(self.input_projection.weight)
                nn.init.zeros_(self.input_projection.bias)
            nn.init.zeros_(self.pre.weight)
            if self.write_head_init_std == 0.0:
                nn.init.zeros_(self.post.weight)
            else:
                nn.init.normal_(self.post.weight, std=self.write_head_init_std)
                # Break symmetry without introducing a common write-gain shift.
                self.post.weight.data.sub_(self.post.weight.data.mean(dim=0, keepdim=True))
            nn.init.constant_(self.pre.bias, -math.log(3.0) if self.read_function == "sigmoid" else 0.0)
            nn.init.zeros_(self.post.bias)
            if self.residual_mixing == "learned":
                nn.init.zeros_(self.res.weight)
                nn.init.constant_(self.res.bias, non_identity_bias)
                self.res.bias.data[0] = 0.0
            elif self.residual_mixing == "kromhc":
                nn.init.zeros_(self.res.weight)
                nn.init.constant_(self.res.bias, non_identity_bias)
                self.res.bias.data[::2] = 0.0
        else:
            nn.init.constant_(self.pre_logits, -math.log(3.0) if self.read_function == "sigmoid" else 0.0)
            if self.write_head_init_std == 0.0:
                nn.init.zeros_(self.post_logits)
            else:
                nn.init.normal_(self.post_logits, std=self.write_head_init_std)
                self.post_logits.data.sub_(self.post_logits.data.mean())
            if self.residual_mixing == "learned":
                nn.init.constant_(self.res_logits, non_identity_bias)
                self.res_logits.data[0] = 0.0
            elif self.residual_mixing == "kromhc":
                nn.init.constant_(self.res_logits, non_identity_bias)
                self.res_logits.data[::2] = 0.0

    def forward(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if streams.ndim != 4:
            raise ValueError(f"streams must have shape [batch, seq, streams, hidden], got {tuple(streams.shape)}")
        if streams.shape[-2:] != (self.num_streams, self.hidden_size):
            raise ValueError(
                "streams trailing shape must be "
                f"({self.num_streams}, {self.hidden_size}), got {tuple(streams.shape[-2:])}"
            )

        if self.routing_mode == "dynamic":
            flat = streams.reshape(*streams.shape[:2], self.num_streams * self.hidden_size)
            routed = self.norm(flat)
            if hasattr(self, "input_projection"):
                routed = self.activation(self.input_projection(routed))
            pre_logits = self.alpha_pre * torch.nn.functional.linear(routed, self.pre.weight) + self.pre.bias
            h_pre = (
                torch.softmax(pre_logits, dim=-1)
                if self.read_function == "softmax"
                else torch.sigmoid(pre_logits)
            )
            h_post = 2.0 * torch.sigmoid(
                self.alpha_post * torch.nn.functional.linear(routed, self.post.weight) + self.post.bias
            )
        else:
            static_pre = (
                torch.softmax(self.pre_logits, dim=-1)
                if self.read_function == "softmax"
                else torch.sigmoid(self.pre_logits)
            )
            h_pre = static_pre.to(dtype=streams.dtype).expand(*streams.shape[:2], self.num_streams)
            h_post = (2.0 * torch.sigmoid(self.post_logits)).to(dtype=streams.dtype).expand(
                *streams.shape[:2], self.num_streams
            )
        if self.residual_mixing == "identity":
            permutation_probs = streams.new_zeros(*streams.shape[:2], self.permutations.shape[0])
            permutation_probs[..., 0] = 1.0
            eye = torch.eye(self.num_streams, dtype=streams.dtype, device=streams.device)
            h_res = eye.expand(*streams.shape[:2], self.num_streams, self.num_streams)
        elif self.residual_mixing == "learned":
            if self.routing_mode == "dynamic":
                res_logits = self.alpha_res * torch.nn.functional.linear(routed, self.res.weight) + self.res.bias
            else:
                res_logits = self.res_logits.to(dtype=streams.dtype).expand(*streams.shape[:2], self.permutations.shape[0])
            permutation_probs = torch.softmax(res_logits, dim=-1)
            h_res = torch.einsum("btp,pij->btij", permutation_probs, self.permutations.to(dtype=permutation_probs.dtype))
        else:  # KromHC: Kronecker product of log2(n) dynamic 2x2 Birkhoff factors.
            if self.routing_mode == "dynamic":
                factor_logits = self.alpha_res * torch.nn.functional.linear(routed, self.res.weight) + self.res.bias
            else:
                factor_logits = self.res_logits.to(dtype=streams.dtype).expand(*streams.shape[:2], -1)
            factor_probs = torch.softmax(factor_logits.reshape(*factor_logits.shape[:2], self.kromhc_num_factors, 2), dim=-1)
            p = factor_probs[..., 0]
            factors = torch.stack((torch.stack((p, 1.0 - p), dim=-1), torch.stack((1.0 - p, p), dim=-1)), dim=-2)
            h_res = factors[..., 0, :, :]
            for idx in range(1, self.kromhc_num_factors):
                factor = factors[..., idx, :, :]
                h_res = (h_res[..., :, None, :, None] * factor[..., None, :, None, :]).reshape(*h_res.shape[:-2], h_res.shape[-2] * 2, h_res.shape[-1] * 2)
            permutation_probs = factor_probs.reshape(*factor_probs.shape[:2], -1)
        h_res = h_res.to(dtype=streams.dtype)
        return h_pre, h_post, permutation_probs, h_res


class MHCLiteBranchWrapper(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        branch: Callable[..., torch.Tensor],
        num_streams: int = 4,
        bottleneck_dim: int | None = None,
        read_function: str = "sigmoid",
        alpha_init: float = 0.01,
        non_identity_bias: float = -8.0,
        residual_mixing: str = "learned",
        routing_mode: str = "dynamic",
    ) -> None:
        super().__init__()
        self.router = MHCLiteRouter(
            hidden_size=hidden_size,
            num_streams=num_streams,
            bottleneck_dim=bottleneck_dim,
            read_function=read_function,
            alpha_init=alpha_init,
            non_identity_bias=non_identity_bias,
            residual_mixing=residual_mixing,
            routing_mode=routing_mode,
        )
        self.branch = branch
        self.last_stats: MHCLiteStats | None = None

    def forward(self, streams: torch.Tensor, *args, collect_stats: bool = False, **kwargs) -> torch.Tensor:
        h_pre, h_post, permutation_probs, h_res = self.router(streams)
        x = torch.einsum("bts,btsh->bth", h_pre.to(streams.dtype), streams)
        branch_output = self.branch(x, *args, **kwargs)
        if isinstance(branch_output, tuple):
            y = branch_output[0]
        else:
            y = branch_output
        mixed = torch.einsum("btij,btjh->btih", h_res, streams)
        updated = mixed + h_post.to(streams.dtype).unsqueeze(-1) * y.unsqueeze(-2)
        if collect_stats:
            self.last_stats = MHCLiteStats(
                h_pre=h_pre.detach(),
                h_post=h_post.detach(),
                permutation_probs=permutation_probs.detach(),
                h_res=h_res.detach(),
            )
        return updated


class StreamExpander(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_streams: int = 4,
        init_std: float = 1e-6,
        use_lss: bool = False,
        scale_init_std: float = 0.0,
    ) -> None:
        super().__init__()
        embeddings = torch.randn(num_streams, hidden_size) * init_std
        embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
        self.stream_embeddings = nn.Parameter(embeddings)
        self.use_lss = bool(use_lss)
        if self.use_lss:
            scale_offsets = torch.randn(num_streams, hidden_size) * scale_init_std
            scale_offsets = scale_offsets - scale_offsets.mean(dim=0, keepdim=True)
            self.stream_scales = nn.Parameter(torch.ones(num_streams, hidden_size) + scale_offsets)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        streams = hidden_states.unsqueeze(-2)
        if self.use_lss:
            streams = streams * self.stream_scales.to(dtype=hidden_states.dtype)
        return streams + self.stream_embeddings.to(dtype=hidden_states.dtype)
