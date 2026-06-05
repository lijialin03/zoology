# -*- coding: utf-8 -*-
"""
Multi-Scale DeltaNet (MS-DeltaNet) — compressor+concat applied to DeltaNet.

Architecture
------------
Following MS-GLA's compressor+concat design, extra key/query capacity comes
from a separate *compressor* pathway (DimGatedPoolCompressor) rather than
fused projection (expand_k).  This is applied on top of DeltaNet's delta rule.

    q_main = q_proj(x)             ∈ R^{K}       (standard DeltaNet)
    q_extra = q_compress(x)        ∈ R^{K/r}    (compressor)
    q = [q_main ‖ q_extra]         ∈ R^{K+K/r}  (concatenate)

Same for k.  v stays at V (rectangular state).  Beta (learned per-head gate
for the delta rule) is also kept standard — no compressor needed.

    S ∈ R^{(K+K/r) × V}
    v̂ = S @ k                        (state read: predicted value)
    Δ  = β · (v - v̂)                (prediction error × gate)
    S = S + k ⊗ Δ                    (state write: outer product)
    o = S @ q                        (output: read from updated state)

Reference:
    - DeltaNet: https://arxiv.org/abs/2406.06484
    - MS-GLA: ../mixers/ms_gla.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

try:
    from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
    from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule
except:
    assert 0, print("Need to install fla: pip install flash-linear-attention")

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


# Re-use DimGatedPoolCompressor from MS-GLA
from zoology.mixers.ms_gla import DimGatedPoolCompressor


def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


class MultiScaleDeltaNet(nn.Module):
    r"""
    Multi-Scale DeltaNet — compressor+concat applied to DeltaNet.

    A separate *compressor* pathway (using DimGatedPoolCompressor) produces
    extra key/query dimensions, concatenated with the standard DeltaNet
    projections.  This gives a single rectangular state of size (K+K/r × V)
    — analogous to MS-GLA but with the delta rule instead of gated accumulation.

    Args:
        mode (str): Which DeltaNet kernel to use. Default: ``chunk``.
        d_model (int): Hidden size of the input. Default: 1024.
        expand_k (float): Expansion ratio for the key dim. Default: 1.0.
        expand_v (float): Expansion ratio for the value dim. Default: 1.0.
        num_heads (int): Number of heads. Default: 4.
        scale_ratio (int): Compression ratio for the auxiliary pathway.
            Extra capacity = K/r per head. Default: 4.
        use_beta (bool): Whether to use beta. Default: ``True``.
        use_gate (bool): Whether to use output gate. Default: ``False``.
        use_short_conv (bool): Whether to use short convolutions (main path only).
            Default: ``True``.
        conv_size (int): Kernel size of the short convolution. Default: 4.
        conv_bias (bool): Whether to use bias in short conv. Default: ``False``.
        allow_neg_eigval (bool): Allow negative eigenvalues. Default: ``False``.
        layer_idx (int, optional): Index of the layer.
        norm_eps (float): Epsilon for RMSNorm. Default: 1e-5.
        qk_activation (str): Activation for q/k. Default: ``silu``.
        qk_norm (str): Normalization for q/k. Default: ``l2``.
    """

    def __init__(
        self,
        mode: str = 'chunk',
        d_model: int = None,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        scale_ratio: int = 4,
        use_beta: bool = True,
        use_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        qk_activation: str = 'silu',
        qk_norm: str = 'l2',
        **kwargs
    ):
        super().__init__()

        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        assert self.qk_activation in ['silu', 'relu', 'elu', 'identity']
        assert self.qk_norm in ['l2', 'sum']

        hidden_size = int(d_model)
        self.hidden_size = d_model
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.scale_ratio = scale_ratio
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx

        self.silu = nn.SiLU()
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # Compressed (extra) key dimension
        assert self.head_k_dim % scale_ratio == 0, (
            f"head_k_dim ({self.head_k_dim}) must be divisible by scale_ratio ({scale_ratio})"
        )
        self.extra_head_k_dim = self.head_k_dim // scale_ratio
        self.extra_key_dim = self.extra_head_k_dim * num_heads  # K/r (full)
        self.aug_head_k_dim = self.head_k_dim + self.extra_head_k_dim  # K + K/r (per head)

        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        # ---------- main pathway (standard DeltaNet) ----------
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # When to use gated-pooling compressor vs plain Linear:
        #   extra_key_dim = hidden_size * expand_k / scale_ratio
        #   DimGatedPoolCompressor outputs hidden_size / scale_ratio
        #   They match iff expand_k == 1.0
        _use_pool = (expand_k == 1.0 and hidden_size % self.extra_key_dim == 0)

        # ---------- compressor pathway (extra dims, gated pooling) ----------
        if _use_pool:
            self.q_compress = DimGatedPoolCompressor(hidden_size, compress_ratio=scale_ratio)
            self.k_compress = DimGatedPoolCompressor(hidden_size, compress_ratio=scale_ratio)
        else:
            self.q_compress = nn.Linear(hidden_size, self.extra_key_dim, bias=False)
            self.k_compress = nn.Linear(hidden_size, self.extra_key_dim, bias=False)

        # ---------- short conv (main path only) ----------
        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu' if qk_activation == 'silu' else None
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu' if qk_activation == 'silu' else None
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation='silu'
            )

        # ---------- beta (per-head gate for delta rule) ----------
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # ---------- output gate / norm ----------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        # change to inference mode for short sequences
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # ---------- main pathway (with optional short conv) ----------
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            position_ids = kwargs.get('position_ids', None)

            q_main, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_q,
                output_final_state=use_cache,
                seq_idx=position_ids)
            k_main, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_k,
                output_final_state=use_cache,
                seq_idx=position_ids)
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_v,
                output_final_state=use_cache,
                seq_idx=position_ids)
        else:
            q_main = self.q_proj(hidden_states)
            k_main = self.k_proj(hidden_states)
            if self.qk_activation == 'silu':
                q_main, k_main = self.silu(q_main), self.silu(k_main)
            v = self.silu(self.v_proj(hidden_states))
            conv_state_q, conv_state_k, conv_state_v = None, None, None

        # ---------- compressor pathway (no short conv) ----------
        q_extra = self.q_compress(hidden_states)
        k_extra = self.k_compress(hidden_states)
        # Apply same activation as main path to ensure compatible feature distributions
        if self.qk_activation == 'silu':
            # silu is applied to q_main/k_main either via short conv (use_short_conv=True)
            # or explicitly (use_short_conv=False). Apply to compressor outputs too.
            q_extra, k_extra = self.silu(q_extra), self.silu(k_extra)

        # ---------- concat main + compressed ----------
        q = torch.cat([q_main, q_extra], dim=-1)  # [B, T, K + K/r]
        k = torch.cat([k_main, k_extra], dim=-1)  # [B, T, K + K/r]

        # reshape to multi-head
        q = rearrange(q, 'b t (h d) -> b t h d', d=self.aug_head_k_dim)
        k = rearrange(k, 'b t (h d) -> b t h d', d=self.aug_head_k_dim)
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)

        # q/k activations (if not already applied via short conv silu)
        if not self.use_short_conv:
            if self.qk_activation == 'relu':
                q, k = q.relu(), k.relu()
            elif self.qk_activation == 'elu':
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation == 'identity':
                pass
            elif self.qk_activation == 'silu':
                # already applied above
                pass
            else:
                raise NotImplementedError

        # q/k normalization
        if self.qk_norm == 'sum':
            q = sum_norm(q).to(q)
            k = sum_norm(k).to(k)
        # 'l2' normalization is handled inside the kernel via use_qk_l2norm_in_kernel

        # ---------- beta ----------
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = q.new_ones(q.shape[0], q.shape[1], q.shape[2])

        if self.allow_neg_eigval:
            beta = beta * 2.

        # dealing with padding
        if attention_mask is not None:
            beta = beta.mul(attention_mask[:, -beta.shape[-2]:, None])

        # ---------- single DeltaNet kernel call ----------
        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        cu_seqlens = kwargs.get('cu_seqlens', None)

        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_delta_rule(
                q=q.to(torch.bfloat16),
                k=k.to(torch.bfloat16),
                v=v.to(torch.bfloat16),
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True if self.qk_norm == 'l2' else False
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_delta_rule(
                q=q.to(torch.bfloat16),
                k=k.to(torch.bfloat16),
                v=v.to(torch.bfloat16),
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True if self.qk_norm == 'l2' else False
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        o = o.float()

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )

        # ---------- output gate / norm ----------
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o

    def state_size(self, **kwargs) -> int:
        """Total state size (rectangular: K+K/r × V per head)."""
        return self.num_heads * self.aug_head_k_dim * self.head_v_dim