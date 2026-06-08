# -*- coding: utf-8 -*-
"""
Multi-Scale Linear Attention (MSLA) — single DeltaNet state with compressed
per-block extra writes.

Architecture
------------
Single DeltaNet state S ∈ R^{D_k × D_v} receives two types of writes:

  1. Per-token writes (standard DeltaNet) at every position t:
       S += β[t] · k[t] ⊗ (v[t] - S @ k[t])

  2. Compressed writes at each block boundary:
       k̄, v̄ = LenGatedPoolCompressor(k_block, v_block)   ← L-axis pooling
       S += β̄ · k̄ ⊗ (v̄ - S @ k̄)                          ← delta write

  Reads use the single state S at every position:
       o[t] = S @ q[t]

  Unlike CSA/HCA (where compressed KV is concatenated to the original KV
  sequence for softmax attention), linear attention's recurrent state has
  no "concatenation" concept — the compressed write injects coarse multi-
  token information directly into the same state via the delta rule.

When ``scale_ratio >= T``, no compressed writes are performed and the
forward is bit-exact with a standard DeltaNet (same kernel dispatch, mode
switching, cache handling, and cu_seqlens support).

Reference:
  - CSA: DeepSeek-V4 Compressed Sparse Attention
  - DeltaNet: https://arxiv.org/abs/2406.06484
  - LenGatedPoolCompressor: ../mixers/cla.py
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

from zoology.mixers.cla import LenGatedPoolCompressor


def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


class MultiScaleLinearAttention(nn.Module):
    r"""
    Multi-Scale Linear Attention — single DeltaNet state with extra
    per-block compressed writes.

    The standard DeltaNet state S ∈ R^{D×D} is updated per-token.
    At each block boundary (every ``scale_ratio`` tokens), an additional
    write is performed using L-axis compressed (k̄, v̄) from the block,
    injecting coarse multi-token information into the same state.

    The ``state_size`` is identical to a standard DeltaNet with the
    same ``expand_k``, ``expand_v``, and ``num_heads`` — no extra
    parameters besides the lightweight ``LenGatedPoolCompressor``.

    When ``scale_ratio >= T``, the forward matches DeltaNet bit-exactly
    (same kernel dispatch, mode switching, cache, cu_seqlens).

    Args:
        mode (str): Kernel mode. Default: ``chunk``.
        d_model (int): Hidden size. Default: ``1024``.
        expand_k (float): Key dimension expansion. Default: ``1.0``.
        expand_v (float): Value dimension expansion. Default: ``1.0``.
        num_heads (int): Number of heads. Default: ``4``.
        scale_ratio (int): L-axis compression ratio. Default: ``4``.
            A compressed write is performed every ``scale_ratio`` tokens.
        use_beta (bool): Beta gate. Default: ``True``.
        use_gate (bool): Output gate. Default: ``False``.
        use_short_conv (bool): Short conv. Default: ``True``.
        conv_size (int): Short conv kernel size. Default: ``4``.
        conv_bias (bool): Short conv bias. Default: ``False``.
        compressed_beta_scale (float): Global multiplier for compressed beta.
            Values < 1.0 dampen compressed writes, > 1.0 amplify them.
            Setting to 0.0 disables compressed writes entirely (making MS-LA
            effectively a standard DeltaNet). Default: ``1.0``.
        use_compressed_gate (bool): Per-head learnable gate for compressed
            writes. Adds ``num_heads`` extra parameters, initialized to 1.0
            and sigmoid-constrained to (0, 1). Applied as:
            ``beta_bar = beta_bar * g_bar.sigmoid()``. Default: ``False``.
        allow_neg_eigval (bool): Allow negative eigenvalues. Default: ``False``.
        layer_idx (int, optional): Layer index.
        norm_eps (float): RMSNorm epsilon. Default: ``1e-5``.
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
        compressed_beta_scale: float = 1.0,
        use_compressed_gate: bool = False,
        allow_neg_eigval: bool = False,
        layer_idx: int = None,
        qk_activation: str = 'silu',
        qk_norm: str = 'l2',
        norm_eps: float = 1e-5,
        **kwargs
    ):
        super().__init__()

        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        assert self.qk_activation in ['silu', 'relu', 'elu', 'identity']
        assert self.qk_norm in ['l2', 'sum']

        hidden_size = int(d_model)
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.scale_ratio = scale_ratio
        self.compressed_beta_scale = compressed_beta_scale
        self.use_compressed_gate = use_compressed_gate
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.silu = nn.SiLU()
        if mode == 'fused_chunk':
            raise NotImplementedError("fused_chunk_delta_rule is now deprecated. Please use `chunk_delta_rule` instead.")
        assert mode in ['chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."
        assert self.hidden_size % num_heads == 0, f"hidden_size must be divisible by num_heads of {num_heads}"
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
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
        
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # =====================================================================
        # L-axis compressors for per-block extra writes
        # =====================================================================
        self.k_compress = LenGatedPoolCompressor(
            d_model=self.key_dim,
            head_dim=self.key_dim,
            compress_ratio=scale_ratio,
            overlap=False,
        )
        self.v_compress = LenGatedPoolCompressor(
            d_model=self.value_dim,
            head_dim=self.value_dim,
            compress_ratio=scale_ratio,
            overlap=False,
        )

        # =====================================================================
        # Compressed write gate/scale — explicit control over the strength of
        # per-block compressed writes into the recurrent state.
        #
        # compressed_beta_scale (float):
        #   A constant multiplier applied to the compressed beta (averaged over
        #   the block). Values < 1.0 dampen compressed writes; > 1.0 amplify
        #   them. Setting to 0.0 disables compressed writes entirely, making
        #   MS-LA bit-exact with DeltaNet even when scale_ratio < T.
        #
        # use_compressed_gate (bool):
        #   When True, introduces a learnable per-head gate g_bar ∈ R^H,
        #   initialized to 1.0, applied as:
        #       beta_bar = beta_bar * g_bar.sigmoid()
        #   The sigmoid constrains the gate to (0, 1), and the 1.0 init
        #   means it starts from "no modification" and gradually adjusts.
        #   The gate is a single scalar per head (num_heads params total),
        #   adding negligible parameter overhead.
        # =====================================================================
        if self.use_compressed_gate:
            # Per-head learnable gate, init to 1.0 → sigmoid(1.0) ≈ 0.731
            self.compressed_gate = nn.Parameter(
                torch.ones(self.num_heads)
            )
        else:
            self.compressed_gate = None

    # ======================================================================
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
        
        # change to inference mode.
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # =====================================================================
        # standard DeltaNet
        # =====================================================================
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            position_ids = kwargs.get('position_ids', None)
            q, conv_state_q = self.q_conv1d(x=self.q_proj(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_q,
                                            output_final_state=use_cache,
                                            seq_idx=position_ids)
            k, conv_state_k = self.k_conv1d(x=self.k_proj(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_k,
                                            output_final_state=use_cache,
                                            seq_idx=position_ids)
            v, conv_state_v = self.v_conv1d(x=self.v_proj(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_v,
                                            output_final_state=use_cache,
                                            seq_idx=position_ids)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            if self.qk_activation == 'silu':
                q, k = self.silu(q), self.silu(k)
            v = self.silu(self.v_proj(hidden_states))
        
        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.qk_activation != 'silu':
            if self.qk_activation == 'relu':
                q, k = q.relu(), k.relu()
            elif self.qk_activation == 'elu':
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation == 'identity':
                pass
            else:
                raise NotImplementedError

        if self.qk_norm == 'sum':
            q = sum_norm(q).to(q)
            k = sum_norm(k).to(k)
        
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = q.new_ones(q.shape[0], q.shape[1], q.shape[2])

        if self.allow_neg_eigval:
            beta = beta * 2.

        if attention_mask is not None:
            beta = beta.mul(attention_mask[:, -beta.shape[-2]:, None])

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        cu_seqlens = kwargs.get('cu_seqlens', None)

        # =====================================================================
        # 7. Main computation — single-state DeltaNet with per-block compressed writes
        # =====================================================================
        if self.scale_ratio >= hidden_states.shape[1]:
            # ── No compressed writes: bit-exact with standard DeltaNet ──
            if mode == 'fused_recurrent':
                o, recurrent_state = fused_recurrent_delta_rule(
                    q=q.to(torch.bfloat16),
                    k=k.to(torch.bfloat16),
                    v=v.to(torch.bfloat16),
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                    # head_first=False,  # not supported in current FLA version
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
                    # head_first=False,  # not supported in current FLA version
                    use_qk_l2norm_in_kernel=True if self.qk_norm == 'l2' else False
                )
            else:
                raise NotImplementedError(f"Not supported mode `{mode}`.")
            o = o.float()                                    # [B, T, H, D_v]
        else:
            # ── Single state: per-token writes + per-block compressed writes ──
            # Build an augmented sequence by inserting one compressed pseudo-token
            # after each block. A single DeltaNet call then applies both normal
            # token writes and compressed boundary writes in the same state.
            if cu_seqlens is not None:
                raise NotImplementedError(
                    "MultiScaleLinearAttention augmented compressed path does not support cu_seqlens."
                )

            T, B = hidden_states.shape[1], hidden_states.shape[0]
            r = self.scale_ratio
            device = q.device
            n_blocks = (T + r - 1) // r
            T_aug = T + n_blocks

            # L-axis pooling → one compressed (k̄, v̄) per block.
            k_flat = rearrange(k, 'b t h d -> b t (h d)')
            v_flat = rearrange(v, 'b t h d -> b t (h d)')
            k_bar = self.k_compress(k_flat)    # [B, N, key_dim]
            v_bar = self.v_compress(v_flat)    # [B, N, value_dim]
            k_bar = rearrange(k_bar, 'b n (h d) -> b n h d',
                              h=self.num_heads, d=self.head_k_dim)
            v_bar = rearrange(v_bar, 'b n (h d) -> b n h d',
                              h=self.num_heads, d=self.head_v_dim)

            # Normalize k_bar for consistent delta writes.
            if self.qk_norm == 'l2':
                k_bar = F.normalize(k_bar.float(), p=2, dim=-1).to(k_bar.dtype)
            elif self.qk_norm == 'sum':
                k_bar = sum_norm(k_bar)

            # Average beta over each block, matching the old per-block loop.
            beta_padded = beta.new_zeros(B, n_blocks * r, self.num_heads)
            beta_padded[:, :T] = beta
            block_lens = torch.full((n_blocks,), r, device=device, dtype=beta.dtype)
            block_lens[-1] = T - (n_blocks - 1) * r
            beta_bar = beta_padded.view(B, n_blocks, r, self.num_heads).sum(dim=2)
            beta_bar = beta_bar / block_lens.view(1, n_blocks, 1)

            # Apply compressed write controls:
            #   1. compressed_beta_scale: global strength multiplier
            #   2. use_compressed_gate: per-head learnable gate (sigmoid-constrained to (0, 1))
            if self.compressed_beta_scale != 1.0:
                beta_bar = beta_bar * self.compressed_beta_scale
            if self.use_compressed_gate and self.compressed_gate is not None:
                beta_bar = beta_bar * self.compressed_gate.sigmoid()  # [H] broadcast over [B, N, H]

            # Interleave real tokens and compressed pseudo-tokens:
            #   real block b → positions [b*(r+1), ..., b*(r+1)+len-1]
            #   compressed b → immediately after that block's last real token.
            real_idx = torch.arange(T, device=device)
            real_pos = real_idx + real_idx // r
            block_idx = torch.arange(n_blocks, device=device)
            block_ends = torch.clamp((block_idx + 1) * r, max=T)
            compressed_pos = block_ends + block_idx

            q_aug = q.new_zeros(B, T_aug, self.num_heads, self.head_k_dim)
            k_aug = k.new_zeros(B, T_aug, self.num_heads, self.head_k_dim)
            v_aug = v.new_zeros(B, T_aug, self.num_heads, self.head_v_dim)
            beta_aug = beta.new_zeros(B, T_aug, self.num_heads)

            q_aug[:, real_pos] = q
            k_aug[:, real_pos] = k
            v_aug[:, real_pos] = v
            beta_aug[:, real_pos] = beta

            # Pseudo-token outputs are discarded; q is set nonzero only to keep
            # the kernel path numerically ordinary when q/k normalization is on.
            q_aug[:, compressed_pos] = k_bar.to(q_aug.dtype)
            k_aug[:, compressed_pos] = k_bar.to(k_aug.dtype)
            v_aug[:, compressed_pos] = v_bar.to(v_aug.dtype)
            beta_aug[:, compressed_pos] = beta_bar

            if mode == 'fused_recurrent':
                o_aug, recurrent_state = fused_recurrent_delta_rule(
                    q=q_aug.to(torch.bfloat16),
                    k=k_aug.to(torch.bfloat16),
                    v=v_aug.to(torch.bfloat16),
                    beta=beta_aug,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    cu_seqlens=None,
                    use_qk_l2norm_in_kernel=(self.qk_norm == 'l2'),
                )
            elif mode == 'chunk':
                o_aug, recurrent_state = chunk_delta_rule(
                    q=q_aug.to(torch.bfloat16),
                    k=k_aug.to(torch.bfloat16),
                    v=v_aug.to(torch.bfloat16),
                    beta=beta_aug,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    cu_seqlens=None,
                    use_qk_l2norm_in_kernel=(self.qk_norm == 'l2'),
                )
            else:
                raise NotImplementedError(f"Not supported mode `{mode}`.")

            o = o_aug[:, real_pos].float()

        # =====================================================================
        # 8. Update cache
        # =====================================================================
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )

        # =====================================================================
        # 9. Output gate / norm / projection
        # =====================================================================
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o#, None, past_key_values

    # ======================================================================
    def state_size(self, sequence_length: int=2048):
        state_size = (
            self.num_heads * self.head_k_dim * self.head_v_dim
        )
        return state_size
