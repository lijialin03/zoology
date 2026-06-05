# -*- coding: utf-8 -*-
"""
Multi-Scale Linear Attention (MSLA) — single-state with per-block compressed writes.

Architecture
------------
CSA 的核心是 Window KV + Compressed KV 提供不同 L 轴分辨率的视图:
  - Window KV: 精细的逐 token 上下文
  - Compressed KV: 粗粒度的全局上下文（L 轴压缩）

在线性注意力（DeltaNet）中, 在同一个状态上做两种粒度的写入:
  S ∈ R^{D×D} (单个状态, state_size 与标准 DeltaNet 完全一致)

  逐 token 写入 (main view, fine):
    S += β · k ⊗ (v − S@k)

  每 block 末额外写入 (compressed view, coarse):
    k̄, v̄ = LenGatedPoolCompressor(k_block, v_block)  ← L 轴门控压缩
    S += β̄ · k̄ ⊗ (v̄ − S@k̄)

每个 token 的读取方式不变: o_t = S @ q_t

核心机制: block 末的压缩写将当前 block 的粗粒度摘要注入 S,
影响后续 token 的读取。 这样 S 同时包含逐 token 的精细信息和
逐 block 的粗粒度信息, 形成多尺度时间感受野。

优势:
  - state_size 不变 (与标准 DeltaNet 相同), 公平对照
  - 无 S_aux, 无额外 einsum 读取, 无 repeat_interleave
  - 与 decode 阶段自然兼容: token 逐个到达, block 满时做一次额外写
  - 行为类似 DeepSeek-V4 CSA 的 KV 缓存: 攒够 compress_ratio 个再压缩写入

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
        # 7. Main computation
        # =====================================================================
        if self.scale_ratio >= hidden_states.shape[1]:
            # ── No compressed writes possible: behave exactly like DeltaNet ──
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
            T = hidden_states.shape[1]
            r = self.scale_ratio
            # ── Block-by-block with compressed writes ──
            n_blocks = (T + r - 1) // r
            outputs = []

            for b in range(n_blocks):
                start = b * r
                end = min((b + 1) * r, T)

                q_block = q[:, start:end]                    # [B, len, H, D_k]
                k_block = k[:, start:end]
                v_block = v[:, start:end]
                beta_block = beta[:, start:end]              # [B, len, H]

                # (a) Per-token DeltaNet on this block
                o_block, block_state = fused_recurrent_delta_rule(
                    q=q_block.to(torch.bfloat16),
                    k=k_block.to(torch.bfloat16),
                    v=v_block.to(torch.bfloat16),
                    beta=beta_block,
                    initial_state=recurrent_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=(self.qk_norm == 'l2')
                )
                outputs.append(o_block.float())              # [B, len, H, D_v]

                # (b) Extra compressed write at block boundary
                k_flat = rearrange(k_block, 'b t h d -> b t (h d)')
                v_flat = rearrange(v_block, 'b t h d -> b t (h d)')

                k_bar = self.k_compress(k_flat)              # [B, 1, key_dim]
                v_bar = self.v_compress(v_flat)              # [B, 1, value_dim]

                k_bar = rearrange(k_bar, 'b n (h d) -> b n h d',
                                  h=self.num_heads, d=self.head_k_dim)
                v_bar = rearrange(v_bar, 'b n (h d) -> b n h d',
                                  h=self.num_heads, d=self.head_v_dim)

                # Per-block beta: mean of per-token betas in this block
                beta_bar = beta_block.mean(dim=1, keepdim=True)  # [B, 1, H]

                # DeltaNet write+read on the same state using compressed (k̄, v̄)
                # o_bar is the read output: state queried by compressed key
                o_bar, block_state = fused_recurrent_delta_rule(
                    q=k_bar.to(torch.bfloat16),
                    k=k_bar.to(torch.bfloat16),
                    v=v_bar.to(torch.bfloat16),
                    beta=beta_bar,
                    initial_state=block_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=(self.qk_norm == 'l2')
                )

                # # Accumulate compressed read into the last token of the block
                # outputs[-1] = outputs[-1] + o_bar.float()

                # Carry final state to next block
                recurrent_state = block_state

            o = torch.cat(outputs, dim=1)                    # [B, T, H, D_v]

            # State is only meaningful for caching when use_cache=True
            if not use_cache:
                recurrent_state = None

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
