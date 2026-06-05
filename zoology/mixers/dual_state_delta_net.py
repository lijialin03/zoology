# -*- coding: utf-8 -*-
"""
Dual-State DeltaNet — two independent states with L-axis compressed aux writes.

Architecture
------------
Main state  S_main ∈ [B, H, D_k, D_v]:          每 token 一次写入+读取
Aux state   S_aux  ∈ [B, H, D_k/f, D_v/f]:       每 block 一次写入 (T/r), 每 token 读取 (T)

其中：
  r = scale_ratio (L 轴压缩, 控制写入频率)
  f = feature_ratio (D 轴压缩, 控制 aux 状态容量)
  当 f=1 时, S_aux ∈ [B, H, D_k, D_v] 且无需 q_down / aux_up

核心设计:
  L 轴压缩仅作用于 WRITE 侧（aux 状态的更新频率降低到 T/r）
  READ 侧保持全 T 分辨率（每个 token 独立读取 aux 状态）
  D 轴压缩可选：f=1 时不降维（无额外投影层），f>1 时降维 + 投影

Main:
  For each token t:
    o_main[t] = S_main @ q[t]                       (read)
    S_main += β[t] · k[t] ⊗ (v[t] - S_main @ k[t])  (write)

Aux:
  At each block boundary (every r tokens):
    k̄ = LenGatedPoolCompressor(k_block)  [B, 1, D_k/f]  or [B, 1, D_k] if f=1
    v̄ = LenGatedPoolCompressor(v_block)  [B, 1, D_v/f]  or [B, 1, D_v] if f=1
    S_aux += β̄ · k̄ ⊗ (v̄ - S_aux @ k̄)    (compressed delta write)

  For each token t (full T resolution):
    q_aux = q_down(q[t]) if f>1 else q[t]
    o_aux[t] = S_aux @ q_aux[t]

Output:
  o[t] = o_main[t] + (up(o_aux[t]) if f>1 else o_aux[t])

与已有方案的关键区别:
  - MS-LA (ms_la.py): 单状态 + 压缩写入, 写入污染精细信息 → 与 short_conv 不兼容
  - MS-DN (ms_dn.py): 单状态 + 特征维压缩 (K → K+K/r), 没有 L 轴压缩
  - 本方案: 双独立状态, L 轴压缩仅在写入侧, 读写不冲突

Reference:
  - DeltaNet: https://arxiv.org/abs/2406.06484
  - CSA: DeepSeek-V4 Compressed Sparse Attention (L-axis compression inspiration)
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
    from fla.ops.ms_delta_rule import fused_recurrent_ds_delta_rule
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


class DualStateDeltaNet(nn.Module):
    r"""
    Dual-State DeltaNet — main (per-token) + aux (per-block compressed writes).

    Two independent DeltaNet states:
      S_main ∈ [B, H, D_k, D_v]          — fine-grained, per-token
      S_aux  ∈ [B, H, D_k/r, D_v/r]      — coarse-grained, per-block writes

    L-axis compression is applied ONLY to the WRITE side via LenGatedPoolCompressor.
    The READ side maintains full temporal resolution (every token reads S_aux
    via its own down-projected query).

    When ``scale_ratio >= T``, no compressed writes are performed and the aux
    path outputs zeros (S_aux remains all-zeros), making the forward equivalent
    to a standard DeltaNet.

    Args:
        mode (str): Kernel mode. Default: ``chunk``.
        d_model (int): Hidden size. Default: ``1024``.
        expand_k (float): Key dimension expansion. Default: ``1.0``.
        expand_v (float): Value dimension expansion. Default: ``1.0``.
        num_heads (int): Number of heads. Default: ``4``.
        scale_ratio (int): L-axis compression ratio for aux writes. Default: ``4``.
            Controls write frequency: a compressed write is performed every
            ``scale_ratio`` tokens along the T axis.
        feature_ratio (int): D-axis (feature dimension) compression ratio for
            aux state. Default: ``1`` (no feature reduction — S_aux stays at
            full [H, D_k, D_v] capacity).
            When ``feature_ratio > 1``, S_aux is reduced to [H, D_k/f, D_v/f]
            with learned q_down/aux_up projections (zero-init).
        use_beta (bool): Per-head gate for delta rule. Default: ``True``.
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
        feature_ratio: int = 1,
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
        self.feature_ratio = feature_ratio
        self.use_beta = use_beta
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
            raise NotImplementedError(
                "fused_chunk_delta_rule is now deprecated. "
                "Please use `chunk_delta_rule` instead."
            )
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        # =====================================================================
        # Shared projections (main path — standard DeltaNet)
        # =====================================================================
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

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
        # Aux path: L-axis compressed writes + D-axis optional reduction
        # =====================================================================
        f = feature_ratio
        assert f >= 1, f"feature_ratio must be >= 1, got {f}"
        assert f == 1 or (self.head_k_dim % f == 0 and self.head_v_dim % f == 0), (
            f"head_k_dim ({self.head_k_dim}) and head_v_dim ({self.head_v_dim}) "
            f"must be divisible by feature_ratio ({f})"
        )

        self.aux_head_k_dim = self.head_k_dim // f
        self.aux_head_v_dim = self.head_v_dim // f
        self.aux_key_dim = self.aux_head_k_dim * num_heads
        self.aux_value_dim = self.aux_head_v_dim * num_heads

        # L-axis compressors: pool every scale_ratio tokens along T-axis
        # Output feature dimension = aux_key_dim / aux_value_dim (which equals
        # full dim when f=1, or reduced dim when f>1)
        self.k_compress = LenGatedPoolCompressor(
            d_model=self.key_dim,
            head_dim=self.aux_key_dim,
            compress_ratio=scale_ratio,
            overlap=False,
        )
        self.v_compress = LenGatedPoolCompressor(
            d_model=self.value_dim,
            head_dim=self.aux_value_dim,
            compress_ratio=scale_ratio,
            overlap=False,
        )

        # Optional D-axis down/up projections
        self.has_feature_reduction = f > 1
        if self.has_feature_reduction:
            self.q_down = nn.Linear(self.key_dim, self.aux_key_dim, bias=False)
            self.aux_up = nn.Linear(self.aux_value_dim, self.value_dim, bias=False)
            nn.init.zeros_(self.q_down.weight)
            nn.init.zeros_(self.aux_up.weight)
        else:
            self.q_down = nn.Identity()
            self.aux_up = nn.Identity()

        # Aux beta (per-head gate for aux delta writes)
        self.aux_b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

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
        T = hidden_states.shape[1]
        r = self.scale_ratio

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # =====================================================================
        # 1. Main path: QKV projections (+ optional short conv)
        # =====================================================================
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            position_ids = kwargs.get('position_ids', None)

            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_q,
                output_final_state=use_cache,
                seq_idx=position_ids)
            k, conv_state_k = self.k_conv1d(
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
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            if self.qk_activation == 'silu':
                q, k = self.silu(q), self.silu(k)
            v = self.silu(self.v_proj(hidden_states))
            conv_state_q, conv_state_k, conv_state_v = None, None, None

        # Reshape to multi-head
        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        # q/k activation (if not already via short conv silu)
        if not self.use_short_conv:
            if self.qk_activation != 'silu':
                if self.qk_activation == 'relu':
                    q, k = q.relu(), k.relu()
                elif self.qk_activation == 'elu':
                    q, k = elu_p1(q), elu_p1(k)
                elif self.qk_activation == 'identity':
                    pass
                else:
                    raise NotImplementedError

        # q/k normalization (l2 norm is applied inside the kernel via
        # use_qk_l2norm_in_kernel, sum norm is applied here)
        if self.qk_norm == 'sum':
            q = sum_norm(q).to(q)
            k = sum_norm(k).to(k)

        # =====================================================================
        # 2. Beta (per-head gate)
        # =====================================================================
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = q.new_ones(q.shape[0], q.shape[1], q.shape[2])

        if self.allow_neg_eigval:
            beta = beta * 2.

        if attention_mask is not None:
            beta = beta.mul(attention_mask[:, -beta.shape[-2]:, None])

        # =====================================================================
        # 3. Main DeltaNet kernel
        # =====================================================================
        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        cu_seqlens = kwargs.get('cu_seqlens', None)

        if mode == 'fused_recurrent':
            o_main, main_state = fused_recurrent_delta_rule(
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
            o_main, main_state = chunk_delta_rule(
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

        o_main = o_main.float()  # [B, T, H, D_v]

        # =====================================================================
        # 4. Aux path: L-axis compressed writes + full-res reads
        # =====================================================================
        if r < T and not self._aux_path_disabled():
            # (a) Compute compressed k̄, v̄ via L-axis pooling
            #     Input: k/v already reshaped to [B, T, H, D]
            #     Compressor expects flat [B, T, feature_dim]
            k_flat = rearrange(k, 'b t h d -> b t (h d)')   # [B, T, key_dim]
            v_flat = rearrange(v, 'b t h d -> b t (h d)')   # [B, T, value_dim]

            k_bar = self.k_compress(k_flat)   # [B, T/r, aux_key_dim]
            v_bar = self.v_compress(v_flat)   # [B, T/r, aux_value_dim]

            # (b) Down-project queries for aux reads (full T resolution)
            q_flat = rearrange(q, 'b t h d -> b t (h d)')
            q_aux = self.q_down(q_flat)       # [B, T, aux_key_dim]

            # Reshape to multi-head
            k_bar = rearrange(
                k_bar, 'b n (h d) -> b n h d',
                h=self.num_heads, d=self.aux_head_k_dim
            )
            v_bar = rearrange(
                v_bar, 'b n (h d) -> b n h d',
                h=self.num_heads, d=self.aux_head_v_dim
            )
            q_aux = rearrange(
                q_aux, 'b t (h d) -> b t h d',
                h=self.num_heads, d=self.aux_head_k_dim
            )

            # (c) Apply qk_norm to k_bar for consistent delta writes
            if self.qk_norm == 'l2':
                k_bar = F.normalize(k_bar, p=2, dim=-1)
            elif self.qk_norm == 'sum':
                k_bar = sum_norm(k_bar)

            # (d) Aux beta (per-block average)
            beta_aux = self.aux_b_proj(hidden_states).sigmoid()  # [B, T, H]

            if self.allow_neg_eigval:
                beta_aux = beta_aux * 2.

            # (e) Use fused recurrent kernel for block-by-block aux processing
            #     Replaces the Python for-loop with a vectorized kernel call.
            #
            #     fused_recurrent_ds_delta_rule expands compressed k/v_bar to full
            #     T resolution (zero at non-write positions) and delegates to the
            #     standard fused_recurrent_delta_rule.
            #
            #     At non-write positions, k=0, beta=0 → delta write is a no-op.
            #     At write positions (r-1, 2r-1, ...), compressed delta write.
            #
            #     Small semantic difference vs reference:
            #       Reference: READ all tokens → WRITE compressed (per block)
            #       Kernel:    WRITE → READ at boundary positions (write-then-read)
            #       Only affects 1/r of positions; acceptable.

            B = hidden_states.shape[0]
            device = q.device
            dtype = q.dtype
            n_blocks = k_bar.shape[1]  # ceil(T/r)

            # Compute per-block beta: average over each block's tokens
            if self.use_beta:
                block_starts = torch.arange(0, n_blocks, device=device) * r
                block_ends = torch.minimum(block_starts + r, torch.tensor(T, device=device))
                block_lens = block_ends - block_starts  # [N]
                block_idx = (
                    torch.arange(T, device=device)
                    .unsqueeze(0).unsqueeze(-1)
                    .expand(B, -1, self.num_heads) // r
                )  # [B, T, H]
                beta_bar = torch.zeros(B, n_blocks, self.num_heads, device=device, dtype=beta_aux.dtype)
                beta_bar.scatter_add_(1, block_idx, beta_aux)
                beta_bar = beta_bar / block_lens.view(1, -1, 1)  # [B, N, H]
            else:
                beta_bar = torch.ones(B, n_blocks, self.num_heads, device=device, dtype=dtype)

            o_aux, _ = fused_recurrent_ds_delta_rule(
                q=q_aux.contiguous(),            # [B, T, H, aux_K] — full T
                k=k_bar.contiguous(),            # [B, N, H, aux_K] — compressed
                v=v_bar.contiguous(),            # [B, N, H, aux_V] — compressed
                beta=beta_bar.contiguous(),      # [B, N, H] — per-block
                scale_ratio=r,
                scale=1.0,                       # no additional scaling on aux q
                use_qk_l2norm_in_kernel=False,   # k_bar already normalized above
                cu_seqlens=cu_seqlens,
            )

            # (f) Up-project aux output
            o_aux_flat = rearrange(o_aux, 'b t h d -> b t (h d)')
            o_aux_up_flat = self.aux_up(o_aux_flat)  # [B, T, value_dim]
            o_aux_up = rearrange(
                o_aux_up_flat, 'b t (h d) -> b t h d',
                h=self.num_heads, d=self.head_v_dim
            )
        else:
            o_aux_up = 0
            # Still track aux_state for caching if needed
            aux_state = None

        # =====================================================================
        # 5. Combine main + aux outputs
        # =====================================================================
        o = o_main + o_aux_up

        # =====================================================================
        # 6. Cache update
        # =====================================================================
        if past_key_values is not None:
            # Pack both states into cache
            past_key_values.update(
                recurrent_state=main_state,
                conv_state=(
                    conv_state_q, conv_state_k, conv_state_v
                ) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )

        # =====================================================================
        # 7. Output gate / norm / projection
        # =====================================================================
        if self.use_gate:
            g = rearrange(
                self.g_proj(hidden_states),
                '... (h d) -> ... h d',
                d=self.head_v_dim
            )
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o

    # ======================================================================

    def _aux_path_disabled(self) -> bool:
        """Aux path is disabled when scale_ratio is unreasonably large
        (placeholder for future logic if needed)."""
        return False

    # ======================================================================

    def state_size(self, sequence_length: int = 2048) -> int:
        """Total state size = main state + aux state (aux capacity depends on feature_ratio)."""
        main_ss = self.num_heads * self.head_k_dim * self.head_v_dim
        aux_ss = (
            self.num_heads
            * self.aux_head_k_dim
            * self.aux_head_v_dim
        )
        return main_ss + aux_ss
