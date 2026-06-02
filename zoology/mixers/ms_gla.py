# -*- coding: utf-8 -*-
"""
Multi-Scale GLA (MS-GLA) — augmented single-state design, based on GLA.

Architecture (CSA-inspired)
----------------------------
Following DeepSeek-V4's CSA design, the "extra" key capacity comes from a
separate *compressor* pathway rather than a fused projection:

    q_main = q_proj(x)             ∈ R^{K}       (standard GLA)
    q_extra = q_compress(x)        ∈ R^{K/r}    (compressor)
    q = [q_main ‖ q_extra]         ∈ R^{K+K/r}  (concatenate)

Same for k and the gate gk.  v stays at V (rectangular state).

    S ∈ R^{(K+K/r) × V}
    g = sigmoid(gk)   (per-key-dimension decay gate)
    S = g * S + k ⊗ v
    o = S @ q         ∈ R^{V}

The compressor uses gated pooling over adjacent hidden-dimension groups
(mirroring GatedPoolCompressor on the L-axis), rather than a plain Linear.
This gives richer feature extraction for the auxiliary pathway.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

try:
    from fla.modules import FusedRMSNormSwishGate, RMSNorm
    from fla.ops.gla import chunk_gla, fused_recurrent_gla
except:
    assert 0, print("Need to install fla: pip install flash-linear-attention")

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


# ---------------------------------------------------------------------------
# D-dim gated-pooling compressor
# ---------------------------------------------------------------------------

class DimGatedPoolCompressor(nn.Module):
    """
    Compresses the hidden dimension via gated pooling over adjacent dim groups.

    For each token independently:
      1. Linear project: D → (1 or 2)×D  (``coff=2`` when overlap enabled)
      2. Split D into groups of size ``compress_ratio``
      3. Within each group, compute gating scores (Linear + softmax + APE)
      4. Weighted sum → D/r

    When *overlap* is True (ratio <= 4), adjacent groups share half their
    dimensions for smoother transitions — mirrors ``GatedPoolCompressor._overlap_transform``
    which shifts the first half of dim-channels one block forward.

    When ``compress_ratio`` does not evenly divide ``d_model``, falls back to
    a plain Linear projection (no gated pooling).
    """

    def __init__(
        self,
        d_model: int,
        compress_ratio: int = 4,
        overlap: bool = False,
        shuffle_groups: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.compress_ratio = compress_ratio
        self.overlap = overlap and compress_ratio <= 4
        self.shuffle_groups = shuffle_groups
        coff = 2 if self.overlap else 1

        # Check if gated pooling is applicable
        if d_model % compress_ratio == 0:
            self.n_groups = d_model // compress_ratio   # D/r (output dim)

            self.wkv = nn.Linear(d_model, coff * d_model, bias=False)
            self.wgate = nn.Linear(d_model, coff * d_model, bias=False)
            # Learnable position code: [n_groups, coff * r]
            self.ape = nn.Parameter(torch.empty(self.n_groups, coff * compress_ratio))
            self.norm = nn.RMSNorm(self.n_groups)
            self.pool = True

            nn.init.normal_(self.ape, mean=0.0, std=0.02)

            if shuffle_groups:
                self.register_buffer('shuffle_idx', torch.randperm(coff * d_model))
        else:
            self.linear = nn.Linear(d_model, d_model // compress_ratio, bias=False)
            self.pool = False

    # ------------------------------------------------------------------
    def _overlap_transform(self, x: torch.Tensor, fill_value: float):
        """
        Shift the overlap half of each group one group forward.

        Input:  [*, n_groups, 2*r]   where r = compress_ratio
        Output: [*, n_groups, 2*r]   — same shape, content rearranged

        The last ``r`` values per group = normal contribution (stays).
        The first ``r`` values per group = overlap contribution (shifted to
        the next group).  Group 0 gets padding because there is no group -1.
        """
        *_, n, coff_r = x.shape
        r = self.compress_ratio
        new = x.new_full((*_, n, 2 * r), fill_value)
        new[..., r:] = x[..., r:]          # normal half: keep in place
        new[..., 1:, :r] = x[..., :-1, :r]  # overlap half: shift right 1 group
        return new

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]

        Returns:
            [B, T, D/r]
        """
        if not self.pool:
            return self.linear(x)

        b, t, d = x.shape
        B = b * t

        kv = self.wkv(x)                                        # [B, T, coff*D]
        gate = self.wgate(x)                                    # [B, T, coff*D]

        # Optional: shuffle D-dim indices before grouping to destroy adjacency
        if self.shuffle_groups:
            kv = kv[..., self.shuffle_idx]
            gate = gate[..., self.shuffle_idx]

        coff_r = (2 if self.overlap else 1) * self.compress_ratio

        kv = kv.view(B, self.n_groups, coff_r)                  # [B*, D/r, 2r] or [B*, D/r, r]
        gate = gate.view(B, self.n_groups, coff_r)              # same

        if self.overlap:
            kv = self._overlap_transform(kv, 0.0)               # [B*, D/r, 2r] (rearranged)
            gate = self._overlap_transform(gate, float("-inf"))  # same

        # Gated pooling: softmax over elements within each group
        gate = (gate + self.ape.unsqueeze(0)).softmax(dim=-1)   # [B*, D/r, coff_r]
        out = (kv * gate).sum(dim=-1)                           # [B*, D/r]
        out = out.view(b, t, self.n_groups)                     # [B, T, D/r]
        out = self.norm(out)

        return out


class MultiScaleGLA(nn.Module):
    r"""
    Multi-Scale GLA — augmented single-state design, based on GLA.

    A separate *compressor* pathway produces extra key/query/gate dimensions,
    concatenated with the standard GLA projections.  This gives a single
    rectangular state of size (K+K/r × V) — analogous to how CSA concatenates
    window and compressed KV pools before attention.

    The compressor uses ``DimGatedPoolCompressor`` — gated pooling over
    adjacent hidden-dimension groups — rather than a plain Linear.  This
    mirrors ``GatedPoolCompressor`` (which pools on the L-axis) for richer
    feature extraction in the auxiliary pathway.  When the dimension math
    does not divide evenly, it falls back to a plain Linear.

    Args:
        mode (str):  Which GLA kernel to use.  Default: ``chunk``.
        d_model (int):  Hidden size of the input.  Default: 1024.
        expand_k (float):  Expansion ratio for the key dim.  Default: 1.0.
        expand_v (float):  Expansion ratio for the value dim.  Default: 1.0.
        num_heads (int):  Number of heads.  Default: 4.
        scale_ratio (int):  Compression ratio for the auxiliary pathway.
            Extra capacity = K/r per head.  Default: 4.
        use_compressor_overlap (bool):  Enable overlapping groups in the D-dim
            compressor (like GatedPoolCompressor).  Default: ``False``.
            Useless! the indexing order of hidden dims does not have a semantic structure.
        use_output_gate (bool):  Whether to use output gate.  Default: ``True``.
        gate_fn (str):  Activation for output gate.  Default: ``swish``.
        gate_logit_normalizer (int):  Normalizer for gate logits after logsigmoid.
            Default: 16.
        gate_low_rank_dim (int):  Low-rank dim for gate projection.  Default: 16.
        clamp_min (float, optional):  Minimum value for gate logits.
        fuse_norm (bool):  Fuse norm and output gate.  Default: ``True``.
        layer_idx (int, optional):  Index of the layer.
        norm_eps (float):  Epsilon for RMSNorm.  Default: 1e-5.
    """

    def __init__(
        self,
        mode: str = 'chunk',
        d_model: int = None,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        scale_ratio: int = 4,
        use_compressor_overlap: bool = False,
        use_output_gate: bool = True,
        gate_fn: str = 'swish',
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        clamp_min: Optional[float] = None,
        fuse_norm: bool = True,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs
    ) -> MultiScaleGLA:
        super().__init__()

        self.mode = mode

        hidden_size = int(d_model)
        self.hidden_size = d_model
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.scale_ratio = scale_ratio
        self.use_compressor_overlap = use_compressor_overlap
        self.use_output_gate = use_output_gate
        self.clamp_min = clamp_min
        self.layer_idx = layer_idx

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

        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        # ---------- main pathway (standard GLA) ----------
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
            self.q_compress = DimGatedPoolCompressor(hidden_size, compress_ratio=scale_ratio, overlap=use_compressor_overlap)
            self.k_compress = DimGatedPoolCompressor(hidden_size, compress_ratio=scale_ratio, overlap=use_compressor_overlap)
        else:
            self.q_compress = nn.Linear(hidden_size, self.extra_key_dim, bias=False)
            self.k_compress = nn.Linear(hidden_size, self.extra_key_dim, bias=False)

        # ---------- gate logits: main + compressor ----------
        self.gk_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
            nn.Linear(gate_low_rank_dim, self.key_dim, bias=True)
        )
        if _use_pool:
            self.gk_compress = DimGatedPoolCompressor(hidden_size, compress_ratio=scale_ratio, overlap=use_compressor_overlap)
        else:
            self.gk_compress = nn.Linear(hidden_size, self.extra_key_dim, bias=False)

        # ---------- output gate ----------
        if use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if gate_fn == 'swish' and fuse_norm and use_output_gate:
            self.g_norm_swish_gate = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            from fla.modules.activations import ACT2FN
            self.g_norm = RMSNorm(hidden_size=self.head_v_dim, eps=norm_eps)
            self.gate_fn = ACT2FN[gate_fn]

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.gate_logit_normalizer = gate_logit_normalizer

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

        # ---------- main pathway ----------
        q_main = self.q_proj(hidden_states)
        k_main = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # ---------- compressor pathway ----------
        q_extra = self.q_compress(hidden_states)
        k_extra = self.k_compress(hidden_states)

        # ---------- gate logits: main + compressor ----------
        gk_main = self.gk_proj(hidden_states)
        gk_extra = self.gk_compress(hidden_states)

        # ---------- concat main + compressed ----------
        q = torch.cat([q_main, q_extra], dim=-1)  # [B, T, K + K/r]
        k = torch.cat([k_main, k_extra], dim=-1)  # [B, T, K + K/r]
        gk = torch.cat([gk_main, gk_extra], dim=-1)  # [B, T, K + K/r]

        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask[:, -v.shape[-2]:, None])

        # reshape to multi-head
        q = rearrange(q, 'b t (h d) -> b t h d', d=self.aug_head_k_dim)
        k = rearrange(k, 'b t (h d) -> b t h d', d=self.aug_head_k_dim)
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)
        gk = rearrange(gk, 'b t (h d) -> b t h d', d=self.aug_head_k_dim)

        # gate activation
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer
        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

        # ---------- single GLA kernel call ----------
        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        cu_seqlens = kwargs.get('cu_seqlens', None)

        if mode == 'fused_recurrent':
            o, final_state = fused_recurrent_gla(
                q=q.to(torch.bfloat16),
                k=k.to(torch.bfloat16),
                v=v.to(torch.bfloat16),
                gk=gk.to(torch.bfloat16),
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        elif mode == 'chunk':
            o, final_state = chunk_gla(
                q=q.to(torch.bfloat16),
                k=k.to(torch.bfloat16),
                v=v.to(torch.bfloat16),
                g=gk.to(torch.bfloat16),
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        o = o.float()

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=final_state,
                conv_state=None,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )

        # ---------- output gate + norm ----------
        if self.use_output_gate:
            g = self.g_proj(hidden_states)
            if self.fuse_norm_and_gate:
                g = rearrange(g, 'b t (h d) -> b t h d', d=self.head_v_dim)
                o = self.g_norm_swish_gate(o, g)
                o = rearrange(o, 'b t h d -> b t (h d)')
            else:
                o = rearrange(self.g_norm(o), 'b t h d -> b t (h d)')
                o = o * self.gate_fn(g)
        else:
            o = rearrange(self.g_norm(o), 'b t h d -> b t (h d)')

        o = self.o_proj(o)
        return o

    def state_size(self, **kwargs) -> int:
        """Total state size (rectangular: K+K/r × V per head)."""
        return self.num_heads * self.aug_head_k_dim * self.head_v_dim