"""
CSA (Compressed Sparse Attention) and HCA (Heavily Compressed Attention) mixers.

Adapted from the DeepSeek-V4 model.py attention design:
  - CSA: compress_ratio=4, overlapping gated pooling + Indexer top-k selection
  - HCA: compress_ratio=128, non-overlapping gated pooling, no sparse selection

Both use a multi-pathway design:
  pathway 1 – sliding-window attention (local, fine-grained)
  pathway 2 – compressed attention (all compressed blocks, long-range coarse)
  pathway 3 – (CSA only) selected attention (top-k compressed blocks, sparse fine)

Pathway outputs are fused via learned scalar weights (sigmoid gates).

Changes from the original version:
  - Added RotaryEmbedding (RoPE) for Q/K
  - Added GQA support (num_kv_heads)
  - Fixed Pathway 3 to use per-query selected attention (_attend_selected)
    instead of flattening all selected blocks into a shared dense KV sequence
  - Added token-level causal safety mask for Pathway 3
  - Gate bias initialized to favor the sliding-window pathway
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from zoology.mixers.deepseek.local_attention import LocalAttention


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Lightweight RoPE for head-first tensors [B, H, T, D]."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x: [batch, heads, seq_len, dim] (head-first)
        seq_len = x.shape[2]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype) + offset
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().view(1, seq_len, 1, -1)
        sin = emb.sin().view(1, seq_len, 1, -1)
        # rotate
        x = x.transpose(1, 2)  # [batch, seq_len, heads, dim]
        x1, x2 = x[..., ::2], x[..., 1::2]
        rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
        out = x * cos + rot * sin
        return out.transpose(1, 2)


# ---------------------------------------------------------------------------
# Shared gated-pooling compressor
# ---------------------------------------------------------------------------

class GatedPoolCompressor(nn.Module):
    """Compresses a sequence via learned gated pooling over ``compress_ratio``
    consecutive tokens.

    When *overlap* is True (ratio <= 4), adjacent compression windows share one
    token for smoother boundaries.  This mirrors ``model.py:Compressor`` without
    quantization, RoPE, Hadamard rotation, or incremental-decode state.
    """

    def __init__(
        self,
        d_model: int,
        head_dim: int,
        compress_ratio: int = 4,
        overlap: bool = False,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.overlap = overlap and compress_ratio <= 4
        coff = 2 if self.overlap else 1

        self.wkv = nn.Linear(d_model, coff * head_dim, bias=False)
        self.wgate = nn.Linear(d_model, coff * head_dim, bias=False)
        self.ape = nn.Parameter(torch.zeros(compress_ratio, coff * head_dim))
        self.norm = nn.LayerNorm(head_dim)

    # ------------------------------------------------------------------
    def _overlap_transform(self, x: torch.Tensor, fill_value: float):
        """Shift the first half of dims one block forward (overlapping windows)."""
        b, n, r, coff_d = x.shape
        d = self.head_dim
        new = x.new_full((b, n, 2 * r, d), fill_value)
        new[:, :, r:, :] = x[:, :, :, d:]        # normal half
        new[:, 1:, :r, :] = x[:, :-1, :, :d]     # overlapping half (shifted)
        return new

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [b, s, d_model]  ->  compressed: [b, n_blocks, head_dim]"""
        b, s, _ = x.shape
        ratio = self.compress_ratio
        coff = 2 if self.overlap else 1

        # pad to multiple of ratio
        remainder = s % ratio
        if remainder > 0:
            x = F.pad(x, (0, 0, 0, ratio - remainder))

        s_pad = x.shape[1]
        n_blocks = s_pad // ratio

        kv = self.wkv(x).view(b, n_blocks, ratio, coff * self.head_dim)
        score = self.wgate(x).view(b, n_blocks, ratio, coff * self.head_dim) + self.ape

        if self.overlap and n_blocks >= 2:
            kv = self._overlap_transform(kv, 0.0)
            score = self._overlap_transform(score, float("-inf"))

        weights = score.softmax(dim=2)
        compressed = (kv * weights).sum(dim=2)          # [b, n_blocks, coff * head_dim]

        # When overlap=True but n_blocks < 2, the output still has coff*head_dim
        # dims (the overlap transform was skipped). Reduce to head_dim via a
        # learned projection so LayerNorm receives the correct shape.
        if self.overlap and n_blocks < 2:
            compressed = compressed.view(b, n_blocks, 2, self.head_dim).sum(dim=2)

        return self.norm(compressed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _causal_comp_mask(s: int, n_comp: int, ratio: int, device: torch.device) -> torch.Tensor:
    """Mask for compressed attention: block j visible when j < ceil((i+1)/ratio)."""
    q_idx = torch.arange(s, device=device).view(1, s, 1)
    c_idx = torch.arange(n_comp, device=device).view(1, 1, n_comp)
    return c_idx < ((q_idx + 1) // ratio)


def _attend(q, k, v, mask, scale, dropout):
    """Standard scaled dot-product attention with mask.

    Supports GQA by repeating k/v along the head dimension when
    q.shape[1] != k.shape[1].

    Returns zeros for query positions that have no visible KV tokens
    (e.g. first ratio tokens in the compressed pathway).
    """
    # GQA: repeat k/v if needed
    if q.shape[1] != k.shape[1]:
        g = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)

    scores = torch.einsum("bhsd,bhzd->bhsz", q, k) * scale
    scores = scores.masked_fill(~mask, float("-inf"))

    # detect rows that are entirely -inf (no visible tokens)
    all_masked = (~mask).all(dim=-1, keepdim=True)  # [b, 1, s, 1]

    # replace all-inf rows with zeros so softmax doesn't produce NaN
    safe_scores = scores.masked_fill(all_masked, 0.0)
    attn = safe_scores.softmax(dim=-1)
    attn = attn.masked_fill(all_masked, 0.0)  # zero out the artificial softmax
    attn = dropout(attn)
    return torch.einsum("bhsz,bhzd->bhsd", attn, v)


def _attend_selected(q, k, v, mask, scale, dropout):
    """Per-query selected attention.

    Each query token only attends to its own selected KV set,
    rather than a shared global KV sequence.

    q   : [b, h, s, d]
    k/v : [b, h, s, n_sel, d]
    mask: [b, 1, s, n_sel] or broadcastable
    """
    # GQA: repeat k/v if needed
    if q.shape[1] != k.shape[1]:
        g = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)

    scores = torch.einsum("bhsd,bhsnd->bhsn", q, k) * scale
    scores = scores.masked_fill(~mask, float("-inf"))

    all_masked = (~mask).all(dim=-1, keepdim=True)
    safe_scores = scores.masked_fill(all_masked, 0.0)
    attn = safe_scores.softmax(dim=-1)
    attn = attn.masked_fill(all_masked, 0.0)
    attn = dropout(attn)
    return torch.einsum("bhsn,bhsnd->bhsd", attn, v)


# ---------------------------------------------------------------------------
# CSA – Compressed Sparse Attention
# ---------------------------------------------------------------------------

class CSAttention(nn.Module):
    """Compressed Sparse Attention.

    Three pathways:
      1. sliding-window attention  (local, fine)
      2. compressed attention      (all compressed blocks, coarse)
      3. selected attention        (top-k compressed blocks via Indexer, sparse)

    Output = gate_1 * path_1 + gate_2 * path_2 + gate_3 * path_3
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        num_kv_heads: int | None = None,
        window_size: int = 16,
        compress_ratio: int = 4,
        index_topk: int = 8,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        self.index_topk = index_topk

        assert d_model % num_heads == 0
        assert num_heads % self.num_kv_heads == 0

        self.Wq = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rotary = RotaryEmbedding(self.head_dim, base=rope_base)

        # --- compressor (shared K+V for pathways 2 & 3) ---
        # In GQA mode the compressor outputs num_kv_heads * head_dim dims.
        self.compressor = GatedPoolCompressor(
            d_model, self.num_kv_heads * self.head_dim, compress_ratio, overlap=True
        )

        # --- Indexer (pathway 3) ---
        self.indexer_q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.indexer_w = nn.Linear(d_model, num_heads, bias=False)
        self.indexer_compressor = GatedPoolCompressor(
            d_model, self.num_kv_heads * self.head_dim, compress_ratio, overlap=True
        )

        # --- pathway combination gates ---
        # Initialize to favor the sliding-window pathway (local patterns first)
        gate_proj = nn.Linear(d_model, 3 * num_heads, bias=True)
        nn.init.zeros_(gate_proj.weight)
        with torch.no_grad():
            gate_proj.bias.copy_(torch.tensor([-2.0, -2.0, 2.0] * num_heads))
        self.gate = nn.Sequential(gate_proj, nn.Sigmoid())

        # --- sliding window (LocalAttention from zoology) ---
        self.sliding_window = LocalAttention(
            dim=self.head_dim,
            window_size=window_size,
            causal=True,
            exact_windowsize=True,
            autopad=True,
            use_rotary_pos_emb=False,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.softmax_scale = self.head_dim ** -0.5

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs):
        b, s, d = x.shape
        h, hk, hd = self.num_heads, self.num_kv_heads, self.head_dim
        ratio = self.compress_ratio
        win = self.window_size
        device = x.device

        # --- QKV projection ---
        q = self.Wq(x).view(b, s, h, hd).transpose(1, 2)      # [b, h, s, hd]
        k = self.Wk(x).view(b, s, hk, hd).transpose(1, 2)     # [b, hk, s, hd]
        v = self.Wv(x).view(b, s, hk, hd).transpose(1, 2)     # [b, hk, s, hd]

        # --- RoPE ---
        q = self.rotary(q)
        k = self.rotary(k)

        # --- shared compressed KV ---
        kv_comp = self.compressor(x)                   # [b, n, hk*hd]
        n_comp = kv_comp.shape[1]
        k_comp = kv_comp.view(b, n_comp, hk, hd).transpose(1, 2)
        v_comp = kv_comp.view(b, n_comp, hk, hd).transpose(1, 2)

        # ================================================================
        # Pathway 1 – sliding window (LocalAttention)
        # ================================================================
        # LocalAttention expects [batch, seq, heads, dim]
        q_seq = q.transpose(1, 2)
        k_seq = k.transpose(1, 2)
        v_seq = v.transpose(1, 2)
        if self.num_kv_groups > 1:
            k_seq = k_seq.repeat_interleave(self.num_kv_groups, dim=2)
            v_seq = v_seq.repeat_interleave(self.num_kv_groups, dim=2)
        out1 = self.sliding_window(q_seq, k_seq, v_seq)
        out1 = out1.transpose(1, 2)  # [b, h, s, hd]

        # ================================================================
        # Pathway 2 – compressed (all blocks)
        # ================================================================
        mask2 = _causal_comp_mask(s, n_comp, ratio, device)  # [1, s, n_comp]
        out2 = _attend(q, k_comp, v_comp, mask2, self.softmax_scale, self.dropout)

        # ================================================================
        # Pathway 3 – selected (top-k compressed blocks via Indexer)
        # ================================================================
        q_idx = self.indexer_q(x).view(b, s, h, hd).transpose(1, 2)
        idx_comp = self.indexer_compressor(x).view(b, n_comp, hk, hd).transpose(1, 2)

        # einsum supports different query / kv head counts
        idx_scores = torch.einsum("b h s d, b k t d -> b h s t", q_idx, idx_comp) * self.softmax_scale
        idx_w = self.indexer_w(x).transpose(1, 2).unsqueeze(-1)     # [b, h, s, 1]
        idx_scores = idx_scores.relu() * idx_w * (h ** -0.5)

        # causal mask at block level
        qpos = torch.arange(s, device=device).view(1, 1, s, 1)
        cpos = torch.arange(n_comp, device=device).view(1, 1, 1, n_comp)
        idx_scores = idx_scores.masked_fill(cpos >= ((qpos + 1) // ratio), float("-inf"))

        num_sel = min(self.index_topk, n_comp)
        _, topk_idxs = idx_scores.topk(num_sel, dim=-1)             # [b, h, s, num_sel]

        # gather per-head selected compressed KV
        b_idx = torch.arange(b, device=device).view(b, 1, 1, 1)
        h_idx = torch.arange(h, device=device).view(1, h, 1, 1)
        # expand compressed KV to query-head dimension for gathering
        k_comp_h = (
            k_comp.repeat_interleave(self.num_kv_groups, dim=1)
            if self.num_kv_groups > 1
            else k_comp
        )
        v_comp_h = (
            v_comp.repeat_interleave(self.num_kv_groups, dim=1)
            if self.num_kv_groups > 1
            else v_comp
        )
        k_sel = k_comp_h[b_idx, h_idx, topk_idxs, :]                  # [b, h, s, num_sel, hd]
        v_sel = v_comp_h[b_idx, h_idx, topk_idxs, :]

        # token-level causal safety mask:
        # each selected block covers tokens [idx*ratio, (idx+1)*ratio-1]
        # the query may only attend to blocks whose last token is <= query position
        block_last_pos = (topk_idxs + 1) * ratio - 1                  # [b, h, s, num_sel]
        q_pos = torch.arange(s, device=device).view(1, 1, s, 1)
        sel_mask = block_last_pos <= q_pos                            # [b, h, s, num_sel]

        out3 = _attend_selected(q, k_sel, v_sel, sel_mask, self.softmax_scale, self.dropout)

        # ================================================================
        # Combine pathways
        # ================================================================
        gates = self.gate(x).view(b, s, h, 3).permute(0, 3, 2, 1).unsqueeze(-1)  # [b, 3, h, s, 1]
        g1, g2, g3 = gates[:, 0], gates[:, 1], gates[:, 2]
        out = g1 * out1 + g2 * out2 + g3 * out3

        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.out_proj(out)

    # ------------------------------------------------------------------
    def state_size(self, sequence_length: int = 2048):
        n_comp = sequence_length // self.compress_ratio
        return (self.window_size + n_comp + self.index_topk) * self.d_model * 2


# ---------------------------------------------------------------------------
# HCA – Heavily Compressed Attention
# ---------------------------------------------------------------------------

class HCAAttention(nn.Module):
    """Heavily Compressed Attention.

    Two pathways:
      1. sliding-window attention  (local, fine)
      2. compressed attention      (all compressed blocks, coarse)

    No Indexer – the compression ratio is large enough that all blocks fit.
    Output = gate_1 * path_1 + gate_2 * path_2
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        num_kv_heads: int | None = None,
        window_size: int = 16,
        compress_ratio: int = 128,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.compress_ratio = compress_ratio

        assert d_model % num_heads == 0
        assert num_heads % self.num_kv_heads == 0

        self.Wq = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.compressor = GatedPoolCompressor(
            d_model, self.num_kv_heads * self.head_dim, compress_ratio, overlap=False
        )
        self.out_proj = nn.Linear(d_model, d_model)

        self.rotary = RotaryEmbedding(self.head_dim, base=rope_base)

        # --- sliding window (LocalAttention from zoology) ---
        self.sliding_window = LocalAttention(
            dim=self.head_dim,
            window_size=window_size,
            causal=True,
            exact_windowsize=True,
            autopad=True,
            use_rotary_pos_emb=False,
            dropout=dropout,
        )

        # pathway combination (2-pathway); favor sliding window initially
        gate_proj = nn.Linear(d_model, 2 * num_heads, bias=True)
        nn.init.zeros_(gate_proj.weight)
        with torch.no_grad():
            gate_proj.bias.copy_(torch.tensor([-2.0, 2.0] * num_heads))
        self.gate = nn.Sequential(gate_proj, nn.Sigmoid())

        self.dropout = nn.Dropout(dropout)
        self.softmax_scale = self.head_dim ** -0.5

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs):
        b, s, d = x.shape
        h, hk, hd = self.num_heads, self.num_kv_heads, self.head_dim
        ratio = self.compress_ratio
        win = self.window_size
        device = x.device

        q = self.Wq(x).view(b, s, h, hd).transpose(1, 2)
        k = self.Wk(x).view(b, s, hk, hd).transpose(1, 2)
        v = self.Wv(x).view(b, s, hk, hd).transpose(1, 2)

        q = self.rotary(q)
        k = self.rotary(k)

        # shared compressed KV
        kv_comp = self.compressor(x)                   # [b, n, hk*hd]
        n_comp = kv_comp.shape[1]
        k_comp = kv_comp.view(b, n_comp, hk, hd).transpose(1, 2)
        v_comp = kv_comp.view(b, n_comp, hk, hd).transpose(1, 2)

        # Pathway 1 – sliding window (LocalAttention)
        q_seq = q.transpose(1, 2)
        k_seq = k.transpose(1, 2)
        v_seq = v.transpose(1, 2)
        if self.num_kv_groups > 1:
            k_seq = k_seq.repeat_interleave(self.num_kv_groups, dim=2)
            v_seq = v_seq.repeat_interleave(self.num_kv_groups, dim=2)
        out1 = self.sliding_window(q_seq, k_seq, v_seq)
        out1 = out1.transpose(1, 2)  # [b, h, s, hd]

        # Pathway 2 – compressed
        mask2 = _causal_comp_mask(s, n_comp, ratio, device)
        out2 = _attend(q, k_comp, v_comp, mask2, self.softmax_scale, self.dropout)

        # Combine
        gates = self.gate(x).view(b, s, h, 2).permute(0, 3, 2, 1).unsqueeze(-1)  # [b, 2, h, s, 1]
        g1, g2 = gates[:, 0], gates[:, 1]
        out = g1 * out1 + g2 * out2

        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.out_proj(out)

    # ------------------------------------------------------------------
    def state_size(self, sequence_length: int = 2048):
        n_comp = max(sequence_length // self.compress_ratio, 1)
        return (self.window_size + n_comp) * self.d_model * 2
