"""
CLA – Compressed Linear Attention.

Fuses CSA's compress+indexer mechanism with DeltaNet's linear (delta-rule)
attention, creating an architecture with:
  - Sparse compression for sequence reduction
  - Linear attention (O(d²) recurrent state) for efficient long-range modeling

3-Pathway Design:
  Pathway 1: Sliding Window (LocalAttention + RoPE)
  Pathway 2: Compressed Linear Attention (avg-pool Q/K/V → DeltaNet → expand)
  Pathway 3: Selected Linear Attention (Indexer top-k → gather DeltaNet output → weighted sum)

Output = g1*out1 + g2*out2 + g3*out3 → out_proj
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule

from zoology.mixers.csa_hca import GatedPoolCompressor, RotaryEmbedding
from zoology.mixers.deepseek.local_attention import LocalAttention


class CompressedLinearAttention(nn.Module):
    """Compressed Linear Attention with 3-pathway design.

    Pathway 1 – Sliding window attention (local, fine-grained).
    Pathway 2 – Compressed linear attention via DeltaNet on avg-pooled tokens.
    Pathway 3 – Selected linear attention via Indexer top-k on DeltaNet output.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        window_size: Sliding window size for pathway 1.
        compress_ratio: Compression factor for pathways 2 & 3.
        index_topk: Number of selected compressed blocks for pathway 3.
        expand_k: DeltaNet key dimension expansion ratio.
        expand_v: DeltaNet value dimension expansion ratio.
        dropout: Attention dropout rate.
        use_beta: Whether to use learned beta gating in DeltaNet.
        rope_base: RoPE base frequency.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        window_size: int = 16,
        compress_ratio: int = 4,
        index_topk: int = 4,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        dropout: float = 0.0,
        use_beta: bool = True,
        rope_base: float = 10000.0,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        self.index_topk = index_topk
        self.use_beta = use_beta

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        assert self.key_dim % num_heads == 0, "key_dim must be divisible by num_heads"
        assert self.value_dim % num_heads == 0, "value_dim must be divisible by num_heads"

        # --- Pathway 1: sliding window QKV ---
        self.Wq = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(d_model, num_heads * self.head_dim, bias=False)

        # --- Pathways 2 & 3: DeltaNet QKV ---
        self.Wq_delta = nn.Linear(d_model, self.key_dim, bias=False)
        self.Wk_delta = nn.Linear(d_model, self.key_dim, bias=False)
        self.Wv_delta = nn.Linear(d_model, self.value_dim, bias=False)

        if use_beta:
            self.b_proj = nn.Linear(d_model, num_heads, bias=False)

        # --- Output projections ---
        self.out_proj = nn.Linear(d_model, d_model)
        self.out_proj_delta = nn.Linear(self.value_dim, d_model)

        # --- RoPE (pathway 1 only) ---
        self.rotary = RotaryEmbedding(self.head_dim, base=rope_base)

        # --- Indexer (pathway 3) ---
        self.indexer_q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.indexer_w = nn.Linear(d_model, num_heads, bias=False)
        self.indexer_compressor = GatedPoolCompressor(
            d_model, num_heads * self.head_dim, compress_ratio, overlap=True
        )

        # --- Pathway combination gates ---
        # Initialize to favor the sliding-window pathway
        gate_proj = nn.Linear(d_model, 3 * num_heads, bias=True)
        nn.init.zeros_(gate_proj.weight)
        with torch.no_grad():
            gate_proj.bias.copy_(torch.tensor([-2.0, -2.0, 2.0] * num_heads))
        self.gate = nn.Sequential(gate_proj, nn.Sigmoid())

        # --- Sliding window ---
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
    def _delta_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Run DeltaNet forward pass.

        Args:
            q, k: [batch, seq, heads, head_k_dim]
            v:   [batch, seq, heads, head_v_dim]
            beta: [batch, seq, heads]

        Returns:
            [batch, seq, heads, head_v_dim]
        """
        seq_len = q.shape[1]
        mode = 'fused_recurrent' if seq_len <= 64 else 'chunk'

        # Apply SiLU activation (DeltaNet default)
        q = F.silu(q)
        k = F.silu(k)
        v = F.silu(v)

        if mode == 'fused_recurrent':
            o, _ = fused_recurrent_delta_rule(
                q=q.to(torch.bfloat16),
                k=k.to(torch.bfloat16),
                v=v.to(torch.bfloat16),
                beta=beta,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            o, _ = chunk_delta_rule(
                q=q.to(torch.bfloat16),
                k=k.to(torch.bfloat16),
                v=v.to(torch.bfloat16),
                beta=beta,
                use_qk_l2norm_in_kernel=True,
            )
        return o.float()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        b, s, d = x.shape
        h, hd = self.num_heads, self.head_dim
        ratio = self.compress_ratio
        device = x.device

        # ================================================================
        # Pathway 1 – Sliding Window
        # ================================================================
        q1 = self.Wq(x).view(b, s, h, hd).transpose(1, 2)  # [b, h, s, hd]
        k1 = self.Wk(x).view(b, s, h, hd).transpose(1, 2)
        v1 = self.Wv(x).view(b, s, h, hd).transpose(1, 2)

        q1 = self.rotary(q1)
        k1 = self.rotary(k1)

        # LocalAttention expects [batch, seq, heads, dim]
        out1 = self.sliding_window(
            q1.transpose(1, 2), k1.transpose(1, 2), v1.transpose(1, 2)
        )
        out1 = out1.transpose(1, 2)  # [b, h, s, hd]

        # ================================================================
        # Shared – compress Q/K/V and run DeltaNet (for pathways 2 & 3)
        # ================================================================
        q2 = self.Wq_delta(x).view(b, s, h, self.head_k_dim)
        k2 = self.Wk_delta(x).view(b, s, h, self.head_k_dim)
        v2 = self.Wv_delta(x).view(b, s, h, self.head_v_dim)

        # Pad sequence to a multiple of compress_ratio
        remainder = s % ratio
        pad_len = ratio - remainder if remainder > 0 else 0
        if remainder > 0:
            q2 = F.pad(q2, (0, 0, 0, 0, 0, pad_len))
            k2 = F.pad(k2, (0, 0, 0, 0, 0, pad_len))
            v2 = F.pad(v2, (0, 0, 0, 0, 0, pad_len))

        s_pad = q2.shape[1]
        n_comp = s_pad // ratio

        # Avg-pool compression along the sequence dimension
        q_comp = q2.view(b, n_comp, ratio, h, self.head_k_dim).mean(dim=2)
        k_comp = k2.view(b, n_comp, ratio, h, self.head_k_dim).mean(dim=2)
        v_comp = v2.view(b, n_comp, ratio, h, self.head_v_dim).mean(dim=2)

        # Beta for compressed sequence
        if self.use_beta:
            if remainder > 0:
                x_pad = F.pad(x, (0, 0, 0, pad_len))
                x_comp = x_pad.view(b, n_comp, ratio, d).mean(dim=2)
            else:
                x_comp = x.view(b, n_comp, ratio, d).mean(dim=2)
            beta = self.b_proj(x_comp).sigmoid()  # [b, n_comp, h]
        else:
            beta = q_comp.new_ones(b, n_comp, h)

        # DeltaNet on compressed tokens
        o_comp = self._delta_forward(q_comp, k_comp, v_comp, beta)
        # o_comp: [b, n_comp, h, head_v_dim]

        # ================================================================
        # Pathway 2 – Compressed Linear Attention (all blocks)
        # ================================================================
        o2 = o_comp.repeat_interleave(ratio, dim=1)  # [b, s_pad, h, head_v_dim]
        if remainder > 0:
            o2 = o2[:, :s]
        out2 = (
            self.out_proj_delta(o2.reshape(b, s, self.value_dim))
            .view(b, s, h, hd)
            .transpose(1, 2)
        )  # [b, h, s, hd]

        # ================================================================
        # Pathway 3 – Selected Linear Attention via Indexer
        # ================================================================
        q_idx = self.indexer_q(x).view(b, s, h, hd).transpose(1, 2)  # [b, h, s, hd]
        idx_comp = (
            self.indexer_compressor(x).view(b, n_comp, h, hd).transpose(1, 2)
        )  # [b, h, n_comp, hd]

        # Indexer scores: per-head dot-product between query and compressed index
        idx_scores = (
            torch.einsum("b h s d, b h t d -> b h s t", q_idx, idx_comp)
            * self.softmax_scale
        )
        idx_w = self.indexer_w(x).transpose(1, 2).unsqueeze(-1)  # [b, h, s, 1]
        idx_scores = idx_scores.relu() * idx_w * (h ** -0.5)

        # Causal mask at block level: query at position i can only see
        # compressed blocks whose last token position < i
        qpos = torch.arange(s, device=device).view(1, 1, s, 1)
        cpos = torch.arange(n_comp, device=device).view(1, 1, 1, n_comp)
        idx_scores = idx_scores.masked_fill(
            cpos >= ((qpos + 1) // ratio), float("-inf")
        )

        # Select top-k compressed blocks
        num_sel = min(self.index_topk, n_comp)
        sel_scores, topk_idxs = idx_scores.topk(num_sel, dim=-1)
        # sel_scores: [b, h, s, num_sel], topk_idxs: [b, h, s, num_sel]

        # Gather DeltaNet compressed output for selected blocks
        b_idx = torch.arange(b, device=device).view(b, 1, 1, 1)
        h_idx = torch.arange(h, device=device).view(1, h, 1, 1)
        o3_sel = o_comp[b_idx, topk_idxs, h_idx]  # [b, h, s, num_sel, head_v_dim]

        # Weighted sum using indexer scores as weights (handle -inf from causal mask)
        valid_mask = ~torch.isinf(sel_scores)  # [b, h, s, num_sel]
        sel_scores_safe = sel_scores.masked_fill(~valid_mask, 0.0)
        sel_weights = sel_scores_safe.softmax(dim=-1)  # [b, h, s, num_sel]
        sel_weights = sel_weights * valid_mask.float()
        sel_weights = sel_weights / (sel_weights.sum(dim=-1, keepdim=True) + 1e-8)
        sel_weights = sel_weights.unsqueeze(-1)  # [b, h, s, num_sel, 1]
        o3 = (sel_weights * o3_sel).sum(dim=-2)  # [b, h, s, head_v_dim]

        out3 = (
            self.out_proj_delta(o3.transpose(1, 2).reshape(b, s, self.value_dim))
            .view(b, s, h, hd)
            .transpose(1, 2)
        )  # [b, h, s, hd]

        # ================================================================
        # Combine pathways via learned scalar gates
        # ================================================================
        gates = (
            self.gate(x).view(b, s, h, 3).permute(0, 3, 2, 1).unsqueeze(-1)
        )  # [b, 3, h, s, 1]
        g1, g2, g3 = gates[:, 0], gates[:, 1], gates[:, 2]
        out = g1 * out1 + g2 * out2 + g3 * out3

        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.out_proj(out)

    # ------------------------------------------------------------------
    def state_size(self, sequence_length: int = 2048) -> int:
        """Compute total state size (for memory accounting).

        Returns:
            State size = window_size * d_model * 2 (sliding window KV cache)
                       + num_heads * head_k_dim * head_v_dim (DeltaNet O(d²) state)
        """
        return (
            self.window_size * self.d_model * 2
            + self.num_heads * self.head_k_dim * self.head_v_dim
        )