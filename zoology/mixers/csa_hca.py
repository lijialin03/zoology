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
from typing import Optional, Literal, Tuple

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
# gated-pooling compressor
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
        **kwargs,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.overlap = overlap and compress_ratio <= 4
        coff = 2 if self.overlap else 1

        self.wkv = nn.Linear(d_model, coff * head_dim, bias=False)
        self.wgate = nn.Linear(d_model, coff * head_dim, bias=False)
        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * head_dim))
        self.norm = nn.RMSNorm(head_dim)

        nn.init.normal_(self.ape, mean=0.0, std=0.02)

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

        # pad to multiple of ratio
        remainder = s % ratio
        pad_len = (ratio - remainder) % ratio
        if remainder > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        kv = self.wkv(x).unflatten(1, (-1, ratio))
        score = self.wgate(x).unflatten(1, (-1, ratio)) + self.ape

        if pad_len > 0:
            valid_mask = torch.zeros(b, s + pad_len, device=x.device, dtype=torch.bool)
            valid_mask[:, :s] = True
            score = score.masked_fill(~valid_mask.view(b, -1, ratio, 1), float("-inf"))

        if self.overlap:
            kv = self._overlap_transform(kv, 0.0)
            score = self._overlap_transform(score, float("-inf"))

        kv = (kv * score.softmax(dim=2)).sum(dim=2)          # [b, n_blocks, head_dim]
        kv = self.norm(kv)
        # todo: 'apply_rotary_emb'

        return kv

# ---------------------------------------------------------------------------
# indexer
# ---------------------------------------------------------------------------

class SimpleIndexer(nn.Module):
    """Lightweight Indexer for small-scale experiments.
    Strips quantization, Hadamard rotation, TP, and decode state.
    Focuses on core routing logic: compress -> score -> mask -> topk.
    """
    def __init__(
        self,
        d_model: int,
        q_lora_rank: int,
        head_dim: int,
        n_heads: int,
        compress_ratio: int = 4,
        topk: int = 8,
        overlap: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.q_lora_rank = q_lora_rank
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.compress_ratio = compress_ratio
        self.topk = topk
        self.routing_scale = (self.head_dim ** -0.5) * (self.n_heads ** -0.5)

        self.compressor = GatedPoolCompressor(
            d_model=d_model,
            head_dim=head_dim,
            compress_ratio=compress_ratio,
            overlap=overlap,
        )

        self.wq_b = nn.Linear(q_lora_rank, n_heads * head_dim, bias=False)
        self.weights_proj = nn.Linear(d_model, n_heads, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        offset: int,
    ) -> torch.Tensor:
        """
        x: [b, s, d_model] 当前层隐藏状态
        qr: [b, s, q_lora_rank] MLA 产出的低秩 Q 状态
        offset: int 压缩块在主 KV Cache 中的起始物理索引
        Returns: topk_idxs [b, s, topk] 物理 KV Cache 索引
        """
        b, s, _ = x.shape

        # 1. 直接获取压缩 KV（函数式调用，透明可微）
        score_kv = self.compressor(x)
        n_blocks = score_kv.shape[1]
        if n_blocks == 0:
            return -torch.ones(b, s, self.topk, device=x.device, dtype=torch.long)

        # 2. 生成路由 Q
        q = self.wq_b(qr)
        q = q.view(b, s, self.n_heads, self.head_dim)  # [b, s, nh, h]
        # todo: 'apply_rotary_emb'

        # 3. 计算相似度 & ReLU 截断负相关
        index_score = torch.einsum("bshd,btd->bsht", q, score_kv)  # [b, s, nh, nb]
        index_score = index_score.relu_()

        # 4. 头加权聚合
        head_weights = self.weights_proj(x) * self.routing_scale  # [b, s, nh]
        index_score = (index_score * head_weights.unsqueeze(-1)).sum(dim=2)  # [b, s, nb]

        # 5. 严格因果掩码
        block_indices = torch.arange(n_blocks, device=x.device)
        token_indices = torch.arange(1, s + 1, device=x.device).unsqueeze(1)
        causal_mask = block_indices.unsqueeze(0) >= (token_indices // self.compress_ratio)
        index_score = index_score.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

        # 6. Top-K 召回
        k = min(self.topk, n_blocks)
        topk_idxs = index_score.topk(k, dim=-1)[1]  # [b, s, k]

        # 7. 物理偏移对齐 & 越界保护
        topk_idxs = topk_idxs + offset
        invalid = topk_idxs >= (token_indices // self.compress_ratio) + offset
        topk_idxs = torch.where(invalid, -1, topk_idxs)

        # 补齐到固定 topk 长度
        if k < self.topk:
            pad = torch.full((b, s, self.topk - k), -1, dtype=torch.long, device=x.device)
            topk_idxs = torch.cat([topk_idxs, pad], dim=-1)

        return topk_idxs


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
# Compressed Attention
# ---------------------------------------------------------------------------


class SimpleCompressedAttention(nn.Module):
    """MLA 架构的压缩稀疏注意力（静态实验版）。
    严格对齐原代码: wq_a→q_norm→wq_b (Q低秩), wkv→kv_norm (KV共享MQA), 
    配合 GatedPoolCompressor + SimpleIndexer / 确定性采样。
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        compress_ratio: int,
        topk: int = None,
        routing_mode: str = "CSA",  # "CSA" | "HCA"
        window_size: int = 32,
        overlap: bool = False,
        q_lora_rank: Optional[int] = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.compress_ratio = compress_ratio
        self.topk = topk
        self.window_size = window_size
        self.routing_mode = routing_mode
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        self.eps = 1e-6

        # MLA 核心投影层（替代标准 MHA 的 q/k/v_proj）
        q_lr = q_lora_rank if q_lora_rank is not None else d_model // 4  # 默认 1/4 降维，与原代码 4096->1024 一致
        self.wq_a = nn.Linear(d_model, q_lr, bias=False)
        self.q_norm = nn.RMSNorm(q_lr, eps=self.eps)
        self.wq_b = nn.Linear(q_lr, d_model, bias=False)  # q_lr -> n_heads * head_dim

        # KV 共享投影 (MQA 风格)
        self.wkv = nn.Linear(d_model, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=self.eps)

        # 输出投影（简化版，可替换为原代码的分组低秩 wo_a+wo_b）
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # 压缩器
        self.compressor = GatedPoolCompressor(
            d_model=d_model,
            head_dim=self.head_dim,
            compress_ratio=compress_ratio,
            overlap=overlap,
        )

        # 路由索引器（仅 CSA 使用）
        if routing_mode == "CSA":
            self.indexer = SimpleIndexer(
                d_model=d_model,
                q_lora_rank=q_lr,
                head_dim=self.head_dim,
                n_heads=n_heads,
                compress_ratio=compress_ratio,
                topk=topk,
                overlap=overlap,
            )
        else:
            self.indexer = None
    
    def _get_window_idxs(self, s: int, device: torch.device) -> torch.Tensor:
        """[1, S, window_size] 滑动窗口索引，-1 表示越界"""
        q_idx = torch.arange(s, device=device).view(1, s, 1)
        w_idx = torch.arange(self.window_size, device=device).view(1, 1, self.window_size)
        idx = q_idx - self.window_size + 1 + w_idx
        mask = idx >= 0
        return torch.where(mask, idx, torch.tensor(-1, device=device, dtype=torch.long))

    def _build_selected_kv(self, kv_window, kv_compress, topk_idxs, s):
        """修正签名，移除多余参数"""
        b, h, _, d = kv_window.shape
        # kv_compress: [B, 1, n_blocks, Dh] -> expand to [B, H, n_blocks, Dh]
        kv_compress_exp = kv_compress.expand(-1, h, -1, -1)
        kv_full = torch.cat([kv_window, kv_compress_exp], dim=2)  # [B, H, S+n_blocks, Dh]
        valid_mask = topk_idxs >= 0
        safe_idxs = topk_idxs.clamp(min=0)
        idxs_exp = safe_idxs.unsqueeze(1).unsqueeze(-1).expand(b, h, s, -1, d)
        kv_full_exp = kv_full.unsqueeze(2).expand(b, h, s, -1, d)
        kv_selected = torch.gather(kv_full_exp, 3, idxs_exp)
        mask = valid_mask.unsqueeze(1)  # [B, 1, S, K]
        return kv_selected, mask

    def _get_all_compress_idxs(self, s: int, n_blocks: int, offset: int, device: torch.device) -> torch.Tensor:
        """
        返回每个 query 可见的所有历史压缩块索引（含因果掩码）。
        形状: [1, s, n_blocks] （后续 expand 到 batch）
        """
        if n_blocks == 0:
            return torch.empty(1, s, 0, dtype=torch.long, device=device)
        
        # query 索引（0..s-1） -> 对应的最大块索引（含当前块）
        # 注意：原代码中 causal mask 是 block_idx < ceil((i+1)/ratio)
        # 这里简化为 block_idx <= i // ratio
        q_idx = torch.arange(s, device=device).view(1, s, 1)          # [1, s, 1]
        block_idx = torch.arange(n_blocks, device=device).view(1, 1, n_blocks)  # [1, 1, n_blocks]
        # 因果掩码：block_idx <= q_idx // ratio
        mask = block_idx <= (q_idx // self.compress_ratio)
        
        # 有效位置填 block_idx + offset，无效填 -1
        idxs = torch.where(mask, block_idx + offset, torch.tensor(-1, device=device, dtype=torch.long))
        return idxs  # [1, s, n_blocks]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape

        # ================= MLA Q 路径 =================
        # 1. 低秩投影 + 归一化 -> qr: [B, S, q_lr]
        qr = self.q_norm(self.wq_a(x))
        # 2. 升维至多头 -> q: [B, H, S, Dh]
        q = self.wq_b(qr).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        # 3. 分布稳定化 (原代码: q *= torch.rsqrt(...))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)

        # ================= MLA KV 路径 (共享 K/V 投影 + 归一化) =================
        kv = self.kv_norm(self.wkv(x))                     # [B, S, Dh]
        kv_window = kv.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # [B, H, S, Dh]

        # 2. 压缩 KV [B, n_blocks, Dh] -> 扩展为 [B, 1, n_blocks, Dh]
        kv_comp = self.compressor(x)
        n_blocks = kv_comp.shape[1]
        kv_comp = kv_comp.unsqueeze(1) if n_blocks > 0 else torch.empty(b, 1, 0, self.head_dim, device=x.device)

        # 3. 生成路由索引
        offset = s  # 压缩块在拼接缓存中的起始物理位置
        if self.routing_mode == "CSA" and self.indexer is not None:
            compress_idxs = self.indexer(x, qr, offset=offset)
        else:
            compress_idxs = self._get_all_compress_idxs(s, n_blocks, offset, x.device)
            compress_idxs = compress_idxs.expand(b, -1, -1)         # [B, S, n_blocks]

        # 4. 窗口索引（因果三角）
        window_idxs = self._get_window_idxs(s, x.device)  # [1, S, window_size]
        window_idxs = window_idxs.expand(b, -1, -1)      # [B, S, window_size]

        # 5. 拼接所有候选索引 [B, S, win+topk]
        all_idxs = torch.cat([window_idxs, compress_idxs], dim=-1)

        # ================= 稀疏注意力计算 =================
        kv_selected, mask = self._build_selected_kv(kv_window, kv_comp, all_idxs, s)

        # 7. 稀疏注意力计算（真正只算选中的 KV！）
        out = _attend_selected(q, kv_selected, kv_selected, mask, self.scale, self.dropout)

        # 8. 输出投影
        out = out.transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(out)

    def state_size(self, sequence_length: int = 2048):
        n_comp = (sequence_length + self.compress_ratio - 1) // self.compress_ratio
        return (self.window_size + n_comp) * self.head_dim

# ==============================================================================
# CSA – Compressed Sparse Attention
# HCA – Heavily Compressed Attention
# ==============================================================================
class SimpleCSA(SimpleCompressedAttention):
    """Compressed Sparse Attention (ratio=4, CSA routing, overlap)"""
    def __init__(self, d_model: int, num_heads: int, index_topk: int = 4, window_size: int = 32, **kwargs):
        super().__init__(
            d_model=d_model,
            n_heads=num_heads, 
            compress_ratio=4, 
            topk=index_topk,
            routing_mode="CSA", 
            window_size=window_size, 
            overlap=True
        )

class SimpleHCA(SimpleCompressedAttention):
    """High-compression Attention (ratio=128, HCA routing, no overlap)"""
    def __init__(self, d_model: int, num_heads: int, window_size: int = 32, **kwargs):
        super().__init__(
            d_model=d_model, 
            n_heads=num_heads, 
            compress_ratio=128,
            routing_mode="HCA", 
            window_size=window_size, 
            overlap=False
        )

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


if __name__ == "__main__":
    def sanity_test(name: str, cls, seqlen=32):
        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cls(d_model=256, num_heads=8, topk=4, window_size=16).to(device)
        x = torch.randn(2, seqlen, 256, device=device)

        # 前向 & 形状
        out = model(x)
        assert out.shape == (2, seqlen, 256)
        assert torch.isfinite(out).all()

        # 梯度检查（多个参数）
        x_grad = torch.randn(2, seqlen, 256, requires_grad=True, device=device)
        model(x_grad).sum().backward()
        assert model.wq_a.weight.grad is not None
        assert model.wq_b.weight.grad is not None
        assert model.wkv.weight.grad is not None
        # compressor 的参数如果有梯度也应检查（取决于是否可微）
        # indexer 的参数可能无梯度（预期），不强制要求

        print(f"[{name}] seqlen={seqlen} OK")

    # 额外测试 seqlen > compress_ratio 的情况
    sanity_test("CSA", SimpleCSA, seqlen=64)   # compress_ratio=4 -> 16 blocks
    sanity_test("HCA", SimpleHCA, seqlen=256)  # compress_ratio=128 -> 2 blocks
